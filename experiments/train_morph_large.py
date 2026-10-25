"""Train MorphNet-L (wavelet2vec.morph_large): vox-synth hybrids with a pretrained Swin-L encoder, on the CPU.

Version 2 of the vox-synth morph, built from the small run's lessons
(docs/MORPH_MODEL.md):

Phase 0 (GPU if available, minutes): judge-view cache (shared with v1) and a
critic-view cache (stereo, up to 20 kHz) of every note in MIDI 36-82.

Phase 1 (GPU if available, minutes): calibrated judges. JudgeCNN and JudgeTCN,
wider than v1, with label smoothing (logits stay finite) and a supervised
contrastive term on their embeddings. The class centres of the embeddings
define a "vocality" axis (synth centre 0, vox centre 1). Leave-one-singer-out
check. The held-out judge J (Wavelet2Vec + logistic) is reused from v1 (same
splits).

Phase 2 (CPU): MorphNet-L, encoder warm-started from KeyboardNet v3. Each pair
gets a random pitch source (synth or vocal: the hybrid's partials keep that
note's phase, so its pitch is as stable as the source's) and is decoded at
alpha = 0.5 and at a second random alpha. Losses per hybrid:
- vocality (both judges) -> alpha, plus a light logit balance;
- presence: both detectors on;
- between / novelty (away from mix and both vocoders), as v1;
- realism: two feature-matching hinge critics (judge view; stereo view up
  to 20 kHz);
- loudness, harmonic anchor, residual budget, gain smoothness.

Across the two alphas, a ranking loss: higher alpha must be more vocal.

Articulation (syllables, vowels): every vocal note has a per-frame target
from the algorithmic vocal detector (silence, i, e, eh, a, o, u, sibilant,
plosive, breath). A frame-wise ArticulationJudge is trained on them (GPU,
minutes). Each pair asks the hybrid either to keep the voice's articulation
(half the pairs), or to sing another vowel / glide between two vowels on the
voice's vowel frames. The articulation loss compares the judge's reading of
the hybrid with that target, so vowels can be automated at inference.

Checkpoints every quarter epoch; ``best.pt`` by fixed metrics only:
- the held-out judge's balance;
- vocality error;
- novelty shortfall;
- both detectors on;
- KID realism in Wavelet2Vec space relative to the plain mix.

Listening demos every epoch, including an alpha sweep.

Pause / resume exactly: create ``PAUSE`` in the run folder (or use the dashboard / ``training_control.py``).
The loop saves its complete state to ``checkpoints/paused.pt`` before its next step and exits. ``--resume``
continues from the newer of ``paused.pt`` and ``last.pt`` with the same data order and random draws.

    PYTHONPATH=src python experiments/train_morph_large.py [--epochs 10 --steps-per-epoch 800 --batch 6]
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_morph as tm  # noqa: E402  (data pipeline and helpers of v1)
import training_control as ctl  # noqa: E402  (pause / resume)
from wavelet2vec.morph import (JUDGE_PLANES, VIEW_STATS, JudgeCNN, JudgeTCN, augment_view,  # noqa: E402
                               judge_view, mix_baseline, vocoder_baseline, vox_probability)
from wavelet2vec.morph_large import (CRITIC_PLANES, FeatureCritic, MorphNetL, backbone_inputs,  # noqa: E402
                                     critic_view, load_v3_backbone, pitch_jitter, vocality)
from wavelet2vec.articulation import (ART_CLASSES, VOWEL_INDEX, VOWELS, ArticulationJudge,  # noqa: E402
                                      apply_vowels, articulation_loss, articulation_targets, morph_vowels,
                                      vowel_automation)
from wavelet2vec.tools.vocal_detector import detect_vocal_parts  # noqa: E402
from wavelet2vec.wavelet_image import WaveletImage, decode  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "morph" / "v2"
V3_CHECKPOINT = HERE / "output" / "keyboard" / "v3" / "checkpoints" / "best.pt"
V1_JUDGES = HERE / "output" / "morph" / "v1" / "judges.pt"
PRESENT = 1.0                              # presence target logit (calibrated judges: sigmoid ~ 0.73)
HYBRIDS = ("hybrid", "hybrid, vocal pitch")  # pitch source 0 = synth, 1 = vocal


# ----------------------------------------------------------------------------------------- caches


@torch.no_grad()
def build_critic_cache(catalog, device, path, batch=24):
    if path.exists():
        return torch.load(path, weights_only=False)
    started, views = time.time(), []
    for start in range(0, len(catalog.notes), batch):
        audio = np.stack([catalog.normalized(i)[0] for i in range(start, min(start + batch, len(catalog.notes)))])
        views.append(critic_view(tm.encode_batch(audio, device)).half().cpu())
    cache = {"views": torch.cat(views)}
    torch.save(cache, path)
    print(f"  critic cache: {len(cache['views'])} notes in {time.time() - started:.0f} s", flush=True)
    return cache


# ----------------------------------------------------------------------------------------- articulation


def build_articulation_cache(catalog, path, log):
    """Per vocal note: ``[10, 120]`` articulation targets from the vocal detector, one vocal-tract scale per
    singer (median of its notes' own estimates)."""
    if path.exists():
        return torch.load(path, weights_only=False)
    started = time.time()
    vox = [i for i, n in enumerate(catalog.notes) if n["kind"] == "vox"]
    scales = {}
    for singer in sorted({catalog.notes[i]["singer"] for i in vox}):
        idx = [i for i in vox if catalog.notes[i]["singer"] == singer]
        found = [detect_vocal_parts(catalog.normalized(i)[0].mean(0), tm.SPEC.sample_rate).scale
                 for i in idx[:: max(1, len(idx) // 24)]]
        scales[singer] = float(np.median(found))
    targets = {}
    for i in vox:
        track = detect_vocal_parts(catalog.normalized(i)[0].mean(0), tm.SPEC.sample_rate,
                                   scale=scales[catalog.notes[i]["singer"]])
        targets[i] = torch.from_numpy(articulation_targets(track)).half()
    cache = {"targets": targets, "scales": scales}
    torch.save(cache, path)
    log(f"  articulation targets: {len(targets)} vocal notes in {time.time() - started:.0f} s, scales {scales}")
    return cache


def _art_view_batch(rng, cache, art, idx):
    views = cache["views"][idx].float()
    rows = views.shape[2]
    for b in range(len(idx)):                       # random low-pass and noise floor (as the judges)
        cut = int(rng.integers(rows - 24, rows + 1))
        if cut < rows:
            views[b, 0, cut:] = -1.0
            views[b, 1:, cut:] = 0.0
        if rng.random() < 0.5:
            views[b, 0] = torch.clamp(views[b, 0], min=float(rng.uniform(-1.0, -0.5)))
    return views, torch.stack([art["targets"][i] for i in idx]).float()


def vowel_accuracy(logits, target):
    """Per item: share of the target's vowel frames where the judged vowel matches the target vowel."""
    mask = target[:, VOWEL_INDEX].sum(1) > 0.5
    hit = logits[:, VOWEL_INDEX].argmax(1) == target[:, VOWEL_INDEX].argmax(1)
    return [float(hit[b][mask[b]].float().mean()) if bool(mask[b].any()) else float("nan") for b in range(len(target))]


def train_articulation_judge(pools, cache, art, device, steps=3000, batch=32, seed=5, log=None):
    train = [i for (split, _), v in pools.vox.items() if split == "train" for i in v]
    val = [i for (split, _), v in pools.vox.items() if split == "val" for i in v]
    torch.manual_seed(seed)
    model = ArticulationJudge().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    started = time.time()
    for step in range(1, steps + 1):
        idx = [train[k] for k in rng.integers(len(train), size=batch)]
        views, targets = _art_view_batch(rng, cache, art, idx)
        targets = targets.to(device)
        weight = 1.0 - 0.7 * targets[:, 0]           # silence frames count less
        loss = articulation_loss(model(views.to(device)), targets, weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 1000 == 0 or step == 20):
            log(f"  articulation judge step {step}/{steps} loss {loss.item():.4f} {time.time() - started:.0f} s")
    model.eval()
    with torch.no_grad():
        views = cache["views"][val].float().to(device)
        targets = torch.stack([art["targets"][i] for i in val]).float().to(device)
        logits = model(views)
        frame_ok = targets.max(1).values > 0.5
        hit = (logits.argmax(1) == targets.argmax(1))[frame_ok].float().mean()
        vowel = np.nanmean(vowel_accuracy(logits, targets))
        per_class = {}
        for k, name in enumerate(ART_CLASSES):
            sel = frame_ok & (targets.argmax(1) == k)
            if bool(sel.any()):
                per_class[name] = {"frames": int(sel.sum()), "recall": round(float((logits.argmax(1)[sel] == k).float().mean()), 4)}
    report = {"held-out takes": {"frame accuracy": round(float(hit), 4), "vowel accuracy": round(float(vowel), 4),
                                 "notes": len(val)}, **{f"class {k}": v for k, v in per_class.items()}}
    return model.cpu(), report


def articulation_batch(rng, pairs, art):
    """Per pair: keep the voice's articulation (half), or one other vowel / a glide between two vowels on
    its vowel frames. Returns ``[B, 10, 120]`` and the mode per pair."""
    out, modes = [], []
    for v, _, _ in pairs:
        target = art["targets"][v].float().numpy()
        r = rng.random()
        if r < 0.5:
            modes.append("keep")
        elif r < 0.75:
            vowel = VOWELS[int(rng.integers(len(VOWELS)))]
            target = apply_vowels(target, vowel_automation([(0.0, vowel)]))
            modes.append("vowel")
        else:
            first, last = rng.choice(len(VOWELS), 2, replace=False)
            target = morph_vowels(target, VOWELS[first], VOWELS[last])
            modes.append("glide")
        out.append(target)
    return torch.from_numpy(np.stack(out)), modes


def swapped_vowels(target):
    """A fixed, different vowel for control checks: three steps along VOWELS from the dominant vowel."""
    dominant = int(target[VOWEL_INDEX].sum(1).argmax())
    return apply_vowels(target, vowel_automation([(0.0, VOWELS[(dominant + 3) % len(VOWELS)])]))


# ----------------------------------------------------------------------------------------- calibrated judges


def supcon(z, y, tau=0.1):
    """Supervised contrastive loss on normalized embeddings."""
    z = F.normalize(z, dim=-1)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return (-(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def train_judge_v2(kind, pools, cache, device, steps=3000, batch=64, seed=0, singers=("shevannai", "own"),
                   smoothing=0.05, contrastive=0.5, log=None):
    torch.manual_seed(seed)
    model = (JudgeCNN(width=32) if kind == "cnn" else JudgeTCN(width=256)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    started = time.time()
    for step in range(1, steps + 1):
        views, labels = tm.judge_batch(rng, pools, cache, batch, singers)
        views = augment_view(views, gen).to(device)
        if rng.random() < 0.5:
            views[:, 0] = torch.clamp(views[:, 0], min=rng.uniform(-1.0, -0.5))
        labels = labels.to(device)
        classes = torch.where(labels[:, 0] > 0, 0, torch.where(labels[:, 1] > 0, 1, 2))
        logits, emb = model(views)
        loss = F.binary_cross_entropy_with_logits(logits, labels * (1 - smoothing) + smoothing / 2)
        loss = loss + contrastive * supcon(emb, classes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 1000 == 0 or step == 20):
            log(f"  judge v2 {kind} {'+'.join(singers)} step {step}/{steps} loss {loss.item():.4f} {time.time() - started:.0f} s")
    return model.eval()


@torch.no_grad()
def embed_views(model, views, device, batch=256):
    outs = [model(views[i:i + batch].float().to(device)) for i in range(0, len(views), batch)]
    return torch.cat([o[0].cpu() for o in outs]), torch.cat([o[1].cpu() for o in outs])


@torch.no_grad()
def centres(model, pools, cache, device, rng):
    vox = sum(pools.vox_split("train").values(), [])
    synth = [i for f in tm._per_family(pools, "train", 40, rng) for i in f]
    _, ev = embed_views(model, cache["views"][vox], device)
    _, es = embed_views(model, cache["views"][synth], device)
    return F.normalize(ev, dim=-1).mean(0), F.normalize(es, dim=-1).mean(0)


def judge_report_v2(model, centre_v, centre_s, pools, cache, device, rng):
    from sklearn.metrics import roc_auc_score
    sets = {"vox (held-out takes)": sum(pools.vox_split("val").values(), []),
            "synth (held-out families)": pools.synth_sample(rng, "val", 600),
            "Exhale (processed vox, reference)": pools.ref["processed_vox"],
            "Harmor vox (synth vox, reference)": pools.ref["synth_vox"]}
    out = {}
    for name, idx in sets.items():
        logits, emb = embed_views(model, cache["views"][idx], device)
        voc = vocality(emb, centre_v, centre_s)
        out[name] = {"notes": len(idx), "p_vox": round(float(vox_probability(logits).mean()), 4),
                     "vocality": round(float(voc.mean()), 4), "vocality_sd": round(float(voc.std()), 4),
                     "vox_present": round(float(torch.sigmoid(logits[:, 0]).mean()), 4),
                     "synth_present": round(float(torch.sigmoid(logits[:, 1]).mean()), 4)}
    logits, emb = embed_views(model, cache["noise"][448:], device)
    out["noise (held-out)"] = {"notes": len(logits), "vocality": round(float(vocality(emb, centre_v, centre_s).mean()), 4),
                               "vox_present": round(float(torch.sigmoid(logits[:, 0]).mean()), 4),
                               "synth_present": round(float(torch.sigmoid(logits[:, 1]).mean()), 4)}
    lv, _ = embed_views(model, cache["views"][sets["vox (held-out takes)"]], device)
    ls, _ = embed_views(model, cache["views"][sets["synth (held-out families)"]], device)
    p = torch.cat([vox_probability(lv), vox_probability(ls)]).numpy()
    y = np.r_[np.ones(len(lv)), np.zeros(len(ls))]
    out["held-out vox vs synth"] = {"auc": round(float(roc_auc_score(y, p)), 4),
                                    "accuracy": round(float(((p > 0.5) == y).mean()), 4)}
    return out


def loso_v2(pools, cache, device, log, steps):
    out = {}
    for train_singer, test_singer in (("shevannai", "own"), ("own", "shevannai")):
        model = train_judge_v2("cnn", pools, cache, device, steps=steps, seed=3, singers=(train_singer,), log=log)
        idx = pools.vox["train", test_singer] + pools.vox["val", test_singer]
        logits, _ = embed_views(model, cache["views"][idx], device)
        p = vox_probability(logits)
        out[f"train {train_singer} -> test {test_singer}"] = {"notes": len(idx), "p_vox": round(float(p.mean()), 4),
                                                             "recall": round(float((p > 0.5).float().mean()), 4)}
    return out


# ----------------------------------------------------------------------------------------- realism (KID)


def kid(x, y):
    """Unbiased squared MMD with the cubic polynomial kernel (as in KID)."""
    d = x.shape[1]
    kxx, kyy, kxy = ((x @ x.T) / d + 1) ** 3, ((y @ y.T) / d + 1) ** 3, ((x @ y.T) / d + 1) ** 3
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


# ----------------------------------------------------------------------------------------- generator step


def split_encoding(enc, index):
    return {k: (v[index] if torch.is_tensor(v) else [x[index] for x in v]) for k, v in enc.items()}


def load_batch(catalog, pairs):
    """CPU images of the vocal and synth notes, and the backbone's input planes for both."""
    audio = np.stack([catalog.normalized(v)[0] for v, _, _ in pairs] + [catalog.normalized(s)[0] for _, s, _ in pairs])
    with torch.no_grad():
        image = tm.encode_batch(audio, torch.device("cpu"))
    midis = [catalog.notes[v]["midi"] for v, _, _ in pairs]
    planes, cond = backbone_inputs(image, midis + midis)
    n = len(pairs)
    return tm.take(image, slice(0, n)), tm.take(image, slice(n, 2 * n)), image, planes, cond, midis


class Judges:
    """Frozen calibrated judges with their vocality axes."""

    def __init__(self, models, centres_):
        self.models, self.centres = models, centres_

    def __call__(self, view):
        out = []
        for m, (cv, cs) in zip(self.models, self.centres):
            logits, emb = m(view)
            out.append((logits, emb, vocality(emb, cv, cs)))
        return out


def hybrid_losses(hybrid, extras, alpha, judges, critics, refs, gen, real_feats, art_judge=None, art_target=None):
    view_h = judge_view(hybrid)
    cview_h = critic_view(hybrid)
    target = torch.logit(alpha.clamp(0.02, 0.98))
    voc_loss, balance, presence, novelty, between, vocs = 0.0, 0.0, 0.0, 0.0, 0.0, []
    for k, (logits, emb, voc) in enumerate(judges(augment_view(view_h, gen))):
        voc_loss = voc_loss + 4 * ((voc - alpha) ** 2).mean()
        balance = balance + F.smooth_l1_loss(logits[:, 0] - logits[:, 1], target)
        presence = presence + (F.softplus(PRESENT - logits[:, 0]) + F.softplus(PRESENT - logits[:, 1])).mean()
        vocs.append(voc)
    for k, (_, emb, _) in enumerate(judges(view_h)):
        e_v, e_s, e_mix, *e_voc = refs["emb"][k]
        span = (1 - F.cosine_similarity(e_v, e_s)).clamp(min=0.05)
        for e in [e_mix] + e_voc:
            novelty = novelty + F.relu(1 - (1 - F.cosine_similarity(emb, e)) / (0.25 * span)).mean()
        between = between + (F.relu((1 - F.cosine_similarity(emb, e_v)) / span - 1)
                             + F.relu((1 - F.cosine_similarity(emb, e_s)) / span - 1)).mean()
    n = len(judges.models)
    realism, matching = 0.0, 0.0
    for critic, view, real in zip(critics, (augment_view(view_h, gen), cview_h), real_feats):
        score, feats = critic(view)
        realism = realism - score.mean()
        if real is not None:
            matching = matching + sum((f.mean(0) - r).abs().mean() / (r.abs().mean() + 1e-6) for f, r in zip(feats, real))
    e_h, e_v, e_s = tm.energy(hybrid), refs["energy_v"], refs["energy_s"]
    loudness = ((10 * torch.log10((e_h + 1e-12) / (0.5 * (e_v + e_s) + 1e-12))) ** 2).mean() / 100
    lh = tm.row_levels(hybrid).gather(1, refs["rows"])
    anchor = (F.relu(refs["anchor_ref"] - 15 - lh) / 10).mean()
    residual = F.relu(extras["residual_energy"] / (e_h + 1e-12) - 0.5).mean()
    level = extras["gain_level"]
    smooth = (level[..., 1:] - level[..., :-1]).abs().mean()
    articulation = torch.zeros(())
    if art_judge is not None:
        weight = (1.0 - art_target[:, 0]) * (0.5 + alpha)[:, None]      # voiced frames; more vocal -> stricter
        articulation = articulation_loss(art_judge(view_h), art_target, weight)
    parts = {"vocality": voc_loss / n, "balance": balance / n, "presence": presence / n, "novelty": novelty / (3 * n),
             "between": between / n, "realism": realism / 2, "matching": matching / 2, "loudness": loudness,
             "anchor": anchor, "residual": residual, "smooth": smooth, "articulation": articulation}
    return parts, vocs, view_h, cview_h


@torch.no_grad()
def reference_embeddings(judges, vocal, synth, alpha_list):
    views = {"v": judge_view(vocal), "s": judge_view(synth), "voc_s": judge_view(vocoder_baseline(synth, vocal)),
             "voc_v": judge_view(vocoder_baseline(vocal, synth))}
    mixes = [judge_view(mix_baseline(vocal, synth, a)) for a in alpha_list]
    out = []
    for m in judges.models:
        e = {k: m(v)[1] for k, v in views.items()}
        out.append([[e["v"], e["s"], m(mx)[1], e["voc_s"], e["voc_v"]] for mx in mixes])
    return out                                   # [judge][alpha index] -> [e_v, e_s, e_mix, e_voc_s, e_voc_v]


# ----------------------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(G, judges, critics, probe, w2v_real, catalog, pools, pairs, art_judge, art, batch=6):
    """Validation on held-out pairs, for both pitch sources, with the baselines alongside."""
    G.eval()
    names = HYBRIDS + ("mix", "vocoder: synth carrier", "vocoder: vocal carrier", "vocal", "synth")
    audio_names = names[:5]
    rows = defaultdict(lambda: defaultdict(list))
    w2v = defaultdict(list)
    alphas = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        vocal, synth, image, planes, cond, midis = load_batch(catalog, chunk)
        alpha = torch.tensor([a for _, _, a in chunk], dtype=torch.float32)
        n = len(chunk)
        enc = G.encode(planes, cond, image)
        ev, es = split_encoding(enc, slice(0, n)), split_encoding(enc, slice(n, 2 * n))
        own = torch.stack([art["targets"][v].float() for v, _, _ in chunk])
        hybrids = [G.decode(ev, es, alpha, vocal, synth, torch.full((n,), k, dtype=torch.long), own, midis)[0]
                   for k in (0, 1)]
        swapped = torch.from_numpy(np.stack([swapped_vowels(art["targets"][v].float().numpy()) for v, _, _ in chunk]))
        controlled = G.decode(ev, es, alpha, vocal, synth, torch.zeros(n, dtype=torch.long), swapped, midis)[0]
        for name, image_ in ((HYBRIDS[0], hybrids[0]), (HYBRIDS[1], hybrids[1]), ("vocal", vocal)):
            rows[name]["articulation_keep_acc"] += vowel_accuracy(art_judge(judge_view(image_)), own)
        rows[HYBRIDS[0]]["articulation_control_acc"] += vowel_accuracy(art_judge(judge_view(controlled)), swapped)
        images = hybrids + [mix_baseline(vocal, synth, alpha), vocoder_baseline(synth, vocal),
                            vocoder_baseline(vocal, synth), vocal, synth]
        views = [judge_view(x) for x in images]
        for jn, model, (cv, cs) in zip(("A", "B"), judges.models, judges.centres):
            outs = [model(v) for v in views]
            span = (1 - F.cosine_similarity(outs[-2][1], outs[-1][1])).clamp(min=1e-3)
            for name, (logits, emb) in zip(names, outs):
                rows[name][f"p_vox_{jn}"] += vox_probability(logits).tolist()
                rows[name][f"vocality_{jn}"] += vocality(emb, cv, cs).tolist()
                both = (torch.sigmoid(logits[:, 0]) > 0.5) & (torch.sigmoid(logits[:, 1]) > 0.5)
                rows[name][f"both_present_{jn}"] += both.float().tolist()
                rows[name][f"presence_{jn}"] += torch.minimum(torch.sigmoid(logits[:, 0]),
                                                              torch.sigmoid(logits[:, 1])).tolist()
            for hi, hname in enumerate(HYBRIDS):
                for name, (_, emb) in zip(names[2:], outs[2:]):
                    rows[hname][f"distance_to_{name}_{jn}"] += ((1 - F.cosine_similarity(outs[hi][1], emb)) / span).tolist()
        for name, v, image in zip(names, views, images):
            rows[name]["critic"] += critics[0](v)[0].tolist()
            rows[name]["pitch_jitter_cents"] += pitch_jitter(image, midis)
            if name in audio_names:
                emb = np.stack([tm.w2v_features(x) for x in decode(image).float().numpy()])
                w2v[name].append(emb)
                rows[name]["p_vox_J"] += probe.predict_proba(emb)[:, 1].tolist()
        alphas += alpha.tolist()
    G.train()
    alpha = np.array(alphas)
    scaler = probe[0]
    summary = {}
    for name, cols in rows.items():
        summary[name] = {k: round(float(np.nanmean(v)), 4) for k, v in cols.items()}
        for j in ("A", "B", "J"):
            if f"p_vox_{j}" in cols:
                summary[name][f"balance_error_{j}"] = round(float(np.abs(np.array(cols[f"p_vox_{j}"]) - alpha).mean()), 4)
        for j in ("A", "B"):
            summary[name][f"vocality_error_{j}"] = round(float(np.abs(np.array(cols[f"vocality_{j}"]) - alpha).mean()), 4)
        if name in w2v:
            summary[name]["kid"] = round(kid(scaler.transform(np.concatenate(w2v[name])), w2v_real), 4)
    lows = [i for i, a in enumerate(alpha) if a < 0.5]
    highs = [i for i, a in enumerate(alpha) if a > 0.5]
    for hname in HYBRIDS:
        h = summary[hname]
        if lows and len(lows) == len(highs):
            voc = np.array(rows[hname]["vocality_A"])
            h["alpha_monotone_A"] = round(float(np.mean([voc[hi] > voc[lo] for lo, hi in zip(lows, highs)])), 4)
        novelty_short = np.mean([max(0.0, 0.25 - h[k]) / 0.25 for k in h
                                 if k.startswith("distance_to_mix") or k.startswith("distance_to_vocoder")])
        h["kid_relative_to_mix"] = round(float(h["kid"] / max(summary["mix"]["kid"], 1e-6)), 4)
        keep = h.get("articulation_keep_acc", 0.0)
        control = summary[HYBRIDS[0]].get("articulation_control_acc", 0.0)
        h["score"] = round(float(h["balance_error_J"] + 0.5 * (h["vocality_error_A"] + h["vocality_error_B"])
                                 + 0.5 * novelty_short + 0.25 * (1 - 0.5 * (h["both_present_A"] + h["both_present_B"]))
                                 + 0.25 * min(h["kid_relative_to_mix"], 4.0) / 4.0
                                 + 0.25 * (1 - keep) + 0.25 * (1 - control)), 4)
    summary["score"] = round(float(np.mean([summary[h]["score"] for h in HYBRIDS])), 4)
    summary["kid_relative_to_mix"] = summary["hybrid"]["kid_relative_to_mix"]
    for name in names:
        summary[name]["realism_gap"] = round(0.5 * (summary["vocal"]["critic"] + summary["synth"]["critic"])
                                             - summary[name]["critic"], 4)
    return summary


@torch.no_grad()
def write_demos(G, catalog, pairs, folder: Path, art):
    """192 kHz float WAVs per pair: the hybrid with synth pitch and with vocal pitch, the two sources, the
    plain mix and the vocoder (synth carrier)."""
    folder.mkdir(parents=True, exist_ok=True)
    G.eval()
    entries = []
    for k, (v, s, a) in enumerate(pairs):
        vocal, synth, image, planes, cond, midis = load_batch(catalog, [(v, s, a)])
        enc = G.encode(planes, cond, image)
        alpha = torch.tensor([a], dtype=torch.float32)
        ev, es = split_encoding(enc, slice(0, 1)), split_encoding(enc, slice(1, 2))
        own = art["targets"][v].float().numpy()
        glide = torch.from_numpy(morph_vowels(own, "a", "o"))[None]
        own = torch.from_numpy(own)[None]
        images = {"hybrid": G.decode(ev, es, alpha, vocal, synth, torch.tensor([0]), own, midis)[0],
                  "hybrid, vocal pitch": G.decode(ev, es, alpha, vocal, synth, torch.tensor([1]), own, midis)[0],
                  "hybrid, vowels a→o": G.decode(ev, es, alpha, vocal, synth, torch.tensor([0]), glide, midis)[0],
                  "vocal": vocal, "synth": synth, "mix": mix_baseline(vocal, synth, alpha),
                  "vocoder": vocoder_baseline(synth, vocal)}
        nv, ns = catalog.notes[v], catalog.notes[s]
        entry = {"pair": k, "midi": nv["midi"], "alpha": a, "vocal": f"{nv['family']}: {nv['file']}",
                 "synth": f"{ns['family']}: {ns['file']}", "files": {}}
        for name, image_ in images.items():
            audio = decode(image_)[0].float().numpy()
            peak = float(np.abs(audio).max())
            if peak > 0.99:
                audio = audio * (0.99 / peak)
            safe = name.replace(", ", "_").replace(" ", "_").replace("→", "-to-")
            path = folder / f"pair{k:02d}_{safe}.wav"
            sf.write(path, audio.T, tm.SPEC.sample_rate, subtype="FLOAT")
            entry["files"][name] = path.name
        entries.append(entry)
    (folder / "demos.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    G.train()
    return entries


# ----------------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=800)
    parser.add_argument("--batch", type=int, default=6, help="pairs per step (each decoded at two alphas)")
    parser.add_argument("--lr", type=float, default=2e-4, help="new parts")
    parser.add_argument("--backbone-lr", type=float, default=2e-5, help="pretrained v3 backbone and head")
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--adv", type=float, default=0.3)
    parser.add_argument("--judge-steps", type=int, default=3000)
    parser.add_argument("--val-pairs", type=int, default=36)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--prep-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--v3", type=Path, default=V3_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start the generator (and critics) from a MorphNet-L checkpoint; new parts stay zero")
    parser.add_argument("--art-steps", type=int, default=3000, help="articulation judge training steps")
    parser.add_argument("--noise-gate", choices=VIEW_STATS, default="cell",
                        help="tonal/noise split of the one-sample cells below 440 Hz: 'cell' (always tonal, as "
                             "trained so far) or 'scale' (semantic-bridge coherence; the notes' harmonic rows stay "
                             "tonal). Keep the run's choice when resuming.")
    args = parser.parse_args()
    faulthandler.enable()                          # a native crash leaves a traceback in the error log
    if sys.platform == "win32":                    # keep the PC awake while training (idle sleep)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)                            # reproducible initialization of the new parts
    prep = torch.device(args.prep_device)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    events = tm.EventLog(args.out / "metrics.jsonl")
    ctl.record_launch(args.out)
    ctl.clear_pause(args.out)                       # a request belongs to a running process, not to a start

    def log(text):
        print(text, flush=True)

    catalog = tm.Catalog()
    cache = tm.build_cache(catalog, prep, OUT.parent / "judge_cache.pt")
    critic_cache = build_critic_cache(catalog, prep, OUT.parent / "critic_cache.pt")
    pools = tm.Pools(catalog, cache["rms"])

    # ---- phase 1: calibrated judges
    judges_path = args.out / "judges_v2.pt"
    if judges_path.exists():
        saved = torch.load(judges_path, map_location="cpu", weights_only=False)
        models = [JudgeCNN(width=32), JudgeTCN(width=256)]
        for m, key in zip(models, ("A", "B")):
            m.load_state_dict(saved[key])
        centres_ = saved["centres"]
        reports = saved["reports"]
        log("calibrated judges loaded")
    else:
        rng = np.random.default_rng(12)
        models = [train_judge_v2("cnn", pools, cache, prep, args.judge_steps, seed=1, log=log),
                  train_judge_v2("tcn", pools, cache, prep, args.judge_steps, seed=2, log=log)]
        centres_ = [centres(m, pools, cache, prep, rng) for m in models]
        reports = {"A (CNN, calibrated)": judge_report_v2(models[0], *centres_[0], pools, cache, prep, rng),
                   "B (TCN, calibrated)": judge_report_v2(models[1], *centres_[1], pools, cache, prep, rng),
                   "leave one singer out (CNN, calibrated)": loso_v2(pools, cache, prep, log, args.judge_steps // 2)}
        torch.save({"A": models[0].cpu().state_dict(), "B": models[1].cpu().state_dict(), "centres": centres_,
                    "reports": reports}, judges_path)
        for name, rep in reports.items():
            events.write("judge", name=name, report=rep)
        v1_reports = torch.load(V1_JUDGES, map_location="cpu", weights_only=False)["reports"]
        events.write("judge", name="J (Wavelet2Vec + logistic, held out; same splits, from v1)",
                     report=v1_reports["J (Wavelet2Vec + logistic, held out)"])
    probe = torch.load(V1_JUDGES, map_location="cpu", weights_only=False)["probe"]
    log(json.dumps(reports, indent=1))

    # ---- articulation: targets for every vocal note, and the frame-wise judge
    art = build_articulation_cache(catalog, OUT.parent / "articulation_cache.pt", log)
    art_path = args.out / "articulation_judge.pt"
    if art_path.exists():
        saved = torch.load(art_path, map_location="cpu", weights_only=False)
        art_judge = ArticulationJudge()
        art_judge.load_state_dict(saved["state"])
        log(f"articulation judge loaded: {saved['report']['held-out takes']}")
    else:
        art_judge, art_report = train_articulation_judge(pools, cache, art, prep, args.art_steps, log=log)
        torch.save({"state": art_judge.state_dict(), "report": art_report}, art_path)
        events.write("judge", name="Articulation judge (frame-wise: silence, i, e, eh, a, o, u, sibilant, plosive, breath)",
                     report=art_report)
        log(json.dumps(art_report, indent=1))
    art_judge.eval()
    art_judge.requires_grad_(False)
    models = [m.cpu().eval() for m in models]
    for m in models:
        m.requires_grad_(False)
    judges = Judges(models, [(cv.cpu(), cs.cpu()) for cv, cs in centres_])

    # ---- realism reference for KID: Wavelet2Vec embeddings of real held-out notes of every class
    real_path = OUT.parent / "w2v_real_val.npy"
    if real_path.exists():
        w2v_real = np.load(real_path)
    else:
        rng = np.random.default_rng(17)
        idx = (sum(pools.vox_split("val").values(), []) + [i for f in tm._per_family(pools, "val", 12, rng) for i in f]
               + list(rng.choice(pools.ref["processed_vox"], 60, replace=False))
               + list(rng.choice(pools.ref["synth_vox"], 40, replace=False)))
        w2v_real = probe[0].transform(np.stack([tm.w2v_features(catalog.normalized(i)[0]) for i in idx]))
        np.save(real_path, w2v_real)
        log(f"KID reference: {len(w2v_real)} real held-out notes")

    # ---- phase 2: MorphNet-L on the CPU
    state, v2_config = load_v3_backbone(args.v3)
    G = MorphNetL(state, v2_config, noise_gate=args.noise_gate)
    del state
    critics = [FeatureCritic(JUDGE_PLANES), FeatureCritic(CRITIC_PLANES)]
    pre = G.pretrained_parameters()
    new = G.new_parameters()
    opt_g = torch.optim.AdamW([{"params": [p for p in pre if p.ndim >= 2], "lr": args.backbone_lr, "weight_decay": 0.05},
                               {"params": [p for p in pre if p.ndim < 2], "lr": args.backbone_lr, "weight_decay": 0.0},
                               {"params": new, "lr": args.lr, "weight_decay": 1e-4}], betas=(0.8, 0.99))
    opt_c = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=args.critic_lr, betas=(0.0, 0.99))
    total_steps = args.epochs * args.steps_per_epoch
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt_g, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))
    weights = {"vocality": 1.0, "balance": 0.2, "presence": 0.5, "novelty": 1.0, "between": 1.0, "matching": 1.0,
               "loudness": 0.5, "anchor": 0.5, "residual": 1.0, "smooth": 0.02, "articulation": 0.5}
    rank_weight = 1.0
    rng = np.random.default_rng(1)
    gen = torch.Generator().manual_seed(1)
    start_epoch, start_step, best, history = 1, 0, float("inf"), []
    real_feats = [None, None]
    resume_parts, resume_seconds = None, 0.0
    warm_from = None
    if args.init_from is not None and not (args.resume and (ckpt / "last.pt").exists()):
        saved = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing = G.load_state_dict(saved["G"], strict=False)
        for c, s_ in zip(critics, saved["critics"]):
            c.load_state_dict(s_)
        warm_from = {"checkpoint": str(args.init_from), "new_parameters": len(missing.missing_keys)}
        log(f"generator initialized from {args.init_from}; {len(missing.missing_keys)} new tensors start at zero")
    state_path = ctl.latest_state(ckpt) if args.resume else None
    if state_path is not None:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        G.load_state_dict(saved["G"])
        for c, s in zip(critics, saved["critics"]):
            c.load_state_dict(s)
        opt_g.load_state_dict(saved["opt_g"])
        opt_c.load_state_dict(saved["opt_c"])
        sched.load_state_dict(saved["sched"])
        ctl.restore_random(saved, rng, gen)
        best, history = saved["best"], saved["history"]
        real_feats = saved.get("real_feats", real_feats)
        done = saved["step_in_epoch"]
        start_epoch, start_step = (saved["epoch"] + 1, 0) if done >= args.steps_per_epoch else (saved["epoch"], done)
        if start_step > 0:
            resume_parts, resume_seconds = saved.get("parts"), saved.get("epoch_seconds", 0.0)
        exact = "torch_generator" in saved and "parts" in saved
        log(f"resumed from {state_path.name} at epoch {start_epoch} step {start_step}"
            f"{'' if exact else ' (older state file: random generators of augmentation/dropout restart)'}")
        events.write("resumed", epoch=start_epoch - 1 + start_step / args.steps_per_epoch, file=state_path.name,
                     exact=exact)
        del saved

    val_rng = np.random.default_rng(31)
    val_pairs = tm.sample_pairs(val_rng, catalog, pools, args.val_pairs, "val", alpha=0.5)
    sweep = tm.sample_pairs(val_rng, catalog, pools, 6, "val")
    val_pairs += [(v, s, 0.25) for v, s, _ in sweep] + [(v, s, 0.75) for v, s, _ in sweep]
    demo_pairs = val_pairs[:6] + [(sweep[0][0], sweep[0][1], a) for a in (0.3, 0.7)]
    piuw = [i for i in pools.vox["train", "own"] if catalog.notes[i]["file"] == tm.PIUW]
    for vi in ([piuw[len(piuw) // 3], piuw[2 * len(piuw) // 3]] if piuw else []):
        midi = catalog.notes[vi]["midi"]
        fam = sorted(f for (f, m) in pools.synth_key["val"] if m == midi)[0]
        demo_pairs.append((vi, pools.synth_key["val"][fam, midi][0], 0.5))
    count = lambda ps: sum(p.numel() for p in ps)
    run = {"task": "morph", "model": "morphnet-L (v2): stable pitch + articulation", "device": "cpu",
           "threads": args.threads, "init_from": warm_from,
           "parameters": {"total": count(G.parameters()), "pretrained_v3": count(pre), "new": count(new),
                          "fusion": count(G.fusion.parameters()), "critics": count(p for c in critics for p in c.parameters())},
           "warm_start": str(args.v3), "batch": args.batch, "alphas_per_pair": 2, "steps_per_epoch": args.steps_per_epoch,
           "epochs_total": args.epochs, "lr": args.lr, "backbone_lr": args.backbone_lr, "loss_weights": weights,
           "rank_weight": rank_weight, "adv": args.adv, "val_families": catalog.val_families,
           "val_vox_files": list(tm.VAL_VOX_FILES),
           "train_notes": sum(len(v) for (k, _), v in pools.vox.items() if k == "train"), "sample_rate": tm.SPEC.sample_rate}
    log(json.dumps(run, indent=1, default=str))
    events.write("run_start", epoch=start_epoch - 1 + start_step / args.steps_per_epoch, resumed=bool(args.resume),
                 size=f"MorphNet-L {count(G.parameters()) / 1e6:.0f}M (v3 Swin-L warm start)", **run)
    quarter = max(1, args.steps_per_epoch // 4)

    def full_state(epoch, step, parts, started):
        """Everything needed to continue at the next step exactly as if never stopped."""
        return {"G": G.state_dict(), "critics": [c.state_dict() for c in critics], "opt_g": opt_g.state_dict(),
                "opt_c": opt_c.state_dict(), "sched": sched.state_dict(), **ctl.random_states(rng, gen),
                "real_feats": real_feats, "parts": parts, "epoch_seconds": time.time() - started,
                "epoch": epoch, "step_in_epoch": step, "best": best, "history": history,
                "v2_config": v2_config, "noise_gate": G.noise_gate}

    def checkpoint(epoch, step, parts, started):
        nonlocal best
        point = epoch - 1 + step / args.steps_per_epoch
        tick = time.time()
        val = evaluate(G, judges, critics, probe, w2v_real, catalog, pools, val_pairs, art_judge, art)
        record = {"epoch": round(point, 3), "train": {k: round(float(np.mean([p[k] for p in parts])), 4) for k in parts[0]}
                  if parts else None, "score": val["score"], "minutes": round((time.time() - started) / 60, 1)}
        history.append(record)
        name = f"{point:g}_epoch.pth"
        torch.save({"G": G.state_dict(), "critics": [c.state_dict() for c in critics], "val": val, "metrics": record,
                    "v2_config": v2_config, "noise_gate": G.noise_gate}, ckpt / name)
        is_best = val["score"] < best
        if is_best:
            best = val["score"]
            shutil.copyfile(ckpt / name, ckpt / "best.pt")
        ctl.save_state(full_state(epoch, step, parts, started), ckpt / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        events.write("val", epoch=point, morph=val)
        events.write("checkpoint", epoch=point, file=name, best=is_best, val_model=val["score"],
                     size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
        h, hv = val["hybrid"], val["hybrid, vocal pitch"]
        log(f"== checkpoint {name}: score {val['score']:.4f}{' (best)' if is_best else ''}  hybrid (synth pitch) p_vox "
            f"A {h['p_vox_A']:.2f} B {h['p_vox_B']:.2f} J {h['p_vox_J']:.2f} vocality A {h['vocality_A']:.2f} "
            f"both on {h['both_present_A']:.2f} jitter {h['pitch_jitter_cents']:.1f} ct | vocal pitch: vocality A "
            f"{hv['vocality_A']:.2f} jitter {hv['pitch_jitter_cents']:.1f} ct | synth {val['synth']['pitch_jitter_cents']:.1f} "
            f"vocal {val['vocal']['pitch_jitter_cents']:.1f} mix {val['mix']['pitch_jitter_cents']:.1f} ct  "
            f"KID/mix {val['kid_relative_to_mix']:.2f}  vowels kept {h.get('articulation_keep_acc', float('nan')):.2f} "
            f"(voice itself {val['vocal'].get('articulation_keep_acc', float('nan')):.2f}) controlled "
            f"{h.get('articulation_control_acc', float('nan')):.2f}  ({time.time() - tick:.0f} s)")
        if step == args.steps_per_epoch:
            demos = write_demos(G, catalog, demo_pairs, args.out / "demos" / f"epoch_{epoch:02d}", art)
            events.write("demos", epoch=point, folder=f"demos/epoch_{epoch:02d}", pairs=demos)

    for epoch in range(start_epoch, args.epochs + 1):
        started, parts = time.time(), []
        if epoch == start_epoch and resume_parts is not None:
            started, parts = time.time() - resume_seconds, list(resume_parts)
        first = start_step + 1 if epoch == start_epoch else 1
        for step in range(first, args.steps_per_epoch + 1):
            if ctl.pause_requested(args.out):           # pause: save the exact state, then stop
                tick = time.time()
                ctl.save_state(full_state(epoch, step - 1, parts, started), ckpt / "paused.pt")
                ctl.clear_pause(args.out)
                point = epoch - 1 + (step - 1) / args.steps_per_epoch
                events.write("paused", epoch=point, global_step=(epoch - 1) * args.steps_per_epoch + step - 1,
                             file="checkpoints/paused.pt")
                log(f"paused at epoch {epoch} after step {step - 1}: state saved to checkpoints/paused.pt "
                    f"({time.time() - tick:.0f} s); continue with --resume")
                return
            tick = time.time()
            pairs = tm.sample_pairs(rng, catalog, pools, args.batch, alpha=0.5)
            vocal, synth, image, planes, cond, midis = load_batch(catalog, pairs)
            n = len(pairs)
            a_low = torch.from_numpy(rng.uniform(0.2, 0.45, n).astype(np.float32))
            a_high = torch.from_numpy(rng.uniform(0.55, 0.8, n).astype(np.float32))
            second = torch.where(torch.from_numpy(rng.random(n) < 0.5), a_low, a_high)
            alphas = [torch.full((n,), 0.5), second]
            pitch = torch.from_numpy(rng.integers(0, 2, n)).long()        # pitch source per pair: synth / vocal
            art_target, _ = articulation_batch(rng, pairs, art)            # keep / other vowel / vowel glide
            data_seconds = time.time() - tick
            global_step = (epoch - 1) * args.steps_per_epoch + step
            adv_weight = args.adv * min(1.0, max(0.0, (global_step - 100) / 200))
            with torch.no_grad():
                rows_h = tm.harmonic_rows(midis, torch.device("cpu"))
                refs_all = reference_embeddings(judges, vocal, synth, alphas)
                lv, ls = tm.row_levels(vocal).gather(1, rows_h), tm.row_levels(synth).gather(1, rows_h)
                base = {"energy_v": tm.energy(vocal), "energy_s": tm.energy(synth), "rows": rows_h,
                        "anchor_ref": torch.maximum(lv, ls)}
            for c in critics:
                c.requires_grad_(False)
            enc = G.encode(planes, cond, image)
            ev, es = split_encoding(enc, slice(0, n)), split_encoding(enc, slice(n, 2 * n))
            total, part_sum, vocs_all, fakes = 0.0, defaultdict(float), [], []
            for ai, alpha in enumerate(alphas):
                hybrid, extras = G.decode(ev, es, alpha, vocal, synth, pitch, art_target, midis)
                refs = dict(base, emb=[refs_all[j][ai] for j in range(len(models))])
                parts_, vocs, view_h, cview_h = hybrid_losses(hybrid, extras, alpha, judges, critics, refs, gen, real_feats,
                                                              art_judge, art_target)
                total = total + sum(weights[k] * v for k, v in parts_.items() if k != "realism") / 2
                total = total + adv_weight * parts_["realism"] / 2
                for k, v in parts_.items():
                    part_sum[k] += (float(v.detach()) if torch.is_tensor(v) else float(v)) / 2
                vocs_all.append(vocs)
                fakes.append((view_h.detach(), cview_h.detach()))
                if ai == 0:
                    with torch.no_grad():
                        logits_a = models[0](view_h)[0]
                        part_sum["p_vox_A"] = float(vox_probability(logits_a).mean())
                        part_sum["vox_present_A"] = float(torch.sigmoid(logits_a[:, 0]).mean())
                        part_sum["synth_present_A"] = float(torch.sigmoid(logits_a[:, 1]).mean())
                        part_sum["vocality_A"] = float(vocs[0].mean())
            sign = torch.sign(alphas[1] - alphas[0])
            rank = sum(F.relu(0.5 * (alphas[1] - alphas[0]).abs() - sign * (v1_ - v0_)).mean()
                       for v0_, v1_ in zip(vocs_all[0], vocs_all[1])) / len(models)
            total = total + rank_weight * rank
            part_sum["rank"] = float(rank)
            opt_g.zero_grad()
            total.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0))
            opt_g.step()
            sched.step()
            # critics: real notes of every class (both cached views of the same notes) vs both hybrids
            for c in critics:
                c.requires_grad_(True)
            idx = _real_indices(rng, pools, 2 * n)
            real_views = [augment_view(cache["views"][idx].float(), gen), critic_cache["views"][idx].float()]
            fake_views = [augment_view(torch.cat([f[0] for f in fakes]), gen), torch.cat([f[1] for f in fakes])]
            c_loss, new_real = 0.0, []
            for c, rv, fv in zip(critics, real_views, fake_views):
                score_r, feats_r = c(rv)
                score_f, _ = c(fv)
                c_loss = c_loss + F.relu(1 - score_r).mean() + F.relu(1 + score_f).mean()
                new_real.append([f.detach().mean(0) for f in feats_r])
            opt_c.zero_grad()
            c_loss.backward()
            opt_c.step()
            real_feats = new_real
            parts.append(dict(part_sum) | {"critic": float(c_loss) / 2, "total": float(total)})
            if step % args.log_every == 0:
                window = parts[-args.log_every:]
                mean = {k: round(float(np.mean([p[k] for p in window])), 4) for k in window[0]}
                events.write("train", epoch=epoch - 1 + step / args.steps_per_epoch, global_step=global_step, losses=mean,
                             lr=sched.get_last_lr()[0], grad_norm={"generator": grad}, adv_weight=adv_weight,
                             sec_per_step=round(time.time() - tick, 2), data_seconds=round(data_seconds, 2))
                log(f"epoch {epoch:3d} step {step:5d}/{args.steps_per_epoch}  vocality {mean['vocality']:.3f} "
                    f"(A {mean['vocality_A']:.2f})  p_vox(A) {mean['p_vox_A']:.2f}  present vox {mean['vox_present_A']:.2f} "
                    f"synth {mean['synth_present_A']:.2f}  rank {mean['rank']:.3f}  novelty {mean['novelty']:.3f}  "
                    f"between {mean['between']:.3f}  match {mean['matching']:.3f}  artic {mean['articulation']:.3f}  "
                    f"critic {mean['critic']:.2f}  "
                    f"{time.time() - tick:.1f} s/step")
            if step % quarter == 0 and step < args.steps_per_epoch:
                checkpoint(epoch, step, parts, started)
        checkpoint(epoch, args.steps_per_epoch, parts, started)
    log(f"done; best score {best:.4f}")


def _real_indices(rng, pools, size):
    n_syn, n_vox, n_exh = size // 2, size // 4, size // 8
    vox = sum(pools.vox_split("train").values(), [])
    return (pools.synth_sample(rng, "train", n_syn) + [vox[i] for i in rng.integers(len(vox), size=n_vox)]
            + [pools.ref["processed_vox"][i] for i in rng.integers(len(pools.ref["processed_vox"]), size=n_exh)]
            + [pools.ref["synth_vox"][i] for i in rng.integers(len(pools.ref["synth_vox"]), size=size - n_syn - n_vox - n_exh)])


if __name__ == "__main__":
    main()
