"""Train the vox-synth morph (MorphNet, wavelet2vec.morph): judges first, then the generator.

Second purpose of note_dataset: short notes heard as partly vocal and partly
synth (about 50 % by the judges), not a plain mix and not a vocoder. Runs on
the GPU by default (the CPU is busy with KeyboardNet v3).

Phase 1, judges (minutes):
- a cache of judge views for every note in MIDI 36-82 (vox, synth, Exhale,
  Harmor vox) plus 512 generated noise notes;
- JudgeCNN and JudgeTCN, heads "vox present" / "synth present", trained on
  clean vox, synth (balanced per envelope class and family) and noise;
- leave-one-singer-out checks (does "voice" transfer between the singers?);
- a held-out judge never used for generation: Wavelet2Vec embedding (mid,
  low-passed at 18 kHz) + logistic regression.

Phase 2, generator: pairs of a vocal note and a synth note at the same key,
vocal amount alpha (0.5 half the time, else 0.2-0.8). Losses:
- balance: both judges' vox-vs-synth logit difference -> logit(alpha), on
  randomly shifted / low-passed views (harder to fool with small changes);
- presence: both "vox present" and "synth present" confidently on (logit >= 2),
  so the "neither" solution (unrecognized, noise-like sound that splits the
  vote evenly) does not pay; seen in the first attempt;
- between: in judge embeddings, the hybrid is no farther from either note
  than the two notes are from each other;
- novelty: in judge embeddings, the hybrid keeps a distance (a quarter of
  the vocal-synth distance) from the plain mix and both vocoders;
- realism: hinge-GAN critic against real notes of every class;
- loudness, harmonic anchor (pitch kept), residual budget, gain smoothness.

Every half epoch: checkpoint ``{epoch:g}_epoch.pth``, validation on held-out
singers' takes and held-out synth families (all three judges, baselines
alongside); every epoch: listening demos (192 kHz WAV: hybrid, vocal, synth,
mix, vocoder).

    PYTHONPATH=src python experiments/train_morph.py [--epochs 20 --steps-per-epoch 1000 --batch 4]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from wavelet2vec.morph import (JUDGE_ROWS, SPEC, Critic, JudgeCNN, JudgeTCN, MorphConfig, MorphNet, augment_view,
                               count_parameters, judge_view, mix_baseline, power, vocoder_baseline, vox_probability)
from wavelet2vec.wavelet_image import WaveletImage, decode, encode

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "keyboard_v3_data"
OUT = HERE / "output" / "morph" / "v1"
KEYS = (36, 82)                          # the vox range: both classes are judged here only
TARGET_RMS = 0.1                         # loudness normalization (-20 dBFS)
VAL_VOX_FILES = ("from_C1_till_B10_kontakt_shalanaja_tze.wav", "from_C1_till_B10_kontakt_shalanaja_koh_.wav",
                 "from_E_real_Vocal_recording_melodyne_sound_EH.wav")
PIUW = "from_C_real_Vocal_recording_melodyne.wav"
PRESENT = 2.0                            # presence target logit for both detectors (sigmoid ~ 0.88)


class EventLog:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, kind, **fields):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": kind, "time": time.time(), **fields}, default=float) + "\n")


# ----------------------------------------------------------------------------------------------- data


class Catalog:
    """Notes of note_dataset in the v3 store (192 kHz stereo float32), grouped for the two phases."""

    def __init__(self, root=DATA, val_synth_families=8, seed=5):
        info = json.loads((root / "index.json").read_text(encoding="utf-8"))
        self.arrays, self.notes = {}, []
        for n in info["notes"]:
            if not KEYS[0] <= n["midi"] <= KEYS[1]:
                continue
            if n["array"] not in self.arrays:
                self.arrays[n["array"]] = np.load(root / "notes" / n["array"], mmap_mode="r")
            kind = n["source_type"]
            singer = ("shevannai" if "shalanaja" in n["file"] else "own") if kind == "vox" else None
            self.notes.append({**n, "kind": kind, "singer": singer})
        rng = np.random.default_rng(seed)
        synth_families = sorted({n["family"] for n in self.notes if n["kind"] == "synth"})
        single = [f for f in synth_families if len({n["file"] for n in self.notes if n["family"] == f}) == 1]
        self.val_families = sorted(rng.choice(single, val_synth_families, replace=False).tolist())
        self.train_families = [f for f in synth_families if f not in self.val_families]

    def audio(self, i):
        n = self.notes[i]
        return np.asarray(self.arrays[n["array"]][n["row"]], dtype=np.float32)

    def normalized(self, i):
        x = self.audio(i)
        rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
        return (x * (TARGET_RMS / max(rms, 1e-6))).astype(np.float32), rms


def noise_note(rng, n=SPEC.audio_length, sr=SPEC.sample_rate):
    """A note-shaped noise burst (white, pink, brown, band-pass or clicks; random envelope and width):
    a negative for both presence heads."""
    freqs = np.fft.rfftfreq(n, 1 / sr)
    freqs[0] = freqs[1]
    kind = rng.choice(["white", "pink", "brown", "band", "band", "clicks"])
    if kind == "clicks":
        x = np.zeros((2, n))
        for t in rng.integers(0, n, rng.integers(3, 60)):
            tail = np.exp(-np.arange(min(4000, n - t)) / rng.uniform(20, 800))
            x[:, t:t + tail.size] += rng.standard_normal((2, 1)) * tail
    else:
        spectrum = rng.standard_normal((2, freqs.size)) + 1j * rng.standard_normal((2, freqs.size))
        shape = {"white": np.ones_like(freqs), "pink": freqs ** -0.5, "brown": freqs ** -1.0}.get(kind)
        if shape is None:
            centre, width = np.exp(rng.uniform(np.log(60), np.log(12000))), rng.uniform(0.1, 1.5)
            shape = np.exp(-0.5 * (np.log2(freqs / centre) / width) ** 2)
        shape = shape * (freqs < rng.uniform(4000, 90000))
        x = np.fft.irfft(spectrum * shape, n=n)
    width = rng.uniform(0, 1)
    mid, side = x.mean(0), (x[0] - x[1]) / 2 * width
    t = np.arange(n) / sr
    attack, decay, sustain = rng.uniform(0.001, 0.2), rng.uniform(0.05, 1.0), rng.uniform(0, 0.8)
    env = np.where(t < attack, t / attack, sustain + (1 - sustain) * np.exp(-(t - attack) / decay))
    env *= np.clip((1.15 - t) / 0.05, 0, 1)
    y = np.stack([mid + side, mid - side]) * env
    return (y * TARGET_RMS / max(np.sqrt((y ** 2).mean()), 1e-9)).astype(np.float32)


def take(image: WaveletImage, index) -> WaveletImage:
    return WaveletImage(image.spec, {c: v[index] for c, v in image.groups.items()}, dict(image.rows))


def encode_batch(audio: np.ndarray, device) -> WaveletImage:
    return encode(torch.from_numpy(audio).to(device), SPEC)


@torch.no_grad()
def build_cache(catalog: Catalog, device, path: Path, noise_count=512, batch=24):
    if path.exists():
        return torch.load(path, weights_only=False)
    started = time.time()
    views, rms = [], []
    for start in range(0, len(catalog.notes), batch):
        idx = range(start, min(start + batch, len(catalog.notes)))
        pairs = [catalog.normalized(i) for i in idx]
        rms += [r for _, r in pairs]
        views.append(judge_view(encode_batch(np.stack([a for a, _ in pairs]), device)).half().cpu())
        if start % (batch * 40) == 0:
            print(f"  judge views {start}/{len(catalog.notes)}  {time.time() - started:.0f} s", flush=True)
    rng = np.random.default_rng(99)
    noise = []
    for start in range(0, noise_count, batch):
        audio = np.stack([noise_note(rng) for _ in range(min(batch, noise_count - start))])
        noise.append(judge_view(encode_batch(audio, device)).half().cpu())
    cache = {"views": torch.cat(views), "rms": np.array(rms), "noise": torch.cat(noise)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"  cache: {len(cache['views'])} notes + {len(cache['noise'])} noise in {time.time() - started:.0f} s", flush=True)
    return cache


class Pools:
    """Index lists for sampling: per class, singer, family, envelope; train / val; audible notes only."""

    def __init__(self, catalog: Catalog, rms, min_rms=1e-4):
        ok = rms > min_rms
        self.vox = defaultdict(list)                       # (split, singer) -> idx
        self.synth = defaultdict(lambda: defaultdict(list))  # split -> (envelope, family) -> idx
        self.synth_key = defaultdict(lambda: defaultdict(list))  # split -> (family, midi) -> idx
        self.ref = defaultdict(list)                       # processed_vox / synth_vox -> idx
        for i, n in enumerate(catalog.notes):
            if not ok[i]:
                continue
            if n["kind"] == "vox":
                self.vox["val" if n["file"] in VAL_VOX_FILES else "train", n["singer"]].append(i)
            elif n["kind"] == "synth":
                split = "val" if n["family"] in catalog.val_families else "train"
                self.synth[split][n["envelope"], n["family"]].append(i)
                self.synth_key[split][n["family"], n["midi"]].append(i)
            else:
                self.ref[n["kind"]].append(i)

    def vox_split(self, split, singers=("shevannai", "own")):
        return {s: self.vox[split, s] for s in singers if self.vox[split, s]}

    def synth_sample(self, rng, split, size):
        groups = self.synth[split]
        by_env = defaultdict(list)
        for env, fam in groups:
            by_env[env].append(fam)
        envs = sorted(by_env)
        out = []
        for _ in range(size):
            env = envs[rng.integers(len(envs))]
            fam = by_env[env][rng.integers(len(by_env[env]))]
            pool = groups[env, fam]
            out.append(pool[rng.integers(len(pool))])
        return out


# ----------------------------------------------------------------------------------------------- judges


def judge_batch(rng, pools: Pools, cache, size, singers=("shevannai", "own"), noise_split=(0, 448)):
    n_vox, n_syn = int(0.4 * size), int(0.4 * size)
    vox = pools.vox_split("train", singers)
    names = sorted(vox)
    vi = [vox[names[k % len(names)]][rng.integers(len(vox[names[k % len(names)]]))] for k in range(n_vox)]
    si = pools.synth_sample(rng, "train", n_syn)
    ni = rng.integers(*noise_split, size - n_vox - n_syn)
    views = torch.cat([cache["views"][vi + si], cache["noise"][ni]]).float()
    labels = torch.zeros(len(views), 2)
    labels[:n_vox, 0] = 1
    labels[n_vox:n_vox + n_syn, 1] = 1
    return views, labels


def train_judge(kind, pools, cache, device, steps=2500, batch=64, seed=0, singers=("shevannai", "own"), log=None):
    torch.manual_seed(seed)
    model = (JudgeCNN() if kind == "cnn" else JudgeTCN()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    started = time.time()
    for step in range(1, steps + 1):
        views, labels = judge_batch(rng, pools, cache, batch, singers)
        views = augment_view(views, gen).to(device)
        if rng.random() < 0.5:   # random noise floor between -60 and -45 dB (recordings vs digital silence)
            floor = rng.uniform(-1.0, -0.5)
            views[:, 0] = torch.clamp(views[:, 0], min=floor)
        logits, _ = model(views)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % 500 == 0 or step == 20):
            log(f"  judge {kind} {'+'.join(singers)} step {step}/{steps} loss {loss.item():.4f} "
                f"{time.time() - started:.0f} s")
    return model.eval()


@torch.no_grad()
def judge_scores(model, views, device, batch=256):
    logits = torch.cat([model(views[i:i + batch].float().to(device))[0].cpu() for i in range(0, len(views), batch)])
    return logits


def judge_report(model, pools, cache, device, rng):
    from sklearn.metrics import roc_auc_score
    sets = {"vox (held-out takes)": sum(pools.vox_split("val").values(), []),
            "synth (held-out families)": pools.synth_sample(rng, "val", 600),
            "Exhale (processed vox, reference)": pools.ref["processed_vox"],
            "Harmor vox (synth vox, reference)": pools.ref["synth_vox"]}
    out = {}
    for name, idx in sets.items():
        logits = judge_scores(model, cache["views"][idx], device)
        out[name] = {"notes": len(idx), "p_vox": round(float(vox_probability(logits).mean()), 4),
                     "vox_present": round(float(torch.sigmoid(logits[:, 0]).mean()), 4),
                     "synth_present": round(float(torch.sigmoid(logits[:, 1]).mean()), 4)}
    noise = judge_scores(model, cache["noise"][448:], device)
    out["noise (held-out)"] = {"notes": len(noise), "vox_present": round(float(torch.sigmoid(noise[:, 0]).mean()), 4),
                               "synth_present": round(float(torch.sigmoid(noise[:, 1]).mean()), 4)}
    pv = judge_scores(model, cache["views"][sets["vox (held-out takes)"]], device)
    ps = judge_scores(model, cache["views"][sets["synth (held-out families)"]], device)
    p = torch.cat([vox_probability(pv), vox_probability(ps)]).numpy()
    y = np.r_[np.ones(len(pv)), np.zeros(len(ps))]
    out["held-out vox vs synth"] = {"auc": round(float(roc_auc_score(y, p)), 4),
                                    "accuracy": round(float(((p > 0.5) == y).mean()), 4)}
    return out


def loso_report(kind, pools, cache, device, log, steps=1200):
    """Leave one singer out: train on one voice, score the other voice's notes (all takes)."""
    out = {}
    for train_singer, test_singer in (("shevannai", "own"), ("own", "shevannai")):
        model = train_judge(kind, pools, cache, device, steps=steps, seed=3, singers=(train_singer,), log=log)
        idx = pools.vox["train", test_singer] + pools.vox["val", test_singer]
        p = vox_probability(judge_scores(model, cache["views"][idx], device))
        out[f"train {train_singer} -> test {test_singer}"] = {"notes": len(idx), "p_vox": round(float(p.mean()), 4),
                                                             "recall": round(float((p > 0.5).float().mean()), 4)}
    return out


def w2v_features(audio: np.ndarray) -> np.ndarray:
    """Held-out judge features: Wavelet2Vec of the mid channel, low-passed at 18 kHz, at 48 kHz."""
    from scipy.signal import butter, resample_poly, sosfiltfilt
    from wavelet2vec import Wavelet2Vec
    global _W2V, _SOS
    if "_W2V" not in globals():
        _W2V, _SOS = Wavelet2Vec(), butter(8, 18000, fs=48000, output="sos")
    mid = resample_poly(audio.mean(0).astype(np.float64), 1, 4)
    return _W2V.embed(sosfiltfilt(_SOS, mid).astype(np.float32), 48000)


def train_w2v_judge(catalog, pools, log, per_family=18, seed=4):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    started = time.time()

    def embed(idx):
        return np.stack([w2v_features(catalog.normalized(i)[0]) for i in idx])

    vox_train = sum(pools.vox_split("train").values(), [])
    synth_train = [i for fam_idx in _per_family(pools, "train", per_family, rng) for i in fam_idx]
    x = np.concatenate([embed(vox_train), embed(synth_train)])
    y = np.r_[np.ones(len(vox_train)), np.zeros(len(synth_train))]
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000))
    probe.fit(x, y)
    log(f"  held-out judge (Wavelet2Vec + logistic): {len(x)} notes embedded in {time.time() - started:.0f} s")
    report = {}
    val_sets = {"vox (held-out takes)": sum(pools.vox_split("val").values(), []),
                "synth (held-out families)": [i for f in _per_family(pools, "val", 12, rng) for i in f],
                "Exhale (processed vox, reference)": list(rng.choice(pools.ref["processed_vox"], 60, replace=False)),
                "Harmor vox (synth vox, reference)": list(rng.choice(pools.ref["synth_vox"],
                                                                     min(40, len(pools.ref["synth_vox"])), replace=False))}
    for name, idx in val_sets.items():
        p = probe.predict_proba(embed(idx))[:, 1]
        report[name] = {"notes": len(idx), "p_vox": round(float(p.mean()), 4), "vox_rate": round(float((p > 0.5).mean()), 4)}
    return probe, report


def _per_family(pools, split, per_family, rng):
    fams = defaultdict(list)
    for (env, fam), idx in pools.synth[split].items():
        fams[fam] += idx
    return [list(rng.choice(idx, min(per_family, len(idx)), replace=False)) for fam, idx in sorted(fams.items())]


# ----------------------------------------------------------------------------------------------- generator


def sample_pairs(rng, catalog, pools, size, split="train", alpha=None):
    vox = pools.vox_split(split)
    singers = sorted(vox)
    pairs = []
    while len(pairs) < size:
        singer = singers[rng.integers(len(singers))]
        vi = vox[singer][rng.integers(len(vox[singer]))]
        midi = catalog.notes[vi]["midi"]
        fams = [f for (f, m) in pools.synth_key[split] if m == midi]
        if not fams:
            continue
        fam = fams[rng.integers(len(fams))]
        si = pools.synth_key[split][fam, midi][rng.integers(len(pools.synth_key[split][fam, midi]))]
        a = alpha if alpha is not None else (0.5 if rng.random() < 0.5 else float(rng.uniform(0.2, 0.8)))
        pairs.append((vi, si, a))
    return pairs


def load_pairs(catalog, pairs, device):
    audio = np.stack([catalog.normalized(v)[0] for v, _, _ in pairs] + [catalog.normalized(s)[0] for _, s, _ in pairs])
    with torch.no_grad():
        image = encode_batch(audio, device)
    n = len(pairs)
    alpha = torch.tensor([a for _, _, a in pairs], device=device, dtype=torch.float32)
    return take(image, slice(0, n)), take(image, slice(n, 2 * n)), alpha


def energy(image: WaveletImage) -> torch.Tensor:
    """[B] core energy, mid + side."""
    core = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)
    return sum(power(v[:, :, :, core]).sum((1, 2, 3, 4)) / v.shape[-1] for v in image.groups.values())


def row_levels(image: WaveletImage) -> torch.Tensor:
    """[B, ROWS] time-averaged mid level per row in dB (core frames)."""
    core = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)
    first = next(iter(image.groups.values()))
    out = torch.zeros(first.shape[0], SPEC.n_bands, device=first.device, dtype=first.real.dtype)
    for c, v in image.groups.items():
        rows = torch.as_tensor(list(image.rows[c]), device=first.device)
        out = out.index_copy(1, rows, power(v[:, 0, :, core]).mean((-1, -2)))
    return 10 * torch.log10(out + 1e-30)


def harmonic_rows(midis, device):
    rows = torch.tensor([[m - SPEC.first_midi + round(12 * math.log2(k)) for k in (1, 2, 3, 4)] for m in midis],
                        device=device)
    return rows.clamp(max=SPEC.n_bands - 1)


def judge_outputs(judges, view):
    return [j(view) for j in judges]


def generator_losses(G, judges, critic, vocal, synth, alpha, midis, weights, gen, adv_weight):
    hybrid, extras = G(vocal, synth, alpha)
    view_h = judge_view(hybrid)
    with torch.no_grad():
        baselines = [mix_baseline(vocal, synth, alpha), vocoder_baseline(synth, vocal), vocoder_baseline(vocal, synth)]
        ref_views = [judge_view(x) for x in [vocal, synth] + baselines]
        ref_emb = [[j(v)[1] for v in ref_views] for j in judges]
    target = torch.logit(alpha.clamp(0.02, 0.98))
    balance, presence, novelty, between = 0.0, 0.0, 0.0, 0.0
    for k, judge in enumerate(judges):
        logits, _ = judge(augment_view(view_h, gen))
        balance = balance + F.smooth_l1_loss(logits[:, 0] - logits[:, 1], target)
        # both detectors confidently on (logit >= PRESENT): "neither" (noise-like, unrecognized) must not pay
        presence = presence + (F.softplus(PRESENT - logits[:, 0]) + F.softplus(PRESENT - logits[:, 1])).mean()
        _, emb = judge(view_h)
        e_v, e_s, *e_b = ref_emb[k]
        span = (1 - F.cosine_similarity(e_v, e_s)).clamp(min=0.05)
        margin = 0.25 * span
        for e in e_b:
            novelty = novelty + F.relu(1 - (1 - F.cosine_similarity(emb, e)) / margin).mean()
        # between the two notes: no farther from either note than they are from each other
        between = between + (F.relu((1 - F.cosine_similarity(emb, e_v)) / span - 1)
                             + F.relu((1 - F.cosine_similarity(emb, e_s)) / span - 1)).mean()
    n = len(judges)
    balance, presence, novelty, between = balance / n, presence / n, novelty / (n * 3), between / n
    realism = -critic(augment_view(view_h, gen)).mean()
    e_h, e_v, e_s = energy(hybrid), energy(vocal), energy(synth)
    loudness = ((10 * torch.log10((e_h + 1e-12) / (0.5 * (e_v + e_s) + 1e-12))) ** 2).mean() / 100
    rows = harmonic_rows(midis, view_h.device)
    lh, lv, ls = (row_levels(x).gather(1, rows) for x in (hybrid, vocal, synth))
    anchor = (F.relu(torch.maximum(lv, ls) - 15 - lh) / 10).mean()
    residual = F.relu(extras["residual_energy"] / (e_h + 1e-12) - 0.5).mean()
    level = extras["gain_level"]
    smooth = (level[..., 1:] - level[..., :-1]).abs().mean()
    parts = {"balance": balance, "presence": presence, "novelty": novelty, "between": between, "realism": realism,
             "loudness": loudness, "anchor": anchor, "residual": residual, "smooth": smooth}
    total = sum(weights[k] * v for k, v in parts.items() if k != "realism") + adv_weight * realism
    with torch.no_grad():
        logits_a = judges[0](view_h)[0]
        monitor = {"p_vox_A": float(vox_probability(logits_a).mean()),
                   "vox_present_A": float(torch.sigmoid(logits_a[:, 0]).mean()),
                   "synth_present_A": float(torch.sigmoid(logits_a[:, 1]).mean())}
    return total, {k: float(v.detach()) for k, v in parts.items()} | monitor, view_h


def real_views(rng, pools, cache, size):
    """Real notes of every class for the critic: synth (family/envelope balanced), vox, Exhale, Harmor vox."""
    n_syn, n_vox, n_exh = size // 2, size // 4, size // 8
    vox = sum(pools.vox_split("train").values(), [])
    idx = (pools.synth_sample(rng, "train", n_syn) + [vox[i] for i in rng.integers(len(vox), size=n_vox)]
           + [pools.ref["processed_vox"][i] for i in rng.integers(len(pools.ref["processed_vox"]), size=n_exh)]
           + [pools.ref["synth_vox"][i] for i in rng.integers(len(pools.ref["synth_vox"]), size=size - n_syn - n_vox - n_exh)])
    return cache["views"][idx].float()


# ----------------------------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(G, judges, critic, probe, catalog, pools, cache, pairs, device, batch=4):
    G.eval()
    names = ("hybrid", "mix", "vocoder: synth carrier", "vocoder: vocal carrier", "vocal", "synth")
    rows = defaultdict(lambda: defaultdict(list))
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        vocal, synth, alpha = load_pairs(catalog, chunk, device)
        hybrid, _ = G(vocal, synth, alpha)
        images = [hybrid, mix_baseline(vocal, synth, alpha), vocoder_baseline(synth, vocal),
                  vocoder_baseline(vocal, synth), vocal, synth]
        views = [judge_view(x) for x in images]
        for jn, judge in zip(("A", "B"), judges):
            outs = [judge(v) for v in views]
            margin = (1 - F.cosine_similarity(outs[4][1], outs[5][1])).clamp(min=1e-3)
            for name, (logits, emb) in zip(names, outs):
                rows[name][f"p_vox_{jn}"] += vox_probability(logits).tolist()
                rows[name][f"presence_{jn}"] += torch.minimum(torch.sigmoid(logits[:, 0]),
                                                              torch.sigmoid(logits[:, 1])).tolist()
                if name != "hybrid":
                    rows["hybrid"][f"distance_to_{name}_{jn}"] += ((1 - F.cosine_similarity(outs[0][1], emb)) / margin).tolist()
        for name, v in zip(names, views):
            rows[name]["critic"] += critic(v).tolist()
        if probe is not None:
            for name, image in zip(names[:4], images[:4]):
                audio = decode(image).float().cpu().numpy()
                rows[name]["p_vox_J"] += probe.predict_proba(np.stack([w2v_features(a) for a in audio]))[:, 1].tolist()
        rows["hybrid"]["alpha"] += alpha.tolist()
    real = torch.cat([cache["views"][sum(pools.vox_split("val").values(), [])],
                      cache["views"][pools.synth_sample(np.random.default_rng(7), "val", 120)]]).float().to(device)
    real_critic = float(critic(real).mean())
    G.train()
    summary = {}
    for name, cols in rows.items():
        summary[name] = {k: round(float(np.mean(v)), 4) for k, v in cols.items() if k != "alpha"}
        summary[name]["realism_gap"] = round(real_critic - summary[name]["critic"], 4)
        for j in ("A", "B", "J"):
            if f"p_vox_{j}" in cols:
                err = np.abs(np.array(cols[f"p_vox_{j}"]) - np.array(rows["hybrid"]["alpha"][:len(cols[f"p_vox_{j}"])]))
                summary[name][f"balance_error_{j}"] = round(float(err.mean()), 4)
    h = summary["hybrid"]
    novelty_short = np.mean([max(0.0, 0.25 - h[k]) / 0.25 for k in h if k.startswith("distance_to_mix")
                             or k.startswith("distance_to_vocoder")])
    errors = [h[f"balance_error_{j}"] for j in ("A", "B", "J") if f"balance_error_{j}" in h]
    summary["score"] = round(float(np.mean(errors) + 0.5 * novelty_short + 0.1 * max(h["realism_gap"], 0.0)), 4)
    return summary


@torch.no_grad()
def write_demos(G, catalog, pairs, device, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    G.eval()
    files = []
    for k, (v, s, a) in enumerate(pairs):
        vocal, synth, alpha = load_pairs(catalog, [(v, s, a)], device)
        hybrid, _ = G(vocal, synth, alpha)
        images = {"hybrid": hybrid, "vocal": vocal, "synth": synth, "mix": mix_baseline(vocal, synth, alpha),
                  "vocoder": vocoder_baseline(synth, vocal)}
        nv, ns = catalog.notes[v], catalog.notes[s]
        entry = {"pair": k, "midi": nv["midi"], "alpha": a, "vocal": f"{nv['family']}: {nv['file']}",
                 "synth": f"{ns['family']}: {ns['file']}", "files": {}}
        for name, image in images.items():
            audio = decode(image)[0].float().cpu().numpy()
            peak = float(np.abs(audio).max())
            if peak > 0.99:
                audio = audio * (0.99 / peak)
            path = folder / f"pair{k:02d}_{name}.wav"
            sf.write(path, audio.T, SPEC.sample_rate, subtype="FLOAT")
            entry["files"][name] = path.name
        files.append(entry)
    (folder / "demos.json").write_text(json.dumps(files, indent=1), encoding="utf-8")
    G.train()
    return files


# ----------------------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--adv", type=float, default=0.3, help="realism (critic) weight after warm-up")
    parser.add_argument("--adv-warmup", type=int, default=200)
    parser.add_argument("--judge-steps", type=int, default=2500)
    parser.add_argument("--val-pairs", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=OUT.parent / "judge_cache.pt", help="judge views, shared by runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-w2v", action="store_true", help="no held-out Wavelet2Vec judge (faster checks)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    events = EventLog(args.out / "metrics.jsonl")

    def log(text):
        print(text, flush=True)

    catalog = Catalog()
    log(f"notes in MIDI {KEYS[0]}-{KEYS[1]}: {len(catalog.notes)}; val synth families: {catalog.val_families}")
    cache = build_cache(catalog, device, args.cache)
    pools = Pools(catalog, cache["rms"])
    counts = {f"vox {s} {k}": len(v) for (k, s), v in pools.vox.items()}
    log(json.dumps(counts))

    # ---- phase 1: judges
    judges_path = args.out / "judges.pt"
    if judges_path.exists():
        saved = torch.load(judges_path, map_location="cpu", weights_only=False)
        judge_a, judge_b = JudgeCNN(), JudgeTCN()
        judge_a.load_state_dict(saved["A"])
        judge_b.load_state_dict(saved["B"])
        probe, reports = saved.get("probe"), saved["reports"]
        log("judges loaded from judges.pt")
    else:
        rng = np.random.default_rng(12)
        judge_a = train_judge("cnn", pools, cache, device, args.judge_steps, seed=1, log=log)
        judge_b = train_judge("tcn", pools, cache, device, args.judge_steps, seed=2, log=log)
        reports = {"A (CNN)": judge_report(judge_a, pools, cache, device, rng),
                   "B (TCN)": judge_report(judge_b, pools, cache, device, rng),
                   "leave one singer out (CNN)": loso_report("cnn", pools, cache, device, log,
                                                             steps=max(args.judge_steps // 2, 10))}
        probe = None
        if not args.skip_w2v:
            probe, reports["J (Wavelet2Vec + logistic, held out)"] = train_w2v_judge(catalog, pools, log)
        torch.save({"A": judge_a.cpu().state_dict(), "B": judge_b.cpu().state_dict(), "probe": probe,
                    "reports": reports}, judges_path)
        for name, rep in reports.items():
            events.write("judge", name=name, report=rep)
    log(json.dumps(reports, indent=1))
    judges = [judge_a.to(device).eval(), judge_b.to(device).eval()]
    for j in judges:
        j.requires_grad_(False)

    # ---- phase 2: generator
    G = MorphNet(MorphConfig()).to(device)
    critic = Critic().to(device)
    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, betas=(0.0, 0.99))
    total_steps = args.epochs * args.steps_per_epoch
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt_g, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))
    weights = {"balance": 1.0, "presence": 1.0, "novelty": 1.0, "between": 1.0, "loudness": 0.5, "anchor": 0.5,
               "residual": 1.0, "smooth": 0.02}
    rng = np.random.default_rng(1)
    gen = torch.Generator().manual_seed(1)
    start_epoch, start_step, best, history = 1, 0, float("inf"), []
    if args.resume and (ckpt / "last.pt").exists():
        state = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
        G.load_state_dict(state["G"])
        critic.load_state_dict(state["critic"])
        opt_g.load_state_dict(state["opt_g"])
        opt_c.load_state_dict(state["opt_c"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["rng"]
        best, history = state["best"], state["history"]
        done = state["step_in_epoch"]
        start_epoch, start_step = (state["epoch"] + 1, 0) if done >= args.steps_per_epoch else (state["epoch"], done)
        log(f"resumed at epoch {start_epoch} step {start_step}")

    val_rng = np.random.default_rng(31)
    val_pairs = sample_pairs(val_rng, catalog, pools, args.val_pairs, "val", alpha=0.5)
    val_pairs += [(v, s, a) for (v, s, _), a in zip(sample_pairs(val_rng, catalog, pools, 8, "val"), [0.25, 0.75] * 4)]
    demo_pairs = val_pairs[:6]
    piuw = [i for i in pools.vox["train", "own"] if catalog.notes[i]["file"] == PIUW]
    for vi in [piuw[len(piuw) // 3], piuw[2 * len(piuw) // 3]] if piuw else []:
        midi = catalog.notes[vi]["midi"]
        fam = sorted(f for (f, m) in pools.synth_key["val"] if m == midi)[0]
        demo_pairs.append((vi, pools.synth_key["val"][fam, midi][0], 0.5))
    run = {"task": "morph", "model": "morphnet-v1", "device": str(device),
           "parameters": {"total": count_parameters(G), "critic": count_parameters(critic),
                          "judge_A": count_parameters(judges[0]), "judge_B": count_parameters(judges[1])},
           "config": G.config.to_dict(), "batch": args.batch, "steps_per_epoch": args.steps_per_epoch,
           "epochs_total": args.epochs, "lr": args.lr, "loss_weights": weights, "adv": args.adv,
           "val_families": catalog.val_families, "val_vox_files": list(VAL_VOX_FILES),
           "train_notes": sum(len(v) for (k, _), v in pools.vox.items() if k == "train"),
           "judge_rows": JUDGE_ROWS, "keys": KEYS, "sample_rate": SPEC.sample_rate}
    log(json.dumps(run, indent=1, default=str))
    events.write("run_start", epoch=start_epoch - 1 + start_step / args.steps_per_epoch, resumed=bool(args.resume),
                 size=f"MorphNet {count_parameters(G) / 1e6:.1f}M", **run)
    half = args.steps_per_epoch // 2

    def checkpoint(epoch, step, parts, started):
        nonlocal best
        point = epoch - 1 + step / args.steps_per_epoch
        val = evaluate(G, judges, critic, probe, catalog, pools, cache, val_pairs, device)
        record = {"epoch": round(point, 3), "train": {k: round(float(np.mean([p[k] for p in parts])), 4) for k in parts[0]}
                  if parts else None, "score": val["score"], "minutes": round((time.time() - started) / 60, 1)}
        history.append(record)
        name = f"{point:g}_epoch.pth"
        torch.save({"G": G.state_dict(), "critic": critic.state_dict(), "config": G.config.to_dict(), "val": val,
                    "metrics": record}, ckpt / name)
        is_best = val["score"] < best
        if is_best:
            best = val["score"]
            shutil.copyfile(ckpt / name, ckpt / "best.pt")
        torch.save({"G": G.state_dict(), "critic": critic.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_c": opt_c.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                    "epoch": epoch, "step_in_epoch": step, "best": best, "history": history}, ckpt / "last.tmp")
        shutil.move(ckpt / "last.tmp", ckpt / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        events.write("val", epoch=point, morph=val)
        events.write("checkpoint", epoch=point, file=name, best=is_best, val_model=val["score"],
                     size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
        h = val["hybrid"]
        log(f"== checkpoint {name}: score {val['score']:.4f}{' (best)' if is_best else ''}  hybrid p_vox "
            f"A {h['p_vox_A']:.2f} B {h['p_vox_B']:.2f} J {h.get('p_vox_J', float('nan')):.2f}  "
            f"mix p_vox A {val['mix']['p_vox_A']:.2f}  realism gap {h['realism_gap']:.2f}")
        if step == args.steps_per_epoch:
            demos = write_demos(G, catalog, demo_pairs, device, args.out / "demos" / f"epoch_{epoch:02d}")
            events.write("demos", epoch=point, folder=f"demos/epoch_{epoch:02d}", pairs=demos)

    for epoch in range(start_epoch, args.epochs + 1):
        started, parts = time.time(), []
        first = start_step + 1 if epoch == start_epoch else 1
        for step in range(first, args.steps_per_epoch + 1):
            tick = time.time()
            pairs = sample_pairs(rng, catalog, pools, args.batch)
            vocal, synth, alpha = load_pairs(catalog, pairs, device)
            midis = [catalog.notes[v]["midi"] for v, _, _ in pairs]
            data_seconds = time.time() - tick
            global_step = (epoch - 1) * args.steps_per_epoch + step
            adv_weight = args.adv * min(1.0, max(0.0, (global_step - args.adv_warmup) / args.adv_warmup))
            critic.requires_grad_(False)                   # generator step: the critic only scores
            total, part, view_h = generator_losses(G, judges, critic, vocal, synth, alpha, midis, weights, gen,
                                                   adv_weight)
            opt_g.zero_grad()
            total.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0))
            opt_g.step()
            sched.step()
            critic.requires_grad_(True)                    # critic step on the detached hybrid
            real = augment_view(real_views(rng, pools, cache, args.batch * 2), gen).to(device)
            fake = augment_view(view_h.detach(), gen)
            c_loss = F.relu(1 - critic(real)).mean() + F.relu(1 + critic(fake)).mean()
            opt_c.zero_grad()
            c_loss.backward()
            opt_c.step()
            parts.append(part | {"critic": float(c_loss), "total": float(total)})
            if step % args.log_every == 0:
                window = parts[-args.log_every:]
                mean = {k: round(float(np.mean([p[k] for p in window])), 4) for k in window[0]}
                mem = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None
                events.write("train", epoch=epoch - 1 + step / args.steps_per_epoch, global_step=global_step,
                             losses=mean, lr=sched.get_last_lr()[0], grad_norm={"generator": grad},
                             adv_weight=adv_weight, sec_per_step=round(time.time() - tick, 2),
                             data_seconds=round(data_seconds, 2), gpu_gb=mem)
                log(f"epoch {epoch:3d} step {step:5d}/{args.steps_per_epoch}  balance {mean['balance']:.3f}  "
                    f"p_vox(A) {mean['p_vox_A']:.2f}  present: vox {mean['vox_present_A']:.2f} synth "
                    f"{mean['synth_present_A']:.2f}  novelty {mean['novelty']:.3f}  between {mean['between']:.3f}  "
                    f"realism {mean['realism']:+.2f}  critic {mean['critic']:.2f}  {time.time() - tick:.2f} s/step"
                    f"{f'  gpu {mem} GB' if mem else ''}")
            if step == half:
                checkpoint(epoch, step, parts, started)
        checkpoint(epoch, args.steps_per_epoch, parts, started)
    log(f"done; best score {best:.4f}")


if __name__ == "__main__":
    main()
