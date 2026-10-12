"""Evaluate MorphDiffusion stage 0: does a diffusion prior give realistic, two-sided vox-synth hybrids?

Writes ``evals/eval_<checkpoint>_<time>.json`` in the run folder (shown in the dashboard's Evaluations),
listening renders in ``demos/eval_<checkpoint>/`` (dashboard Listening), and a go/no-go verdict against the
criteria of docs/MORPH_DIFFUSION.md:

1. class fidelity: vox / synth samples judged as their class (vocality near 1 / 0), judges A and B;
2. realism: KID in judge-A embeddings against real held-out notes of the class, relative to the KID between
   real training and held-out notes;
3. memorization: generated vox samples' distance to their nearest training note, relative to the same
   distance for real held-out vox notes;
4. composition grid: "and" / "interp" x alpha 0.25, 0.5, 0.75 x guidance 2, 4, 6: vocality and its error,
   both detectors on (A, B), P(vox), KID against all real held-out notes, diversity;
5. references in the same metrics: the plain mix and the vocoder of real held-out pairs, and optionally
   MorphNet-L's hybrids of the same pairs (``--morphnet``, on the CPU);
6. alpha automated within the note (0 -> 1): vocality of the first vs the last third;
7. renders: composed samples imposed on the held-out pairs' synth notes (stable pitch).

    PYTHONPATH=src python experiments/eval_morph_diffusion.py [--checkpoint .../best.pt] [--morphnet .../best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_morph as tm  # noqa: E402
from train_morph_diffusion import OUT, FeatureData, judge_features, kid, load_judges  # noqa: E402
from wavelet2vec.morph import mix_baseline, vocoder_baseline  # noqa: E402
from wavelet2vec.morph_diffusion import build_denoiser, ddim_sample, render_features  # noqa: E402
from wavelet2vec.morph_large import critic_view  # noqa: E402
from wavelet2vec.wavelet_image import WaveletImage, decode, encode  # noqa: E402

CRITERIA = {
    "fidelity": "vox samples vocality_A >= 0.7 and synth samples vocality_A <= 0.3",
    "realism": "KID(generated vox / synth vs held-out) <= 3x KID(real training vs held-out)",
    "composition": "best 'and' setting at alpha 0.5: both detectors on (A) >= 0.5 and |vocality_A - 0.5| <= 0.2",
    "references": "that setting: both-on >= plain mix's and vocoder's (and MorphNet-L's if given), KID <= mix's",
    "automation": "alpha 0 -> 1 within the note: vocality_A last third - first third >= 0.3",
    "memorization": "median nearest-training distance of generated vox >= 0.5x that of real held-out vox",
}


def summarize(j, alpha=None, ref_embeddings=None):
    out = {}
    for name in ("A", "B"):
        out[f"p_vox_{name}"] = round(float(j[name]["p_vox"].mean()), 4)
        out[f"vocality_{name}"] = round(float(j[name]["vocality"].mean()), 4)
        out[f"both_present_{name}"] = round(float(j[name]["both"].float().mean()), 4)
        if alpha is not None:
            out[f"vocality_error_{name}"] = round(float((j[name]["vocality"] - alpha).abs().mean()), 4)
    e = F.normalize(j["A"]["embedding"], dim=-1)
    out["diversity_A"] = round(float((1 - e @ e.T).sum() / max(len(e) * (len(e) - 1), 1)), 4)
    if ref_embeddings is not None:                  # KID needs at least 3 notes on each side
        enough = len(j["A"]["embedding"]) > 2 and len(ref_embeddings) > 2
        out["kid_A"] = round(kid(j["A"]["embedding"], ref_embeddings), 4) if enough else float("nan")
    return out


def nearest_distance(x, bank):
    x, bank = F.normalize(x, dim=-1), F.normalize(bank, dim=-1)
    return (1 - x @ bank.T).min(1).values


@torch.no_grad()
def pair_views(catalog, pairs, device, morphnet=None, art=None):
    """Critic views of the vocal, synth, plain mix, vocoder (and MorphNet-L hybrid) of held-out pairs."""
    out = {k: [] for k in ("vocal", "synth", "mix", "vocoder", "morphnet")}
    images = []
    for v, s, a in pairs:
        audio = np.stack([catalog.normalized(v)[0], catalog.normalized(s)[0]])
        image = encode(torch.from_numpy(audio).to(device), tm.SPEC)
        vocal = WaveletImage(image.spec, {c: x[0:1] for c, x in image.groups.items()}, dict(image.rows))
        synth = WaveletImage(image.spec, {c: x[1:2] for c, x in image.groups.items()}, dict(image.rows))
        alpha = torch.tensor([a], device=device)
        for name, im in (("vocal", vocal), ("synth", synth), ("mix", mix_baseline(vocal, synth, alpha)),
                         ("vocoder", vocoder_baseline(synth, vocal))):
            out[name].append(critic_view(im).cpu())
        images.append(synth)
        if morphnet is not None:
            from wavelet2vec.morph_large import generate_hybrid
            hybrid = generate_hybrid(morphnet, audio[0], audio[1], catalog.notes[v]["midi"], a, "synth",
                                     None, sample_rate=tm.SPEC.sample_rate)
            h = encode(torch.from_numpy(hybrid[None]).float(), tm.SPEC)
            out["morphnet"].append(critic_view(h))
    return {k: torch.cat(v) for k, v in out.items() if v}, images


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=OUT)
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: <run>/checkpoints/best.pt")
    parser.add_argument("--n", type=int, default=64, help="samples per setting")
    parser.add_argument("--steps", type=int, default=40, help="DDIM steps")
    parser.add_argument("--pairs", type=int, default=24, help="held-out pairs for references and renders")
    parser.add_argument("--morphnet", type=Path, default=None, help="MorphNet-L checkpoint to compare (CPU, slow)")
    parser.add_argument("--renders", type=int, default=6)
    parser.add_argument("--max-batch", type=int, default=8, help="denoiser chunk size while sampling")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    ckpt_path = args.checkpoint or args.run / "checkpoints" / "best.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_denoiser(saved.get("meta") or {"arch": "unet", "chans": saved["chans"]})
    model.load_state_dict(saved["ema"])
    model = model.to(device)
    model.eval()
    data = FeatureData()
    models, centres = load_judges(device)
    gen = torch.Generator(device=device).manual_seed(123)
    rng = np.random.default_rng(123)
    started = time.time()
    report = {"checkpoint": str(ckpt_path), "settings": {"n": args.n, "steps": args.steps}, "criteria": CRITERIA}

    def embed(idx):
        return judge_features(models, centres, data.views[idx].float().to(device))

    # ---- 1-3: classes, realism, memorization
    real_val = {k: embed(v) for k, v in data.val_by_class.items() if len(v) > 4}
    real_all = torch.cat([j["A"]["embedding"] for j in real_val.values()])
    train_vox = embed(data.train_by_class["vox"])["A"]["embedding"]
    classes = {}
    for name in ("vox", "synth", "processed_vox", "synth_vox"):
        keys = torch.from_numpy(rng.choice(data.midi[data.val_by_class.get(name) or data.train_by_class[name]],
                                           args.n).astype(np.float32))
        feats = ddim_sample(model, keys, {"mode": "label", "label": name, "scale": 3.0}, args.steps, gen, device, args.max_batch)
        j = judge_features(models, centres, feats)
        ref = real_val[name]["A"]["embedding"] if name in real_val else real_all
        train_idx = data.train_by_class[name]
        sub = [train_idx[k] for k in rng.choice(len(train_idx), min(len(train_idx), 256), replace=False)]
        classes[name] = summarize(j, ref_embeddings=ref)
        classes[name]["kid_real_train_vs_heldout"] = round(kid(embed(sub)["A"]["embedding"], ref), 4)
        if name == "vox":
            gen_nn = nearest_distance(j["A"]["embedding"], train_vox)
            real_nn = nearest_distance(real_val["vox"]["A"]["embedding"], train_vox)
            classes[name]["memorization_ratio"] = round(float(gen_nn.median() / real_nn.median().clamp(min=1e-6)), 4)
    report["classes"] = classes

    # ---- 5: references on held-out pairs (same judges, same KID space)
    pairs = tm.sample_pairs(np.random.default_rng(31), data.catalog, data.pools, args.pairs, "val", alpha=0.5)
    morphnet, art = None, None
    if args.morphnet is not None:
        from wavelet2vec.morph_large import load_morph_large
        morphnet = load_morph_large(args.morphnet)
    views, carriers = pair_views(data.catalog, pairs, device, morphnet, art)
    references = {}
    for name, v in views.items():
        references[name] = summarize(judge_features(models, centres, v.to(device)), 0.5, real_all)
    report["references"] = references
    pair_keys = torch.tensor([data.catalog.notes[v]["midi"] for v, _, _ in pairs], dtype=torch.float32)

    # ---- 4: composition grid (keys of the held-out pairs, repeated)
    grid = {}
    keys = pair_keys[torch.from_numpy(rng.integers(len(pairs), size=args.n))]
    for mode in ("and", "interp"):
        for alpha in (0.25, 0.5, 0.75):
            for scale in (2.0, 4.0, 6.0):
                feats = ddim_sample(model, keys, {"mode": mode, "alpha": alpha, "scale": scale}, args.steps, gen, device, args.max_batch)
                grid[f"{mode} alpha={alpha} g={scale:g}"] = summarize(judge_features(models, centres, feats), alpha, real_all)
    report["composition"] = grid

    # ---- 6: alpha automated within the note
    curve = torch.linspace(0.0, 1.0, 120)
    feats = ddim_sample(model, keys[:max(8, args.n // 2)], {"mode": "and", "alpha": curve, "scale": 4.0},
                        args.steps, gen, device, args.max_batch)
    thirds = {}
    for part, sl in (("first third", slice(0, 40)), ("last third", slice(80, 120))):
        masked = feats.clone()
        keep = torch.zeros(120, dtype=torch.bool)
        keep[sl] = True
        masked[:, :2, :, ~keep] = -1.0
        masked[:, 2:, :, ~keep] = 0.0
        thirds[part] = round(float(judge_features(models, centres, masked)["A"]["vocality"].mean()), 4)
    report["automation"] = thirds

    # ---- verdict
    and_05 = {k: v for k, v in grid.items() if k.startswith("and alpha=0.5")}
    best_key = max(and_05, key=lambda k: and_05[k]["both_present_A"] - and_05[k]["vocality_error_A"])
    best = and_05[best_key]
    passes = {
        "fidelity": classes["vox"]["vocality_A"] >= 0.7 and classes["synth"]["vocality_A"] <= 0.3,
        "realism": all(classes[c]["kid_A"] <= 3 * max(classes[c]["kid_real_train_vs_heldout"], 1e-3) for c in ("vox", "synth")),
        "composition": best["both_present_A"] >= 0.5 and abs(best["vocality_A"] - 0.5) <= 0.2,
        "references": (best["both_present_A"] >= max(references["mix"]["both_present_A"], references["vocoder"]["both_present_A"],
                                                     references.get("morphnet", {}).get("both_present_A", 0.0))
                       and best["kid_A"] <= references["mix"]["kid_A"]),
        "automation": thirds["last third"] - thirds["first third"] >= 0.3,
        "memorization": classes["vox"]["memorization_ratio"] >= 0.5,
    }
    report["verdict"] = {"best_and_setting": best_key, "passes": passes,
                         "decision": "go: build stage 1" if all(passes.values()) else
                                     "no-go as is: " + ", ".join(k for k, ok in passes.items() if not ok)}

    # ---- 7: renders for listening
    if args.renders:
        folder = args.run / "demos" / f"eval_{ckpt_path.stem}"
        folder.mkdir(parents=True, exist_ok=True)
        g_render = torch.Generator(device=device).manual_seed(7)
        entries = []
        for k in range(min(args.renders, len(pairs))):
            v, s, _ = pairs[k]
            key = pair_keys[k:k + 1]
            carrier = WaveletImage(carriers[k].spec, {c: x.cpu() for c, x in carriers[k].groups.items()}, dict(carriers[k].rows))
            samples = {"and α=0.5 (g=4)": {"mode": "and", "alpha": 0.5, "scale": 4.0},
                       "interp α=0.5": {"mode": "interp", "alpha": 0.5, "scale": 3.0},
                       "α automated 0→1": {"mode": "and", "alpha": curve, "scale": 4.0},
                       "vox sample": {"mode": "label", "label": "vox", "scale": 3.0},
                       "synth sample": {"mode": "label", "label": "synth", "scale": 3.0}}
            entry = {"pair": k, "midi": int(key[0]), "alpha": 0.5,
                     "vocal": f"(prior samples; no vocal note used) key of {data.catalog.notes[v]['file']}",
                     "synth": f"carrier: {data.catalog.notes[s]['family']}: {data.catalog.notes[s]['file']}", "files": {}}
            renders = {"carrier (synth note)": carrier}
            for name, guidance in samples.items():
                feats = ddim_sample(model, key, guidance, args.steps, g_render, device, args.max_batch)
                renders[name] = render_features(feats[0].cpu(), carrier, seed=k)
            for name, image in renders.items():
                audio = decode(image)[0].double().numpy()
                audio *= 0.99 / max(float(np.abs(audio).max()), 0.99)
                safe = name.replace(" ", "_").replace("α", "alpha").replace("→", "-to-").replace("(", "").replace(")", "").replace("=", "")
                path = folder / f"pair{k:02d}_{safe}.wav"
                sf.write(path, audio.T, tm.SPEC.sample_rate, subtype="FLOAT")
                entry["files"][name] = path.name
            entries.append(entry)
        (folder / "demos.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
        tm.EventLog(args.run / "metrics.jsonl").write("demos", epoch=saved.get("metrics", {}).get("epoch", 0),
                                                      folder=f"demos/{folder.name}", pairs=entries)
    report["minutes"] = round((time.time() - started) / 60, 1)
    evals = args.run / "evals"
    evals.mkdir(exist_ok=True)
    path = evals / f"eval_{ckpt_path.stem}_{time.strftime('%Y%m%d-%H%M')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    tm.EventLog(args.run / "metrics.jsonl").write("evaluation", file=path.name, verdict=report["verdict"])
    print(json.dumps({"classes": classes, "references": references, "best_and": {best_key: best},
                      "automation": thirds, "verdict": report["verdict"]}, indent=1))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
