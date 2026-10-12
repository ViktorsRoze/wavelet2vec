"""Evaluate a KeyboardNet v2 checkpoint: accuracy, head diagnostics, backbone diagnostics, probes.

Accuracy: test families × (1, 2, 3 examples) × distance buckets, against
shifting the nearest example, the blend of all examples (the baseline the
head corrects), and KeyboardNet v1 (1-2 examples).

Head:
- error by frequency region and time region;
- how much of the needed correction (truth - baseline) the head delivers:
  slope, correlation, magnitude ratio;
- auxiliary predictions (IF deviation, coherence, width) against copying the
  shifted nearest example;
- ablations: high-resolution stem path off, backbone path off (fused FPN
  features zeroed), conditioning off.

Backbone:
- per-block residual update ratio |update| / |x| for attention and MLP
  (adaLN-Zero blocks start at 0: near-zero blocks are unused);
- attention entropy per stage (1 = uniform, lower = focused);
- effective rank of pooled stage features;
- linear (ridge) probes from pooled features of the input, each Swin stage
  and the fused head map. They are fitted on training-family episodes and
  scored on test-family episodes, for pitch, interval, brightness, decay,
  level, source type and envelope class. Probes are repeated for an
  untrained network and earlier epochs, to show what training added.

    PYTHONPATH=src python experiments/evaluate_keyboard_v2.py [--checkpoint best.pt] [--probe-checkpoints init,1,3,6]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_keyboard import BUCKETS, OUT, audible_error, split_families
from train_keyboard_v2 import evaluation_episodes, load_rich_cache
from wavelet2vec.keyboard import DB_SCALE, ENV_BANDS, ENV_FRAMES, FLOOR_DB, load_keyboard_model, predict_envelope, shift_envelope
from wavelet2vec.keyboard_v2 import (AUX_TARGETS, BASE_PLANE, KeyboardV2Config, MAX_EXAMPLES, build_model_v2,
                                     load_keyboard_v2, network_inputs_v2)

FREQ_REGIONS = {"<100 Hz": (12, 43), "100-500 Hz": (43, 71), "0.5-2 kHz": (71, 95), "2-8 kHz": (95, 119),
                "8-20 kHz": (119, 136)}  # MIDI ranges -> band = midi - 12
TIME_REGIONS = {"attack 0-50 ms": (0, 5), "50-300 ms": (5, 30), "0.3-1.2 s": (30, 120)}
AUX_NAMES = {1: "if_deviation", 2: "phase_coherence", 6: "stereo_width"}


def splits(meta, notes):
    train_families, test_families = split_families(meta)
    files_by_family = defaultdict(list)
    for file in notes:
        files_by_family[meta[file]["family"]].append(file)
    rng = np.random.default_rng(8)
    pool = [f for f in train_families if len(files_by_family[f]) == 1]
    val = sorted(rng.choice(pool, 5, replace=False).tolist())
    return [f for f in train_families if f not in val], val, test_families, files_by_family


def run_parts(model, planes, cond, ablate=None):
    """KeyboardNetV2.forward, step by step, returning pooled internals; ``ablate`` in
    {None, "no_stem", "no_backbone", "no_conditioning"}."""
    if ablate == "no_conditioning":
        cond = torch.zeros_like(cond)
    c = model.cond(cond)
    full = model.stem_full(planes)
    half = model.stem_half(full)
    if ablate == "no_stem":
        full, half = torch.zeros_like(full), torch.zeros_like(half)
    x = model.patch(planes)
    b = x.shape[0]
    x = model.patch_norm(x.flatten(2).transpose(1, 2))
    stages = []
    for i, blocks in enumerate(model.stages):
        for block in blocks:
            x = block(x, c)
        res = model.resolutions[i]
        stages.append(model.stage_norms[i](x).transpose(1, 2).reshape(b, -1, *res))
        if i < 3:
            x = model.merges[i](x)
    top = stages[3]
    pooled = [F.interpolate(p(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for p in model.ppm]
    levels = [None, None, None, model.ppm_fuse(torch.cat([top] + pooled, 1))]
    for i in (2, 1, 0):
        up = F.interpolate(levels[i + 1], size=stages[i].shape[-2:], mode="bilinear", align_corners=False)
        levels[i] = model.fpn[i](model.lateral[i](stages[i]) + up)
    size = levels[0].shape[-2:]
    fused = model.fuse(torch.cat([levels[0]] + [F.interpolate(l, size=size, mode="bilinear", align_corners=False)
                                                for l in levels[1:]], 1))
    h = model.fuse_film(fused, c)
    fused_pooled = h.mean((-2, -1))
    if ablate == "no_backbone":
        h = torch.zeros_like(h)
    h = F.interpolate(h, size=half.shape[-2:], mode="bilinear", align_corners=False)
    h = model.refine_half[1](model.refine_half[0](torch.cat([h, half], 1), c), c)
    h = F.interpolate(h, size=full.shape[-2:], mode="bilinear", align_corners=False)
    h = model.refine_full[1](model.refine_full[0](torch.cat([h, full], 1), c), c)
    out = model.out(h)[:, :, :ENV_BANDS, :ENV_FRAMES]
    limit = model.config.max_correction_db / DB_SCALE
    envelope = planes[:, BASE_PLANE, :ENV_BANDS, :ENV_FRAMES] + limit * torch.tanh(out[:, 0] / limit)
    return envelope, out[:, 1:], {"stages": [s.mean((-2, -1)) for s in stages], "fused": fused_pooled}


def batches(episodes, notes, size=16):
    for i in range(0, len(episodes), size):
        chunk = episodes[i:i + size]
        planes, conds, bases, truths, nearest = [], [], [], [], []
        for file, examples, target, _ in chunk:
            feats = [notes[file][m].astype(np.float32) for m in examples]
            p, c, base = network_inputs_v2(feats, examples, target)
            planes.append(p)
            conds.append(c)
            bases.append(base)
            truths.append(notes[file][target].astype(np.float32))
            near = int(np.argmin([abs(target - m) for m in examples]))
            nearest.append(np.stack([np.roll(feats[near][ch], 0) for ch in range(feats[near].shape[0])]))
        yield chunk, torch.from_numpy(np.stack(planes)), torch.from_numpy(np.stack(conds)), np.stack(bases), np.stack(truths)


def accuracy_and_head(model, episodes, notes, meta, v1, envelopes_v1, ablation_episodes):
    rows = []
    region_err = defaultdict(lambda: defaultdict(list))
    corr = {"needed": [], "applied": []}
    aux_err = {name: {"model": [], "copy_nearest": []} for name in AUX_NAMES.values()}
    with torch.no_grad():
        for chunk, planes, conds, bases, truths in batches(episodes, notes):
            envelope, aux, _ = run_parts(model, planes, conds)
            pred = envelope.numpy() * DB_SCALE
            aux = aux.numpy()
            for j, (file, examples, target, bucket) in enumerate(chunk):
                truth = truths[j][0] * DB_SCALE
                mask = truth > truth.max() - 70.0
                near = examples[int(np.argmin([abs(target - m) for m in examples]))]
                shifted, _ = shift_envelope(notes[file][near][0].astype(np.float32) * DB_SCALE, target - near)
                row = {"family": meta[file]["family"], "k": len(examples), "bucket": f"{bucket[0]}-{bucket[1]}",
                       "model": audible_error(pred[j], truth), "shift_nearest": audible_error(shifted, truth),
                       "blend": audible_error(bases[j], truth)}
                if v1 is not None and len(examples) <= 2:
                    v1_pred, _ = predict_envelope(v1, [envelopes_v1[file][m] for m in examples], examples, target)
                    row["v1"] = audible_error(v1_pred, truth)
                rows.append(row)
                for name, (lo, hi) in FREQ_REGIONS.items():
                    for tname, (t0, t1) in TIME_REGIONS.items():
                        m = np.zeros_like(mask)
                        m[max(lo - 12, 0):max(hi - 12, 0), t0:t1] = True
                        m &= mask
                        if m.any():
                            region_err[f"{name} | {tname}"]["model"].append(float(np.abs(pred[j] - truth)[m].mean()))
                            region_err[f"{name} | {tname}"]["blend"].append(float(np.abs(bases[j] - truth)[m].mean()))
                idx = np.flatnonzero(mask)
                pick = np.random.default_rng(j).choice(idx, min(400, idx.size), replace=False)
                corr["needed"].append((truth - bases[j]).ravel()[pick])
                corr["applied"].append((pred[j] - bases[j]).ravel()[pick])
                d = target - near
                for a, ch in enumerate(AUX_TARGETS):
                    copy = np.zeros((ENV_BANDS, ENV_FRAMES), dtype=np.float32)
                    src = notes[file][near][ch].astype(np.float32)
                    if d >= 0:
                        copy[d:] = src[:ENV_BANDS - d]
                    else:
                        copy[:d] = src[-d:]
                    tgt = truths[j][ch]
                    aux_err[AUX_NAMES[ch]]["model"].append(float(np.abs(aux[j, a] - tgt)[mask].mean()))
                    aux_err[AUX_NAMES[ch]]["copy_nearest"].append(float(np.abs(copy - tgt)[mask].mean()))
        ablations = {}
        for ablate in ("no_stem", "no_backbone", "no_conditioning"):
            errs, full_errs = [], []
            for chunk, planes, conds, bases, truths in batches(ablation_episodes, notes):
                envelope, _, _ = run_parts(model, planes, conds, ablate)
                normal, _, _ = run_parts(model, planes, conds)
                for j in range(len(chunk)):
                    truth = truths[j][0] * DB_SCALE
                    errs.append(audible_error(envelope[j].numpy() * DB_SCALE, truth))
                    full_errs.append(audible_error(normal[j].numpy() * DB_SCALE, truth))
            ablations[ablate] = {"ablated": round(float(np.mean(errs)), 2), "full_model": round(float(np.mean(full_errs)), 2),
                                 "n": len(errs)}
    needed, applied = np.concatenate(corr["needed"]), np.concatenate(corr["applied"])
    head = {
        "correction": {
            "slope_applied_vs_needed": round(float((applied * needed).sum() / (needed ** 2).sum()), 3),
            "correlation": round(float(np.corrcoef(applied, needed)[0, 1]), 3),
            "mean_abs_needed_db": round(float(np.abs(needed).mean()), 2),
            "mean_abs_applied_db": round(float(np.abs(applied).mean()), 2),
            "residual_after_head_db": round(float(np.abs(needed - applied).mean()), 2)},
        "error_by_region": {k: {m: round(float(np.mean(v)), 2) for m, v in d.items()} | {"n": len(d["model"])}
                            for k, d in sorted(region_err.items())},
        "aux": {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in aux_err.items()},
        "ablations": ablations,
    }
    return rows, head


def summarize(rows):
    methods = [m for m in ("model", "v1", "shift_nearest", "blend") if any(m in r for r in rows)]
    table = {}
    for k in range(1, MAX_EXAMPLES + 1):
        for lo, hi in BUCKETS:
            sel = [r for r in rows if r["k"] == k and r["bucket"] == f"{lo}-{hi}"]
            if sel:
                table[f"{k}-shot {lo}-{hi} st"] = {m: round(float(np.mean([r[m] for r in sel if m in r])), 2)
                                                   for m in methods if any(m in r for r in sel)} | {"n": len(sel)}
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    table["per_family"] = {f: {m: round(float(np.mean([r[m] for r in rs])), 2) for m in ("model", "shift_nearest", "blend")}
                           for f, rs in sorted(fam.items())}
    table["overall"] = {m: round(float(np.mean([r[m] for r in rows if m in r])), 2) for m in methods}
    table["overall_k_le_2"] = {m: round(float(np.mean([r[m] for r in rows if m in r and r["k"] <= 2])), 2)
                               for m in methods}
    return table


def backbone_health(model, episodes, notes):
    blocks = [b for stage in model.stages for b in stage]
    attn = [b.attn for b in blocks]
    for b in blocks:
        b.stats = []
    for a in attn:
        a.stats = []
    pooled = defaultdict(list)
    with torch.no_grad():
        for _, planes, conds, _, _ in batches(episodes, notes):
            _, _, parts = run_parts(model, planes, conds)
            for i, s in enumerate(parts["stages"]):
                pooled[f"stage{i + 1}"].append(s.numpy())
    utilization, index = [], 0
    for si, stage in enumerate(model.stages):
        for bi, block in enumerate(stage):
            utilization.append({"stage": si + 1, "block": bi + 1,
                                "attn_update_ratio": round(float(np.mean([s["attn"] for s in block.stats])), 4),
                                "mlp_update_ratio": round(float(np.mean([s["mlp"] for s in block.stats])), 4),
                                "attention_entropy": round(float(np.mean(block.attn.stats)), 3)})
            index += 1
    for b in blocks:
        b.stats = None
    for a in attn:
        a.stats = None

    def effective_rank(x):
        x = x - x.mean(0)
        s = np.linalg.svd(x, compute_uv=False) ** 2
        return float(s.sum() ** 2 / (s ** 2).sum())

    ranks = {k: {"effective_rank": round(effective_rank(np.concatenate(v)), 1), "dims": int(np.concatenate(v).shape[1])}
             for k, v in pooled.items()}
    per_stage = defaultdict(list)
    for u in utilization:
        per_stage[u["stage"]].append(u)
    stage_summary = {f"stage{s}": {"blocks": len(us),
                                   "attn_update_ratio": round(float(np.mean([u["attn_update_ratio"] for u in us])), 4),
                                   "mlp_update_ratio": round(float(np.mean([u["mlp_update_ratio"] for u in us])), 4),
                                   "attention_entropy": round(float(np.mean([u["attention_entropy"] for u in us])), 3),
                                   "near_idle_blocks": sum(u["attn_update_ratio"] < 0.01 and u["mlp_update_ratio"] < 0.01
                                                           for u in us)}
                     for s, us in per_stage.items()}
    return {"per_block": utilization, "per_stage": stage_summary, "feature_rank": ranks}


def probe_targets(episodes, notes, meta):
    y = defaultdict(list)
    for file, examples, target, _ in episodes:
        env = notes[file][target][0].astype(np.float32) * DB_SCALE
        audible = env > env.max() - 70.0
        weights = np.where(audible, 10 ** (env / 10), 0.0).sum(1)
        y["target_pitch_midi"].append(target)
        y["nearest_interval_abs"].append(min(abs(target - m) for m in examples))
        y["brightness_band"].append(float((weights * np.arange(ENV_BANDS)).sum() / (weights.sum() + 1e-30)))
        frame_level = 10 * np.log10((10 ** (env / 10)).sum(0) + 1e-30)
        peak = int(np.argmax(frame_level))
        below = np.flatnonzero(frame_level[peak:] < frame_level[peak] - 30.0)
        y["decay_frames_to_-30dB"].append(float(below[0] if below.size else ENV_FRAMES - peak))
        y["level_peak_db"].append(float(env.max()))
        y["source_type"].append(meta[file]["source_type"])
        y["envelope_class"].append(meta[file]["envelope"])
    return y


def ridge_probe(x_train, y_train, x_test, y_test, classify):
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    if classify:
        classes = sorted(set(y_train))
        t = np.array([[1.0 if v == c else 0.0 for c in classes] for v in y_train])
    else:
        t = np.asarray(y_train, dtype=np.float64)[:, None]
        t_mu, t_sd = t.mean(), t.std() + 1e-9
        t = (t - t_mu) / t_sd
    best = None
    folds = np.arange(len(a)) % 3
    for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
        score = []
        for f in range(3):
            tr, va = folds != f, folds == f
            w = np.linalg.solve(a[tr].T @ a[tr] + lam * np.eye(a.shape[1]), a[tr].T @ t[tr])
            score.append(((a[va] @ w - t[va]) ** 2).mean())
        if best is None or np.mean(score) < best[0]:
            best = (np.mean(score), lam)
    w = np.linalg.solve(a.T @ a + best[1] * np.eye(a.shape[1]), a.T @ t)
    pred = b @ w
    if classify:
        labels = [classes[i] for i in pred.argmax(1)]
        majority = max(set(y_train), key=list(y_train).count)
        return {"accuracy": round(float(np.mean([p == q for p, q in zip(labels, y_test)])), 3),
                "majority_baseline": round(float(np.mean([q == majority for q in y_test])), 3)}
    pred = pred[:, 0] * t_sd + t_mu
    y = np.asarray(y_test, dtype=np.float64)
    return {"r2": round(float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}


def probe_features(model, episodes, notes):
    feats = defaultdict(list)
    with torch.no_grad():
        for _, planes, conds, _, _ in batches(episodes, notes):
            _, _, parts = run_parts(model, planes, conds)
            region = planes[:, :, :ENV_BANDS, :ENV_FRAMES]
            feats["input"].append(torch.cat([region.mean((-2, -1)), region.std((-2, -1))], 1).numpy())
            for i, s in enumerate(parts["stages"]):
                feats[f"stage{i + 1}"].append(s.numpy())
            feats["head_fused"].append(parts["fused"].numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}


def probes(model, train_eps, test_eps, notes, meta):
    y_train, y_test = probe_targets(train_eps, notes, meta), probe_targets(test_eps, notes, meta)
    f_train, f_test = probe_features(model, train_eps, notes), probe_features(model, test_eps, notes)
    out = {}
    for name in f_train:
        out[name] = {target: ridge_probe(f_train[name], y_train[target], f_test[name], y_test[target],
                                         classify=target in ("source_type", "envelope_class"))
                     for target in y_train}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=OUT / "v2")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--per-cell", type=int, default=100)
    parser.add_argument("--ablation-per-cell", type=int, default=40)
    parser.add_argument("--probe-checkpoints", default="init,1,3,6",
                        help="comma list: 'init' (untrained) and checkpoint epochs, e.g. 1,1.5,3")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    notes, meta = load_rich_cache(OUT / "features_rich.npz")
    train_f, val_f, test_f, files_by_family = splits(meta, notes)
    test_eps = evaluation_episodes(notes, files_by_family, test_f, per_cell=args.per_cell, seed=21)
    ablation_eps = evaluation_episodes(notes, files_by_family, test_f, per_cell=args.ablation_per_cell, seed=22)
    probe_train = evaluation_episodes(notes, files_by_family, train_f, per_cell=32, seed=23)
    probe_test = evaluation_episodes(notes, files_by_family, test_f, per_cell=24, seed=24)

    ckpt_dir = args.run / "checkpoints"
    model, extra = load_keyboard_v2(ckpt_dir / args.checkpoint)
    v1, _ = load_keyboard_model(OUT / "keyboard_v1.pt")
    data = np.load(OUT / "envelopes.npz")
    info = json.loads((OUT / "envelopes.json").read_text(encoding="utf-8"))
    envelopes_v1 = defaultdict(dict)
    for env, midi, index in zip(data["envelopes"], data["midi"], data["file_index"]):
        envelopes_v1[info["files"][index]][int(midi)] = env.astype(np.float32)

    print("accuracy + head ...", flush=True)
    rows, head = accuracy_and_head(model, test_eps, notes, meta, v1, envelopes_v1, ablation_eps)
    print("backbone health ...", flush=True)
    health = backbone_health(model, ablation_eps[:160], notes)
    print("probes ...", flush=True)
    probe_results = {}
    for name in args.probe_checkpoints.split(","):
        if name == "init":
            torch.manual_seed(0)
            probe_model = build_model_v2(KeyboardV2Config(**{k: v for k, v in model.config.__dict__.items()
                                                             if k in KeyboardV2Config.__dataclass_fields__})).eval()
        else:
            path = ckpt_dir / f"{name}_epoch.pth"
            if not path.exists():  # older naming
                path = ckpt_dir / f"epoch_{int(float(name)):03d}.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            probe_model = build_model_v2(model.config).eval()
            probe_model.load_state_dict(state["state"])
        probe_results["untrained" if name == "init" else f"epoch {name}"] = probes(probe_model, probe_train, probe_test,
                                                                                    notes, meta)
        print(f"  probes {name} done", flush=True)
    report = {"checkpoint": args.checkpoint, "checkpoint_extra": extra, "test_families": test_f, "val_families": val_f,
              "accuracy": summarize(rows), "head": head, "backbone": health, "probes": probe_results,
              "episodes": {"test": len(test_eps), "ablation": len(ablation_eps), "probe_train": len(probe_train),
                           "probe_test": len(probe_test)},
              "minutes": round((time.time() - started) / 60, 1)}
    out_dir = args.run / "evals"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    path = out_dir / f"eval_{Path(args.checkpoint).stem}_{stamp}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report -> {path} ({report['minutes']} min)")
    print(json.dumps({"overall": report["accuracy"]["overall"], "k<=2": report["accuracy"]["overall_k_le_2"],
                      "ablations": head["ablations"], "correction": head["correction"],
                      "backbone": health["per_stage"]}, indent=1))


if __name__ == "__main__":
    main()
