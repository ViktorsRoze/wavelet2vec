"""Train KeyboardNet v2 (Swin backbone + UPerNet head) on CPU, by epochs, with checkpoints.

An epoch is as many episodes as there are target notes in the training
families. Families are sampled uniformly, as in v1, so the 38-file Harmor
family does not dominate. An episode is one target note plus one, two or
three example notes from the same file (a third each); evaluation and demos
report all three scenarios.

Splits (whole families; nothing from a held-out family is ever trained on):
- test: the same 10 families as v1 (including the own-voice takes);
- validation: 5 more single-file families, scored every epoch; the best epoch
  is kept.

Every half epoch and every epoch saves:
- ``checkpoints/<epochs>_epoch.pth``, e.g. ``1.5_epoch.pth``, ``2_epoch.pth``:
  model weights and the metrics at that point;
- ``checkpoints/last.pt``: the full resumable state (weights, optimizer,
  schedule, sampler RNG, step inside the epoch), overwritten each time;
- ``checkpoints/best.pt``: the best validation point's weights.

Training probes go to ``metrics.jsonl`` (read by
experiments/training_dashboard.py):
- every ``--log-every`` steps: loss parts, learning rate, per-group gradient
  norms and update/weight ratios, per-stage block update ratios and
  attention entropy;
- every checkpoint: validation by example count and distance, and ridge
  probes of input / stage / head features (fitted on training families,
  scored on validation families; test families stay untouched).

``--resume`` continues from last.pt, mid-epoch if it was saved at a half epoch. At the end, the best epoch is scored on
the test families with v1's metric and baselines, and demos are rendered.

    PYTHONPATH=src python experiments/train_keyboard_v2.py --epochs 50 [--size large] [--resume]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_keyboard import BUCKETS, OUT, audible_error, note_audio, split_families
from wavelet2vec.keyboard import DB_SCALE, ENV_RATE, FLOOR_DB, render_note, shift_envelope
from wavelet2vec.keyboard_v2 import (AUX_TARGETS, MAX_EXAMPLES, KeyboardV2Config, build_model_v2, count_parameters,
                                     network_inputs_v2, predict_envelope_v2, save_keyboard_v2)

AUDIBLE_DB = 70.0
AUX_WEIGHT = 0.25


def load_rich_cache(path):
    data = np.load(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    files = info["files"]
    notes = defaultdict(dict)
    for features, midi, index in zip(data["envelopes"], data["midi"], data["file_index"]):
        notes[files[index]][int(midi)] = features  # float16 [7, 124, 120]
    return notes, info["meta"]


def episode(rng, notes, files_by_family, families):
    while True:
        family = families[rng.integers(len(families))]
        file = files_by_family[family][rng.integers(len(files_by_family[family]))]
        midis = sorted(notes[file])
        if len(midis) < 3:
            continue
        target = midis[rng.integers(len(midis))]
        reach = int(rng.choice([12, 24, 48], p=[0.6, 0.3, 0.1]))
        others = [m for m in midis if m != target and abs(m - target) <= reach]
        k = int(rng.integers(1, MAX_EXAMPLES + 1))
        if len(others) >= k:
            return file, [int(m) for m in rng.choice(others, k, replace=False)], int(target)


def make_batch(rng, size, notes, files_by_family, families, augment=True):
    planes, conds, targets = [], [], []
    for _ in range(size):
        file, examples, target = episode(rng, notes, files_by_family, families)
        feats = [notes[file][m].astype(np.float32) for m in examples]
        truth = notes[file][target].astype(np.float32)
        if augment:  # one random gain for examples and target: level-relative learning
            gain = rng.uniform(-12.0, 12.0) / DB_SCALE
            for f in feats + [truth]:
                audible = f[0] > FLOOR_DB / DB_SCALE + 1e-3
                f[0] = np.where(audible, np.maximum(f[0] + gain, FLOOR_DB / DB_SCALE), f[0])
        p, c, _ = network_inputs_v2(feats, examples, target)
        planes.append(p)
        conds.append(c)
        targets.append(truth)
    return (torch.from_numpy(np.stack(planes)), torch.from_numpy(np.stack(conds)),
            torch.from_numpy(np.stack(targets)))


def loss_fn(envelope, aux, truth):
    target_db = truth[:, 0] * DB_SCALE
    peak = target_db.amax(dim=(-2, -1), keepdim=True)
    audible = (target_db > peak - AUDIBLE_DB).float()
    weight = audible + 0.1 * (1 - audible)
    main = (weight * (envelope - truth[:, 0]).abs()).sum() / weight.sum()
    aux_target = truth[:, list(AUX_TARGETS)]
    aux_loss = (audible[:, None] * (aux - aux_target).abs()).sum() / (audible.sum() * len(AUX_TARGETS) + 1e-6)
    return main + AUX_WEIGHT * aux_loss, main.detach() * DB_SCALE, aux_loss.detach()


def evaluation_episodes(notes, files_by_family, families, per_cell=150, seed=11):
    """Fixed episodes per (example count, distance bucket): the nearest example lies in the
    bucket, any further examples within the bucket's upper distance."""
    rng = np.random.default_rng(seed)
    episodes = []
    for k in range(1, MAX_EXAMPLES + 1):
        for lo, hi in BUCKETS:
            cell, tries = [], 0
            while len(cell) < per_cell and tries < per_cell * 300:
                tries += 1
                family = families[rng.integers(len(families))]
                file = files_by_family[family][rng.integers(len(files_by_family[family]))]
                midis = sorted(notes[file])
                target = midis[rng.integers(len(midis))]
                near = [m for m in midis if m != target and lo <= abs(m - target) <= hi]
                if not near:
                    continue
                first = int(rng.choice(near))
                rest = [m for m in midis if m not in (target, first) and lo <= abs(m - target) <= hi]
                if len(rest) < k - 1:
                    continue
                chosen = [first] + [int(m) for m in rng.choice(rest, k - 1, replace=False)] if k > 1 else [first]
                cell.append((file, chosen, target, (lo, hi)))
            episodes += cell
    return episodes


def evaluate(model, episodes, notes, meta):
    model.eval()
    rows = []
    for file, examples, target, bucket in episodes:
        feats = [notes[file][m].astype(np.float32) for m in examples]
        truth = notes[file][target][0].astype(np.float32) * DB_SCALE
        pred, blend = predict_envelope_v2(model, feats, examples, target)
        nearest = int(np.argmin([abs(target - m) for m in examples]))
        shifted, _ = shift_envelope(feats[nearest][0] * DB_SCALE, target - examples[nearest])
        rows.append({"family": meta[file]["family"], "k": len(examples), "bucket": f"{bucket[0]}-{bucket[1]}",
                     "model": audible_error(pred, truth), "shift_nearest": audible_error(shifted, truth),
                     "blend": audible_error(blend, truth)})
    model.train()
    return rows


def summarize(rows):
    table = {}
    for k in range(1, MAX_EXAMPLES + 1):
        for lo, hi in BUCKETS:
            sel = [r for r in rows if r["k"] == k and r["bucket"] == f"{lo}-{hi}"]
            if sel:
                table[f"{k}-shot {lo}-{hi} st"] = {m: round(float(np.mean([r[m] for r in sel])), 2)
                                                   for m in ("model", "shift_nearest", "blend")} | {"n": len(sel)}
    per_family = defaultdict(list)
    for r in rows:
        per_family[r["family"]].append(r)
    table["per_family"] = {f: {m: round(float(np.mean([r[m] for r in rs])), 2) for m in ("model", "shift_nearest")}
                           for f, rs in sorted(per_family.items())}
    table["overall"] = {m: round(float(np.mean([r[m] for r in rows])), 2) for m in ("model", "shift_nearest", "blend")}
    return table


def render_demo_v2(model, file, example_midis, notes, meta, out_dir):
    import soundfile as sf
    audio = note_audio(file, meta)
    targets = [m for m in sorted(notes[file]) if m in audio]
    feats = [notes[file][m].astype(np.float32) for m in example_midis]
    generated, original = [], []
    for target in targets:
        nearest = min(example_midis, key=lambda m: abs(m - target))
        pred, _ = predict_envelope_v2(model, feats, example_midis, target)
        generated.append(render_note(audio[nearest], nearest, target, pred))
        original.append(audio[target])
    stem = Path(file).stem.removeprefix("from_C1_till_B10_") + "_from_" + "_".join(map(str, example_midis))
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = np.concatenate(generated, 1)
    joined *= min(1.0, 0.98 / max(np.abs(joined).max(), 1e-9))
    sf.write(out_dir / f"{stem}__keyboardnet_v2.wav", joined.T.astype(np.float32), ENV_RATE, subtype="FLOAT")
    return {"file": file, "examples": example_midis, "targets": targets}


class EventLog:
    """Append-only JSON lines for the dashboard: one event per line, with a wall-clock time."""

    def __init__(self, path):
        self.path = Path(path)

    def write(self, kind, **fields):
        event = {"type": kind, "time": time.time(), **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=float) + "\n")


def parameter_groups(model):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if name.startswith("stages."):
            groups[f"stage{int(name.split('.')[1]) + 1}"].append(p)
        elif name.startswith(("patch", "merges", "stage_norms")):
            groups["patch+merge"].append(p)
        elif name.startswith("stem"):
            groups["stem"].append(p)
        elif name.startswith("cond"):
            groups["conditioning"].append(p)
        else:
            groups["head"].append(p)
    return dict(groups)


def set_recording(model, on):
    for stage in model.stages:
        for block in stage:
            block.stats = [] if on else None
            block.attn.stats = [] if on else None


def collect_recording(model):
    """Per stage: mean residual update ratio (attention, MLP) and normalized attention entropy."""
    blocks, entropy = {}, {}
    for i, stage in enumerate(model.stages):
        attn = [s["attn"] for b in stage for s in (b.stats or [])]
        mlp = [s["mlp"] for b in stage for s in (b.stats or [])]
        ent = [e for b in stage for e in (b.attn.stats or [])]
        blocks[f"stage{i + 1}"] = {"attn": round(float(np.mean(attn)), 5) if attn else None,
                                   "mlp": round(float(np.mean(mlp)), 5) if mlp else None}
        entropy[f"stage{i + 1}"] = round(float(np.mean(ent)), 4) if ent else None
    return blocks, entropy


def training_probes(model, probe_train, probe_val, notes, meta):
    """Ridge probes on pooled features (fit: training families, score: validation families)."""
    from evaluate_keyboard_v2 import probe_features, probe_targets, ridge_probe  # lazy: avoids a circular import
    was_training = model.training
    model.eval()
    y_train, y_val = probe_targets(probe_train, notes, meta), probe_targets(probe_val, notes, meta)
    f_train, f_val = probe_features(model, probe_train, notes), probe_features(model, probe_val, notes)
    model.train(was_training)
    return {name: {target: ridge_probe(f_train[name], y_train[target], f_val[name], y_val[target],
                                       classify=target in ("source_type", "envelope_class"))
                   for target in y_train} for name in f_train}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--size", default="large", choices=("tiny", "base", "large"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--epoch-samples", type=int, default=None, help="override the epoch length (dry runs)")
    parser.add_argument("--val-episodes", type=int, default=30, help="per (k, distance) cell")
    parser.add_argument("--out", type=Path, default=OUT / "v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-demos", action="store_true")
    parser.add_argument("--log-every", type=int, default=50, help="steps between training-probe events")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    notes, meta = load_rich_cache(OUT / "features_rich.npz")
    train_families, test_families = split_families(meta)
    files_by_family = defaultdict(list)
    for file in notes:
        files_by_family[meta[file]["family"]].append(file)
    # Validation comes from single-file families only, so multi-file families (the
    # Shevannai voice, Exhale, the Serum and Harmor patch families) stay in training.
    rng_split = np.random.default_rng(8)
    pool = [f for f in train_families if len(files_by_family[f]) == 1]
    val_families = sorted(rng_split.choice(pool, 5, replace=False).tolist())
    train_families = [f for f in train_families if f not in val_families]
    train_notes = sum(len(notes[f]) for fam in train_families for f in files_by_family[fam])
    epoch_samples = args.epoch_samples or train_notes
    steps_per_epoch = max(epoch_samples // args.batch, 1)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    config = KeyboardV2Config(size=args.size)
    model = build_model_v2(config)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "bias_table" in name else decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.999))
    total_steps = steps_per_epoch * args.epochs
    warmup = min(steps_per_epoch, max(total_steps // 20, 1))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(s / total_steps, 1.0))))
    rng = np.random.default_rng(1)
    start_epoch, start_step, best, history = 1, 0, float("inf"), []
    if args.resume and (ckpt / "last.pt").exists():
        state = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        rng.bit_generator.state = state["rng"]
        best, history = state["best"], state["history"]
        done = state.get("step_in_epoch", steps_per_epoch)  # older checkpoints: whole epochs only
        start_epoch, start_step = (state["epoch"] + 1, 0) if done >= steps_per_epoch else (state["epoch"], done)
        print(f"resumed at epoch {start_epoch} step {start_step}", flush=True)

    val_episodes = evaluation_episodes(notes, files_by_family, val_families, per_cell=args.val_episodes, seed=13)
    probe_train = evaluation_episodes(notes, files_by_family, train_families, per_cell=16, seed=31)
    probe_val = evaluation_episodes(notes, files_by_family, val_families, per_cell=12, seed=32)
    run_info = {"parameters": count_parameters(model), "train_families": len(train_families),
                "val_families": val_families, "test_families": test_families, "train_notes": train_notes,
                "epoch_samples": epoch_samples, "steps_per_epoch": steps_per_epoch, "batch": args.batch,
                "lr": args.lr, "size": args.size, "epochs_total": args.epochs}
    print(json.dumps(run_info, indent=1), flush=True)
    events = EventLog(args.out / "metrics.jsonl")
    events.write("run_start", epoch=start_epoch - 1 + start_step / steps_per_epoch, resumed=bool(args.resume),
                 **run_info)
    groups = parameter_groups(model)
    half = steps_per_epoch // 2

    def checkpoint(epoch, step, mains, auxes, started):
        nonlocal best
        point = epoch - 1 + step / steps_per_epoch
        val = summarize(evaluate(model, val_episodes, notes, meta))
        probe = training_probes(model, probe_train, probe_val, notes, meta)
        record = {"epoch": round(point, 3), "train_env_db": round(float(np.mean(mains)), 3) if mains else None,
                  "train_aux": round(float(np.mean(auxes)), 4) if auxes else None, "val": val["overall"],
                  "val_by_k": {key: v for key, v in val.items() if "shot" in key},
                  "minutes": round((time.time() - started) / 60, 1)}
        history.append(record)
        name = f"{point:g}_epoch.pth"
        torch.save({"config": vars(config) | {"refine_dims": list(config.refine_dims)}, "state": model.state_dict(),
                    "metrics": record}, ckpt / name)
        is_best = val["overall"]["model"] < best
        if is_best:
            best = val["overall"]["model"]
            save_keyboard_v2(ckpt / "best.pt", model, extra={"epoch": point, "val": val["overall"],
                                                              "test_families": test_families})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, "step_in_epoch": step, "best": best,
                    "history": history}, ckpt / "last.tmp")
        shutil.move(ckpt / "last.tmp", ckpt / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        events.write("val", epoch=point, val=val["overall"], val_by_k=record["val_by_k"],
                     val_per_family=val["per_family"])
        events.write("probe", epoch=point, results=probe)
        events.write("checkpoint", epoch=point, file=name, best=is_best, val_model=val["overall"]["model"],
                     size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
        print(f"== checkpoint {name}: train env {record['train_env_db']} dB, val {val['overall']} "
              f"(best {best:.2f}{', new best' if is_best else ''})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started, mains, auxes = time.time(), [], []
        first = start_step + 1 if epoch == start_epoch else 1
        for step in range(first, steps_per_epoch + 1):
            log_now = step % args.log_every == 0
            if log_now:
                set_recording(model, True)
            tick = time.time()
            planes, cond, truth = make_batch(rng, args.batch, notes, files_by_family, train_families)
            envelope, aux = model(planes, cond)
            loss, main, aux_loss = loss_fn(envelope, aux, truth)
            optimizer.zero_grad()
            loss.backward()
            if log_now:
                grad_norms = {g: float(torch.sqrt(sum((q.grad ** 2).sum() for q in ps if q.grad is not None)))
                              for g, ps in groups.items()}
                before = {g: [q.detach().clone() for q in ps] for g, ps in groups.items()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            mains.append(float(main))
            auxes.append(float(aux_loss))
            if log_now:
                ratios = {g: float(torch.sqrt(sum(((q.detach() - r) ** 2).sum() for q, r in zip(ps, before[g])))
                                   / torch.sqrt(sum((r ** 2).sum() for r in before[g])))
                          for g, ps in groups.items()}
                del before
                blocks, entropy = collect_recording(model)
                set_recording(model, False)
                window = mains[-args.log_every:]
                events.write("train", epoch=epoch - 1 + step / steps_per_epoch, global_step=schedule.last_epoch,
                             env_db=round(float(np.mean(window)), 4),
                             aux=round(float(np.mean(auxes[-args.log_every:])), 5),
                             lr=schedule.get_last_lr()[0], grad_norm=grad_norms, update_ratio=ratios,
                             block_update=blocks, attention_entropy=entropy,
                             sec_per_step=round(time.time() - tick, 2))
                print(f"epoch {epoch:3d} step {step:5d}/{steps_per_epoch}  env {np.mean(window):5.2f} dB  "
                      f"aux {np.mean(auxes[-args.log_every:]):.3f}  lr {schedule.get_last_lr()[0]:.2e}  "
                      f"{time.time() - started:7.0f} s", flush=True)
            if step == half:
                checkpoint(epoch, step, mains, auxes, started)
        checkpoint(epoch, steps_per_epoch, mains, auxes, started)

    from wavelet2vec.keyboard_v2 import load_keyboard_v2
    model, extra = load_keyboard_v2(ckpt / "best.pt")
    test_rows = evaluate(model, evaluation_episodes(notes, files_by_family, test_families), notes, meta)
    report = {"best_epoch": extra["epoch"], "parameters": count_parameters(model), "test": summarize(test_rows),
              "history": history, "val_families": val_families, "test_families": test_families}
    if not args.skip_demos:
        demos = []
        for family in ("Own voice (sung takes)", "Harmor_synth_preset_rhodes", "Serum_preset1"):
            if family not in files_by_family:
                continue
            file = sorted(files_by_family[family])[0]
            midis = sorted(notes[file])
            n = len(midis)
            for given in ([midis[n // 2]], [midis[n // 4], midis[3 * n // 4]],
                          [midis[n // 4], midis[n // 2], midis[3 * n // 4]]):
                if len(set(given)) == len(given) and n > len(given) + 1:
                    demos.append(render_demo_v2(model, file, given, notes, meta, args.out / "demos"))
        report["demos"] = demos
    (args.out / "keyboard_v2_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["test"], indent=1))


if __name__ == "__main__":
    main()
