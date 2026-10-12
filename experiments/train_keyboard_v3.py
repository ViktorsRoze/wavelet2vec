"""Train KeyboardNet v3 on CPU: 192 kHz stereo, lossless wavelet image, warm-started from v2.

Notes come from experiments/output/keyboard_v3_data (exact 192 kHz stereo audio),
encoded on the fly into the lossless wavelet image (wavelet2vec.wavelet_image).
The v2 checkpoint (default: v2 best.pt) initializes the Swin-L backbone and the
head; the new lossless path starts at zero (wavelet2vec.keyboard_v3).

Episodes, splits (test = v2's 10 families, validation = v2's 5) and the epoch
definition are v2's. Losses, all masked above each note's source Nyquist
(44.1 kHz sources have no content above 22 kHz):
- mid and side cell level (dB), weighted to the audible part of the target;
- a phase-aware complex term: |output - target| / |target| over audible cells;
- multi-resolution log-spectral L1 on the decoded 192 kHz audio (mid, side);
- v2's auxiliary IF deviation and coherence maps.

Checkpoints at every half epoch and epoch (``<epochs>_epoch.pth``, best.pt,
resumable last.pt); training probes and validation go to metrics.jsonl for
experiments/training_dashboard.py.

    PYTHONPATH=src python experiments/train_keyboard_v3.py --epochs 20 [--resume]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from train_keyboard import BUCKETS, OUT, split_families
from train_keyboard_v2 import EventLog, collect_recording, episode, evaluation_episodes, set_recording
from wavelet2vec.keyboard import DB_SCALE, FLOOR_DB
from wavelet2vec.keyboard_v2 import AUX_TARGETS, MAX_EXAMPLES, KeyboardV2Config, load_keyboard_v2
from wavelet2vec.keyboard_v3 import (CORE, FRAMES, ROWS, SPEC, KeyboardNetV3, cell_features, cell_levels_db,
                                     count_parameters_v3, inputs_v3, power, row_nyquist_mask, shift_image)
from wavelet2vec.wavelet_image import WaveletImage, decode, encode

DATA = OUT.parent / "keyboard_v3_data"
AUDIBLE_DB = 70.0
STFT_SIZES = ((2048, 512), (8192, 2048))
HIGH_ROW = int(np.argmax(np.array([440 * 2 ** ((12 + r - 69) / 12) for r in range(ROWS)]) > 24000))


class NoteStore:
    def __init__(self, root=DATA):
        info = json.loads((root / "index.json").read_text(encoding="utf-8"))
        self.arrays = {}
        self.notes = defaultdict(dict)   # file -> midi -> (array, row, source_rate)
        self.meta = {}
        for n in info["notes"]:
            if n["array"] not in self.arrays:
                self.arrays[n["array"]] = np.load(root / "notes" / n["array"], mmap_mode="r")
            self.notes[n["file"]][n["midi"]] = (n["array"], n["row"], n["source_rate"])
            self.meta[n["file"]] = {k: n[k] for k in ("family", "source_type", "synth", "envelope")}

    def audio(self, file, midi):
        array, row, rate = self.notes[file][midi]
        return np.asarray(self.arrays[array][row]), rate


def build_batch(store, episodes, rng=None, augment=False):
    """Encode one batch of episodes: model inputs, carriers, targets and masks."""
    needed = []
    for file, examples, target, *_ in episodes:
        needed += [(file, m) for m in examples] + [(file, target)]
    order = {key: i for i, key in enumerate(dict.fromkeys(needed))}
    audio = np.stack([store.audio(*key)[0] for key in order])
    gains = np.ones(len(order), dtype=np.float32)
    if augment:  # one random gain per episode, shared by its examples and target
        for file, examples, target, *_ in episodes:
            g = 10 ** (rng.uniform(-12, 12) / 20)
            for m in examples + [target]:
                gains[order[(file, m)]] = g
    audio = torch.from_numpy(audio * gains[:, None, None])
    image = encode(audio, SPEC)
    feats = cell_features(image)
    planes, conds, bases, nearest_index, shifts, targets, rates = [], [], [], [], [], [], []
    for file, examples, target, *_ in episodes:
        p, c, base, near = inputs_v3([feats[order[(file, m)]] for m in examples], examples, target)
        planes.append(p)
        conds.append(c)
        bases.append(base)
        nearest = sorted(examples, key=lambda m: abs(target - m))[0]
        nearest_index.append(order[(file, nearest)])
        shifts.append(target - nearest)
        targets.append(order[(file, target)])
        rates.append(store.notes[file][target][2])
    pick = lambda idx: WaveletImage(SPEC, {c: v[idx] for c, v in image.groups.items()}, dict(image.rows))
    carrier = shift_image(pick(nearest_index), shifts)
    target_image = pick(targets)
    return {"planes": torch.from_numpy(np.stack(planes)), "cond": torch.from_numpy(np.stack(conds)),
            "base_db": torch.from_numpy(np.stack(bases)), "carrier": carrier, "target": target_image,
            "target_audio": audio[targets], "target_features": feats[targets],
            "nyquist": row_nyquist_mask(rates).float(), "rates": rates}


def stft_loss(output_audio, target_audio, rates):
    """Log-spectral L1 (dB / 40) on mid and side, bins above each source's Nyquist excluded."""
    total = 0.0
    for n_fft, hop in STFT_SIZES:
        window = torch.hann_window(n_fft, dtype=output_audio.dtype)
        freqs = torch.fft.rfftfreq(n_fft, 1 / SPEC.sample_rate)
        mask = (freqs[None, :] <= 0.95 * torch.tensor(rates, dtype=torch.float32)[:, None] / 2).float()[:, None, :, None]
        for x, y in ((output_audio, target_audio),):
            ms_x = torch.stack([(x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2], 1)
            ms_y = torch.stack([(y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2], 1)
            sx = torch.stft(ms_x.flatten(0, 1), n_fft, hop, window=window, return_complex=True)
            sy = torch.stft(ms_y.flatten(0, 1), n_fft, hop, window=window, return_complex=True)
            dx = 10 * torch.log10(power(sx) + 1e-12).view(x.shape[0], 2, *sx.shape[-2:])
            dy = 10 * torch.log10(power(sy) + 1e-12).view(x.shape[0], 2, *sy.shape[-2:])
            peak = dy.amax(dim=(-2, -1), keepdim=True)
            weight = mask * torch.where(dy > peak - AUDIBLE_DB, 1.0, 0.1)
            total = total + (weight * (dx - dy).abs()).sum() / weight.sum() / DB_SCALE
    return total / len(STFT_SIZES)


def losses(model_out, batch):
    out_image, envelope, aux = model_out
    target = batch["target"]
    pred_db, true_db = cell_levels_db(out_image), cell_levels_db(target)
    peak = true_db[:, 0].amax(dim=(-2, -1), keepdim=True)
    audible = (true_db[:, 0] > peak - AUDIBLE_DB).float() * batch["nyquist"][:, :, None]
    weight = audible + 0.1 * (batch["nyquist"][:, :, None] - audible)
    mid = (weight * (pred_db[:, 0] - true_db[:, 0]).abs()).sum() / weight.sum() / DB_SCALE
    side = (weight * (pred_db[:, 1] - true_db[:, 1]).abs()).sum() / weight.sum() / DB_SCALE
    num, den = 0.0, 0.0
    for c, values in out_image.groups.items():
        rows = torch.as_tensor(out_image.rows[c])
        m = audible[:, rows][:, None, :, :, None]
        diff = values[:, :, :, CORE] - target.groups[c][:, :, :, CORE]
        num = num + (torch.sqrt(power(diff) + 1e-20) * m).sum()
        den = den + (torch.sqrt(power(target.groups[c][:, :, :, CORE]) + 1e-20) * m).sum()
    complex_term = num / (den + 1e-12)
    spectral = stft_loss(decode(out_image), batch["target_audio"], batch["rates"])
    feats = batch["target_features"]
    aux_idx = [AUX_TARGETS.index(1), AUX_TARGETS.index(2)]
    aux_loss = (audible[:, None] * (aux[:, aux_idx] - feats[:, [1, 2]]).abs()).sum() / (2 * audible.sum() + 1e-6)
    total = mid + 0.5 * side + 0.2 * complex_term + 0.5 * spectral + 0.1 * aux_loss
    mid, side, complex_term, spectral, aux_loss = (v.detach() for v in (mid, side, complex_term, spectral, aux_loss))
    return total, {"mid_db": float(mid) * DB_SCALE, "side_db": float(side) * DB_SCALE,
                   "complex": float(complex_term), "spectral_db": float(spectral) * DB_SCALE, "aux": float(aux_loss)}


def evaluate(model, store, episodes, batch_size=8):
    """Cell-level error (dB, audible, below source Nyquist): model vs shifted carrier vs blend; plus
    ultrasonic rows (192 kHz sources only), side channel, and decoded-audio spectral error."""
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(episodes), batch_size):
            chunk = episodes[i:i + batch_size]
            batch = build_batch(store, chunk)
            out_image, envelope, aux = model(batch["planes"], batch["cond"], batch["carrier"])
            pred, carrier, truth = cell_levels_db(out_image), cell_levels_db(batch["carrier"]), cell_levels_db(batch["target"])
            spec_model = [float(stft_loss(decode(out_image)[[j]], batch["target_audio"][[j]], [batch["rates"][j]])) * DB_SCALE
                          for j in range(len(chunk))]
            spec_carrier = [float(stft_loss(decode(batch["carrier"])[[j]], batch["target_audio"][[j]], [batch["rates"][j]])) * DB_SCALE
                            for j in range(len(chunk))]
            for j, (file, examples, target, bucket) in enumerate(chunk):
                t = truth[j, 0]
                mask = (t > t.max() - AUDIBLE_DB) & (batch["nyquist"][j][:, None] > 0)
                high = mask.clone()
                high[:HIGH_ROW] = False
                side_mask = mask & (truth[j, 1] > FLOOR_DB + 1)
                err = lambda a: float((a - t).abs()[mask].mean())
                row = {"family": store.meta[file]["family"], "k": len(examples), "bucket": f"{bucket[0]}-{bucket[1]}",
                       "model": err(pred[j, 0]), "carrier": err(carrier[j, 0]), "blend": err(batch["base_db"][j]),
                       "spectral_model": spec_model[j], "spectral_carrier": spec_carrier[j]}
                if high.any():
                    row["ultrasonic_model"] = float((pred[j, 0] - t).abs()[high].mean())
                    row["ultrasonic_carrier"] = float((carrier[j, 0] - t).abs()[high].mean())
                if side_mask.any():
                    row["side_model"] = float((pred[j, 1] - truth[j, 1]).abs()[side_mask].mean())
                    row["side_carrier"] = float((carrier[j, 1] - truth[j, 1]).abs()[side_mask].mean())
                rows.append(row)
    model.train()
    return rows


def summarize(rows):
    keys = ("model", "carrier", "blend", "spectral_model", "spectral_carrier", "ultrasonic_model",
            "ultrasonic_carrier", "side_model", "side_carrier")
    mean = lambda rs, k: round(float(np.mean([r[k] for r in rs if k in r])), 2) if any(k in r for r in rs) else None
    table = {"overall": {k: mean(rows, k) for k in keys}}
    for k in range(1, MAX_EXAMPLES + 1):
        for lo, hi in BUCKETS:
            sel = [r for r in rows if r["k"] == k and r["bucket"] == f"{lo}-{hi}"]
            if sel:
                table[f"{k}-shot {lo}-{hi} st"] = {m: mean(sel, m) for m in ("model", "carrier", "blend")} | {"n": len(sel)}
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    table["per_family"] = {f: {m: mean(rs, m) for m in ("model", "carrier", "blend")} for f, rs in sorted(fam.items())}
    return table


def parameter_groups(model):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if name.startswith(("detail_embed", "phase", "residual")):
            groups["new lossless path"].append(p)
        elif name.startswith("net.stages."):
            groups[f"stage{int(name.split('.')[2]) + 1}"].append(p)
        elif name.startswith(("net.patch", "net.merges", "net.stage_norms")):
            groups["patch+merge"].append(p)
        elif name.startswith("net.stem"):
            groups["stem"].append(p)
        elif name.startswith("net.cond"):
            groups["conditioning"].append(p)
        else:
            groups["head"].append(p)
    return dict(groups)


def write_demo(model, store, file, given, out_dir):
    """Generate every other note of ``file`` from ``given`` notes; write original and generated 192 kHz WAVs."""
    targets = [m for m in sorted(store.notes[file]) if m not in given]
    generated, original = [], []
    for i in range(0, len(targets), 8):
        chunk = [(file, list(given), t, (0, 0)) for t in targets[i:i + 8]]
        batch = build_batch(store, chunk)
        with torch.no_grad():
            out_image, _, _ = model(batch["planes"], batch["cond"], batch["carrier"])
        generated.append(decode(out_image).numpy())
        original.append(batch["target_audio"].numpy())
    gen = np.concatenate([g for g in np.concatenate(generated)], axis=1)
    orig = np.concatenate([o for o in np.concatenate(original)], axis=1)
    stem = Path(file).stem.removeprefix("from_C1_till_B10_") + "_from_" + "_".join(map(str, given))
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / f"{stem}__original.wav", orig.T.astype(np.float32), SPEC.sample_rate, subtype="FLOAT")
    gen = gen * min(1.0, 0.98 / max(np.abs(gen).max(), 1e-9))
    sf.write(out_dir / f"{stem}__keyboardnet_v3.wav", gen.T.astype(np.float32), SPEC.sample_rate, subtype="FLOAT")
    return {"file": file, "given": list(given), "targets": targets}


def main():
    import training_control as ctl                  # pause / resume exactly (experiments/training_control.py)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of the warm-started v2 weights")
    parser.add_argument("--new-lr", type=float, default=5e-4, help="learning rate of the new lossless path")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warm-start", type=Path, default=OUT / "v2" / "checkpoints" / "best.pt")
    parser.add_argument("--size", default=None, help="only without --warm-start: tiny/base/large")
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=20, help="per (k, distance) cell")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=OUT / "v3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-demos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    store = NoteStore()
    meta = store.meta
    train_families, test_families = split_families(meta)
    files_by_family = defaultdict(list)
    for file in store.notes:
        files_by_family[meta[file]["family"]].append(file)
    rng_split = np.random.default_rng(8)
    pool = [f for f in train_families if len(files_by_family[f]) == 1]
    val_families = sorted(rng_split.choice(pool, 5, replace=False).tolist())
    train_families = [f for f in train_families if f not in val_families]
    train_notes = sum(len(store.notes[f]) for fam in train_families for f in files_by_family[fam])
    epoch_samples = args.epoch_samples or train_notes
    steps_per_epoch = max(epoch_samples // args.batch, 1)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    ctl.record_launch(args.out)
    ctl.clear_pause(args.out)

    if args.size:
        model = KeyboardNetV3(None, KeyboardV2Config(size=args.size))
        warm = None
    else:
        v2, extra = load_keyboard_v2(args.warm_start)
        model = KeyboardNetV3(v2.state_dict(), v2.config)
        warm = {"checkpoint": str(args.warm_start), "v2_epoch": extra.get("epoch")}
        del v2
    groups = parameter_groups(model)
    new = groups.get("new lossless path", [])
    new_ids = {id(p) for p in new}
    old_decay = [p for p in model.parameters() if id(p) not in new_ids and p.ndim >= 2]
    old_no_decay = [p for p in model.parameters() if id(p) not in new_ids and p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": old_decay, "weight_decay": 0.05, "lr": args.lr},
                                   {"params": old_no_decay, "weight_decay": 0.0, "lr": args.lr},
                                   {"params": new, "weight_decay": 0.0, "lr": args.new_lr}])
    total_steps = steps_per_epoch * args.epochs
    warmup = min(200, max(total_steps // 20, 1))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(s / total_steps, 1.0))))
    rng = np.random.default_rng(1)
    start_epoch, start_step, best, history = 1, 0, float("inf"), []
    resume_parts, resume_seconds = None, 0.0
    state_path = ctl.latest_state(ckpt) if args.resume else None
    if state_path is not None:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        ctl.restore_random(state, rng, None)
        best, history = state["best"], state["history"]
        done = state.get("step_in_epoch", steps_per_epoch)
        start_epoch, start_step = (state["epoch"] + 1, 0) if done >= steps_per_epoch else (state["epoch"], done)
        if start_step > 0:
            resume_parts, resume_seconds = state.get("parts"), state.get("epoch_seconds", 0.0)
        print(f"resumed from {state_path.name} at epoch {start_epoch} step {start_step}", flush=True)
        del state

    val_episodes = evaluation_episodes(store.notes, files_by_family, val_families, per_cell=args.val_episodes, seed=13)
    run_info = {"model": "keyboard-v3", "parameters": count_parameters_v3(model), "warm_start": warm,
                "train_families": len(train_families), "val_families": val_families, "test_families": test_families,
                "train_notes": train_notes, "epoch_samples": epoch_samples, "steps_per_epoch": steps_per_epoch,
                "batch": args.batch, "lr": args.lr, "new_lr": args.new_lr, "epochs_total": args.epochs,
                "sample_rate": SPEC.sample_rate, "rows": ROWS, "frames": FRAMES,
                "size": "large (from v2)" if warm else args.size}
    print(json.dumps(run_info, indent=1, default=str), flush=True)
    events = EventLog(args.out / "metrics.jsonl")
    events.write("run_start", epoch=start_epoch - 1 + start_step / steps_per_epoch, resumed=bool(args.resume),
                 **{k: v for k, v in run_info.items() if k != "parameters"},
                 parameters={"total": run_info["parameters"]["total"]})
    half = steps_per_epoch // 2

    def full_state(epoch, step, parts, started):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                **ctl.random_states(rng, None), "parts": parts, "epoch_seconds": time.time() - started,
                "epoch": epoch, "step_in_epoch": step, "best": best, "history": history}

    def checkpoint(epoch, step, parts, started):
        nonlocal best
        point = epoch - 1 + step / steps_per_epoch
        val = summarize(evaluate(model, store, val_episodes))
        record = {"epoch": round(point, 3), "train": {k: round(float(np.mean([p[k] for p in parts])), 4) for k in parts[0]}
                  if parts else None, "val": val["overall"], "minutes": round((time.time() - started) / 60, 1)}
        history.append(record)
        name = f"{point:g}_epoch.pth"
        torch.save({"state": model.state_dict(), "metrics": record, "v2_config": model.net.config.__dict__}, ckpt / name)
        is_best = val["overall"]["model"] < best
        if is_best:
            best = val["overall"]["model"]
            shutil.copyfile(ckpt / name, ckpt / "best.pt")
        ctl.save_state(full_state(epoch, step, parts, started), ckpt / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        val_short = {"model": val["overall"]["model"], "shift_nearest": val["overall"]["carrier"],
                     "blend": val["overall"]["blend"]}
        events.write("val", epoch=point, val=val_short, val_full=val["overall"],
                     val_by_k={k: v for k, v in val.items() if "shot" in k}, val_per_family=val["per_family"])
        events.write("checkpoint", epoch=point, file=name, best=is_best, val_model=val["overall"]["model"],
                     size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
        print(f"== checkpoint {name}: val {val['overall']} (best {best:.2f}{', new best' if is_best else ''})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started, parts = time.time(), []
        if epoch == start_epoch and resume_parts is not None:
            started, parts = time.time() - resume_seconds, list(resume_parts)
        first = start_step + 1 if epoch == start_epoch else 1
        for step in range(first, steps_per_epoch + 1):
            if ctl.pause_requested(args.out):           # pause: save the exact state, then stop
                ctl.save_state(full_state(epoch, step - 1, parts, started), ckpt / "paused.pt")
                ctl.clear_pause(args.out)
                events.write("paused", epoch=epoch - 1 + (step - 1) / steps_per_epoch, file="checkpoints/paused.pt")
                print(f"paused at epoch {epoch} after step {step - 1}: state saved to checkpoints/paused.pt", flush=True)
                return
            log_now = step % args.log_every == 0
            if log_now:
                set_recording(model.net, True)
            tick = time.time()
            episodes = [episode(rng, store.notes, files_by_family, train_families) + ((0, 0),) for _ in range(args.batch)]
            batch = build_batch(store, episodes, rng, augment=True)
            data_seconds = time.time() - tick
            total, part = losses(model(batch["planes"], batch["cond"], batch["carrier"]), batch)
            optimizer.zero_grad()
            total.backward()
            if log_now:
                grad_norms = {g: float(torch.sqrt(sum((q.grad ** 2).sum() for q in ps if q.grad is not None)))
                              for g, ps in groups.items()}
                before = {g: [q.detach().clone() for q in ps] for g, ps in groups.items()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            parts.append(part)
            if log_now:
                ratios = {g: float(torch.sqrt(sum(((q.detach() - r) ** 2).sum() for q, r in zip(ps, before[g])))
                                   / (torch.sqrt(sum((r ** 2).sum() for r in before[g])) + 1e-12))
                          for g, ps in groups.items()}
                del before
                blocks, entropy = collect_recording(model.net)
                set_recording(model.net, False)
                window = parts[-args.log_every:]
                mean = {k: round(float(np.mean([p[k] for p in window])), 4) for k in window[0]}
                events.write("train", epoch=epoch - 1 + step / steps_per_epoch, global_step=schedule.last_epoch,
                             env_db=mean["mid_db"], losses=mean, lr=schedule.get_last_lr()[0], grad_norm=grad_norms,
                             update_ratio=ratios, block_update=blocks, attention_entropy=entropy,
                             sec_per_step=round(time.time() - tick, 2), data_seconds=round(data_seconds, 2))
                print(f"epoch {epoch:3d} step {step:5d}/{steps_per_epoch}  mid {mean['mid_db']:5.2f} dB  "
                      f"side {mean['side_db']:5.2f} dB  spectral {mean['spectral_db']:5.2f} dB  complex {mean['complex']:.3f}  "
                      f"lr {schedule.get_last_lr()[0]:.2e}  {time.time() - started:7.0f} s", flush=True)
            if step == half:
                checkpoint(epoch, step, parts, started)
        checkpoint(epoch, steps_per_epoch, parts, started)

    state = torch.load(ckpt / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state"])
    test = summarize(evaluate(model, store, evaluation_episodes(store.notes, files_by_family, test_families,
                                                                per_cell=60, seed=21)))
    report = {"best": state["metrics"], "test": test, "history": history, "val_families": val_families,
              "test_families": test_families}
    if not args.skip_demos:
        demos = []
        for family in ("Own voice (sung takes)", "Harmor_synth_preset_rhodes", "Serum_preset1"):
            if family in files_by_family:
                file = sorted(files_by_family[family])[0]
                midis = sorted(store.notes[file])
                n = len(midis)
                for given in ([midis[n // 2]], [midis[n // 4], midis[3 * n // 4]],
                              [midis[n // 4], midis[n // 2], midis[3 * n // 4]]):
                    if len(set(given)) == len(given) and n > len(given) + 1:
                        demos.append(write_demo(model, store, file, given, args.out / "demos"))
        report["demos"] = demos
    (args.out / "keyboard_v3_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(test["overall"], indent=1))


if __name__ == "__main__":
    main()
