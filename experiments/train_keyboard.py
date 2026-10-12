"""Train KeyboardNet v1 on CPU: from 1-2 notes of a sound, predict its other notes.

Episodes come from the envelope cache built by keyboard_dataset.py. Each
step picks a family, then a file, a target note and one or two example
notes from the same file. Examples sit mostly within an octave of the
target, sometimes up to four octaves away. Whole sound families are held out
for testing, including the author's own sung takes, so the test measures
sounds the model has never heard.

Metric: mean absolute dB error of the target envelope, over its audible part
(within 70 dB of the note's peak). Compared against shifting the nearest
example (plain pitch shift) and, with two examples, a distance-weighted blend
of both. Demos render whole keyboards at 48 kHz for listening.

    PYTHONPATH=src python experiments/train_keyboard.py [--steps 6000]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from wavelet2vec.audio_io import resample
from wavelet2vec.keyboard import (DB_SCALE, ENV_RATE, NOTE_SAMPLES, KeyboardConfig, build_model, describe,
                                  network_inputs, predict_envelope, render_note, save_keyboard_model,
                                  shift_envelope)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "keyboard"
ROOT = HERE.parent / "note_dataset"
AUDIBLE_DB = 70.0
BUCKETS = ((1, 5), (6, 12), (13, 24), (25, 48))


def load_cache():
    data = np.load(OUT / "envelopes.npz")
    info = json.loads((OUT / "envelopes.json").read_text(encoding="utf-8"))
    files = info["files"]
    notes = defaultdict(dict)
    for env, midi, index in zip(data["envelopes"], data["midi"], data["file_index"]):
        notes[files[index]][int(midi)] = env.astype(np.float32)
    return notes, info["meta"]


def split_families(meta, n_test=9, seed=7):
    families = sorted({m["family"] for m in meta.values()})
    rng = np.random.default_rng(seed)
    pool = [f for f in families if f not in ("Harmor simple saw pluck", "Own voice (sung takes)")]
    test = set(rng.choice(pool, n_test, replace=False)) | {"Own voice (sung takes)"}
    return [f for f in families if f not in test], sorted(test)


def sample_episode(rng, notes_by_file, files_by_family, families):
    while True:
        family = families[rng.integers(len(families))]
        file = files_by_family[family][rng.integers(len(files_by_family[family]))]
        midis = sorted(notes_by_file[file])
        if len(midis) < 3:
            continue
        target = midis[rng.integers(len(midis))]
        reach = int(rng.choice([12, 24, 48], p=[0.6, 0.3, 0.1]))
        others = [m for m in midis if m != target and abs(m - target) <= reach]
        k = 1 if rng.random() < 0.5 or len(others) < 2 else 2
        if len(others) < k:
            continue
        chosen = list(rng.choice(others, k, replace=False))
        return file, [int(m) for m in chosen], int(target)


def make_batch(rng, size, notes_by_file, files_by_family, families):
    planes, conds, targets = [], [], []
    for _ in range(size):
        file, examples, target = sample_episode(rng, notes_by_file, files_by_family, families)
        p, c, _ = network_inputs([notes_by_file[file][m] for m in examples], examples, target)
        planes.append(p)
        conds.append(c)
        targets.append(notes_by_file[file][target])
    return (torch.from_numpy(np.stack(planes)), torch.from_numpy(np.stack(conds)),
            torch.from_numpy(np.stack(targets)))


def weighted_l1(pred, target_db):
    target = target_db / DB_SCALE
    peak = target_db.amax(dim=(-2, -1), keepdim=True)
    weight = torch.where(target_db > peak - AUDIBLE_DB, 1.0, 0.1)
    return (weight * (pred - target).abs()).sum() / weight.sum()


def audible_error(pred_db, target_db):
    mask = target_db > target_db.max() - AUDIBLE_DB
    return float(np.abs(pred_db - target_db)[mask].mean())


def evaluation_episodes(notes_by_file, files_by_family, families, per_cell=150, seed=11):
    rng = np.random.default_rng(seed)
    episodes = []
    for k in (1, 2):
        for lo, hi in BUCKETS:
            cell, tries = [], 0
            while len(cell) < per_cell and tries < per_cell * 200:
                tries += 1
                family = families[rng.integers(len(families))]
                file = files_by_family[family][rng.integers(len(files_by_family[family]))]
                midis = sorted(notes_by_file[file])
                target = midis[rng.integers(len(midis))]
                near = [m for m in midis if m != target and lo <= abs(m - target) <= hi]
                if not near:
                    continue
                first = int(rng.choice(near))
                if k == 1:
                    cell.append((file, [first], target, (lo, hi)))
                    continue
                second_pool = [m for m in midis if m not in (target, first) and abs(m - target) <= hi]
                if second_pool:
                    cell.append((file, [first, int(rng.choice(second_pool))], target, (lo, hi)))
            episodes += cell
    return episodes


def evaluate(model, episodes, notes_by_file, meta):
    rows = []
    model.eval()
    for file, examples, target, bucket in episodes:
        envs = [notes_by_file[file][m] for m in examples]
        truth = notes_by_file[file][target]
        pred, blend = predict_envelope(model, envs, examples, target)
        nearest = int(np.argmin([abs(target - m) for m in examples]))
        shifted, _ = shift_envelope(envs[nearest], target - examples[nearest])
        rows.append({"family": meta[file]["family"], "k": len(examples), "bucket": f"{bucket[0]}-{bucket[1]}",
                     "model": audible_error(pred, truth), "shift_nearest": audible_error(shifted, truth),
                     "blend": audible_error(blend, truth)})
    model.train()
    return rows


def summarize(rows):
    table = {}
    for k in (1, 2):
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


def note_audio(file, meta):
    """{midi: [2, NOTE_SAMPLES] at 48 kHz} with the same slicing rules as the cache."""
    data, rate = sf.read(ROOT / file, dtype="float64", always_2d=True)
    audio = resample(data.T, rate, ENV_RATE)
    notes = {}
    if "_real_Vocal_" in file:
        for row in csv.DictReader(open(ROOT / (Path(file).stem + ".notes.csv"), encoding="utf-8")):
            start = max(int(round((float(row["start_s"]) - 0.1) * ENV_RATE)), 0)
            piece = audio[:, start:start + NOTE_SAMPLES]
            if piece.shape[1] == NOTE_SAMPLES:
                notes.setdefault(int(row["midi"]), piece)
        return notes
    row = next(r for r in csv.DictReader(open(ROOT / "dataset_audit.csv", encoding="utf-8")) if r["file"] == file)
    origin, beat = float(row["grid_origin_ms"]) / 1000, float(row["beat_ms"]) / 1000
    for k in range(120):
        start = int(round((origin + k * beat) * ENV_RATE))
        if 0 <= start and start + NOTE_SAMPLES <= audio.shape[1]:
            notes[12 + k] = audio[:, start:start + NOTE_SAMPLES]
    return notes


def render_demo(model, file, example_midis, notes_by_file, meta, out_dir):
    audio = note_audio(file, meta)
    targets = [m for m in sorted(notes_by_file[file]) if m in audio]
    envs = [notes_by_file[file][m] for m in example_midis]
    original, shifted_only, generated = [], [], []
    for target in targets:
        nearest = min(example_midis, key=lambda m: abs(m - target))
        pred, _ = predict_envelope(model, envs, example_midis, target)
        plain, _ = shift_envelope(notes_by_file[file][nearest], target - nearest)
        generated.append(render_note(audio[nearest], nearest, target, pred))
        shifted_only.append(render_note(audio[nearest], nearest, target, plain))
        original.append(audio[target])
    stem = Path(file).stem.removeprefix("from_C1_till_B10_") + "_from_" + "_".join(map(str, example_midis))
    out_dir.mkdir(parents=True, exist_ok=True)
    peak = max(np.abs(np.concatenate(original, 1)).max(), 1e-9)
    for name, notes in (("original", original), ("pitch_shift_only", shifted_only), ("keyboardnet_v1", generated)):
        joined = np.concatenate(notes, 1)
        joined = joined * min(1.0, 0.98 / max(np.abs(joined).max(), 1e-9)) if name != "original" else joined
        sf.write(out_dir / f"{stem}__{name}.wav", joined.T.astype(np.float32), ENV_RATE, subtype="FLOAT")
    return {"file": file, "examples": example_midis, "targets": targets, "peak": float(peak)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    notes_by_file, meta = load_cache()
    train_families, test_families = split_families(meta)
    files_by_family = defaultdict(list)
    for file in notes_by_file:
        files_by_family[meta[file]["family"]].append(file)
    print(f"{len(notes_by_file)} files; train families {len(train_families)}, test families {len(test_families)}: {test_families}")

    model = build_model(KeyboardConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1)
    test_episodes = evaluation_episodes(notes_by_file, files_by_family, test_families)
    train_episodes = evaluation_episodes(notes_by_file, files_by_family, train_families, per_cell=40, seed=12)
    history, started = [], time.time()
    for step in range(1, args.steps + 1):
        planes, cond, target = make_batch(rng, args.batch, notes_by_file, files_by_family, train_families)
        loss = weighted_l1(model(planes, cond), target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        history.append(float(loss.detach()) * DB_SCALE)
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}  loss {np.mean(history[-250:]):6.2f} dB  {time.time() - started:7.0f} s", flush=True)
        if step % 2000 == 0 and step < args.steps:
            print("  test overall:", summarize(evaluate(model, test_episodes[::6], notes_by_file, meta))["overall"], flush=True)
    minutes = (time.time() - started) / 60
    test = summarize(evaluate(model, test_episodes, notes_by_file, meta))
    held_in = summarize(evaluate(model, train_episodes, notes_by_file, meta))
    report = {"model": describe(model), "steps": args.steps, "batch": args.batch, "train_minutes": round(minutes, 1),
              "device": "cpu", "train_families": train_families, "test_families": test_families,
              "test": test, "train_families_eval": {k: v for k, v in held_in.items() if k != "per_family"},
              "loss_curve_db": [round(float(np.mean(history[i:i + 250])), 3) for i in range(0, len(history), 250)]}
    save_keyboard_model(OUT / "keyboard_v1.pt", model, extra={"test_families": test_families})
    demos = []
    for family in ("Own voice (sung takes)",) + tuple(f for f in test_families if f != "Own voice (sung takes)")[:2]:
        file = sorted(files_by_family[family])[0]
        midis = sorted(notes_by_file[file])
        middle = midis[len(midis) // 2]
        demos.append(render_demo(model, file, [middle], notes_by_file, meta, OUT / "demos"))
        if len(midis) > 4:
            pair = [midis[len(midis) // 4], midis[3 * len(midis) // 4]]
            demos.append(render_demo(model, file, pair, notes_by_file, meta, OUT / "demos"))
    report["demos"] = demos
    (OUT / "keyboard_v1_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "train_minutes")}, indent=1))
    print(json.dumps(report["test"], indent=1))


if __name__ == "__main__":
    main()
