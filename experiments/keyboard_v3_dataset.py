"""KeyboardNet v3 data: every usable note as exact stereo audio at 192 kHz.

The lossless wavelet image (wavelet2vec.wavelet_image) is computed on the fly
from these notes (~0.03 s per stereo note). Precomputed images would be ~10 MB
per note (~150 GB in float32), while the audio itself is 1.8 MB.

Per source file: ``notes/<stem>.npy`` float32 ``[n_notes, 2 (L, R), 230400]``,
1.2 s at 192 kHz, plus one row per note in ``index.json``: file, MIDI note,
family, source type, envelope class, source sample rate (content above the
source's Nyquist is empty and must not be scored) and onset offset.

Note selection is v1/v2's (keyboard_dataset.py):
- the README exclusions (silent/cut slots, MIDI 128-131 repeats, Korg from
  MIDI 124, arpeggio/sequence patches);
- recordings on their fitted grids;
- sung takes from their notes tables, starting 100 ms before the pitch onset.

Rates are converted explicitly with polyphase resampling: 44.1 kHz x640/147,
96 kHz x2; 192 kHz notes are copied unchanged.

    PYTHONPATH=src python experiments/keyboard_v3_dataset.py [--workers 12]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from keyboard_dataset import EXCLUDED_FILES, SILENT_PEAK
from wavelet2vec.audio_io import resample

HERE = Path(__file__).resolve().parent
RATE = 192000
NOTE = int(1.2 * RATE)


def file_notes(job):
    kind, payload, root, out_dir = job
    if kind == "grid":
        row = payload
        name = row["file"]
        data, rate = sf.read(root / name, dtype="float64", always_2d=True)
        origin, beat = float(row["grid_origin_ms"]) / 1000, float(row["beat_ms"]) / 1000
        clamped, recording = row["clamped_above_midi127"] == "True", row["origin"] == "recording"
        picks = []
        for k in range(120):
            midi = 12 + k
            if (clamped and midi >= 128) or (recording and midi >= 124):
                continue
            start = int(round((origin + k * beat) * rate))
            length = int(round(1.2 * rate))
            if start < 0 or start + length > data.shape[0]:
                continue
            if np.abs(data[start:start + length].mean(1)).max() < SILENT_PEAK:
                continue
            picks.append((midi, start, length, 0.0))
    else:
        wav, table = payload
        name = wav.name
        data, rate = sf.read(wav, dtype="float64", always_2d=True)
        picks = []
        for r in csv.DictReader(open(table, encoding="utf-8")):
            start = max(int(round((float(r["start_s"]) - 0.1) * rate)), 0)
            picks.append((int(r["midi"]), start, int(round(1.2 * rate)), 0.1))
    notes, kept = [], []
    for midi, start, length, lead in picks:
        piece = data[start:start + length].T
        if piece.shape[1] < length // 2:
            continue
        if rate != RATE:
            piece = resample(piece, rate, RATE)
        out = np.zeros((2, NOTE), dtype=np.float32)
        n = min(NOTE, piece.shape[1])
        out[:, :n] = piece[-2:, :n] if piece.shape[0] >= 2 else np.repeat(piece[:1, :n], 2, 0)
        notes.append(out)
        kept.append((midi, lead))
    if notes:
        np.save(out_dir / (Path(name).stem + ".npy"), np.stack(notes))
    return name, rate, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=HERE.parent / "note_dataset")
    parser.add_argument("--out", type=Path, default=HERE / "output" / "keyboard_v3_data")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    notes_dir = args.out / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    audit = [r for r in csv.DictReader(open(args.root / "dataset_audit.csv", encoding="utf-8"))
             if r["file"] not in EXCLUDED_FILES]
    sung = sorted((w, w.with_name(w.stem + ".notes.csv")) for w in args.root.glob("from_*_real_Vocal_*.wav")
                  if w.with_name(w.stem + ".notes.csv").exists())
    meta = {r["file"]: {"family": r["family"], "source_type": r["source_type"], "synth": r["synth"],
                        "envelope": r["envelope"], "origin": r["origin"]} for r in audit}
    for wav, _ in sung:
        meta[wav.name] = {"family": "Own voice (sung takes)", "source_type": "vox", "synth": "voice",
                          "envelope": "isolated sung notes", "origin": "recording"}
    jobs = [("grid", r, args.root, notes_dir) for r in audit] + [("sung", s, args.root, notes_dir) for s in sung]
    with Pool(args.workers) as pool:
        results = pool.map(file_notes, jobs, chunksize=1)
    index = []
    for name, rate, kept in results:
        for row, (midi, lead) in enumerate(kept):
            index.append({"file": name, "array": Path(name).stem + ".npy", "row": row, "midi": midi,
                          "source_rate": rate, "lead_in_s": lead, **meta[name]})
    (args.out / "index.json").write_text(json.dumps({"sample_rate": RATE, "note_samples": NOTE, "notes": index},
                                                    indent=0), encoding="utf-8")
    size = sum(p.stat().st_size for p in notes_dir.glob("*.npy")) / 1e9
    print(f"{len({i['file'] for i in index})} files, {len(index)} notes, {size:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
