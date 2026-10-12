"""Build the KeyboardNet envelope cache from note_dataset.

One envelope per usable note: ``[124 semitone bands, 120 x 10 ms]`` dB at
48 kHz (see wavelet2vec.keyboard). Applies the README's exclusions:
- silent and cut slots;
- MIDI 128-131 in files that repeat MIDI 127;
- Korg recordings from MIDI 124 up;
- arpeggiated / sequenced patches.

Grid files use the audit's grid (exact for renders, fitted for recordings).
Sung takes use their notes tables: each note starts 100 ms before its pitch
onset to keep the consonant.

``--features rich`` stores KeyboardNet v2's 7-channel spectral / phase /
wave / stereo features (wavelet2vec.keyboard_v2) instead of v1 envelopes.

    PYTHONPATH=src python experiments/keyboard_dataset.py [--features rich]
"""
from __future__ import annotations

import os

# One FFT/BLAS thread per worker process: the pool already uses every core,
# and nested threading made the first build ~4x slower.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from wavelet2vec.audio_io import resample
from wavelet2vec.keyboard import ENV_RATE, NOTE_SAMPLES, note_envelope
from wavelet2vec.keyboard_v2 import rich_note_features

HERE = Path(__file__).resolve().parent
EXCLUDED_FILES = {  # not playable note-to-note: the pattern depends on time, not on the key
    "from_C1_till_B10_massiv_x__preset_4.wav", "from_C1_till_B10_massiv_x__preset_5.wav",
    "from_C1_till_B10_massiv_x__preset_8.wav", "from_C1_till_B10_Korg_monolog_analog_synth_preset_3_sequance.wav",
}
SILENT_PEAK = 1e-5


def describe_note(note, features):
    return rich_note_features(note) if features == "rich" else note_envelope(note.mean(0))


def grid_notes(row, root, features):
    data, rate = sf.read(root / row["file"], dtype="float64", always_2d=True)
    stereo = resample(data.T, rate, ENV_RATE)
    origin = float(row["grid_origin_ms"]) / 1000.0
    beat = float(row["beat_ms"]) / 1000.0
    clamped = row["clamped_above_midi127"] == "True"
    recording = row["origin"] == "recording"
    notes = []
    for k in range(120):
        midi = 12 + k
        if (clamped and midi >= 128) or (recording and midi >= 124):
            continue
        start = int(round((origin + k * beat) * ENV_RATE))
        if start < 0 or start + NOTE_SAMPLES > stereo.shape[1]:
            continue
        note = stereo[:, start:start + NOTE_SAMPLES]
        if np.abs(note.mean(0)).max() < SILENT_PEAK:
            continue
        notes.append((midi, describe_note(note, features)))
    return notes


def sung_notes(wav, table, features):
    data, rate = sf.read(wav, dtype="float64", always_2d=True)
    stereo = resample(data.T, rate, ENV_RATE)
    notes = []
    for row in csv.DictReader(open(table, encoding="utf-8")):
        start = int(round((float(row["start_s"]) - 0.1) * ENV_RATE))
        start = max(start, 0)
        note = stereo[:, start:start + NOTE_SAMPLES]
        if note.shape[1] < NOTE_SAMPLES // 2:
            continue
        if note.shape[1] < NOTE_SAMPLES:
            note = np.pad(note, ((0, 0), (0, NOTE_SAMPLES - note.shape[1])))
        notes.append((int(row["midi"]), describe_note(note, features)))
    return notes


def work(job):
    kind, payload, root, features = job
    if kind == "grid":
        return payload["file"], grid_notes(payload, root, features)
    wav, table = payload
    return wav.name, sung_notes(wav, table, features)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=HERE.parent / "note_dataset")
    parser.add_argument("--features", choices=("envelope", "rich"), default="envelope")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    args.out = args.out or HERE / "output" / "keyboard" / ("features_rich.npz" if args.features == "rich" else "envelopes.npz")
    audit = [r for r in csv.DictReader(open(args.root / "dataset_audit.csv", encoding="utf-8"))
             if r["file"] not in EXCLUDED_FILES]
    sung = sorted((w, w.with_name(w.stem + ".notes.csv")) for w in args.root.glob("from_*_real_Vocal_*.wav")
                  if w.with_name(w.stem + ".notes.csv").exists())
    meta = {r["file"]: {"family": r["family"], "source_type": r["source_type"], "synth": r["synth"],
                        "envelope": r["envelope"], "origin": r["origin"], "sample_rate": int(r["sample_rate"]),
                        "category": r["category_from_name"]} for r in audit}
    for wav, _ in sung:
        meta[wav.name] = {"family": "Own voice (sung takes)", "source_type": "vox", "synth": "voice",
                          "envelope": "isolated sung notes", "origin": "recording",
                          "sample_rate": sf.info(wav).samplerate, "category": "vox"}
    jobs = [("grid", r, args.root, args.features) for r in audit] + [("sung", s, args.root, args.features) for s in sung]
    with Pool(args.workers) as pool:
        results = pool.map(work, jobs, chunksize=1)
    files = [name for name, notes in results if notes]
    envelopes, midis, file_index = [], [], []
    for name, notes in results:
        if not notes:
            continue
        for midi, env in notes:
            envelopes.append(env.astype(np.float16))
            midis.append(midi)
            file_index.append(files.index(name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, envelopes=np.stack(envelopes), midi=np.array(midis, dtype=np.int16),
             file_index=np.array(file_index, dtype=np.int16))
    args.out.with_suffix(".json").write_text(json.dumps({"files": files, "meta": {f: meta[f] for f in files}},
                                                        indent=1), encoding="utf-8")
    print(f"{len(files)} files, {len(midis)} notes, {len({meta[f]['family'] for f in files})} families -> {args.out}")


if __name__ == "__main__":
    main()
