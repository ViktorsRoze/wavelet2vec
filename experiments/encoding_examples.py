"""Encoding examples for the training dashboard: what KeyboardNet v3 encodes, for three notes.

Three unrelated notes from different source resolutions are pushed through the
real v3 encoding, and every stage is written for
experiments/training_dashboard.py:

- the stereo waveform (min/max envelope);
- the lossless wavelet image: mid and side cell levels, 149 semitone rows x 120
  frames, and how many exact complex samples each row stores per 10 ms (the
  multirate layout);
- which rows are scored (below the note's source Nyquist);
- the seven feature planes the backbone reads, computed from the exact cells;
- the exact key change: the same note moved +5 semitones inside the image (the
  carrier the model reshapes);
- measured round trips (float64, float32) per channel;
- the stereo image: overall and per-row correlation and balance of left and
  right, from the exact cells.

Planes are stored as 8-bit values with their range, so the page draws them
without large downloads. The three notes' audio is also saved
(encoding_examples_audio.npy), so the dashboard can draw sharp, zoomable views
and goniometers on demand at screen resolution.

    PYTHONPATH=src python experiments/encoding_examples.py [--out experiments/output/keyboard/v3]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from wavelet2vec.keyboard import DB_SCALE
from wavelet2vec.keyboard_v3 import (BANDS, CENTERS, PER_FRAME, SPEC, cell_features, row_nyquist_mask, shift_image)
from wavelet2vec.wavelet_image import decode, describe, encode, magnitude_view

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "keyboard_v3_data"
EXAMPLES = (  # unrelated sounds, one per source resolution
    ("from_C1_till_B10_Serum_pluck_16_voices_expander_fx.wav", 72, "Synth pluck, 16-voice unison with expander FX"),
    ("from_C_real_Vocal_recording_melodyne.wav", 57, "Own voice, sung “Piuw”"),
    ("from_C1_till_B10_kontakt_piano_gigant.wav", 45, "Acoustic piano (Kontakt library)"),
)
NAMES = "C C# D D# E F F# G G# A A# B".split()
SHIFT = 5


def pack(plane, lo, hi, kind, unit, label, describe_text):
    q = np.clip((np.asarray(plane, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    data = np.round(np.nan_to_num(q) * 254).astype(np.uint8)   # 255 = "no value" (quiet cell)
    data[~np.isfinite(np.asarray(plane, dtype=np.float64))] = 255
    return {"data": base64.b64encode(data.tobytes()).decode("ascii"), "min": lo, "max": hi, "kind": kind,
            "unit": unit, "label": label, "about": describe_text}


def stereo_stats(image):
    """Left/right power, balance and phase correlation per row (whole note) and overall, from the exact cells."""
    spec = image.spec
    core = slice(spec.pad_frames, spec.pad_frames + spec.n_frames)
    pl, pr, cross = (np.zeros(spec.n_bands) for _ in range(3))
    for c, values in image.groups.items():
        v = values[0].numpy()[:, :, core]                   # [2, rows, frames, c]
        left, right = v[0] + v[1], v[0] - v[1]
        rows = list(map(int, image.rows[c]))
        pl[rows] = (np.abs(left) ** 2).sum((-1, -2)) / c
        pr[rows] = (np.abs(right) ** 2).sum((-1, -2)) / c
        cross[rows] = np.real(left * np.conj(right)).sum((-1, -2)) / c
    both = pl + pr
    level = 10 * np.log10(both + 1e-30)
    level -= level.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.where(both > 0, (pr - pl) / both, 0.0)
        corr = np.where((pl > 0) & (pr > 0), np.clip(cross / np.sqrt(pl * pr), -1, 1), 0.0)
    total = {"balance": round(float((pr.sum() - pl.sum()) / max(both.sum(), 1e-30)), 4),
             "correlation": round(float(np.clip(cross.sum() / np.sqrt(max(pl.sum() * pr.sum(), 1e-60)), -1, 1)), 4)}
    return {**total, "per_row": {"level_db": [round(float(x), 1) for x in level],
                                 "balance": [round(float(x), 4) for x in balance],
                                 "correlation": [round(float(x), 4) for x in corr]}}


def snr(a, b):
    err = float(((a - b) ** 2).sum())
    sig = float((a ** 2).sum())
    if sig == 0.0:
        return None if err == 0.0 else float("-inf")
    return round(10 * np.log10(sig / max(err, 1e-300)), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "output" / "keyboard" / "v3")
    args = parser.parse_args()
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))["notes"]
    examples, audio_all = [], []
    for file, midi, label in EXAMPLES:
        note = next(n for n in index if n["file"] == file and n["midi"] == midi)
        audio64 = torch.from_numpy(np.asarray(np.load(DATA / "notes" / note["array"], mmap_mode="r")[note["row"]],
                                              dtype=np.float64))[None]
        audio_all.append(audio64[0].numpy().astype(np.float32))
        image64 = encode(audio64, SPEC)
        back64 = decode(image64)
        image32 = encode(audio64.float(), SPEC)
        back32 = decode(image32).double()
        side = lambda x: (x[:, 0] - x[:, 1]) / 2
        trips = {"float64": {"left": snr(audio64[:, 0], back64[:, 0]), "right": snr(audio64[:, 1], back64[:, 1]),
                             "side": snr(side(audio64), side(back64))},
                 "float32": {"left": snr(audio64[:, 0], back32[:, 0]), "right": snr(audio64[:, 1], back32[:, 1]),
                             "side": snr(side(audio64), side(back32))}}
        levels = magnitude_view(image64)[0].numpy()                 # [2, rows, frames] dB
        peak = float(levels[0].max())
        quiet = levels[0] < peak - 90.0
        feats = cell_features(image64)[0].numpy()                   # v2-normalized planes
        def masked(p):
            p = p.astype(np.float64).copy()
            p[quiet] = np.nan
            return p
        carrier = magnitude_view(shift_image(image64, [SHIFT]))[0, 0].numpy()
        nyquist = row_nyquist_mask([note["source_rate"]])[0].numpy()
        x = audio64[0].numpy()
        bins = np.array_split(np.arange(x.shape[1]), 600)
        wave = {ch: {"min": [round(float(x[i][b].min()), 4) for b in bins], "max": [round(float(x[i][b].max()), 4) for b in bins]}
                for i, ch in enumerate(("left", "right"))}
        ms_side = float(((x[0] - x[1]) ** 2).sum()) / max(float(((x[0] + x[1]) ** 2).sum()), 1e-30)
        planes = {
            "level_mid": pack(levels[0] - peak, -100, 0, "seq", "dB", "Wavelet image, mid: cell level",
                              "Mean power of the exact complex samples in each cell (mid = (L+R)/2), relative to the note's peak."),
            "level_side": pack(levels[1] - peak, -100, 0, "seq", "dB", "Wavelet image, side: cell level",
                               "Side = (L−R)/2, same reference. Empty for mono sources: stereo is stored exactly, not estimated."),
            "carrier": pack(carrier - peak, -100, 0, "seq", "dB", f"Key change: moved +{SHIFT} semitones (carrier)",
                            "Each row moved up by the interval with the exact heterodyne phase ramp and resampled to its new cell size. "
                            "This is what the model reshapes into the target note."),
            "if_deviation": pack(masked(feats[1]), -0.06, 0.06, "div", "relative", "Instantaneous-frequency deviation",
                                 "Where the partial sits inside its semitone band: tanh((IF − centre) / centre); ±0.059 = ±1 semitone."),
            "phase_coherence": pack(masked(feats[2]), 0, 1, "seq", "", "Phase coherence",
                                    "Stability of the demodulated phase within the cell: 1 = steady partial, low = noise. "
                                    "Rows with one sample per cell read 1 by construction."),
            "baseband_re": pack(masked(feats[3]), -1, 1, "div", "", "Demodulated baseband, real part",
                                "Cell mean of the band signal with its carrier removed, divided by its RMS: phase relative to the band carrier."),
            "baseband_im": pack(masked(feats[4]), -1, 1, "div", "", "Demodulated baseband, imaginary part",
                                "Together with the real part: the phase angle of each cell, exactly invariant under the key change."),
            "modulation_depth": pack(masked(feats[5]), 0, 1.5, "seq", "", "Modulation depth",
                                     "Amplitude wobble inside each 10 ms cell (std / mean of |samples|): beating, unison, tremolo. "
                                     "0 where a cell holds a single sample."),
            "stereo_width": pack(masked(feats[6] * DB_SCALE), -60, 10, "seq", "dB", "Stereo width",
                                 "Side energy relative to mid energy per cell; −60 dB = mono."),
        }
        examples.append({
            "file": file, "label": label, "family": note["family"], "midi": midi,
            "note": f"{NAMES[midi % 12]}{midi // 12 - 1}", "fundamental_hz": round(440 * 2 ** ((midi - 69) / 12), 2),
            "source_rate": note["source_rate"], "stored_rate": SPEC.sample_rate,
            "side_to_mid_db": round(10 * np.log10(ms_side + 1e-30), 1), "round_trip_db": trips,
            "nyquist_rows": [int(v) for v in nyquist], "waveform": wave, "planes": planes,
            "stereo": stereo_stats(image64),
            "peak_db": round(peak, 1)})
    payload = {"encoding": "lossless stereo wavelet image (wavelet2vec.wavelet_image)", "spec": describe(SPEC),
               "rows": SPEC.n_bands, "frames": SPEC.n_frames, "hop_ms": SPEC.hop_ms,
               "row_midi": [b.midi for b in BANDS], "row_hz": [round(float(c), 2) for c in CENTERS],
               "row_samples_per_cell": PER_FRAME, "shift_semitones": SHIFT, "examples": examples,
               "model_inputs": "Backbone input: 28 planes on a 160 x 128 grid = for up to 3 given notes, the 7 feature planes "
                               "shifted to the target key + a validity mask each; plus the distance-weighted blend, row and "
                               "frame coordinates and the nearest-minus-second difference. The lossless path additionally "
                               "embeds the carrier's exact cells per octave group."}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "encoding_examples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    np.save(args.out / "encoding_examples_audio.npy", np.stack(audio_all))   # [3, 2 (L, R), samples] at 192 kHz
    print(f"{path} ({path.stat().st_size / 1e6:.2f} MB)")
    for e in examples:
        print(f"  {e['label']}: {e['source_rate']} Hz, {e['note']}, float32 round trip {e['round_trip_db']['float32']}, "
              f"stereo correlation {e['stereo']['correlation']}, balance {e['stereo']['balance']}")


if __name__ == "__main__":
    main()
