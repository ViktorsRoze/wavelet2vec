"""Annotate a free (sung or played) take: which notes, where, how in tune.

For recordings that are not laid out on the C1..B10 slot grid, such as a
voice singing along with the FL Studio score. pYIN tracks f0 on a 24 kHz
copy (analysis only). Held notes are runs of one rounded MIDI note (merging
gaps < 50 ms) that last >= 0.2 s and are louder than --min-level-db. A new
"pass" starts when the melody drops by 5 or more semitones (the singer
restarting lower).

Where a pass is chromatic, its onsets fix the score's beat grid (one note per
1.2 s beat, MIDI 12 + beat). Each note then gets the score note that was
playing when it started. Writes <stem>.notes.csv next to the audio; with
--embed-markers, also adds named regions (cue + LIST/adtl chunks) to the WAV
without touching its audio samples.

    python experiments/annotate_sung_notes.py note_dataset/from_C_real_Vocal_recording_melodyne.wav --embed-markers
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from wavelet2vec.tools import detect_vocal_parts

NAMES = "C C# D D# E F F# G G# A A# B".split()
BEAT_S = 1.2  # 50 BPM


def sci_name(midi):
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def fl_name(midi):
    return f"{NAMES[midi % 12]}{midi // 12}"


def track(mono, rate, hop_s=0.01):
    import librosa

    down = rate // 24000 if rate % 24000 == 0 else None
    y, sr = (resample_poly(mono, 1, down), rate // down) if down and down > 1 else (mono, rate)
    hop = int(sr * hop_s)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=1100, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][:len(f0)]
    return librosa.hz_to_midi(f0), voiced, 20 * np.log10(rms + 1e-12), hop_s


def segment(midi, voiced, level_db, hop_s, min_level_db, min_dur_s=0.2, max_gap_s=0.05):
    notes, current = [], None
    for i, (m, v, db) in enumerate(zip(midi, voiced, level_db)):
        key = int(np.round(m)) if v and np.isfinite(m) and db > min_level_db else None
        if key is not None and current and key == current["midi"] and (i - current["last"]) * hop_s <= max_gap_s:
            current["last"] = i
            current["frames"].append(i)
            continue
        if current:
            notes.append(current)
        current = {"midi": key, "first": i, "last": i, "frames": [i]} if key is not None else None
    if current:
        notes.append(current)
    notes = [n for n in notes if (n["last"] - n["first"] + 1) * hop_s >= min_dur_s]
    for n in notes:
        frames = np.array(n["frames"])
        n.update(start_s=n["first"] * hop_s, end_s=(n["last"] + 1) * hop_s,
                 cents=float(np.median(midi[frames] - n["midi"]) * 100),
                 pitch_sd_cents=float(np.std(midi[frames]) * 100), level_db=float(np.median(level_db[frames])))
    return notes


def assign_passes(notes, drop=5):
    run = 1
    for i, n in enumerate(notes):
        if i and n["midi"] <= notes[i - 1]["midi"] - drop:
            run += 1
        n["pass"] = run


def fit_score_grid(notes):
    """Beat-0 time of the score, from the longest chromatic stretch (sung = score note)."""
    best, current = [], [notes[0]]
    for a, b in zip(notes, notes[1:]):
        current = current + [b] if b["midi"] == a["midi"] + 1 else [b]
        if len(current) > len(best):
            best = current
    beats = np.array([n["midi"] - 12 for n in best], dtype=float)
    onsets = np.array([n["start_s"] for n in best])
    origin = float(np.median(onsets - beats * BEAT_S))
    return origin, best


def embed_regions(path, regions):
    """Append cue + LIST/adtl (labl + ltxt 'rgn ') chunks; audio bytes stay identical."""
    data = bytearray(Path(path).read_bytes())
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    while pos + 8 <= len(data):  # refuse to stack a second set of markers
        cid, size = struct.unpack("<4sI", data[pos:pos + 8])
        if cid == b"cue " or (cid == b"LIST" and data[pos + 8:pos + 12] == b"adtl"):
            raise ValueError("file already has cue/adtl markers")
        pos += 8 + size + (size & 1)
    cue = struct.pack("<I", len(regions))
    adtl = b"adtl"
    for i, (start, length, label) in enumerate(regions, 1):
        cue += struct.pack("<II4sIII", i, start, b"data", 0, 0, start)
        text = label.encode("ascii") + b"\0"
        labl = struct.pack("<I", i) + text
        adtl += b"labl" + struct.pack("<I", len(labl)) + labl + (b"\0" if len(labl) & 1 else b"")
        ltxt = struct.pack("<II4sHHHH", i, length, b"rgn ", 0, 0, 0, 0)
        adtl += b"ltxt" + struct.pack("<I", len(ltxt)) + ltxt
    chunks = b"cue " + struct.pack("<I", len(cue)) + cue + b"LIST" + struct.pack("<I", len(adtl)) + adtl
    if len(data) & 1:
        data += b"\0"
    data += chunks
    data[4:8] = struct.pack("<I", len(data) - 8)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_bytes(bytes(data))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--min-level-db", type=float, default=-45.0)
    parser.add_argument("--embed-markers", action="store_true")
    parser.add_argument("--syllable", default="", help='what was sung on every note, e.g. "Piuw"')
    parser.add_argument("--vowel-scale", type=float, default=None,
                        help="vocal-tract scale for the vowel templates (default: estimated from the take)")
    args = parser.parse_args()
    audio, rate = sf.read(args.wav, dtype="float64", always_2d=True)
    digest = hashlib.sha256(audio.astype(np.float32).tobytes()).hexdigest()
    midi, voiced, level_db, hop_s = track(audio.mean(1), rate)
    notes = segment(midi, voiced, level_db, hop_s, args.min_level_db)
    assign_passes(notes)
    origin, chromatic = fit_score_grid(notes)
    parts = detect_vocal_parts(audio.T, rate, scale=args.vowel_scale)
    rows = []
    for i, n in enumerate(notes, 1):
        score_midi = 12 + int(np.floor((n["start_s"] - origin) / BEAT_S + 0.5))
        # 120 ms before the pitch starts holds the consonant onset (e.g. the "P" of "Piuw").
        heard = parts.summarize(n["start_s"] - 0.12, n["end_s"] + 0.1)
        rows.append(dict(
            note_index=i, sung_pass=n["pass"],
            start_s=round(n["start_s"], 3), end_s=round(n["end_s"], 3), dur_s=round(n["end_s"] - n["start_s"], 3),
            start_sample=int(round(n["start_s"] * rate)), end_sample=int(round(n["end_s"] * rate)),
            midi=n["midi"], note_sci=sci_name(n["midi"]), note_fl=fl_name(n["midi"]), syllable=args.syllable,
            cents=round(n["cents"], 1), pitch_sd_cents=round(n["pitch_sd_cents"], 1), level_db=round(n["level_db"], 1),
            score_midi=score_midi, score_note_fl=fl_name(score_midi), interval_to_score=n["midi"] - score_midi,
            in_chromatic_stretch=n in chromatic,
            vowel_detected=heard["vowel"], vowel_share=heard["share"], vowel_start=heard["start_vowel"],
            vowel_end=heard["end_vowel"], sibilant_frames=heard["sibilant_frames"],
            plosive_frames=heard["parts"].get("plosive", 0)))
    out = args.wav.with_name(args.wav.stem + ".notes.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} notes, {len(chromatic)} in the chromatic stretch "
          f"({chromatic[0]['midi']}..{chromatic[-1]['midi']}), score beat 0 at {origin:.3f} s, "
          f"vowel scale {parts.scale:.2f} -> {out.name}")
    if args.embed_markers:
        embed_regions(args.wav, [(r["start_sample"], r["end_sample"] - r["start_sample"],
                                  f"{r['note_sci']} (FL {r['note_fl']}) pass {r['sung_pass']}") for r in rows])
        check, _ = sf.read(args.wav, dtype="float64", always_2d=True)
        if hashlib.sha256(check.astype(np.float32).tobytes()).hexdigest() != digest:
            raise RuntimeError("audio changed while embedding markers")
        print(f"embedded {len(rows)} named regions; audio samples verified unchanged")


if __name__ == "__main__":
    main()
