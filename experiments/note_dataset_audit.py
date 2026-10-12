"""Audit note_dataset: one note per beat at 50 BPM, FL C1..B10 (MIDI 12..131).

Reads every WAV once and measures, per beat slot: level, onset, tail at the
slot end, stereo side level, band energy, a harmonic-comb pitch estimate and
the log-frequency shift to the next note (+100 cents expected). Also decodes
FL Studio cue/tlst metadata. Writes a per-file CSV next to the audio and a
per-note CSV under experiments/output. Originals are only read.

    python experiments/note_dataset_audit.py [--root note_dataset] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

BPM, N_NOTES, FIRST_MIDI = 50, 120, 12  # FL Studio names MIDI 12 "C1" (MIDI 60 = C5)
SILENT = 1e-5  # slot peak below -100 dBFS
NAMES = "C C# D D# E F F# G G# A A# B".split()
SYNTHS = [("Harmor", "Harmor"), ("Serum", "Serum"), ("Sakura", "Sakura"), ("T_synth", "T_synth"),
          ("avangers", "Avengers"), ("iZotope_Iris2", "iZotope Iris 2"), ("kontakt", "Kontakt"),
          ("massiv_x", "Massive X"), ("repro-1", "Repro-1"), ("synthplant", "Synplant"), ("zeta+", "Zeta+"),
          ("Korg_monolog", "Korg monologue")]


def fl_name(midi):
    return f"{NAMES[midi % 12]}{midi // 12}"


def midi_hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def stem(path):
    return path.stem.removeprefix("from_C1_till_B10_")


def synth_of(name):
    return next((label for prefix, label in SYNTHS if name.startswith(prefix)), "unknown")


def family_of(name):
    """Files sharing one base patch; keep a family on one side of any split."""
    if name.startswith("Harmor") and "square" in name:
        return "Harmor simple square pluck"
    if name.startswith("Harmor") and "simple_saw" in name:
        return "Harmor simple saw pluck"
    if name.startswith(("Serum_saw", "Serum_pluck")):
        return "Serum saw/pluck patch"
    if "shalanaja" in name:
        return "Shevannai (Kontakt)"
    if name.startswith("kontakt_vox"):
        return "Exhale (Kontakt)"
    if name.startswith("Korg_monolog_analog_synth_preset_1_bass"):  # two takes of one patch
        return "Korg monologue preset 1 bass"
    if name.startswith("T_synth"):
        return "T_synth preset 1"
    return name


def category_of(name):
    """Category stated by the filename; unlabelled presets are pluck/bass/bell/lead per the author."""
    for token, category in (("pluck", "pluck"), ("bass", "bass"), ("vox", "vox"), ("shalanaja", "vox"),
                            ("rhodes", "keys"), ("piano", "keys"), ("organ", "organ"), ("string", "string"),
                            ("sequance", "sequence"), ("sequence", "sequence")):
        if token in name.lower():
            return category
    return ""


def source_type_of(name):
    """vox/synth split for the hybrid-morph purpose.

    vox: clean human voice ("shalanaja" = Best Service Shevannai, soprano,
    no vibrato, no effects). processed_vox: Output Exhale, vocals heavily
    processed with LFO and distortion. synth_vox: Harmor voice imitations.
    processed_vox and synth_vox are already part-way hybrids, so they are
    reference points rather than classifier training classes. synth:
    every non-vocal sound (including the sampled piano).
    """
    if "shalanaja" in name:
        return "vox"
    if name.startswith("kontakt_vox"):
        return "processed_vox"
    if "_vox" in name:
        return "synth_vox"
    return "synth"


def fl_markers(path):
    """FL Studio cue offsets (samples) and the MIDI note stored per cue in 'tlst'."""
    cues, notes = {}, []
    with open(path, "rb") as f:
        end, pos = path.stat().st_size, 12
        while pos + 8 <= end:
            f.seek(pos)
            cid, size = struct.unpack("<4sI", f.read(8))
            if cid == b"cue ":
                body = f.read(size)
                for i in range(struct.unpack("<I", body[:4])[0]):
                    ident, *_, offset = struct.unpack("<II4sIII", body[4 + 24 * i:28 + 24 * i])
                    cues[ident] = offset
            elif cid == b"tlst":
                body = f.read(size)
                notes = [body[4 + 24 * i + 13] for i in range(struct.unpack("<I", body[:4])[0])]
            pos += 8 + size + (size & 1)
    return [cues[k] for k in sorted(cues)], notes


def fl_score_notes(path):
    """Notes of an FL Studio score (.fsc): FLhd/FLdt events, 24-byte records in event 224."""
    data = Path(path).read_bytes()
    if data[:4] != b"FLhd":
        raise ValueError("not an FL Studio score/project file")
    ppq = struct.unpack("<H", data[12:14])[0]
    pos, notes = 22, []
    while pos < len(data):
        event = data[pos]
        pos += 1
        if event < 192:  # fixed-size events: 1, 2 or 4 bytes
            pos += (1, 2, 4)[event // 64]
            continue
        size = shift = 0
        while True:
            byte = data[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        if event == 224:
            for i in range(0, size, 24):
                start, _, _, length, key, _, fine, _, release, _, pan, velocity, _, _ = struct.unpack(
                    "<IHHIHHBBBBBBBB", data[pos + i:pos + i + 24])
                notes.append(dict(beat=start / ppq, beats=length / ppq, midi=key, velocity=velocity,
                                  release=release, pan=pan, fine_pitch=fine))
        pos += size
    return ppq, notes


def _peaks(freqs, mag, fs):
    idx = np.clip(np.searchsorted(freqs, fs), 1, len(freqs) - 2)
    return np.maximum(np.maximum(mag[idx - 1], mag[idx]), mag[idx + 1])


def _comb(freqs, mag, f0, fmax):
    # Harmonics minus half-way points: penalizes sub- and super-octave candidates.
    h = np.arange(1, int(fmax // f0) + 1)
    if len(h) == 0 or f0 < 12:
        return -np.inf
    return float(_peaks(freqs, mag, h * f0).mean() - _peaks(freqs, mag, (h - 0.5) * f0).mean())


def _logspec(power, freqs, rate):
    top = min(0.95 * rate / 2, 20000.0)
    grid = 20.0 * 2 ** (np.arange(0, 1200 * np.log2(top / 20.0), 5) / 1200)  # 5-cent bins
    db = 10 * np.log10(np.interp(grid, freqs, power) + 1e-30)
    db = np.maximum(db, db.max() - 80)
    return db - db.mean()


def _shift_cents(a, b, max_lag=80):
    n = min(len(a), len(b))
    best, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(0, -lag):n - max(0, lag)]
        y = b[max(0, lag):n - max(0, -lag)]
        x, y = x - x.mean(), y - y.mean()
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
        if c > best_c:
            best, best_c = lag, c
    return best * 5, best_c


def _fft_lag(a, b, max_lag=260):
    n = len(a)
    c = np.fft.irfft(np.fft.rfft(a, 2 * n).conj() * np.fft.rfft(b, 2 * n), 2 * n)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    i = int(np.argmax(c))
    return (i - max_lag) * 5, float(c[i])


def _within_slot_jump(seg, onset, rate):
    """Largest spectral shift (cents) of any later 150 ms frame against the first.

    Arpeggios, octave layers decaying at different rates and switching
    detuned oscillators all show up here; vibrato and tremolo stay < 150.
    """
    flen, hop = int(0.15 * rate), int(0.1 * rate)
    nfft = 1 << int(np.ceil(np.log2(flen * 2)))
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    frames = []
    for start in range(onset, len(seg) - flen, hop):
        w = seg[start:start + flen]
        power = np.abs(np.fft.rfft(w * np.hanning(flen), nfft)) ** 2
        frames.append((_logspec(power[freqs >= 40], freqs[freqs >= 40], rate), power.sum()))
    if len(frames) < 2:
        return 0
    loudest = max(e for _, e in frames)
    jumps = [lag for lag, corr in (_fft_lag(frames[0][0], spec) for spec, e in frames[1:] if e > loudest * 1e-3)
             if corr > 0.4]
    return max(jumps, key=abs, default=0)


def software_of(path):
    """LIST/INFO ISFT tag: 'FL Studio ...' for renders, e.g. 'SOUND FORGE ...' for recordings."""
    with open(path, "rb") as f:
        end, pos = Path(path).stat().st_size, 12
        while pos + 8 <= end:
            f.seek(pos)
            cid, size = struct.unpack("<4sI", f.read(8))
            if cid == b"LIST":
                body = f.read(size)
                i = body.find(b"ISFT")
                if body[:4] == b"INFO" and i >= 0:
                    n = struct.unpack("<I", body[i + 4:i + 8])[0]
                    return body[i + 8:i + 8 + n].split(b"\0")[0].decode("latin1")
            pos += 8 + size + (size & 1)
    return ""


def fit_grid(mono, rate):
    """Beat-0 position and beat length (samples) of a recorded, hand-trimmed take.

    Detects one onset per beat (first 1 ms block within 30 dB of the beat's
    peak), assigns beat index round(t / 1.2 s), keeps indices 0..119 and fits
    a line. Valid while the take starts within half a beat of beat 0 and drift
    stays far below half a beat. Returns origin, beat, onset residuals (ms).
    """
    blk = int(rate * 0.001)
    env = np.abs(mono[:len(mono) // blk * blk]).reshape(-1, blk).max(1)
    db = 20 * np.log10(env + 1e-12)
    period = rate * 60 / BPM / blk
    rise = np.maximum(db[5:] - db[:-5], 0)
    fold = np.bincount((np.arange(len(rise)) % period).astype(int), weights=rise, minlength=int(period) + 1)
    phase = int(np.argmax(np.convolve(fold, np.ones(5), "same")))
    times = {}
    for k in range(-1, N_NOTES + 8):
        a = int(round(phase + k * period)) - 60
        if a < 0 or a + 400 > len(db):
            continue
        seg = db[a:a + 400]
        if seg.max() < -80:
            continue
        t = (a + int(np.argmax(seg > seg.max() - 30))) * blk
        index = int(round(t / (rate * 60 / BPM)))
        if 0 <= index < N_NOTES:
            times[index] = t
    ks = np.array(sorted(times))
    ts = np.array([times[k] for k in ks], dtype=float)
    beat, origin = np.polyfit(ks, ts, 1)
    residual_ms = (ts - (origin + beat * ks)) / rate * 1000
    return origin, beat, residual_ms


def audit_file(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    x = data.T.astype(np.float64)
    mono, side = x.mean(0), (x[0] - x[-1]) / 2
    render_beat, nyq = int(round(rate * 60 / BPM)), rate / 2
    software = software_of(path)
    if software.startswith("FL Studio"):  # sample-exact render
        origin, beat, residual_ms = 0.0, float(render_beat), np.zeros(1)
    else:  # recording: fit the grid
        origin, beat, residual_ms = fit_grid(mono, rate)
    nfft = 1 << int(np.ceil(np.log2(rate * 4)))  # ~0.2 Hz bins
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    edges = (0, 20000, 24000, 48000, 96000)
    band = np.zeros(4)
    rows, specs = [], {}
    for k in range(N_NOTES):
        midi = FIRST_MIDI + k
        lo, hi = int(round(origin + k * beat)), min(int(round(origin + (k + 1) * beat)), x.shape[1])
        row = dict(beat=k, midi=midi, note=fl_name(midi), slot_start=lo)
        if lo < 0 or hi - lo < int(beat) - 1:  # slot cut off by the file start or end
            rows.append({**row, "silent": 1, "cut": 1})
            continue
        env = np.abs(x[:, lo:hi]).max(0)
        peak = float(env.max())
        row["peak_db"] = round(20 * np.log10(peak + 1e-12), 2)
        if peak < SILENT:
            rows.append({**row, "silent": 1})
            continue
        seg = mono[lo:hi]
        onset = int(np.argmax(env > peak * 0.01))
        w = seg[onset + int(0.01 * rate):onset + int(0.6 * rate)]
        spec = np.fft.rfft(w * np.hanning(len(w)), nfft)
        power = np.abs(spec) ** 2
        for i in range(4):
            band[i] += power[(freqs >= edges[i]) & (freqs < edges[i + 1])].sum()
        mag = np.sqrt(np.abs(spec))
        mag /= np.median(mag[(freqs > 10) & (freqs < min(nyq * 0.95, 20000))]) + 1e-30
        fmax = min(nyq * 0.95, 30000.0, max(16 * midi_hz(midi), 3000.0))
        offsets = np.arange(-36, 37)
        scores = [_comb(freqs, mag, midi_hz(midi + o), fmax) for o in offsets]
        off = int(offsets[int(np.argmax(scores))])
        cents_grid = np.arange(-50, 51, 2)
        cents = int(cents_grid[int(np.argmax([_comb(freqs, mag, midi_hz(midi + off) * 2 ** (c / 1200), fmax)
                                              for c in cents_grid]))])
        specs[k] = _logspec(power, freqs, rate)
        blk = max(1, int(0.005 * rate))
        coarse = env[:len(env) // blk * blk].reshape(-1, blk).max(1)
        audible = np.nonzero(coarse > peak * 1e-3)[0]
        tail = env[-int(0.05 * rate):].max()
        rows.append({**row, "silent": 0, "onset_ms": round(onset / rate * 1000, 2),
                     "decay60_ms": float((audible[-1] + 1) * 5) if len(audible) else 0.0,
                     "tail_db": round(20 * np.log10(tail / peak + 1e-12), 1),
                     "side_db": round(10 * np.log10((np.mean(side[lo:hi] ** 2) + 1e-24) / (np.mean(seg ** 2) + 1e-24)), 1),
                     "centroid_hz": round(float((freqs * power).sum() / (power.sum() + 1e-30)), 1),
                     "pitch_offset": off, "pitch_cents": cents,
                     "within_slot_jump_cents": _within_slot_jump(seg, onset, rate) if 24 <= midi <= 119 else ""})
    for row in rows:
        k = row["beat"]
        if k in specs and k + 1 in specs:
            row["shift_to_next_cents"], corr = _shift_cents(specs[k], specs[k + 1])
            row["shift_corr"] = round(corr, 3)
    cues, cue_notes = fl_markers(path)
    info = dict(file=path.name, sample_rate=rate, channels=x.shape[0], frames=x.shape[1],
                subtype=sf.info(path).subtype, peak_dbfs=round(20 * np.log10(np.abs(x).max() + 1e-12), 2),
                over_unity=int((np.abs(x) > 1).sum()),
                band_db=[round(10 * np.log10(e / (band.sum() + 1e-30) + 1e-30), 1) for e in band],
                cue_count=len(cues), cues_on_grid=all(c % render_beat == 0 for c in cues),
                cue_note_minus_beat=sorted({n - c // render_beat for c, n in zip(cues, cue_notes)}),
                software=software, grid_origin_ms=round(origin / rate * 1000, 2),
                beat_ms=round(beat / rate * 1000, 4),
                onset_jitter_ms=round(float(np.median(np.abs(residual_ms))), 2),
                onset_jitter_max_ms=round(float(np.abs(residual_ms).max()), 2))
    return info, rows


def summarize(info, rows):
    live = [r for r in rows if not r["silent"]]
    name = stem(Path(info["file"]))
    mid = [r["pitch_offset"] for r in live if 36 <= r["midi"] <= 107]  # FL octaves 3..8
    dominant, count = Counter(mid).most_common(1)[0] if mid else (None, 0)
    shifts = [r["shift_to_next_cents"] for r in live if "shift_to_next_cents" in r]
    by_midi = {r["midi"]: r.get("shift_to_next_cents") for r in live}
    jumps = [r["within_slot_jump_cents"] for r in live if r["within_slot_jump_cents"] != ""]
    moving = [round(j / 100) for j in jumps if abs(j) >= 150]
    tail = float(np.median([r["tail_db"] for r in live])) if live else None
    band = info["band_db"]
    high = info["sample_rate"] > 50000
    return dict(
        file=info["file"], synth=synth_of(name), family=family_of(name), category_from_name=category_of(name),
        source_type=source_type_of(name),
        sample_rate=info["sample_rate"],
        seconds=round(info["frames"] / info["sample_rate"], 3), cue_markers=info["cue_count"],
        live_notes=len(live), live_from=fl_name(live[0]["midi"]) if live else "",
        live_to=fl_name(live[-1]["midi"]) if live else "",
        clamped_above_midi127=all(by_midi.get(m) == 0 for m in (127, 128, 129, 130)),
        semitone_steps_ok=round(float(np.mean([abs(s - 100) <= 10 for s in shifts])), 3) if shifts else None,
        sounding_offset_semitones=dominant, offset_agreement=round(count / len(mid), 2) if mid else None,
        moving_slots=round(len(moving) / len(jumps), 2) if jumps else None,
        main_jump_semitones=Counter(moving).most_common(1)[0][0] if moving else "",
        peak_dbfs=info["peak_dbfs"],
        envelope="" if tail is None else "sustained" if tail > -20 else "rings into next slot" if tail > -60
        else "decays within slot",
        tail_at_slot_end_db=None if tail is None else round(tail, 1),
        stereo_side_db=round(float(np.median([r["side_db"] for r in live])), 1) if live else None,
        hf_20_24k_db=band[1], hf_24_48k_db=band[2] if high else "", hf_48_96k_db=band[3] if high else "",
        origin="render" if info["software"].startswith("FL Studio") else "recording",
        grid_origin_ms=info["grid_origin_ms"], beat_ms=info["beat_ms"],
        onset_jitter_ms=info["onset_jitter_ms"], onset_jitter_max_ms=info["onset_jitter_max_ms"],
        cut_slots=sum(1 for r in rows if r.get("cut")))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=here.parent / "note_dataset")
    parser.add_argument("--notes-out", type=Path, default=here / "output" / "note_dataset_audit")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    # Only takes laid out on the C1..B10 slot grid; free takes (e.g. the sung
    # vocal) are described by experiments/annotate_sung_notes.py instead.
    files = sorted(args.root.glob("from_C1_till_B10_*.wav"))
    if not files:
        raise SystemExit(f"no WAV files in {args.root}")
    for score in sorted(args.root.glob("*.fsc")):  # the piano roll used for every render
        ppq, notes = fl_score_notes(score)
        on_grid = all(n["beat"] == n["midi"] - FIRST_MIDI for n in notes)
        print(f"{score.name}: {len(notes)} notes, PPQ {ppq}, MIDI {notes[0]['midi']}..{notes[-1]['midi']}, "
              f"one per beat: {on_grid}, lengths {sorted({n['beats'] for n in notes})} beats, "
              f"velocities {sorted({n['velocity'] for n in notes})}")
    with Pool(args.workers) as pool:
        results = pool.map(audit_file, files, chunksize=1)
    summaries = [summarize(info, rows) for info, rows in results]
    with open(args.root / "dataset_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    args.notes_out.mkdir(parents=True, exist_ok=True)
    keys = ["file", "beat", "midi", "note", "slot_start", "cut", "silent", "peak_db", "onset_ms", "decay60_ms", "tail_db", "side_db",
            "centroid_hz", "pitch_offset", "pitch_cents", "within_slot_jump_cents", "shift_to_next_cents", "shift_corr"]
    with open(args.notes_out / "notes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for info, rows in results:
            writer.writerows({"file": info["file"], **r} for r in rows)
    (args.notes_out / "files.json").write_text(json.dumps([info for info, _ in results], indent=1), encoding="utf-8")
    print(f"{len(files)} files, {sum(s['live_notes'] for s in summaries)} live notes -> {args.root / 'dataset_audit.csv'}")


if __name__ == "__main__":
    main()
