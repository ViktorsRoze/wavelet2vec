"""Audit one annotated drum performance; preserve originals and report uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from wavelet2vec.drum_analysis import analyze_drum_track
from wavelet2vec.feature_cache import source_fingerprint, implementation_fingerprint


def snr(target, estimate):
    signal = float(np.sum(target*target))
    error = float(np.sum((target-estimate)**2))
    return 10*np.log10(signal/max(error, signal*1e-30)) if signal else None


def audit(root, output, bpm=128., bars=4):
    files = sorted(root.glob("*.wav"))
    if not files:
        raise ValueError("no WAV files")
    infos = {path.name: sf.info(path) for path in files}
    rates = {info.samplerate for info in infos.values()}
    if len(rates) != 1 or any(info.channels != 2 for info in infos.values()):
        raise ValueError("this experiment requires a common native rate and stereo tracks")
    rate = rates.pop()
    body_samples = round(bars*4*60/bpm*rate)
    if any(info.frames < body_samples for info in infos.values()):
        raise ValueError("recording shorter than declared musical body")
    solos = {path.stem[5:]: sf.read(path, dtype="float64", always_2d=True)[0][:body_samples].T
             for path in files if path.stem.startswith("only_")}
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "One shared performance; native solo-reference diagnostics, no independent test set.",
              "bpm": bpm, "bars": bars, "beats_per_bar": 4, "body_samples": body_samples, "sample_rate": rate,
              "master_processing": "User reports master limiter; exact processing settings unknown.",
              "implementation": implementation_fingerprint(), "files": [], "solos": {}, "mixtures": []}
    for path in files:
        info = infos[path.name]
        report["files"].append({"file": path.name, "sha256": source_fingerprint(path), "sample_rate": info.samplerate,
            "channels": info.channels, "frames": info.frames, "subtype": info.subtype,
            "tail_samples_beyond_four_bars": info.frames-body_samples})
    for name, audio in solos.items():
        print("Analyzing", name, flush=True)
        detail = analyze_drum_track(audio, rate, bpm=bpm)
        report["solos"][name] = detail
        (output/f"events_{name}.json").write_text(json.dumps(detail, indent=2, allow_nan=False), encoding="utf-8")
    for path in files:
        if path.stem.startswith("only_"):
            continue
        names = path.stem.split("+")
        if set(names)-set(solos):
            raise ValueError(f"unknown source labels in {path.name}")
        mix = sf.read(path, dtype="float64", always_2d=True)[0][:body_samples].T
        summed = sum(solos[name] for name in names)
        correlation = sum(correlate(a, b, mode="full", method="fft") for a, b in zip(mix, summed))
        lags = correlation_lags(body_samples, body_samples)
        valid = np.abs(lags) <= round(.02*rate)
        best_lag = int(lags[valid][np.argmax(correlation[valid])])
        gain = float(np.sum(mix*summed)/max(np.sum(summed*summed), 1e-30))
        error = mix-summed
        report["mixtures"].append({"file": path.name, "sources": names,
            "sum_reference_snr_db": snr(mix, summed), "peak_sum_error": float(np.max(np.abs(error))),
            "sum_reference_gain": gain, "best_global_lag_samples": best_lag,
            "gain_adjusted_snr_db": snr(mix, gain*summed),
            "additive_at_library_tolerance": bool(np.allclose(mix, summed, rtol=1e-5, atol=1e-6))})
        previous_path = root/("+".join(names[:-1])+".wav") if len(names) > 2 else root/("only_"+names[0]+".wav")
        if previous_path.exists():
            previous = sf.read(previous_path, dtype="float64", always_2d=True)[0][:body_samples].T
            report["mixtures"][-1]["cumulative_difference_source"] = names[-1]
            report["mixtures"][-1]["cumulative_difference_snr_db"] = snr(solos[names[-1]], mix-previous)
            report["mixtures"][-1]["cumulative_difference_scope"] = "Uses another mixture render; not separation from one file."
    report_path = output/"audit.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary = []
    for name, detail in report["solos"].items():
        events = detail["events"]
        pitches = [row["reference_comparison"]["pitch_shift_semitones"] for row in events
                   if row["reference_comparison"].get("pitch_shift_semitones") is not None]
        fitted_levels = [row["reference_comparison"]["fitted_level_change_db"] for row in events
                         if row["reference_comparison"].get("fitted_level_change_db") is not None]
        summary.append({"source": name, "events": len(events), "repeat_groups": detail["repeat_groups"],
            "level_span_db": max(r["level_dbfs"] for r in events)-min(r["level_dbfs"] for r in events),
            "balance_range": [min(r["stereo_energy_balance"] for r in events), max(r["stereo_energy_balance"] for r in events)],
            "accepted_pitch_fits": len(pitches), "relative_pitch_range": [min(pitches), max(pitches)] if pitches else None})
        summary[-1]["matched_gain_span_db"] = max(fitted_levels)-min(fitted_levels) if len(fitted_levels) > 1 else None
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output/"events.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["source", "event_index", "time_seconds", "bar_number", "beat_position", "relative_level_db",
                  "stereo_energy_balance", "right_over_left_db", "pitch_shift_semitones", "waveform_fit",
                  "fitted_level_change_db", "comparison_window_exact_pcm_match", "candidate_repeat_group"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, detail in report["solos"].items():
            for index, event in enumerate(detail["events"]):
                values = {**event, **event["reference_comparison"], "source": name, "event_index": index}
                writer.writerow({key: values.get(key) for key in fields})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(solos), 3, figsize=(15, 2.25*len(solos)), constrained_layout=True)
    for row, (name, audio) in enumerate(solos.items()):
        events = report["solos"][name]["events"]
        stride = 100
        t = np.arange(0, audio.shape[-1], stride)/rate
        axes[row, 0].plot(t, audio[0, ::stride], linewidth=.5)
        for event in events:
            axes[row, 0].axvline(event["time_seconds"], color="tab:red", alpha=.4, linewidth=.6)
        axes[row, 0].set_ylabel(name)
        times = [e["time_seconds"] for e in events]
        axes[row, 1].scatter(times, [e["relative_level_db"] for e in events], s=10)
        axes[row, 2].scatter(times, [e["stereo_energy_balance"] for e in events], s=10)
        for axis in axes[row]:
            axis.set_xlim(0, bars*4*60/bpm)
            axis.grid(alpha=.2)
        axes[row, 2].set_ylim(-1.05, 1.05)
    for axis, title in zip(axes[0], ("Native waveform / estimated events", "Level relative to loudest event (dB)", "Stereo energy balance (left -1, right +1)")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds")
    fig.savefig(output/"events_overview.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (name, detail) in zip(axes.flat, report["solos"].items()):
        accepted = [event for event in detail["events"] if event["reference_comparison"].get("pitch_shift_semitones") is not None]
        axis.scatter([e["time_seconds"] for e in accepted], [e["reference_comparison"]["pitch_shift_semitones"] for e in accepted], s=16)
        axis.set_title(f"{name}: {len(accepted)}/{len(detail['events'])} strong reference fits")
        axis.set_xlim(0, bars*4*60/bpm)
        axis.set_xlabel("Seconds")
        axis.set_ylabel("Relative playback-rate semitones")
        values = [e["reference_comparison"]["pitch_shift_semitones"] for e in accepted]
        if values and max(abs(value) for value in values) < .1:
            axis.set_ylim(-.1, .1)
        axis.grid(alpha=.2)
    fig.savefig(output/"relative_pitch.png", dpi=130)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("drum_substem_dataset_BPM128_4bars"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/drum_dataset_2026-09-22"))
    args = parser.parse_args()
    audit(args.dataset, args.output)
