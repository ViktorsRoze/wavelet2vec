"""Build a local listening/report page from the completed real drum experiments."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

from wavelet2vec.drum_analysis import detect_percussive_events
from wavelet2vec.feature_cache import source_fingerprint


def event_agreement(reference, detected, tolerance=.03):
    reference, detected = sorted(reference), sorted(detected)
    i = j = matches = 0
    while i < len(reference) and j < len(detected):
        if abs(reference[i]-detected[j]) <= tolerance:
            matches += 1; i += 1; j += 1
        elif detected[j] < reference[i]:
            j += 1
        else:
            i += 1
    precision = matches/len(detected) if detected else 0.
    recall = matches/len(reference) if reference else 0.
    return {"reference_events": len(reference), "estimated_events": len(detected), "matched": matches,
            "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}


def build(root, dataset):
    audit = json.loads((root/"audit.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    labels = {"separation": "32 bands, Q=2", "separation_balanced": "32 bands, source balanced", "separation_q8": "96 bands, Q=8"}
    runs = {key: json.loads((root/key/"results.json").read_text()) for key in labels}
    first = runs["separation"]
    for row in audit["files"]:
        assert source_fingerprint(dataset/row["file"]) == row["sha256"] == first["original_hashes"][row["file"]]
    order = first["source_order"]
    # Verify that the three training variants received identical decoded audio.
    def identities(folder):
        data = json.loads((root/folder/"training_manifest.json").read_text())
        return [(r["track_id"], r["mixture"]["pcm_hash"], {k: None if v is None else v["pcm_hash"] for k, v in r["sources"].items()}) for r in data["records"]]
    assert all(identities(key) == identities("separation") for key in runs)
    diagnostics = {"scope": "Agreement with automatically detected solo events, not MIDI/manual ground truth; fourth-bar interior only.",
                   "tolerance_seconds": .03, "excluded_edge_seconds": .05, "sources": {}}
    best = runs["separation_q8"]
    rate = best["native_sample_rate"]
    start, end = best["evaluation_sample_interval"]
    for index, name in enumerate(order):
        rendered = root/"separation_q8"/best["rendered_files"][index]
        audio, _ = sf.read(rendered, dtype="float64", always_2d=True)
        detected = detect_percussive_events(audio[start:end].T, rate)["events"]
        lo, hi = start/rate+.05, end/rate-.05
        actual = [e["time_seconds"]+start/rate for e in detected if lo <= e["time_seconds"]+start/rate <= hi]
        expected = [e["time_seconds"] for e in audit["solos"][name]["events"] if lo <= e["time_seconds"] <= hi]
        diagnostics["sources"][name] = event_agreement(expected, actual)
    (root/"separated_event_agreement.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    rows = []
    for name in order:
        row = {"source": name}
        for key, result in runs.items():
            row[key] = result["results"][-1]["methods"]["model"]["sources"][name]["snr_db"]
        row["reference_ratio_mask"] = runs["separation_q8"]["results"][-1]["methods"]["reference_ratio_mask"]["sources"][name]["snr_db"]
        rows.append(row)
    (root/"comparison.json").write_text(json.dumps({"same_training_pcm": True, "sources": rows}, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(order))
    for index, (key, label) in enumerate(labels.items()):
        ax.bar(positions+(index-1)*.22, [r[key] for r in rows], width=.22, label=label)
    ax.plot(positions, [r["reference_ratio_mask"] for r in rows], "ko--", label="Reference-informed masks (privileged)")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(positions, order, rotation=20)
    ax.set_ylabel("Gain-sensitive waveform SNR (dB); silence = 0 dB")
    ax.set_title("Fourth bar of the eight-source mix: development diagnostic, one performance")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(root/"separation_comparison.png", dpi=140)
    plt.close(fig)
    def table(headers, body):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(h)}</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in values)+"</tr>" for values in body)+"</tbody></table>"
    def audio(path):
        return f'<audio controls preload="none" src="{html.escape(quote(path))}"></audio>'
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>Wavelet2vec drum dataset report</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border:1px solid #ccc;text-align:left}th{background:#eef2f6}img{max-width:100%}audio{max-width:100%;width:270px}section{padding:16px 0;border-top:1px solid #ccc}.note{background:#fff2d9;padding:16px}</style>',
        '<h1>First real drum substem experiment</h1><p>128 BPM, four bars, 4/4. Eight solos and seven cumulative mixes. Native stereo 44.1 kHz / 16-bit inputs; originals unchanged.</p>',
        '<p class="note">Three models trained on bars 1–3, 300 updates each. Bar 4 was used for development comparisons, including choosing a finer wavelet configuration. This is one shared performance, not an independent test set. The master limiter means solo sums are not exact mixture targets. Reference masks use the target solos and are not a blind separator.</p>',
        '<h2>Measured solo events</h2><p>Labels come from filenames. Onsets and transformations are estimates, not MIDI ground truth. Kick pitch is stable after correcting onset-alignment bias. Level windows can be affected by attack timing; accepted fitted gains compensate for timing/pitch. Energy balance is not the DAW pan knob.</p>',
        table(["Source", "Events", "Raw level span (dB)", "Stereo energy balance", "Pitch fits", "Relative pitch span (semitones)"],
              [[r["source"], r["events"], f'{r["level_span_db"]:.2f}', f'{r["balance_range"][0]:+.3f} to {r["balance_range"][1]:+.3f}',
                f'{r["accepted_pitch_fits"]}/{r["events"]}', "unresolved" if r["accepted_pitch_fits"] <= 1 else f'{r["relative_pitch_range"][0]:+.3f} to {r["relative_pitch_range"][1]:+.3f}'] for r in summary]),
        '<p><a href="events.csv">All event measurements (CSV)</a> · <a href="audit.json">Full audit (JSON)</a></p>',
        '<img src="events_overview.png" alt="Waveforms, estimated onsets, levels and stereo balances">',
        '<img src="relative_pitch.png" alt="Accepted relative playback-rate pitch measurements">',
        '<h2>Separation results</h2><p>Scores below are for bar 4 of the complete mix, with identical target units. Positive SNR beats a silent output; near-zero scores can still mean a mostly muted source. Quiet drums remain weak.</p>',
        table(["Source", *labels.values(), "Privileged Q8 masks"], [[r["source"], *[f'{r[k]:.2f} dB' for k in labels], f'{r["reference_ratio_mask"]:.2f} dB'] for r in rows]),
        '<img src="separation_comparison.png" alt="Source SNR comparison">',
        '<h2>Listen to the outputs</h2><p>Players cover the full render, including training bars 1–3. The score table above covers only bar 4. Levels are preserved; no listening normalization was applied. Files are named by source slot; the labels below follow the saved source order.</p>']
    for index, name in enumerate(order):
        parts.append(f'<section><h3>{html.escape(name)}</h3><p>Solo reference</p>'+audio(f'../../../{dataset.name}/only_{name}.wav'))
        for folder, label in labels.items():
            parts.append(f'<p>{html.escape(label)}</p>'+audio(f'{folder}/{runs[folder]["rendered_files"][index]}'))
        parts.append('</section>')
    parts.append('<h2>What this establishes</h2><p>The real audio runs through verified loading, native-rate hybrid encoding, supervised learning, waveform rendering and solo-reference loss measurement. It does not yet deliver clean eight-way separation, blind drum classification, exact MIDI velocity recovery or automatic identification of synthesizer versus sampled origins.</p></html>')
    (root/"LISTEN_AND_RESULTS.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"sources": rows, "event_agreement": diagnostics["sources"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/benchmarks/drum_dataset_2026-09-22"))
    parser.add_argument("--dataset", type=Path, default=Path("drum_substem_dataset_BPM128_4bars"))
    args = parser.parse_args()
    build(args.results, args.dataset)
