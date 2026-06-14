"""Regenerate every figure and the metrics report from an audio library.

    python -m experiments.run_experiments --audio-dir ../soundgen/example_audio

Outputs land in ``experiments/output/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments import metrics, viz
from experiments.dataset import (
    IngestReport,
    canonical_rate,
    list_tracks,
    load_canonical,
    sliding_windows,
)
from experiments.invertible import (
    WaveletMeta,
    complex_stft,
    complex_wavelet,
    inverse_complex_stft,
    inverse_complex_wavelet,
    load_coefficients,
    quantize_complex,
    reconstruction_stats,
    save_coefficients,
)
from experiments.mel_baseline import (
    log_mel_spectrogram,
    mel_chroma,
    mel_summary_vector,
    predicted_pitch_class,
)

# Full-resolution complex coefficients are heavy (n_bands × n_samples); cap the
# round-trip snippet length so the lossless demo stays fast on the 5 s vocals.
ROUNDTRIP_MAX_SECONDS = 1.5
from wavelet2vec import Wavelet2Vec

# Folders whose filename labels are a single played note (strict scoring) vs.
# a song key (tolerant scoring — the snippet need not voice the tonic alone).
STRICT_PITCH_FOLDERS = {"bass", "harmonic_stabs_one_shots", "single_note_vocal"}
TOLERANT_PITCH_FOLDERS = {"vocal_shots", "synth_loops_100BPM", "synth_loops_120BPM"}


def _window_mono(audio, start, end):
    segment = audio.mono[start:end]
    peak = np.abs(segment).max()
    return segment / peak if peak > 1e-8 else segment


def _run_roundtrip(by_folder, rate, out):
    """Lossless audio<->image round trips for a representative snippet per folder.

    Verifies both the exact STFT and the constant-Q complex wavelet frame,
    saves magnitude/phase figures, a 'phase matters' demo, reconstructed WAVs,
    and a bit-depth quantization study (what an image format would cost).
    """
    from wavelet2vec.audio_io import save_audio  # local import; only needed here

    rt_dir = out / "roundtrip"
    rt_dir.mkdir(parents=True, exist_ok=True)
    # The lossless float32 round trips are large (~37 MB each); keep them in
    # their own folder, separate from the lightweight figures.
    tiff_dir = rt_dir / "perfect_tiff"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    max_n = int(ROUNDTRIP_MAX_SECONDS * rate)
    snr = {"stft": [], "wavelet": []}
    storage = {}
    base_snrs = []
    example_done = False

    for folder, group in by_folder.items():
        track = group[len(group) // 2]
        audio = load_canonical(track, rate)
        mono = audio.mono[:max_n]
        peak = np.abs(mono).max()
        if peak < 1e-8:
            continue
        mono = mono / peak

        stft_c, stft_meta = complex_stft(mono, n_fft=1024)
        stft_rec = inverse_complex_stft(stft_c, stft_meta)
        stft_stats = reconstruction_stats(mono, stft_rec)
        snr["stft"].append(stft_stats["snr_db"])

        wav_c, wav_meta = complex_wavelet(mono, rate, n_bands=128)
        wav_rec = inverse_complex_wavelet(wav_c, wav_meta)
        wav_stats = reconstruction_stats(mono, wav_rec)
        snr["wavelet"].append(wav_stats["snr_db"])

        viz.roundtrip_figure(
            mono, rate, wav_c, wav_rec, rt_dir / f"{folder}_wavelet.png",
            title=f"{folder} / {track.path.name}", transform="constant-Q complex wavelet",
            freq_axis=wav_meta.centers, snr_db=wav_stats["snr_db"],
        )

        # Base case per folder: lossless float32 image on disk -> reload ->
        # reconstruct audio from the image alone. The picture is the audio.
        base_path = tiff_dir / f"{folder}_coefficients_float32.tiff"
        save_coefficients(wav_c, base_path, extra=wav_meta.to_dict(), dtype="float32")
        loaded_c, extra = load_coefficients(base_path)
        base_rec = inverse_complex_wavelet(loaded_c, WaveletMeta.from_dict(extra))
        save_audio(tiff_dir / f"{folder}_original.wav", mono[np.newaxis, :], rate)
        save_audio(tiff_dir / f"{folder}_from_image.wav", base_rec[np.newaxis, :], rate)
        base_snrs.append(reconstruction_stats(mono, base_rec)["snr_db"])

        if not example_done:
            viz.roundtrip_figure(
                mono, rate, stft_c, stft_rec, rt_dir / f"{folder}_stft.png",
                title=f"{folder} / {track.path.name}", transform="complex STFT",
                snr_db=stft_stats["snr_db"],
            )
            # Phase-matters: reconstruct from magnitude only (phase zeroed).
            mag_only = inverse_complex_wavelet(np.abs(wav_c).astype(np.complex128), wav_meta)
            viz.phase_matters_figure(
                mono, rate, wav_rec, mag_only, rt_dir / f"{folder}_phase_matters.png",
                title=f"{folder} / {track.path.name}",
            )
            # Storage-format comparison (SNR computed in memory; no huge writes).
            storage = {
                "float64 (.npy, bit-exact)": reconstruction_stats(mono, inverse_complex_wavelet(wav_c, wav_meta))["snr_db"],
                "float32 (.tiff, base case)": base_snrs[-1],
                "16-bit (.png)": reconstruction_stats(mono, inverse_complex_wavelet(quantize_complex(wav_c, 16), wav_meta))["snr_db"],
                "8-bit (.png)": reconstruction_stats(mono, inverse_complex_wavelet(quantize_complex(wav_c, 8), wav_meta))["snr_db"],
            }
            example_done = True

    return {
        "stft_mean_snr_db": round(float(np.mean(snr["stft"])), 1) if snr["stft"] else None,
        "wavelet_mean_snr_db": round(float(np.mean(snr["wavelet"])), 1) if snr["wavelet"] else None,
        "n_snippets": len(snr["stft"]),
        "max_seconds": ROUNDTRIP_MAX_SECONDS,
        "base_case": "float32 image (.tiff) per folder",
        "base_image_roundtrip_snr_db": round(float(np.mean(base_snrs)), 1) if base_snrs else None,
        "base_images_written": len(base_snrs),
        "storage_format_snr_db": {fmt: round(float(v), 1) for fmt, v in storage.items()},
    }


def _pooled_embeddings(audio, embedder, sample_rate):
    """Mean-pooled wavelet embedding and mel summary over sliding windows."""
    bounds = sliding_windows(audio.n_samples, sample_rate, bpm=audio.track.bpm)
    wav_vectors, mel_vectors = [], []
    for start, end in bounds:
        mono = _window_mono(audio, start, end)
        if mono.shape[0] < 64:
            continue
        stereo_segment = audio.stereo[:, start:end]
        wav_vectors.append(embedder.embed(stereo_segment, sample_rate))
        log_mel, _, _ = log_mel_spectrogram(mono, sample_rate)
        mel_vectors.append(mel_summary_vector(log_mel))
    if not wav_vectors:
        return None, None, len(bounds)
    return np.mean(wav_vectors, axis=0), np.mean(mel_vectors, axis=0), len(bounds)


def main() -> None:
    parser = argparse.ArgumentParser(description="wavelet2vec experiment & visualization suite.")
    parser.add_argument("--audio-dir", default="../soundgen/example_audio")
    parser.add_argument("--output", default="experiments/output")
    parser.add_argument("--analysis-rate", type=int, default=None)
    parser.add_argument("--limit-per-folder", type=int, default=None, help="Cap files/folder (fast runs).")
    args = parser.parse_args()

    out = Path(args.output)
    (out / "stages").mkdir(parents=True, exist_ok=True)
    (out / "comparison").mkdir(parents=True, exist_ok=True)

    tracks = list_tracks(args.audio_dir)
    if not tracks:
        raise SystemExit(f"No audio found under {args.audio_dir}")
    rate = canonical_rate(tracks, args.analysis_rate)
    embedder = Wavelet2Vec()
    report = IngestReport()
    print(f"{len(tracks)} tracks; canonical rate = {rate} Hz")

    by_folder = defaultdict(list)
    for track in tracks:
        by_folder[track.folder].append(track)
    if args.limit_per_folder:
        for folder in by_folder:
            by_folder[folder] = by_folder[folder][: args.limit_per_folder]

    # One accurate ingestion pass over every track (the per-experiment loops
    # below only touch subsets, so they would mis-count).
    for track in tracks:
        report.note(track, rate)

    results: dict = {"canonical_rate": rate, "n_tracks": sum(len(v) for v in by_folder.values())}

    # --- Stage figure + a comparison gallery (one representative per folder) ---
    gallery = []
    for folder, group in by_folder.items():
        track = group[len(group) // 2]
        audio = load_canonical(track, rate)
        viz.stage_figure(
            audio.mono, rate, out / "stages" / f"{folder}.png",
            title=f"{folder} / {track.path.name}", embedder=embedder,
        )
        gallery.append((folder, audio.mono, rate))
    viz.comparison_figure(gallery[:6], out / "comparison" / "mel_vs_wavelet.png")
    print(f"wrote {len(by_folder)} stage figures + comparison gallery")

    # --- Pitch recovery: mel chroma vs wavelet harmonic, scored on labels ---
    pitch_rows = defaultdict(lambda: {"labels": [], "mel": [], "wav": []})
    for folder, group in by_folder.items():
        if folder not in STRICT_PITCH_FOLDERS | TOLERANT_PITCH_FOLDERS:
            continue
        for track in group:
            if track.pitch_class is None:
                continue
            audio = load_canonical(track, rate)
            mel_pred = predicted_pitch_class(mel_chroma(audio.mono, rate))
            wav_chroma = embedder.embed_components(audio.stereo, rate)["harmonic"][:12]
            pitch_rows[folder]["labels"].append(track.pitch_class)
            pitch_rows[folder]["mel"].append(mel_pred)
            pitch_rows[folder]["wav"].append(int(np.argmax(wav_chroma)))

    pitch_report = {}
    all_labels, all_mel, all_wav = [], [], []
    for folder, data in pitch_rows.items():
        tolerant = folder in TOLERANT_PITCH_FOLDERS
        res = metrics.score_pitch_recovery(data["labels"], data["mel"], data["wav"], tolerant=tolerant)
        pitch_report[folder] = {
            "n": res.n, "scoring": "tolerant" if tolerant else "strict",
            "mel_accuracy": round(res.mel_accuracy, 3),
            "wavelet_accuracy": round(res.wavelet_accuracy, 3),
        }
        all_labels += data["labels"]; all_mel += data["mel"]; all_wav += data["wav"]
    if all_labels:
        overall = metrics.score_pitch_recovery(all_labels, all_mel, all_wav, tolerant=False)
        pitch_report["overall_strict"] = {
            "n": overall.n,
            "mel_accuracy": round(overall.mel_accuracy, 3),
            "wavelet_accuracy": round(overall.wavelet_accuracy, 3),
        }
    results["pitch_recovery"] = pitch_report
    print("pitch recovery:", json.dumps(pitch_report, indent=2))

    # --- Library map + cluster quality (wavelet vs mel baseline) ---
    # Balance per folder: kNN/silhouette are biased toward large classes, and
    # folder sizes here range from 6 to 100+. Cap so the comparison is fair.
    cluster_cap = 20
    wav_lib, mel_lib, lib_labels = [], [], []
    for folder, group in by_folder.items():
        for track in group[:cluster_cap]:
            audio = load_canonical(track, rate)
            wav_vec, mel_vec, _ = _pooled_embeddings(audio, embedder, rate)
            if wav_vec is None:
                continue
            wav_lib.append(wav_vec); mel_lib.append(mel_vec); lib_labels.append(folder)
    wav_lib, mel_lib = np.asarray(wav_lib), np.asarray(mel_lib)
    results["clustering_balanced_cap"] = cluster_cap

    viz.projection_figure(
        {"wavelet2vec": metrics.pca_2d(wav_lib), "mel baseline": metrics.pca_2d(mel_lib)},
        lib_labels, out / "library_map.png", title="PCA of library (color = folder)",
    )
    results["clustering"] = {
        "wavelet2vec": {
            "knn_purity": round(metrics.knn_label_purity(wav_lib, lib_labels), 3),
            "silhouette": round(metrics.silhouette_score(wav_lib, lib_labels), 3),
        },
        "mel_baseline": {
            "knn_purity": round(metrics.knn_label_purity(mel_lib, lib_labels), 3),
            "silhouette": round(metrics.silhouette_score(mel_lib, lib_labels), 3),
        },
    }
    print("clustering:", json.dumps(results["clustering"], indent=2))

    # --- Sliding-window development strip for one loop ---
    loops = by_folder.get("synth_loops_120BPM") or by_folder.get("synth_loops_100BPM")
    if loops:
        track = loops[0]
        audio = load_canonical(track, rate)
        bounds = sliding_windows(audio.n_samples, rate, bpm=track.bpm)
        window_vecs = [embedder.embed(audio.stereo[:, s:e], rate) for s, e in bounds if e - s >= 64]
        if len(window_vecs) >= 2:
            viz.development_strip(
                np.asarray(window_vecs), embedder.sections["spectral"],
                out / "development_strip.png", title=track.path.name, section_name="spectral",
            )
            results["development_windows"] = len(window_vecs)

    # --- Lossless audio <-> image round trip (full complex coefficients) ---
    results["roundtrip"] = _run_roundtrip(by_folder, rate, out)

    results["ingest_summary"] = report.summary()
    results["ingest_conversions"] = report.converted[:20]
    print(report.summary())
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(out, results)
    print(f"\nwrote {out/'metrics.json'} and {out/'REPORT.md'}")


def _write_report(out: Path, results: dict) -> None:
    lines = ["# Experiment Results\n", f"Canonical rate: {results['canonical_rate']} Hz · "
             f"{results['n_tracks']} tracks\n"]
    pitch = results.get("pitch_recovery", {})
    if pitch:
        lines.append("## Pitch recovery (mel chroma vs wavelet harmonic)\n")
        lines.append("| folder | n | scoring | mel acc | wavelet acc |")
        lines.append("| --- | --- | --- | --- | --- |")
        for folder, row in pitch.items():
            lines.append(f"| {folder} | {row['n']} | {row.get('scoring','-')} | "
                         f"{row['mel_accuracy']} | {row['wavelet_accuracy']} |")
        lines.append("")
    clustering = results.get("clustering", {})
    if clustering:
        cap = results.get("clustering_balanced_cap")
        lines.append(f"## Cluster quality by folder (balanced ≤{cap}/folder; higher = better)\n")
        lines.append("| method | kNN purity | silhouette |")
        lines.append("| --- | --- | --- |")
        for method, row in clustering.items():
            lines.append(f"| {method} | {row['knn_purity']} | {row['silhouette']} |")
        lines.append("")
    rt = results.get("roundtrip")
    if rt:
        lines.append("## Lossless audio↔image round trip (full complex coefficients)\n")
        lines.append(f"First {rt['max_seconds']} s of {rt['n_snippets']} snippets. "
                     "Higher SNR = closer to bit-exact; ~300 dB is float64 machine precision.\n")
        lines.append("| transform | mean reconstruction SNR (dB) |")
        lines.append("| --- | --- |")
        lines.append(f"| complex STFT | {rt['stft_mean_snr_db']} |")
        lines.append(f"| constant-Q complex wavelet | {rt['wavelet_mean_snr_db']} |")
        lines.append("")
        base_snr = rt.get("base_image_roundtrip_snr_db")
        if base_snr is not None:
            n_imgs = rt.get("base_images_written", 0)
            lines.append(f"**Base case — lossless float image (per folder).** For each of the "
                         f"{n_imgs} folders, the coefficients are saved as a float32 image on "
                         f"disk, reloaded, and inverted — mean reconstruction **{base_snr} dB** "
                         f"(beyond 24-bit audio). The picture is the audio. See "
                         f"`roundtrip/perfect_tiff/<folder>_coefficients_float32.tiff` and "
                         f"`roundtrip/perfect_tiff/<folder>_from_image.wav`.\n")
        storage = rt.get("storage_format_snr_db", {})
        if storage:
            lines.append("Reconstruction SNR by storage format (one snippet):\n")
            lines.append("| format | reconstruction SNR (dB) |")
            lines.append("| --- | --- |")
            for fmt, snr in storage.items():
                lines.append(f"| {fmt} | {snr} |")
            lines.append("")
    lines.append("Figures: `stages/<folder>.png`, `comparison/mel_vs_wavelet.png`, "
                 "`library_map.png`, `development_strip.png`, `roundtrip/<folder>_wavelet.png`, "
                 "`roundtrip/<folder>_phase_matters.png`.")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
