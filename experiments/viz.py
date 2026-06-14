"""Matplotlib figures for the wavelet2vec stages and the mel-vs-wavelet study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.dataset import PITCH_CLASS_NAMES
from experiments.mel_baseline import log_mel_spectrogram
from wavelet2vec import Wavelet2Vec
from wavelet2vec.filterbank import log_spaced_frequencies, morlet_scalogram

_SECTION_TITLES = {
    "spectral": "spectral — band mean±std (timbre)",
    "modulation": "modulation — texture (band-group × mod-band)",
    "transient": "transient — envelope + attack/decay/crest",
    "harmonic": "harmonic — chroma + pitch scalars",
    "phase": "phase — IF dev / onset coherence / waveshape",
    "stereo": "stereo — width / coherence / balance",
    "conv": "conv — wavelet-init conv features",
}


def _scalogram(mono: np.ndarray, sample_rate: int, n_bands: int = 96):
    centers = log_spaced_frequencies(27.5, 0.45 * sample_rate, n_bands)
    mag = morlet_scalogram(mono, sample_rate, centers)
    return mag, centers


def _time_freq(ax, data, centers, sample_rate, n_samples, title, cmap="magma"):
    extent = [0, n_samples / sample_rate, 0, data.shape[0]]
    # Percentile contrast so sparse-energy content (e.g. a bass note) is still
    # legible, and mel vs scalogram are shown on comparable dynamic range.
    finite = data[np.isfinite(data)]
    vmin = np.percentile(finite, 5) if finite.size else 0.0
    vmax = np.percentile(finite, 99.5) if finite.size else 1.0
    ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    ax.set_title(title, fontsize=9, loc="left")
    tick_idx = np.linspace(0, len(centers) - 1, 5).astype(int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{centers[i]:.0f}" for i in tick_idx], fontsize=7)
    ax.set_ylabel("Hz", fontsize=7)


def stage_figure(
    mono: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    *,
    title: str = "",
    embedder: Wavelet2Vec | None = None,
) -> Path:
    """Full pipeline for one snippet: waveform → mel → scalogram → sections → vector."""
    embedder = embedder or Wavelet2Vec()
    components = embedder.embed_components(mono, sample_rate)
    embedding = embedder.embed(mono, sample_rate)
    n = mono.shape[0]

    fig = plt.figure(figsize=(11, 15))
    gs = fig.add_gridspec(10, 1, hspace=0.85)

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(n) / sample_rate, mono, lw=0.5, color="#1f77b4")
    ax.set_title(f"waveform — {title}", fontsize=9, loc="left")
    ax.set_xlim(0, n / sample_rate)
    ax.set_xlabel("s", fontsize=7)

    log_mel, mel_centers, _ = log_mel_spectrogram(mono, sample_rate)
    _time_freq(fig.add_subplot(gs[1]), log_mel, mel_centers, sample_rate, n, "mel spectrogram (baseline)")

    mag, centers = _scalogram(mono, sample_rate)
    _time_freq(fig.add_subplot(gs[2]), np.log1p(mag), centers, sample_rate, n, "constant-Q Morlet scalogram (wavelet front end)")

    # spectral
    ax = fig.add_subplot(gs[3])
    half = components["spectral"].shape[0] // 2
    ax.plot(components["spectral"][:half], color="#d62728", label="mean")
    ax.fill_between(np.arange(half), 0, components["spectral"][half:], alpha=0.3, color="#ff7f0e", label="std")
    ax.set_title(_SECTION_TITLES["spectral"], fontsize=9, loc="left")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("wavelet band", fontsize=7)

    # modulation heatmap
    ax = fig.add_subplot(gs[4])
    mod = components["modulation"].reshape(8, 8)
    ax.imshow(mod, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(_SECTION_TITLES["modulation"], fontsize=9, loc="left")
    ax.set_xlabel("modulation band (0.5→64 Hz)", fontsize=7)
    ax.set_ylabel("band group", fontsize=7)

    # transient
    ax = fig.add_subplot(gs[5])
    env_points = embedder.config.envelope_points
    ax.plot(components["transient"][:env_points], color="#2ca02c")
    ax.set_title(_SECTION_TITLES["transient"] + f"  (attack={components['transient'][env_points]:.2f})", fontsize=9, loc="left")
    ax.set_xlabel("normalized time", fontsize=7)

    # harmonic chroma
    ax = fig.add_subplot(gs[6])
    chroma = components["harmonic"][:12]
    bars = ax.bar(PITCH_CLASS_NAMES, chroma, color="#9467bd")
    top = int(np.argmax(chroma))
    bars[top].set_color("#e377c2")
    ax.set_title(_SECTION_TITLES["harmonic"] + f"  (peak={PITCH_CLASS_NAMES[top]}, harmonicity={components['harmonic'][12]:.2f})", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)

    # phase + stereo on one row each
    ax = fig.add_subplot(gs[7])
    ax.plot(components["phase"], color="#8c564b")
    ax.set_title(_SECTION_TITLES["phase"], fontsize=9, loc="left")
    ax.set_xlabel("phase feature index", fontsize=7)

    ax = fig.add_subplot(gs[8])
    ax.plot(components["stereo"], color="#17becf")
    ax.set_title(_SECTION_TITLES["stereo"], fontsize=9, loc="left")
    ax.set_xlabel("stereo feature index", fontsize=7)

    # final vector
    ax = fig.add_subplot(gs[9])
    ax.imshow(embedding[np.newaxis, :], aspect="auto", cmap="coolwarm")
    ax.set_title(f"final embedding ({embedder.dim} dims)", fontsize=9, loc="left")
    offset = 0
    for name, sl in embedder.sections.items():
        offset = sl.stop
        ax.axvline(offset - 0.5, color="k", lw=0.5)
        ax.text(sl.start, 1.6, name, fontsize=6, rotation=0)
    ax.set_yticks([])

    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def comparison_figure(
    snippets: list[tuple[str, np.ndarray, int]],
    output_path: str | Path,
) -> Path:
    """Mel spectrogram vs Morlet scalogram, side by side, for several snippets."""
    n = len(snippets)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    for row, (label, mono, sr) in enumerate(snippets):
        log_mel, mel_centers, _ = log_mel_spectrogram(mono, sr)
        _time_freq(axes[row][0], log_mel, mel_centers, sr, mono.shape[0], f"mel — {label}")
        mag, centers = _scalogram(mono, sr)
        _time_freq(axes[row][1], np.log1p(mag), centers, sr, mono.shape[0], f"Morlet scalogram — {label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def development_strip(
    window_embeddings: np.ndarray,
    section_slice: slice,
    output_path: str | Path,
    *,
    title: str = "",
    section_name: str = "",
) -> Path:
    """How one section evolves across beat-aligned windows of a loop."""
    section = window_embeddings[:, section_slice].T  # [features, windows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(section, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"{section_name} development across windows — {title}", fontsize=10, loc="left")
    ax.set_xlabel("beat-aligned window")
    ax.set_ylabel(f"{section_name} feature")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def roundtrip_figure(
    mono: np.ndarray,
    sample_rate: int,
    coeffs: np.ndarray,
    reconstructed: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "",
    transform: str = "",
    freq_axis: np.ndarray | None = None,
    snr_db: float | None = None,
) -> Path:
    """Magnitude image, phase image (cyclic), and the lossless round trip.

    Shows that the full complex coefficients (magnitude + phase) are the audio:
    the overlaid original and reconstruction are indistinguishable.
    """
    magnitude = np.log1p(np.abs(coeffs))
    phase = np.angle(coeffs)
    n = mono.shape[0]
    extent = [0, n / sample_rate, 0, coeffs.shape[0]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0][0].imshow(magnitude, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0][0].set_title(f"{transform} magnitude  log|coeff|", fontsize=9, loc="left")
    # Phase on a cyclic colormap so the ±pi wrap is continuous.
    axes[0][1].imshow(phase, origin="lower", aspect="auto", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0][1].set_title(f"{transform} phase  angle(coeff) ∈ [−π, π]", fontsize=9, loc="left")
    for ax in (axes[0][0], axes[0][1]):
        if freq_axis is not None:
            idx = np.linspace(0, len(freq_axis) - 1, 5).astype(int)
            ax.set_yticks(idx)
            ax.set_yticklabels([f"{freq_axis[i]:.0f}" for i in idx], fontsize=7)
            ax.set_ylabel("Hz", fontsize=7)
        ax.set_xlabel("s", fontsize=7)

    times = np.arange(n) / sample_rate
    axes[1][0].plot(times, mono, color="#1f77b4", lw=0.6, label="original")
    axes[1][0].plot(times, reconstructed[:n], color="#d62728", lw=0.6, ls="--", label="reconstructed")
    snr_text = f"  (SNR={snr_db:.0f} dB)" if snr_db is not None else ""
    axes[1][0].set_title(f"waveform: original vs reconstructed{snr_text}", fontsize=9, loc="left")
    axes[1][0].legend(fontsize=7, loc="upper right")
    axes[1][0].set_xlim(0, n / sample_rate)

    error = mono - reconstructed[:n]
    axes[1][1].plot(times, error, color="#7f7f7f", lw=0.5)
    axes[1][1].set_title(f"reconstruction error (max |e|={np.max(np.abs(error)):.1e})", fontsize=9, loc="left")
    axes[1][1].set_xlim(0, n / sample_rate)

    fig.suptitle(f"{transform} lossless round trip — {title}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def phase_matters_figure(
    mono: np.ndarray,
    sample_rate: int,
    full_reconstruction: np.ndarray,
    magnitude_only_reconstruction: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "",
) -> Path:
    """Why phase is kept: full-complex reconstructs perfectly, magnitude-only fails."""
    n = mono.shape[0]
    times = np.arange(n) / sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, mono, color="#1f77b4", lw=0.6)
    axes[0].set_title(f"original — {title}", fontsize=9, loc="left")
    axes[1].plot(times, full_reconstruction[:n], color="#2ca02c", lw=0.6)
    axes[1].set_title("reconstructed from full complex coefficients (magnitude + phase) — lossless", fontsize=9, loc="left")
    axes[2].plot(times, magnitude_only_reconstruction[:n], color="#d62728", lw=0.6)
    axes[2].set_title("reconstructed from magnitude only (phase discarded) — destroyed", fontsize=9, loc="left")
    axes[2].set_xlabel("s", fontsize=7)
    for ax in axes:
        ax.set_xlim(0, n / sample_rate)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def projection_figure(
    points_by_method: dict[str, np.ndarray],
    labels: list[str],
    output_path: str | Path,
    *,
    title: str = "",
) -> Path:
    """2D scatter (one panel per method) colored by folder label."""
    unique = sorted(set(labels))
    color_map = {name: plt.cm.tab10(i % 10) for i, name in enumerate(unique)}
    colors = [color_map[name] for name in labels]

    methods = list(points_by_method)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6), squeeze=False)
    for col, method in enumerate(methods):
        pts = points_by_method[method]
        ax = axes[0][col]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=14, alpha=0.7)
        ax.set_title(f"{method} — {title}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=color_map[n], label=n) for n in unique]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(unique), 6), fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)
