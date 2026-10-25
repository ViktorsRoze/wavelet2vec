"""Figures for docs/BRIDGING_SIGNAL_AND_SEMANTICS.md.

Every number shown comes from experiments/semantic_bridge_results.json (run
semantic_bridge_experiments.py first) or is computed here from the plotted data.

  1. semantic_bridge_overview.png   - a composite scene: raw vs reassigned level, existing vs bridge coherence
  2. semantic_bridge_bass_coherence.png - white-noise response per band; harmonic vs noise cells on real notes
  3. semantic_bridge_reassignment.png  - bass click localization and a 60 ms double click
  4. semantic_bridge_embedding.png     - glide direction, invariance checks, family identification on real notes
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wavelet2vec.morph import _cell_stats  # noqa: E402
from wavelet2vec.semantic_bridge import SPEC, harmonic_noise_gate, reassigned_primitives, semantic_planes  # noqa: E402
from wavelet2vec.wavelet_image import band_layout, encode  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "semantic_bridge_results.json"
OUT = ROOT.parent / "docs" / "figures"
CORE = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)
C1 = sum(1 for b in band_layout(SPEC) if b.per_frame == 1)          # 58 one-sample rows, below 440 Hz

# light theme; categorical slots in fixed order (validated reference palette)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "grid.color": GRID, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _stereo(mono: torch.Tensor) -> torch.Tensor:
    return torch.stack([mono, mono], dim=0)[None]


def _freq_ticks(max_row: int):
    rows, labels = [], []
    for hz in (20, 40, 80, 160, 440, 1000, 2500, 5000):
        r = int(round(69.0 + 12.0 * math.log2(hz / 440.0) - SPEC.first_midi))
        if 0 <= r < max_row:
            rows.append(r)
            labels.append(f"{hz} Hz" if hz < 1000 else f"{hz / 1000:g} kHz")
    return rows, labels


def _boundary(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(C1 - 0.5, color=INK_2, linestyle="--", linewidth=0.9)


def _bar_labels(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.015 if h >= 0 else -0.06), fmt.format(h),
                ha="center", va="bottom", fontsize=8, color=INK_2)


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_overview():
    """Composite scene: bass click (0.18 s), 55 Hz saw note (0.28-0.78 s), noise burst (0.84-1.12 s)."""
    n, sr = SPEC.audio_length, SPEC.sample_rate
    t = torch.arange(n, dtype=torch.float32) / sr
    torch.manual_seed(42)
    sig = torch.zeros(n)
    c = int(0.18 * sr)
    sig[c - 32:c + 32] += 0.95 * torch.hann_window(64)
    env = ((t >= 0.28) & (t <= 0.78)).float() * torch.clamp(
        torch.sin(math.pi * torch.clamp((t - 0.28) / 0.5, 0, 1)), min=0) ** 0.5
    for h in range(1, 14):
        sig += env * (0.38 / h ** 0.85) * torch.sin(2 * math.pi * 55.0 * h * (t + 0.04 * torch.clamp(t - 0.28, min=0) ** 2))
    sig += ((t >= 0.84) & (t <= 1.12)).float() * 0.22 * torch.randn(n)
    img = encode(_stereo(sig), SPEC)
    planes = semantic_planes(img)[0].numpy()
    existing = np.zeros((SPEC.n_bands, SPEC.total_frames), dtype=np.float32)
    for cnt, values in img.groups.items():
        _, _, re, im, _ = _cell_stats(values)
        existing[list(img.rows[cnt])] = torch.sqrt(re ** 2 + im ** 2)[0].numpy()
    active = np.clip((planes[0] + 1.0) * 2.5, 0.0, 1.0)                 # fade silent cells out, as the views do
    existing = existing[:, CORE] * active
    bridge = planes[3]
    burst = slice(86, 110)                                              # frames inside the noise burst
    loud = active[:C1, burst] > 0.5
    rows_ticks, labels = _freq_ticks(102)
    extent = [0.0, 1.2, 0, 102]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    panels = [(planes[0], "A. Cell level (mid), dB re peak", "magma", (-1, 1)),
              (planes[2], "B. Reassigned level (sharp_db)", "magma", (-1, 1)),
              (existing, "C. Existing per-cell coherence (judge/critic views)", "Blues", (0, 1)),
              (bridge, "D. Semantic-bridge coherence (plane 3)", "Blues", (0, 1))]
    for ax, (data, title, cmap, (lo, hi)) in zip(axes.flat, panels):
        shown = (data[:102] - 1.0) * 30.0 if cmap == "magma" else data[:102]
        im = ax.imshow(shown, origin="lower", aspect="auto", extent=extent, cmap=cmap,
                       vmin=-60 if cmap == "magma" else lo, vmax=0 if cmap == "magma" else hi)
        _boundary(ax)
        ax.set_title(title, loc="left")
        ax.set_yticks(rows_ticks, labels)
        fig.colorbar(im, ax=ax, label="dB re peak" if cmap == "magma" else "coherence")
    for ax, data in ((axes[1, 0], existing), (axes[1, 1], bridge)):
        value = float(data[:C1, burst][loud].mean())
        ax.text(0.98, 0.04, f"noise burst below 440 Hz: mean {value:.2f}", transform=ax.transAxes, ha="right",
                fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID))
        ax.set_xlabel("time (s)")
    axes[0, 0].text(0.01, (C1 + 1) / 102, "440 Hz: one sample per cell below", transform=axes[0, 0].transAxes,
                    fontsize=8, color="white")
    fig.suptitle("Click, 55 Hz saw note and noise burst at 192 kHz", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "semantic_bridge_overview.png")


def figure_bass_coherence(results):
    """Left: per-band mean on white noise. Right: harmonic vs noise cells on real notes (AUC)."""
    n, sr = SPEC.audio_length, SPEC.sample_rate
    torch.manual_seed(101)
    img = encode(_stereo(0.25 * torch.randn(n)), SPEC)
    prim = reassigned_primitives(img)
    gate = harmonic_noise_gate(img)
    sl = slice(SPEC.pad_frames + 12, SPEC.pad_frames + SPEC.n_frames - 12)
    existing = np.zeros(SPEC.n_bands)
    for cnt, values in img.groups.items():
        _, _, re, im, _ = _cell_stats(values)
        existing[list(img.rows[cnt])] = torch.sqrt(re ** 2 + im ** 2)[0, :, sl].mean(-1).numpy()
    r_max = 95
    x = np.arange(r_max)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(x, existing[:r_max], color=ORANGE, lw=2, label="existing per-cell coherence")
    ax.plot(x, prim.phase_coherence[0, :r_max, sl].mean(-1).numpy(), color=BLUE, lw=2, label="bridge coherence")
    ax.plot(x, gate[0, :r_max, sl].mean(-1).numpy(), color=AQUA, lw=2, label="bridge harmonic gate")
    _boundary(ax, horizontal=False)
    ax.text(C1 - 1.5, 1.02, "one sample per cell  |", ha="right", fontsize=8, color=INK_2)
    ticks, labels = _freq_ticks(r_max)
    ax.set_xticks(ticks, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean on white noise (lower = noise found)")
    ax.set_title("White noise, per band", loc="left")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax = axes[1]
    real = results.get("experiment_6_real_notes")
    if real is None:
        ax.text(0.5, 0.5, "run semantic_bridge_experiments.py with note_dataset\nfor the real-note comparison",
                ha="center", va="center", transform=ax.transAxes, color=INK_2)
        ax.axis("off")
    else:
        part = real["a_harmonic_vs_noise_cells_below_440hz"]
        names = list(part)
        xs = np.arange(len(names))
        w = 0.36
        _bar_labels(ax, ax.bar(xs - w / 2, [part[k]["auc_vox"] for k in names], w, color=BLUE, label="vocal notes"))
        _bar_labels(ax, ax.bar(xs + w / 2, [part[k]["auc_synth"] for k in names], w, color=ORANGE, label="synth notes"))
        ax.axhline(0.5, color=INK_2, lw=0.9, linestyle=":")
        ax.text(len(names) - 0.5, 0.51, "chance", ha="right", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(xs, [k.replace("semantic-bridge ", "bridge ").replace("existing per-cell", "existing per-cell\n")
                           .replace("bridge harmonic", "bridge\nharmonic").replace("bridge coherence", "bridge\ncoherence")
                           for k in names])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC, harmonic vs noise cells")
        n_notes = real["notes"]["cells_notes_per_class"]
        ax.set_title(f"Real notes below 440 Hz ({n_notes} vocal, {n_notes} synth, {n_notes} noise)", loc="left")
        ax.grid(axis="y")
        ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "semantic_bridge_bass_coherence.png")


def figure_reassignment(results):
    """A bass click at 0.50 s and a double click 60 ms apart: raw vs reassigned power."""
    r = results["experiment_2_heisenberg_trap"]
    n, sr = SPEC.audio_length, SPEC.sample_rate
    win = torch.hann_window(64)
    one, two = torch.zeros(n), torch.zeros(n)
    one[int(0.5 * sr) - 32:int(0.5 * sr) + 32] = 0.95 * win
    for at in (0.47, 0.53):
        two[int(at * sr) - 32:int(at * sr) + 32] = 0.95 * win
    prim = reassigned_primitives(encode(torch.cat([_stereo(one), _stereo(two)]), SPEC))
    raw, sharp = prim.mid_power[0, :55, CORE].numpy(), prim.sharp_mid_power[0, :55, CORE].numpy()
    lo, hi = 30, 70
    extent = [0.30, 0.70, 0, 55]
    ticks, labels = _freq_ticks(55)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    spread = r["single_impulse_rms_spread_ms"]
    for ax, data, title in ((axes[0, 0], raw, f"A. Cell power: RMS spread {spread['raw_wavelet_power_ms']:.1f} ms"),
                            (axes[0, 1], sharp, f"B. Reassigned power: RMS spread {spread['reassigned_sharp_power_ms']:.1f} ms")):
        im = ax.imshow(10 * np.log10(data[:, lo:hi] / (data.max() + 1e-30) + 1e-6), origin="lower", aspect="auto",
                       extent=extent, cmap="magma", vmin=-45, vmax=0)
        ax.set_yticks(ticks, labels)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, label="dB re peak")
    frames = np.arange(lo, hi) * SPEC.hop_ms / 1000
    share = r["single_impulse_30ms_energy_fraction"]
    ax = axes[1, 0]
    for data, color, name, key in ((raw, ORANGE, "cell power", "raw_wavelet_power"),
                                   (sharp, BLUE, "reassigned", "reassigned_sharp_power")):
        env = data[16:41, lo:hi].sum(0)
        ax.plot(frames, env / env.sum(), color=color, lw=2, marker="o", ms=3.5,
                label=f"{name}: {100 * share[key]:.1f}% within ±1 frame")
    ax.axvspan(0.49, 0.51, color=GRID, alpha=0.7)
    ax.set_title("C. Click at 0.50 s, 41-165 Hz: energy per frame", loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("share of energy")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    p2v = r["double_impulse_60ms_peak_to_valley_db"]
    ax = axes[1, 1]
    for data, color, name, key in ((prim.mid_power, ORANGE, "cell power", "raw_wavelet_power_db"),
                                   (prim.sharp_mid_power, BLUE, "reassigned", "reassigned_sharp_power_db")):
        level = data[1, 28:47, CORE].mean(0).numpy()[lo:hi]
        ax.plot(frames, 10 * np.log10(level / level.max() + 1e-6), color=color, lw=2, marker="o", ms=3.5,
                label=f"{name}: peak to valley {p2v[key]:.1f} dB")
    for at in (0.47, 0.53):
        ax.axvline(at, color=INK_2, lw=0.9, linestyle=":")
    ax.set_ylim(-25, 2)
    ax.set_title("D. Two clicks 60 ms apart, 82-233 Hz", loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("level (dB re peak)")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "semantic_bridge_reassignment.png")


def figure_embedding(results):
    """A: glide direction plane on up/down chirps. B: invariance checks. C: family identification on real notes."""
    n, sr = SPEC.audio_length, SPEC.sample_rate
    t = torch.arange(n, dtype=torch.float32) / sr
    k = math.log(8.0)
    up = 0.5 * torch.sin(math.pi * t) ** 2 * torch.sin(2 * math.pi * 110.0 * (torch.exp(k * t) - 1) / k)
    planes = semantic_planes(encode(torch.cat([_stereo(up), _stereo(torch.flip(up, [0]))]), SPEC)).numpy()
    e3 = results["experiment_3_jtfs_chirp_and_modulation"]["up_vs_down_chirp"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1.1]})
    ax = axes[0]
    im = ax.imshow(np.concatenate([planes[0, 10, 25:80], planes[1, 10, 25:80]], axis=1), origin="lower",
                   aspect="auto", extent=[0, 2.4, 25, 80], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axvline(1.2, color=INK, lw=1)
    ax.text(0.6, 77, f"rising: mean {e3['mean_jtfs_tilt_up']:+.2f}", ha="center", fontsize=8.5)
    ax.text(1.8, 77, f"falling: mean {e3['mean_jtfs_tilt_down']:+.2f}", ha="center", fontsize=8.5)
    ax.set_title("A. Glide direction plane (jtfs_tilt)", loc="left")
    ax.set_xlabel("rising 110-880 Hz glide | the same, reversed (s)")
    ax.set_ylabel("semitone row")
    fig.colorbar(im, ax=ax, label="tilt")
    ax = axes[1]
    e4 = results["experiment_4_invariance_vs_sensitivity"]
    keys = [("shift_2p7ms_similarity", "2.7 ms\nshift"), ("shift_43ms_similarity", "43 ms\nshift"),
            ("phase_rotation_90deg_similarity", "90° phase\nrotation"), ("sine_vs_sawtooth_110hz", "sine vs\nsaw"),
            ("sine_vs_square_110hz", "sine vs\nsquare"), ("mono_vs_antiphase_stereo", "mono vs\nanti-phase")]
    xs = np.arange(len(keys))
    w = 0.38
    _bar_labels(ax, ax.bar(xs - w / 2, [e4[k]["raw_complex_carrier_cosine"] for k, _ in keys], w, color=ORANGE,
                           label="raw complex cells (110 Hz row)"))
    _bar_labels(ax, ax.bar(xs + w / 2, [e4[k]["semantic_embedding_cosine"] for k, _ in keys], w, color=BLUE,
                           label="semantic embedding"))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(xs, [label for _, label in keys])
    ax.set_ylim(-0.45, 1.15)
    ax.set_ylabel("cosine similarity to the reference note")
    ax.set_title("B. 110 Hz note vs. changed copies", loc="left")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[2]
    real = results.get("experiment_6_real_notes")
    if real is None:
        ax.text(0.5, 0.5, "run with note_dataset for this panel", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        emb = real["c_embeddings"]
        names = ["semantic embedding", "Wavelet2Vec", "mean spectrum"]
        xs = np.arange(len(names))
        _bar_labels(ax, ax.bar(xs - w / 2, [emb[k]["family_1nn"] for k in names], w, color=ORANGE, label="raw"))
        _bar_labels(ax, ax.bar(xs + w / 2, [emb[f"{k}, standardized"]["family_1nn"] for k in names], w, color=BLUE,
                               label="standardized per dimension"))
        ax.axhline(emb["chance_family_1nn"], color=INK_2, lw=0.9, linestyle=":")
        ax.text(len(names) - 0.5, emb["chance_family_1nn"] + 0.01, "chance", ha="right", fontsize=8, color=INK_2)
        ax.set_xticks(xs, [k.replace(" ", "\n", 1) if k != "Wavelet2Vec" else k for k in names])
        ax.set_ylim(0, 1)
        ax.set_ylabel("nearest neighbour is the same family")
        notes = real["notes"]
        ax.set_title(f"C. {notes['embedding_synth_families']} synth families, notes ≥ 7 semitones apart", loc="left")
        ax.grid(axis="y")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "semantic_bridge_embedding.png")


def generate_all_figures() -> list[Path]:
    if not RESULTS.exists():
        raise SystemExit("run experiments/semantic_bridge_experiments.py first")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [figure_overview(), figure_bass_coherence(results), figure_reassignment(results), figure_embedding(results)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    generate_all_figures()
