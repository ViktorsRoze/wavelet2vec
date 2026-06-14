"""Generate the committed README figures from a synthetic (non-copyrighted) signal.

    python -m experiments.make_docs_figures

Writes JPGs into docs/figures/ — small, display-only, safe to commit (no
sample-library audio is used).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import viz
from experiments.invertible import complex_wavelet, inverse_complex_wavelet, reconstruction_stats

# Real examples from the user's own library. Relative to --audio-dir.
EXAMPLE_REL = "fx_at_120BPM/sw2_fx120_ghostrider.wav"   # FX sweep, evolving texture
EXAMPLE_SECONDS = 4.0
KICK_REL = "kicks/AA viktor rose A1 2.wav"  # kick: low body (A1) + bright attack
KICK_SECONDS = 0.42


def synthetic_signal(sample_rate: int = 44100, seconds: float = 1.0) -> np.ndarray:
    """A harmonic tone with vibrato plus a sharp transient — exercises
    magnitude (harmonics), phase (transient), and frequency (vibrato/pitch)."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    vibrato = 220.0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(vibrato) / sample_rate
    tone = sum(np.sin(k * phase) / k for k in range(1, 8))
    tone[5000:5060] += 2.0  # a click, so phase/transient structure is visible
    return (tone / np.abs(tone).max()).astype(np.float64)


def _roundtrip(mono, sr, path, title):
    coeffs, meta = complex_wavelet(mono, sr, n_bands=128)
    reconstructed = inverse_complex_wavelet(coeffs, meta)
    snr = reconstruction_stats(mono, reconstructed)["snr_db"]
    viz.roundtrip_figure(
        mono, sr, coeffs, reconstructed, path,
        title=title, transform="constant-Q complex wavelet",
        freq_axis=meta.centers, snr_db=snr,
    )
    print(f"wrote {path} (round trip ~{snr:.0f} dB)")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the committed README figures.")
    parser.add_argument("--audio-dir", default="../soundgen/example_audio",
                        help="Library for the real-example figure (the user's own audio).")
    args = parser.parse_args()

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    sr = 44100

    # Synthetic showcase + the magnitude/phase/frequency anatomy table.
    mono = synthetic_signal(sr)
    meta = _roundtrip(mono, sr, out / "roundtrip_showcase.jpg",
                      "synthetic tone (harmonics + vibrato + transient)")
    viz.anatomy_table_figure(mono, sr, out / "coefficient_anatomy.jpg", n_bands=128)
    print(f"wrote {out/'coefficient_anatomy.jpg'}")

    # Real examples from the user's own library.
    from experiments.dataset import Track, load_canonical

    def _load(rel, seconds):
        path = Path(args.audio_dir) / rel
        if not path.exists():
            print(f"(skipped; not found at {path})")
            return None, None
        track = Track(path=path, folder=path.parent.name, duration=0.0,
                      native_sr=sr, channels=2, subtype="PCM_24")
        clip = load_canonical(track, sr).mono[: int(seconds * sr)]
        return clip / (np.abs(clip).max() + 1e-12), path

    # Hero pipeline figure on a kick (low body + bright attack).
    kick, kpath = _load(KICK_REL, KICK_SECONDS)
    if kick is not None:
        viz.pipeline_figure(kick, sr, out / "pipeline_kick.jpg",
                            title=f"kick drum — {kpath.name}")
        print(f"wrote {out/'pipeline_kick.jpg'}")

    # FX sweep round trip (evolving texture).
    fx, fxpath = _load(EXAMPLE_REL, EXAMPLE_SECONDS)
    if fx is not None:
        _roundtrip(fx, sr, out / "roundtrip_fx_example.jpg",
                   f"real FX one-shot — {fxpath.name} (first {EXAMPLE_SECONDS:.0f} s)")


if __name__ == "__main__":
    main()
