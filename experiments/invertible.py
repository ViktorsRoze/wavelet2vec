"""Lossless transforms for the experiment suite.

The constant-Q complex wavelet frame was promoted to the library proper —
:mod:`wavelet2vec.frame` (roadmap C-1). This module re-exports it for the
experiment code and keeps the exact complex STFT, which remains an
experiment-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet2vec.frame import (  # noqa: F401  (re-exports)
    STORAGE_FORMATS,
    FrameSpec,
    WaveletMeta,
    _frame_windows,
    analyze,
    complex_wavelet,
    inverse_complex_wavelet,
    load_coefficients,
    quantize_complex,
    reconstruction_stats,
    save_coefficients,
    synthesize,
)


# --------------------------------------------------------------------------
# Exact complex STFT (COLA Hann, 75% overlap) — experiment baseline
# --------------------------------------------------------------------------
@dataclass
class STFTMeta:
    n_fft: int
    hop: int
    length: int
    pad: int


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)  # periodic (COLA)


def complex_stft(mono: np.ndarray, n_fft: int = 1024, hop: int | None = None) -> tuple[np.ndarray, STFTMeta]:
    """Full complex STFT, shape ``[n_fft//2+1, frames]``. Lossless with iSTFT."""
    mono = np.asarray(mono, dtype=np.float64)
    hop = hop or n_fft // 4
    pad = n_fft
    padded = np.concatenate([np.zeros(pad), mono, np.zeros(pad + n_fft)])
    window = _hann(n_fft)
    starts = range(0, len(padded) - n_fft + 1, hop)
    frames = np.stack([np.fft.rfft(padded[s : s + n_fft] * window) for s in starts], axis=1)
    return frames, STFTMeta(n_fft=n_fft, hop=hop, length=mono.shape[0], pad=pad)


def inverse_complex_stft(coeffs: np.ndarray, meta: STFTMeta) -> np.ndarray:
    """Exact inverse of :func:`complex_stft`."""
    window = _hann(meta.n_fft)
    total = meta.pad + meta.length + meta.pad + meta.n_fft
    out = np.zeros(total)
    norm = np.zeros(total)
    for index in range(coeffs.shape[1]):
        frame = np.fft.irfft(coeffs[:, index], n=meta.n_fft)
        start = index * meta.hop
        out[start : start + meta.n_fft] += frame * window
        norm[start : start + meta.n_fft] += window**2
    norm[norm < 1e-12] = 1.0
    recovered = out / norm
    return recovered[meta.pad : meta.pad + meta.length]
