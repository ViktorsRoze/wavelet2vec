"""Self-contained mel-spectrogram baseline (numpy/scipy only).

This is the traditional representation wavelet2vec is compared against. Kept
dependency-light on purpose so the comparison needs no extra audio framework.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from experiments.dataset import PITCH_CLASS_NAMES


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """Triangular mel filterbank ``[n_mels, n_fft//2 + 1]``."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.searchsorted(fft_freqs, hz_points)

    filters = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        filters[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    del bins
    return filters


def log_mel_spectrogram(
    mono: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 64,
    f_min: float = 27.5,
    f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (log_mel ``[n_mels, frames]``, center_freqs_hz, frame_times_s)."""
    f_max = f_max if f_max is not None else 0.45 * sample_rate
    _, times, spectrum = stft(
        mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None
    )
    power = np.abs(spectrum) ** 2
    filters = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
    mel_power = filters @ power
    log_mel = np.log1p(mel_power)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    centers = mel_to_hz(mel_points[1:-1])
    return log_mel, centers, times


def mel_chroma(mono: np.ndarray, sample_rate: int, *, n_fft: int = 8192, hop: int = 2048) -> np.ndarray:
    """12-bin chroma from a linear power spectrogram folded by pitch class.

    A large FFT is used so low notes resolve — this is the mel/STFT-family
    baseline for pitch recovery (the comparison point for wavelet2vec's
    autocorrelation+chroma harmonic section).
    """
    _, _, spectrum = stft(
        mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None
    )
    power = (np.abs(spectrum) ** 2).mean(axis=1)
    freqs = np.linspace(0.0, sample_rate / 2.0, power.shape[0])
    mask = (freqs >= 55.0) & (freqs <= 5000.0)
    chroma = np.zeros(12)
    midi = 69.0 + 12.0 * np.log2(np.clip(freqs[mask], 1e-9, None) / 440.0)
    classes = np.round(midi).astype(int) % 12
    np.add.at(chroma, classes, power[mask])
    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def mel_summary_vector(log_mel: np.ndarray) -> np.ndarray:
    """Per-band mean+std — a magnitude-only analogue of wavelet2vec ``spectral``.

    The mel baseline embedding used for the clustering comparison (E1/E2).
    """
    mean = log_mel.mean(axis=1)
    std = log_mel.std(axis=1)
    vector = np.concatenate([mean, std])
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 1e-12 else vector.astype(np.float32)


def predicted_pitch_class(chroma: np.ndarray) -> int:
    return int(np.argmax(chroma))


def pitch_class_name(pitch_class: int) -> str:
    return PITCH_CLASS_NAMES[pitch_class % 12]
