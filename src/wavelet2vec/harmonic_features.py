from __future__ import annotations

import numpy as np

HARMONIC_SCALAR_KEYS = ("harmonicity", "pitch_height", "spectral_flatness")


def _chroma(mono: np.ndarray, sample_rate: int, f_min: float = 27.5, f_max: float = 5000.0) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sample_rate)
    mask = (freqs >= f_min) & (freqs <= min(f_max, sample_rate / 2))
    chroma = np.zeros(12, dtype=np.float64)
    if mask.any():
        midi = 69.0 + 12.0 * np.log2(freqs[mask] / 440.0)
        pitch_classes = np.round(midi).astype(int) % 12
        np.add.at(chroma, pitch_classes, spectrum[mask])
    total = chroma.sum()
    if total > 0:
        chroma = chroma / total
    return np.sqrt(chroma).astype(np.float32)


def autocorrelation_pitch(
    mono: np.ndarray,
    sample_rate: int,
    pitch_min_hz: float = 50.0,
    pitch_max_hz: float = 1000.0,
) -> tuple[float, float]:
    """Returns (harmonicity in [0, 1], estimated f0 in Hz; 0.0 when unpitched)."""
    centered = mono - mono.mean()
    spectrum = np.fft.rfft(centered, n=2 * centered.shape[0])
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[: centered.shape[0]]
    if autocorr[0] <= 1e-12:
        return 0.0, 0.0
    autocorr = autocorr / autocorr[0]

    lag_min = max(int(sample_rate / pitch_max_hz), 1)
    lag_max = min(int(sample_rate / pitch_min_hz), centered.shape[0] - 1)
    if lag_max <= lag_min:
        return 0.0, 0.0

    window = autocorr[lag_min:lag_max]
    best = int(window.argmax())
    harmonicity = float(np.clip(window[best], 0.0, 1.0))
    f0 = sample_rate / (lag_min + best)
    return harmonicity, f0


def _spectral_flatness(band_energies: np.ndarray) -> float:
    positive = band_energies + 1e-10
    geometric = np.exp(np.log(positive).mean())
    return float(np.clip(geometric / positive.mean(), 0.0, 1.0))


def harmonic_features(mono: np.ndarray, scalogram: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pitch and tonality content: what the snippet means musically.

    Returns ``[12 + 3]``: a chroma vector (energy per pitch class, so two
    sounds playing the same note or chord agree regardless of octave or
    timbre), then harmonicity (tonal vs. noisy), pitch height, and spectral
    flatness (noisiness of the spectral envelope).
    """
    chroma = _chroma(mono, sample_rate)
    harmonicity, f0 = autocorrelation_pitch(mono, sample_rate)
    pitch_height = (
        float(np.clip(np.log2(f0 / 27.5) / 9.0, 0.0, 1.0)) if harmonicity > 0.1 and f0 > 0 else 0.0
    )
    flatness = _spectral_flatness(scalogram.mean(axis=1))
    scalars = np.array([harmonicity, pitch_height, flatness], dtype=np.float32)
    return np.concatenate([chroma, scalars])
