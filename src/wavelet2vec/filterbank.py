from __future__ import annotations

import numpy as np


def log_spaced_frequencies(f_min: float, f_max: float, n_bands: int) -> np.ndarray:
    """Geometrically spaced center frequencies (constant-Q layout)."""
    return np.geomspace(f_min, f_max, n_bands)


def analytic_envelope(signal: np.ndarray, smooth_samples: int = 1) -> np.ndarray:
    """Magnitude of the analytic signal computed via FFT (Hilbert envelope)."""
    n_samples = signal.shape[-1]
    spectrum = np.fft.rfft(signal)
    full = np.zeros(n_samples, dtype=np.complex128)
    full[: spectrum.shape[0]] = spectrum * 2.0
    full[0] = spectrum[0]
    envelope = np.abs(np.fft.ifft(full))
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope = np.convolve(envelope, kernel, mode="same")
    return envelope.astype(np.float32)


def analytic_band_signals(
    signal: np.ndarray,
    sample_rate: int,
    center_frequencies: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Complex analytic band signals via Gaussian (Morlet) filters in frequency.

    Only the positive half of the spectrum is kept, so the inverse FFT yields
    the analytic signal per band: its magnitude is the band envelope and its
    angle is the band phase. Returns ``[n_bands, n_samples]`` complex64.
    """
    n_samples = signal.shape[-1]
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    bands = np.empty((len(center_frequencies), n_samples), dtype=np.complex64)
    full = np.zeros(n_samples, dtype=np.complex128)
    half = freqs.shape[0]
    for index, (center, sigma) in enumerate(zip(center_frequencies, sigmas)):
        window = np.exp(-0.5 * ((freqs - center) / max(sigma, 1e-6)) ** 2)
        full[:] = 0.0
        full[:half] = spectrum * window * 2.0
        bands[index] = np.fft.ifft(full).astype(np.complex64)
    return bands


def constant_q_sigmas(center_frequencies: np.ndarray, bandwidth_scale: float = 1.0) -> np.ndarray:
    if len(center_frequencies) > 1:
        ratio = float(center_frequencies[1] / center_frequencies[0])
    else:
        ratio = 2.0 ** 0.125
    return center_frequencies * (ratio - 1.0) * bandwidth_scale


def morlet_scalogram(
    signal: np.ndarray,
    sample_rate: int,
    center_frequencies: np.ndarray,
    bandwidth_scale: float = 1.0,
    return_complex: bool = False,
) -> np.ndarray:
    """Constant-Q complex Morlet scalogram via frequency-domain filtering.

    Returns ``[n_bands, n_samples]``: analytic band signals (complex64) when
    ``return_complex`` is set, otherwise their magnitudes (float32).
    """
    sigmas = constant_q_sigmas(center_frequencies, bandwidth_scale)
    bands = analytic_band_signals(signal, sample_rate, center_frequencies, sigmas)
    if return_complex:
        return bands
    return np.abs(bands).astype(np.float32)
