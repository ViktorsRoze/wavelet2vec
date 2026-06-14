from __future__ import annotations

import numpy as np

from wavelet2vec.filterbank import analytic_band_signals
from wavelet2vec.harmonic_features import autocorrelation_pitch


def instantaneous_frequency_features(
    complex_scalogram: np.ndarray,
    center_frequencies: np.ndarray,
    sample_rate: int,
    n_groups: int = 8,
) -> np.ndarray:
    """Energy-weighted instantaneous-frequency deviation statistics per band group.

    The instantaneous frequency (time derivative of analytic phase) is
    time-shift invariant, unlike raw phase. Its deviation from the band center
    separates steady tones (near zero) from vibrato (oscillating) and noise
    (large, erratic). Returns ``[2 * n_groups]``: mean absolute deviation and
    deviation spread per group, both relative to the band center frequency.
    """
    phase = np.unwrap(np.angle(complex_scalogram), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate / (2.0 * np.pi)
    weights = np.abs(complex_scalogram[:, :-1]).astype(np.float64) ** 2
    centers = center_frequencies[:, None]
    deviation = np.clip((inst_freq - centers) / centers, -1.0, 1.0)

    band_weight = weights.sum(axis=1) + 1e-12
    band_mad = (weights * np.abs(deviation)).sum(axis=1) / band_weight
    band_mean = (weights * deviation).sum(axis=1) / band_weight
    band_std = np.sqrt(
        (weights * (deviation - band_mean[:, None]) ** 2).sum(axis=1) / band_weight
    )

    n_bands = complex_scalogram.shape[0]
    group_size = max(n_bands // n_groups, 1)
    grouped = np.zeros(2 * n_groups, dtype=np.float32)
    for group in range(n_groups):
        start = group * group_size
        stop = n_bands if group == n_groups - 1 else start + group_size
        group_energy = band_weight[start:stop].sum() + 1e-12
        grouped[group] = float((band_weight[start:stop] * band_mad[start:stop]).sum() / group_energy)
        grouped[n_groups + group] = float(
            (band_weight[start:stop] * band_std[start:stop]).sum() / group_energy
        )
    return grouped


def onset_phase_coherence(complex_scalogram: np.ndarray, min_active_bands: int = 4) -> np.ndarray:
    """Cross-band phase alignment over time.

    At a genuine transient, all wavelet bands lock phase at the attack instant
    (the classical phase-deviation onset detector), while noise and sustained
    tones show no broadband alignment. This separates a true percussive attack
    from a mere loudness bump. Returns ``[max, mean, fraction above 0.5]``.
    """
    magnitude = np.abs(complex_scalogram)
    phasors = complex_scalogram / (magnitude + 1e-12)
    active = magnitude > 0.05 * magnitude.max()
    n_active = active.sum(axis=0)
    coherence = np.abs((phasors * active).sum(axis=0)) / np.maximum(n_active, 1)
    coherence[n_active < min_active_bands] = 0.0
    return np.array(
        [coherence.max(), coherence.mean(), float((coherence > 0.5).mean())],
        dtype=np.float32,
    )


def harmonic_phase_signature(
    mono: np.ndarray,
    sample_rate: int,
    n_harmonics: int = 8,
    min_harmonicity: float = 0.15,
) -> np.ndarray:
    """Pitch-synchronous relative phase of harmonics: the waveshape descriptor.

    For a pitched sound with fundamental f0, the quantity ``phi_k - k * phi_1``
    (phase of harmonic k minus k times the fundamental phase) is exactly
    time-shift invariant, because a delay tau rotates phi_k by ``-2*pi*k*f0*tau``
    and the combination cancels. This is the relative-phase profile of one
    waveform period — it distinguishes sounds with identical magnitude spectra
    but different waveshapes (e.g. a saw versus its phase-scrambled twin),
    which magnitude-only representations cannot. Unpitched input yields zeros.

    Returns ``[3 * (n_harmonics - 1)]``: per harmonic k >= 2, the real part,
    imaginary part, and stability of the mean coupling phasor, each scaled by
    that harmonic's amplitude share.
    """
    output = np.zeros(3 * (n_harmonics - 1), dtype=np.float32)
    harmonicity, f0 = autocorrelation_pitch(mono, sample_rate)
    if harmonicity < min_harmonicity or f0 <= 0:
        return output

    max_k = min(n_harmonics, int(0.45 * sample_rate / f0))
    if max_k < 2:
        return output

    frequencies = f0 * np.arange(1, max_k + 1)
    sigmas = np.full(max_k, 0.25 * f0)
    bands = analytic_band_signals(mono, sample_rate, frequencies, sigmas)
    magnitudes = np.abs(bands)
    phases = np.angle(bands)

    amplitude = magnitudes.mean(axis=1)
    share = amplitude / (amplitude.sum() + 1e-12)
    fundamental_phase = phases[0]
    fundamental_magnitude = magnitudes[0]
    for k in range(2, max_k + 1):
        weight = magnitudes[k - 1] * fundamental_magnitude
        phasor = (weight * np.exp(1j * (phases[k - 1] - k * fundamental_phase))).sum()
        phasor = phasor / (weight.sum() + 1e-12)
        index = 3 * (k - 2)
        scale = float(share[k - 1])
        output[index] = scale * float(phasor.real)
        output[index + 1] = scale * float(phasor.imag)
        output[index + 2] = scale * float(np.abs(phasor))
    return output


def phase_features(
    mono: np.ndarray,
    complex_scalogram: np.ndarray,
    center_frequencies: np.ndarray,
    sample_rate: int,
    n_groups: int = 8,
    n_harmonics: int = 8,
) -> np.ndarray:
    """Shift-invariant phase information: what raw phase values cannot give.

    Returns ``[2 * n_groups + 3 + 3 * (n_harmonics - 1)]``.
    """
    return np.concatenate(
        [
            instantaneous_frequency_features(
                complex_scalogram, center_frequencies, sample_rate, n_groups=n_groups
            ),
            onset_phase_coherence(complex_scalogram),
            harmonic_phase_signature(mono, sample_rate, n_harmonics=n_harmonics),
        ]
    )
