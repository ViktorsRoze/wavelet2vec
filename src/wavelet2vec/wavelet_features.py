from __future__ import annotations

import numpy as np


def spectral_features(scalogram: np.ndarray) -> np.ndarray:
    """Time-averaged wavelet band statistics describing spectral color.

    Returns ``[2 * n_bands]``: per-band mean log energy (the multi-scale
    spectral envelope) followed by per-band temporal standard deviation
    (how unsteady each band is over the snippet).
    """
    log_mag = np.log1p(scalogram)
    band_mean = log_mag.mean(axis=1)
    band_std = log_mag.std(axis=1)
    return np.concatenate([band_mean, band_std]).astype(np.float32)


def modulation_features(
    scalogram: np.ndarray,
    sample_rate: int,
    envelope_rate: int = 256,
    n_band_groups: int = 8,
    n_modulation_bands: int = 8,
    mod_f_min: float = 0.5,
    mod_f_max: float = 64.0,
) -> np.ndarray:
    """Second-order (scattering-style) modulation spectrum of band envelopes.

    Band envelopes are downsampled, level-normalized, and Fourier transformed;
    the modulation energy is pooled over log-spaced modulation-frequency bands
    and over groups of wavelet bands. This captures texture qualities such as
    roughness, tremolo, grain density, and flutter that a plain spectrum misses.

    Returns ``[n_band_groups * n_modulation_bands]``.
    """
    n_bands, n_samples = scalogram.shape
    hop = max(int(sample_rate // envelope_rate), 1)
    n_frames = n_samples // hop
    output = np.zeros((n_band_groups, n_modulation_bands), dtype=np.float32)
    if n_frames < 4:
        return output.reshape(-1)

    envelopes = scalogram[:, : n_frames * hop].reshape(n_bands, n_frames, hop).mean(axis=2)
    # Normalize by the global envelope level (not per band): per-band
    # normalization would divide near-silent bands by a near-zero mean and
    # amplify numerical noise into spurious modulation energy.
    envelopes = envelopes / (envelopes.mean() + 1e-12)

    mod_spectrum = np.abs(np.fft.rfft(envelopes, axis=1))
    mod_freqs = np.fft.rfftfreq(n_frames, d=hop / sample_rate)
    edges = np.geomspace(mod_f_min, mod_f_max, n_modulation_bands + 1)

    group_size = max(n_bands // n_band_groups, 1)
    for group_index in range(n_band_groups):
        start = group_index * group_size
        stop = n_bands if group_index == n_band_groups - 1 else start + group_size
        group_spectrum = mod_spectrum[start:stop].mean(axis=0)
        for mod_index in range(n_modulation_bands):
            mask = (mod_freqs >= edges[mod_index]) & (mod_freqs < edges[mod_index + 1])
            if mask.any():
                output[group_index, mod_index] = np.log1p(group_spectrum[mask].mean())
    return output.reshape(-1)
