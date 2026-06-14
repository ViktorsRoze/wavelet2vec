from __future__ import annotations

import numpy as np

from wavelet2vec.filterbank import analytic_band_signals, constant_q_sigmas


def neutral_stereo_features(n_groups: int = 8) -> np.ndarray:
    """The stereo vector of a perfectly mono (centered, coherent) sound."""
    features = np.zeros(3 + 3 * n_groups, dtype=np.float32)
    features[3 + n_groups : 3 + 2 * n_groups] = 1.0  # full inter-channel coherence
    return features


def stereo_features(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    center_frequencies: np.ndarray,
    bandwidth_scale: float = 1.0,
    n_groups: int = 8,
) -> np.ndarray:
    """Spatial image of a stereo snippet, invariant to level and time shifts.

    All quantities compare the two channels against each other, so a common
    delay or gain cancels. Returns ``[3 + 3 * n_groups]``:

    - global width (side vs. mid energy), decorrelation, balance (pan)
    - per band group: width, inter-channel coherence, balance — capturing
      frequency-dependent image (e.g. wide hats over a mono kick drum)
    """
    eps = 1e-12
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_energy = float((mid**2).sum())
    side_energy = float((side**2).sum())
    width = side_energy / (mid_energy + side_energy + eps)

    left_energy = float((left**2).sum())
    right_energy = float((right**2).sum())
    correlation = float((left * right).sum() / (np.sqrt(left_energy * right_energy) + eps))
    decorrelation = float(np.clip(0.5 * (1.0 - correlation), 0.0, 1.0))
    balance = (right_energy - left_energy) / (left_energy + right_energy + eps)

    sigmas = constant_q_sigmas(center_frequencies, bandwidth_scale)
    left_bands = analytic_band_signals(left, sample_rate, center_frequencies, sigmas)
    right_bands = analytic_band_signals(right, sample_rate, center_frequencies, sigmas)

    left_band_energy = (np.abs(left_bands) ** 2).sum(axis=1)
    right_band_energy = (np.abs(right_bands) ** 2).sum(axis=1)
    cross = (left_bands * np.conj(right_bands)).sum(axis=1)
    mid_band_energy = (np.abs(0.5 * (left_bands + right_bands)) ** 2).sum(axis=1)
    side_band_energy = (np.abs(0.5 * (left_bands - right_bands)) ** 2).sum(axis=1)

    n_bands = len(center_frequencies)
    group_size = max(n_bands // n_groups, 1)
    group_width = np.zeros(n_groups, dtype=np.float32)
    group_coherence = np.zeros(n_groups, dtype=np.float32)
    group_balance = np.zeros(n_groups, dtype=np.float32)
    for group in range(n_groups):
        start = group * group_size
        stop = n_bands if group == n_groups - 1 else start + group_size
        mid_g = mid_band_energy[start:stop].sum()
        side_g = side_band_energy[start:stop].sum()
        left_g = left_band_energy[start:stop].sum()
        right_g = right_band_energy[start:stop].sum()
        group_width[group] = side_g / (mid_g + side_g + eps)
        group_coherence[group] = float(
            np.abs(cross[start:stop].sum()) / (np.sqrt(left_g * right_g) + eps)
        )
        group_balance[group] = (right_g - left_g) / (left_g + right_g + eps)

    return np.concatenate(
        [
            np.array([width, decorrelation, balance], dtype=np.float32),
            group_width,
            group_coherence,
            group_balance,
        ]
    )
