"""M-4 spatial character + M-5 width designer."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import (
    apply_spatial_character,
    design_width,
    extract_spatial_character,
    spatial_matrix,
)

SR = 44100
N = SR  # 1 second


def _snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = np.sum((reference - test) ** 2)
    if noise == 0:
        return np.inf
    return 10.0 * np.log10(np.sum(reference**2) / noise)


def _stereo(correlated: float, seed: int = 0) -> np.ndarray:
    """Stereo noise: 1.0 = dual mono, 0.0 = independent L/R."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(N)
    own = rng.standard_normal((2, N))
    mix = np.sqrt(max(1.0 - correlated**2, 0.0))
    return np.stack([correlated * common + mix * own[0], correlated * common + mix * own[1]])


def _mid_bands(centers: np.ndarray) -> np.ndarray:
    return (centers > 1000.0) & (centers < 15000.0)


# --- M-4 extract -------------------------------------------------------------------
def test_extract_width_extremes():
    mono = _stereo(1.0)
    assert extract_spatial_character(mono, SR).width.max() < 0.05

    independent = _stereo(0.0)
    character = extract_spatial_character(independent, SR)
    mask = _mid_bands(character.centers_hz)
    assert np.abs(character.width[mask].mean() - 0.5) < 0.1

    x = np.random.default_rng(3).standard_normal(N)
    antiphase = np.stack([x, -x])
    assert extract_spatial_character(antiphase, SR).width.min() > 0.95


def test_extract_balance():
    rng = np.random.default_rng(4)
    left_only = np.stack([rng.standard_normal(N), 1e-8 * rng.standard_normal(N)])
    character = extract_spatial_character(left_only, SR)
    assert character.balance.mean() < -0.9


# --- M-4 apply ---------------------------------------------------------------------
def test_apply_narrows_and_preserves_mono():
    wide = _stereo(0.3, seed=1)
    reference = extract_spatial_character(_stereo(1.0, seed=2), SR)  # mono-like target
    out = apply_spatial_character(wide, SR, reference)
    measured = extract_spatial_character(out, SR)
    mask = _mid_bands(measured.centers_hz)
    assert measured.width[mask].mean() < 0.1
    # Mono downmix untouched (Side-only operation).
    assert _snr_db(wide.mean(axis=0), out.mean(axis=0)) > 120.0


def test_apply_amount_zero_is_identity():
    audio = _stereo(0.5, seed=5)
    out = apply_spatial_character(audio, SR, extract_spatial_character(_stereo(1.0), SR), 0.0)
    assert _snr_db(audio, out) > 120.0


def test_apply_moves_width_toward_reference():
    narrow = _stereo(0.95, seed=6)  # a little side energy to scale up
    wide_reference = extract_spatial_character(_stereo(0.0, seed=7), SR)
    before = extract_spatial_character(narrow, SR)
    after = extract_spatial_character(apply_spatial_character(narrow, SR, wide_reference), SR)
    mask = _mid_bands(before.centers_hz)
    assert after.width[mask].mean() > before.width[mask].mean() + 0.2
    assert np.abs(after.width[mask].mean() - wide_reference.width[mask].mean()) < 0.15


def test_apply_match_balance():
    rng = np.random.default_rng(8)
    centered = np.stack([rng.standard_normal(N), rng.standard_normal(N)])
    right_heavy = extract_spatial_character(
        np.stack([0.3 * rng.standard_normal(N), rng.standard_normal(N)]), SR
    )
    out = apply_spatial_character(centered, SR, right_heavy, match_balance=True)
    measured = extract_spatial_character(out, SR)
    mask = _mid_bands(measured.centers_hz)
    assert measured.balance[mask].mean() > 0.5


# --- M-5 design_width --------------------------------------------------------------
def test_design_width_creates_width_from_mono():
    mono = np.random.default_rng(9).standard_normal(N)
    out = design_width(mono, SR, 0.3, coherence_floor=0.0)
    assert out.shape == (2, N)
    measured = extract_spatial_character(out, SR)
    mask = _mid_bands(measured.centers_hz)
    assert np.abs(measured.width[mask].mean() - 0.3) < 0.08
    # Mono downmix is the original signal (antiphase construction).
    assert _snr_db(mono, out.mean(axis=0)) > 120.0
    # Physics: delay decorrelation drops coherence to ~1 - 2*width.
    matrix = spatial_matrix(out, SR)
    coherence = matrix["coherence"][0][_mid_bands(matrix["centers_hz"])]
    assert np.abs(coherence.mean() - 0.4) < 0.15


def test_design_width_coherence_floor_caps_width():
    mono = np.random.default_rng(10).standard_normal(N)
    out = design_width(mono, SR, 0.5, coherence_floor=0.6)  # cap: (1-0.6)/2 = 0.2
    measured = extract_spatial_character(out, SR)
    mask = _mid_bands(measured.centers_hz)
    assert measured.width[mask].mean() < 0.25
    matrix = spatial_matrix(out, SR)
    coherence = matrix["coherence"][0][_mid_bands(matrix["centers_hz"])]
    assert coherence.mean() > 0.5


def test_design_width_narrows_wide_input():
    wide = _stereo(0.0, seed=11)
    out = design_width(wide, SR, 0.05)
    measured = extract_spatial_character(out, SR)
    mask = _mid_bands(measured.centers_hz)
    assert measured.width[mask].mean() < 0.12
    assert _snr_db(wide.mean(axis=0), out.mean(axis=0)) > 120.0


def test_design_width_deterministic_and_per_band():
    mono = np.random.default_rng(12).standard_normal(N)
    per_band = np.full(48, 0.1)
    per_band[30:] = 0.4  # wide highs, tight lows
    first = design_width(mono, SR, per_band)
    second = design_width(mono, SR, per_band)
    assert np.array_equal(first, second)
    measured = extract_spatial_character(first, SR)
    assert measured.width[36:44].mean() > measured.width[16:24].mean() + 0.1
