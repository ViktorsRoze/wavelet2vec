"""Spectral edit primitive: depth, selectivity, and localization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavelet2vec.frame import analyze
from wavelet2vec.spectral_edit import (
    attenuate_region,
    isolate_regions,
    phase_rotate_regions,
    phase_scramble_regions,
    process_regions,
    region_mask,
)

SR = 22050


def _band_rms(audio, f_lo, f_hi, t0, t1):
    coeffs, spec = analyze(audio, SR, n_bands=96)
    bands = (spec.centers >= f_lo) & (spec.centers <= f_hi)
    n0, n1 = int(t0 * SR), int(t1 * SR)
    return float(np.sqrt((np.abs(coeffs[bands, n0:n1]) ** 2).mean()))


def test_erase_is_deep_selective_and_localized():
    t = np.arange(SR) / SR
    mix = np.sin(2 * math.pi * 500 * t) + np.sin(2 * math.pi * 2000 * t)

    edited = attenuate_region(mix, SR, 0.30, 0.60, 1400.0, 2800.0, gain_db=-80.0)

    # Depth: the 2 kHz component inside the window drops massively.
    before = _band_rms(mix, 1700, 2300, 0.35, 0.55)
    after = _band_rms(edited, 1700, 2300, 0.35, 0.55)
    assert 20 * math.log10(after / before) < -40

    # Selectivity: the 500 Hz component in the same window barely moves.
    before_low = _band_rms(mix, 400, 650, 0.35, 0.55)
    after_low = _band_rms(edited, 400, 650, 0.35, 0.55)
    assert abs(20 * math.log10(after_low / before_low)) < 1.0

    # Localization: far outside the window (+ apron) the waveform is intact
    # to numerical silence (< -120 dBFS difference).
    far = slice(int(0.75 * SR), SR)
    assert float(np.abs(edited[far] - mix[far]).max()) < 1e-6
    head = slice(0, int(0.15 * SR))
    assert float(np.abs(edited[head] - mix[head]).max()) < 1e-6


def test_partial_attenuation_and_multichannel():
    t = np.arange(SR // 2) / SR
    tone = np.sin(2 * math.pi * 3000 * t)
    stereo = np.stack([tone, tone * 0.5])

    ducked = attenuate_region(stereo, SR, 0.1, 0.4, 2000.0, 4500.0, gain_db=-12.0)
    assert ducked.shape == stereo.shape
    ratio = _band_rms(ducked[0], 2500, 3600, 0.15, 0.35) / _band_rms(stereo[0], 2500, 3600, 0.15, 0.35)
    assert 20 * math.log10(ratio) == pytest.approx(-12.0, abs=2.5)


def _mix():
    t = np.arange(SR) / SR
    return np.sin(2 * math.pi * 500 * t) + np.sin(2 * math.pi * 2000 * t)


def test_isolate_regions_splits_exactly():
    mix = _mix()
    regions = [(0.30, 0.60, 1400.0, 2800.0)]
    inside, outside = isolate_regions(mix, SR, regions)
    # Linear synthesis: the split reconstructs the input.
    assert float(np.abs(inside + outside - mix).max()) < 1e-6
    # "Inside" holds the 2 kHz content in the window, little of the 500 Hz.
    assert _band_rms(inside, 1700, 2300, 0.35, 0.55) > 0.3
    assert _band_rms(inside, 400, 650, 0.35, 0.55) < 0.05


def test_process_regions_applies_effect_only_inside():
    mix = _mix()
    regions = [(0.30, 0.60, 1400.0, 2800.0)]

    processed = process_regions(mix, SR, regions, lambda x: x * 0.25)  # -12 dB duck
    before = _band_rms(mix, 1700, 2300, 0.40, 0.50)
    after = _band_rms(processed, 1700, 2300, 0.40, 0.50)
    assert 20 * math.log10(after / before) == pytest.approx(-12.0, abs=1.5)
    # Far outside the window the signal is untouched.
    far = slice(int(0.75 * SR), SR)
    assert float(np.abs(processed[far] - mix[far]).max()) < 1e-6

    with pytest.raises(ValueError):
        process_regions(mix, SR, regions, lambda x: x[:10])  # shape-changing effect


def test_phase_rotate_180_flips_region_content():
    mix = _mix()
    regions = [(0.30, 0.60, 1400.0, 2800.0)]
    inside, _outside = isolate_regions(mix, SR, regions)
    flipped = phase_rotate_regions(mix, SR, regions, 180.0)
    # 180° on the isolated band == subtract twice the inside content.
    expected = mix - 2.0 * inside
    assert float(np.abs(flipped - expected).max()) < 1e-3
    far = slice(int(0.75 * SR), SR)
    assert float(np.abs(flipped[far] - mix[far]).max()) < 1e-6


def test_phase_rotate_360_is_identity():
    mix = _mix()
    regions = [(0.20, 0.70, 300.0, 3000.0)]
    rotated = phase_rotate_regions(mix, SR, regions, 360.0)
    assert float(np.abs(rotated - mix).max()) < 1e-6


def test_phase_scramble_keeps_magnitude_kills_coherence():
    mix = _mix()
    regions = [(0.20, 0.70, 1400.0, 2800.0)]
    scrambled = phase_scramble_regions(mix, SR, regions, seed=1)
    # Magnitude in-band roughly preserved (randomizing phase across the
    # redundant, overlapping-band frame costs a few dB to synthesis
    # cancellation — inherent, and fine for a creative smear); the
    # waveform is genuinely, audibly changed.
    before = _band_rms(mix, 1700, 2300, 0.30, 0.60)
    after = _band_rms(scrambled, 1700, 2300, 0.30, 0.60)
    assert -10.0 < 20 * math.log10(after / before) < 3.0
    assert float(np.abs(scrambled - mix).max()) > 0.1
    # Localized to the region + wavelet apron. A full ±π scramble perturbs
    # more than a gentle attenuation, so its apron is a touch wider: 150 ms
    # past the window it is already below -60 dBFS.
    far = slice(int(0.85 * SR), SR)
    assert float(np.abs(scrambled[far] - mix[far]).max()) < 1e-3


def test_region_mask_union_and_bounds():
    _coeffs, spec = analyze(_mix(), SR, n_bands=96)
    mask = region_mask(spec.centers, SR, SR,
                       [(0.1, 0.3, 500.0, 1000.0), (0.5, 0.7, 2000.0, 4000.0)])
    assert mask.shape == (96, SR)
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    # Two disjoint blobs: energy in both windows, none between them in time.
    assert mask[:, int(0.2 * SR)].max() > 0.9
    assert mask[:, int(0.6 * SR)].max() > 0.9
    assert mask[:, int(0.4 * SR)].max() < 0.05
