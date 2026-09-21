"""M-6 multichannel character modes: shared / per_channel / mid_side."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import (
    apply_character,
    blend,
    extract_character,
    perform,
    slice_grid,
)

SR = 22050
N = SR  # 1 second


def _tone_noise(low_gain: float, high_gain: float, seed: int) -> np.ndarray:
    """Mono signal with adjustable low/high balance (below/above ~1.5 kHz)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(N)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(N, 1.0 / SR)
    gains = np.where(freqs < 1500.0, low_gain, high_gain)
    shaped = np.fft.irfft(spectrum * gains, n=N)
    return shaped / np.abs(shaped).max() * 0.8


def _character_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = extract_character(a, SR), extract_character(b, SR)
    return float(np.abs(ca.log_band_energies - cb.log_band_energies).mean())


@pytest.fixture(scope="module")
def reference_character():
    return extract_character(_tone_noise(1.0, 1.0, seed=0), SR)


def test_all_modes_shape_and_amount_zero(reference_character):
    stereo = np.stack([_tone_noise(2.0, 0.5, 1), _tone_noise(0.5, 2.0, 2)])
    for mode in ("shared", "per_channel", "mid_side"):
        out = apply_character(stereo, SR, reference_character, 0.0, channel_mode=mode)
        assert out.shape == stereo.shape
        assert np.abs(out - stereo).max() < 1e-4, mode


def test_unknown_mode_and_mid_side_channel_count(reference_character):
    stereo = np.stack([_tone_noise(1, 1, 1), _tone_noise(1, 1, 2)])
    with pytest.raises(ValueError):
        apply_character(stereo, SR, reference_character, channel_mode="stereo")
    three = np.stack([_tone_noise(1, 1, s) for s in (1, 2, 3)])
    with pytest.raises(ValueError):
        apply_character(three, SR, reference_character, channel_mode="mid_side")
    # per_channel accepts any channel count.
    out = apply_character(three, SR, reference_character, channel_mode="per_channel")
    assert out.shape == three.shape


def test_per_channel_converges_channels_where_shared_cannot(reference_character):
    # L bass-heavy, R treble-heavy: shared gains keep their difference,
    # per-channel matching pulls both toward the same reference.
    stereo = np.stack([_tone_noise(2.0, 0.4, 3), _tone_noise(0.4, 2.0, 4)])
    distance_in = _character_distance(stereo[0], stereo[1])

    shared = apply_character(stereo, SR, reference_character, channel_mode="shared")
    per_channel = apply_character(stereo, SR, reference_character, channel_mode="per_channel")

    distance_shared = _character_distance(shared[0], shared[1])
    distance_per = _character_distance(per_channel[0], per_channel[1])
    assert distance_per < 0.7 * distance_in
    assert abs(distance_shared - distance_in) < 0.15 * distance_in


def test_mid_side_downmix_equals_morphed_mid(reference_character):
    stereo = np.stack([_tone_noise(2.0, 0.5, 5), _tone_noise(0.5, 2.0, 6)])
    mid = (stereo[0] + stereo[1]) / 2.0
    out = apply_character(stereo, SR, reference_character, channel_mode="mid_side")
    mid_alone = apply_character(mid, SR, reference_character)[0]
    # Image-preserving invariant: the mono downmix of the result is exactly
    # the mid processed as mono material.
    assert np.allclose(out.mean(axis=0), mid_alone, atol=1e-6)


def test_blend_and_perform_accept_channel_mode(reference_character):
    stereo = np.stack([_tone_noise(2.0, 0.5, 7), _tone_noise(0.5, 2.0, 8)])
    mixed, out_sr = blend(
        [stereo, _tone_noise(1, 1, 9)], [SR, SR], [0.6, 0.4], channel_mode="mid_side"
    )
    assert mixed.shape == stereo.shape and out_sr == SR

    from wavelet2vec import ADSR

    starts = slice_grid(N, SR, bpm=120.0, division=4)
    out = perform(
        stereo,
        SR,
        reference_character,
        starts,
        character_adsr=ADSR(attack=0.01, decay=0.05, sustain=0.7, release=0.05),
        channel_mode="mid_side",
    )
    assert out.shape == stereo.shape
    assert np.all(np.isfinite(out))
    assert np.abs(out - stereo.astype(np.float32)).max() > 1e-3  # it did something
