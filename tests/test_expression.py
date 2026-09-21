"""G-11 deterministic: expression automation over the imagination axes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from wavelet2vec import AXIS_RANGES, AutomationCurve, render_expression
from wavelet2vec.datasets import axis_labels
from wavelet2vec.scenes import render_note
from wavelet2vec.slicing import slice_grid

SR = 22050


@pytest.fixture(scope="module")
def phrase():
    """8 identical dull pad notes on an 8th grid at 120 BPM."""
    b, a = butter(6, 3000 / (SR / 2), "low")
    note = filtfilt(b, a, render_note(52, 0.25, SR, instrument="saw"))
    loop = np.tile(note, 8).astype(np.float64)
    starts = slice_grid(loop.shape[0], SR, bpm=120.0, division=8)
    return loop, starts


def _bounds(loop, starts):
    return np.concatenate([starts, [loop.shape[0]]])


def test_rising_brightness_and_attack_evolve_per_note(phrase):
    loop, starts = phrase
    lanes = {"brightness": np.linspace(0, 1, len(starts)),
             "attack": np.linspace(0, 1, len(starts))}
    out = render_expression(loop, SR, starts, lanes).astype(np.float64)
    assert out.shape == loop.shape

    bounds = _bounds(loop, starts)
    brightness, crest = [], []
    for i in range(len(starts)):
        seg = out[bounds[i]: bounds[i + 1]]
        brightness.append(axis_labels(seg, SR)["brightness"])
        crest.append(np.abs(seg).max() / (np.sqrt(np.mean(seg**2)) + 1e-12))
    # First note is identity; later notes get progressively more air + attack.
    assert brightness[0] < 0.01
    assert brightness[7] > brightness[3] > brightness[0]
    assert crest[7] > crest[0] * 1.3


def test_zero_lanes_leave_interior_bit_exact(phrase):
    loop, starts = phrase
    out = render_expression(loop, SR, starts,
                            {"brightness": np.zeros(len(starts))}).astype(np.float64)
    bounds = _bounds(loop, starts)
    interior = slice(int(bounds[2]) + 300, int(bounds[3]) - 300)
    assert np.allclose(out[interior], loop[interior], atol=1e-6)


def test_per_note_values_hit_only_their_note(phrase):
    loop, starts = phrase
    values = np.zeros(len(starts))
    values[4] = 1.0  # only the fifth note
    out = render_expression(loop, SR, starts, {"brightness": values}).astype(np.float64)
    bounds = _bounds(loop, starts)
    for i in (1, 6):
        interior = slice(int(bounds[i]) + 300, int(bounds[i + 1]) - 300)
        assert np.allclose(out[interior], loop[interior], atol=1e-6)
    changed = slice(int(bounds[4]) + 300, int(bounds[5]) - 300)
    assert not np.allclose(out[changed], loop[changed], atol=1e-4)


def test_automation_curve_lane(phrase):
    loop, starts = phrase
    curve = AutomationCurve.from_points([(0.0, 0.0), (1.0, 1.0)])
    out = render_expression(loop, SR, starts, {"brightness": curve}).astype(np.float64)
    bounds = _bounds(loop, starts)
    first = axis_labels(out[bounds[0]: bounds[1]], SR)["brightness"]
    last = axis_labels(out[bounds[7]: bounds[8]], SR)["brightness"]
    assert last > first + 0.1


def test_tonality_and_noisiness_lanes(phrase):
    loop, starts = phrase
    noisy = render_expression(loop, SR, starts,
                              {"noisiness": np.ones(len(starts))}).astype(np.float64)
    tonal = render_expression(loop, SR, starts,
                              {"tonality": np.ones(len(starts))}).astype(np.float64)
    h_noisy = axis_labels(noisy, SR)["harmonicity"]
    h_tonal = axis_labels(tonal, SR)["harmonicity"]
    assert h_tonal > h_noisy  # the two lanes pull opposite directions


def test_character_lane_requires_character(phrase):
    loop, starts = phrase
    with pytest.raises(ValueError):
        render_expression(loop, SR, starts, {"character": np.ones(len(starts))})


def test_unknown_lane_raises(phrase):
    loop, starts = phrase
    with pytest.raises(ValueError):
        render_expression(loop, SR, starts, {"sparkle": np.ones(len(starts))})


def test_stereo_shape_and_ranges_override(phrase):
    loop, starts = phrase
    stereo = np.stack([loop, loop * 0.8])
    out = render_expression(stereo, SR, starts,
                            {"brightness": np.linspace(0, 1, len(starts))},
                            ranges={"brightness": (0.0, 1.0)})  # tamer ceiling
    assert out.shape == stereo.shape
    assert "brightness" in AXIS_RANGES  # defaults untouched by the override
    assert AXIS_RANGES["brightness"] == (0.0, 2.0)
