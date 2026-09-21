"""H-3 structure segmentation + H-4 loop-point detection."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import find_loop_points, render_loop, segment_structure

SR = 22050


def _tone(freq, seconds, sr=SR):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * sr)) / sr)


# --- H-3 --------------------------------------------------------------------------
def test_segment_structure_three_sections():
    rng = np.random.default_rng(0)
    a = _tone(220.0, 3.0) * 0.8
    noise = rng.standard_normal(int(3.0 * SR)) * 0.3
    b = _tone(880.0, 3.0) * (0.4 + 0.4 * np.sign(np.sin(2 * np.pi * 4 * np.arange(int(3.0 * SR)) / SR)))
    segments = segment_structure(np.concatenate([a, noise, b]), SR)

    assert len(segments.boundaries_s) == 2
    assert segments.boundaries_s[0] == pytest.approx(3.0, abs=0.75)
    assert segments.boundaries_s[1] == pytest.approx(6.0, abs=0.75)
    assert len(segments.sections) == 3
    assert segments.sections[0][0] == 0.0
    assert segments.sections[-1][1] == pytest.approx(9.0, abs=0.01)


def test_segment_structure_uniform_clip_has_no_boundaries():
    segments = segment_structure(_tone(330.0, 6.0), SR)
    assert segments.boundaries_s == []
    assert segments.sections == [(0.0, pytest.approx(6.0, abs=0.01))]


def test_segment_structure_short_clip():
    segments = segment_structure(_tone(330.0, 0.8), SR)
    assert segments.boundaries_s == []
    assert len(segments.sections) == 1


# --- H-4 --------------------------------------------------------------------------
def _pattern_loop(pattern_s=0.75, repeats=6, seed=1):
    """A percussive pattern tiled exactly; ground-truth loop = pattern_s."""
    rng = np.random.default_rng(seed)
    n = int(pattern_s * SR)
    pattern = np.zeros(n)
    for position in (0.0, 0.25, 0.5):
        k = int(position * n)
        burst = rng.standard_normal(600) * np.exp(-np.arange(600) / 80.0)
        pattern[k : k + 600] += burst
    tone = np.sin(2 * np.pi * 165.0 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.3 * SR))
    pattern += 0.6 * tone
    return np.tile(pattern / np.abs(pattern).max(), repeats)


def test_find_loop_points_recovers_pattern_length():
    audio = _pattern_loop()
    loop = find_loop_points(audio, SR, min_loop_s=0.3)
    assert loop.loop_s == pytest.approx(0.75, abs=0.02)
    assert loop.periodicity > 0.9
    assert loop.seam_error < 0.15
    assert 0 <= loop.start_sample < loop.end_sample <= audio.shape[0]


def test_find_loop_points_seam_is_continuous():
    audio = _pattern_loop(pattern_s=0.6, repeats=8, seed=2)
    loop = find_loop_points(audio, SR, min_loop_s=0.3)
    # Concatenating the loop with itself must not jump at the seam.
    cut = audio[loop.start_sample : loop.end_sample]
    doubled = np.concatenate([cut, cut])
    seam = cut.shape[0]
    step = np.abs(np.diff(doubled[seam - 2 : seam + 2])).max()
    typical = np.abs(np.diff(audio)).mean() + 5 * np.abs(np.diff(audio)).std()
    assert step < typical


def test_find_loop_points_min_length_respected():
    audio = _pattern_loop(pattern_s=0.5, repeats=8, seed=3)
    loop = find_loop_points(audio, SR, min_loop_s=0.8)  # fundamental 0.5 is too short
    assert loop.loop_s >= 0.8 - 0.02
    assert loop.loop_s == pytest.approx(1.0, abs=0.05)  # falls to the 2x multiple


def test_find_loop_points_too_short_raises():
    with pytest.raises(ValueError):
        find_loop_points(_tone(220.0, 0.5), SR, min_loop_s=1.0)


def test_render_loop_length_and_seam():
    audio = _pattern_loop(pattern_s=0.75, repeats=6, seed=4)
    loop = find_loop_points(audio, SR, min_loop_s=0.3)
    rendered = render_loop(audio, SR, loop, repeats=3)
    assert rendered.shape[0] == 3 * (loop.end_sample - loop.start_sample)
    # No jump at the blended seams.
    seam = loop.end_sample - loop.start_sample
    step = np.abs(np.diff(rendered[seam - 4 : seam + 4])).max()
    typical = np.abs(np.diff(audio)).mean() + 5 * np.abs(np.diff(audio)).std()
    assert step < typical
    # Stereo path keeps shape.
    stereo = np.stack([audio, audio * 0.7])
    rendered_stereo = render_loop(stereo, SR, loop, repeats=2)
    assert rendered_stereo.shape == (2, 2 * (loop.end_sample - loop.start_sample))
