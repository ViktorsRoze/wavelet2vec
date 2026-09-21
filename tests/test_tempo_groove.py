"""R-6 tempo estimation + R-8 groove transfer + R-7 rhythm embedding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavelet2vec import (
    RHYTHM_DIM,
    analyze_rhythm,
    apply_groove,
    cosine_similarity,
    estimate_tempo,
    groove_offsets,
    profile_embedding,
    rhythm_embedding,
    swing_offsets,
)

SR = 22050


def _drum_loop(bpm=120.0, bars=4, swing_fraction=0.5):
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars)
    t = np.arange(n) / SR
    x = np.zeros(n)
    rng = np.random.default_rng(0)
    for bar in range(bars):
        for quarter in range(4):
            beat_start = (bar * 4 + quarter) * beat
            k = int(beat_start * SR)
            length = int(0.09 * SR)
            x[k : k + length] += np.sin(2 * math.pi * 60 * t[:length]) * np.exp(-np.arange(length) / SR * 40)
            for frac in (0.0, swing_fraction):
                h = int((beat_start + frac * beat) * SR)
                hat_len = int(0.03 * SR)
                noise = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / SR * 120)
                x[h : h + hat_len] += np.diff(noise, prepend=0.0) * 0.5
    return x / np.abs(x).max()


# --- R-6 tempo -------------------------------------------------------------------
def test_estimate_tempo_synthetic():
    for bpm in (100.0, 120.0, 140.0):
        estimate, confidence = estimate_tempo(_drum_loop(bpm=bpm), SR)
        assert estimate == pytest.approx(bpm, abs=2.0), f"expected {bpm}, got {estimate}"
        assert confidence > 0.2


def test_estimate_tempo_swung_loop_still_correct():
    estimate, _confidence = estimate_tempo(_drum_loop(bpm=120.0, swing_fraction=2 / 3), SR)
    assert estimate == pytest.approx(120.0, abs=2.0)


# --- R-8 groove ---------------------------------------------------------------------
def test_swing_offsets_values():
    offsets = swing_offsets(0.5, bpm=120.0)
    assert np.allclose(offsets, 0.0)
    offsets = swing_offsets(2 / 3, bpm=120.0)
    beat_ms = 500.0
    assert offsets[2] == pytest.approx((2 / 3 - 0.5) * beat_ms)
    assert offsets[0] == 0.0 and offsets[4] == 0.0  # on-beats untouched


def test_apply_groove_strength_zero_is_identity():
    loop = _drum_loop()
    out = apply_groove(loop, SR, 120.0, swing_offsets(2 / 3, 120.0), strength=0.0)
    assert np.array_equal(out, loop)


def test_apply_swing_changes_measured_swing():
    straight = _drum_loop(bpm=120.0, swing_fraction=0.5)
    swung = apply_groove(straight, SR, 120.0, swing_offsets(2 / 3, 120.0))
    assert swung.shape == straight.shape
    profile = analyze_rhythm(swung, SR, bpm=120.0)
    assert profile.swing_percent["high"] == pytest.approx(66.7, abs=5.0)
    # And the original reads straight.
    baseline = analyze_rhythm(straight, SR, bpm=120.0)
    assert baseline.swing_percent["high"] == pytest.approx(50.0, abs=4.0)


def test_groove_transfer_from_analyzed_loop():
    source = analyze_rhythm(_drum_loop(bpm=120.0, swing_fraction=2 / 3), SR, bpm=120.0)
    target = _drum_loop(bpm=120.0, swing_fraction=0.5)
    grooved = apply_groove(target, SR, 120.0, groove_offsets(source))
    measured = analyze_rhythm(grooved, SR, bpm=120.0)
    # The straight loop inherits most of the source's swing.
    assert measured.swing_percent["high"] is not None
    assert measured.swing_percent["high"] > 58.0


def test_apply_groove_multichannel_shape():
    loop = _drum_loop(bars=2)
    stereo = np.stack([loop, loop * 0.8])
    out = apply_groove(stereo, SR, 120.0, swing_offsets(0.6, 120.0))
    assert out.shape == stereo.shape


# --- R-7 rhythm embedding ---------------------------------------------------------
def _hat_loop(bpm=120.0, bars=4):
    """A very different groove: no kick, hats on every 16th."""
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars)
    x = np.zeros(n)
    rng = np.random.default_rng(1)
    for slot in range(bars * 16):
        k = int(slot * beat / 4 * SR)
        hat_len = int(0.03 * SR)
        noise = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / SR * 120)
        x[k : k + hat_len] += np.diff(noise, prepend=0.0)
    return x / np.abs(x).max()


def test_rhythm_embedding_shape_and_finite():
    vector = rhythm_embedding(_drum_loop(), SR, bpm=120.0)
    assert vector.shape == (RHYTHM_DIM,)
    assert np.all(np.isfinite(vector))
    # Estimated-BPM path and silence both stay finite.
    assert np.all(np.isfinite(rhythm_embedding(_drum_loop(), SR)))
    assert np.all(np.isfinite(rhythm_embedding(np.zeros(SR), SR, bpm=120.0)))


def test_rhythm_embedding_matches_profile_embedding():
    loop = _drum_loop()
    direct = rhythm_embedding(loop, SR, bpm=120.0)
    via_profile = profile_embedding(analyze_rhythm(loop, SR, bpm=120.0))
    assert np.array_equal(direct, via_profile)


def test_rhythm_embedding_tempo_invariant_groove_search():
    same_slow = rhythm_embedding(_drum_loop(bpm=100.0), SR, bpm=100.0)
    same_fast = rhythm_embedding(_drum_loop(bpm=140.0), SR, bpm=140.0)
    other = rhythm_embedding(_hat_loop(bpm=100.0), SR, bpm=100.0)

    cross_tempo = cosine_similarity(same_slow, same_fast)
    cross_groove = cosine_similarity(same_slow, other)
    assert cross_tempo > 0.9  # same groove, 100 vs 140 BPM
    assert cross_tempo > cross_groove + 0.1  # groove identity beats tempo identity


def test_rhythm_embedding_hears_swing():
    straight = rhythm_embedding(_drum_loop(swing_fraction=0.5), SR, bpm=120.0)
    swung = rhythm_embedding(_drum_loop(swing_fraction=2 / 3), SR, bpm=120.0)
    reswung = rhythm_embedding(_drum_loop(swing_fraction=2 / 3), SR, bpm=120.0)
    assert cosine_similarity(swung, reswung) == pytest.approx(1.0)
    assert cosine_similarity(straight, swung) < 0.95