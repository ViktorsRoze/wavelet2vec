"""Roadmap groundwork tests: frame API (C-1), HPSS (P-1), transient designer
(G-7), rhythm analysis (R-1..R-5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavelet2vec import (
    analyze,
    analyze_rhythm,
    hpss,
    reconstruction_stats,
    shape_transients,
    synthesize,
    transient_split,
)

SR = 22050


def _tone(freq=220.0, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * math.pi * freq * t)


def _clicks(seconds=1.0, every=0.25):
    x = np.zeros(int(SR * seconds))
    for k in range(int(seconds / every)):
        p = int(k * every * SR)
        x[p : p + 40] = np.hanning(40)
    return x


# --- C-1: frame API -------------------------------------------------------------
def test_frame_round_trip_mono_and_multichannel():
    mono = _tone()
    coeffs, spec = analyze(mono, SR, n_bands=96)
    assert coeffs.shape == (96, mono.shape[0])
    assert reconstruction_stats(mono, synthesize(coeffs, spec))["snr_db"] > 250

    stereo = np.stack([mono, _clicks()])
    coeffs2, spec2 = analyze(stereo, SR, n_bands=64)
    assert coeffs2.shape == (2, 64, mono.shape[0])
    out = synthesize(coeffs2, spec2)
    assert out.shape == stereo.shape
    assert reconstruction_stats(stereo, out)["snr_db"] > 250


def test_frame_compat_aliases_still_importable():
    from experiments.invertible import WaveletMeta, complex_wavelet, inverse_complex_wavelet

    x = _tone(seconds=0.3)
    coeffs, meta = complex_wavelet(x, SR, n_bands=32)
    assert isinstance(meta.to_dict(), dict)
    rebuilt = WaveletMeta.from_dict(meta.to_dict())
    assert reconstruction_stats(x, inverse_complex_wavelet(coeffs, rebuilt))["snr_db"] > 250


# --- P-1: HPSS --------------------------------------------------------------------
def test_hpss_sum_is_exact_and_layers_make_sense():
    mix = _tone() * 0.7 + _clicks() * 0.8
    harmonic, percussive = hpss(mix, SR)
    # Exactness: complementary masks + linear synthesis.
    assert reconstruction_stats(mix, harmonic + percussive)["snr_db"] > 200

    # A pure tone should land mostly in the harmonic layer, clicks mostly percussive.
    tone_h, tone_p = hpss(_tone(), SR)
    assert float(np.sum(tone_h**2)) > 4 * float(np.sum(tone_p**2))
    click_h, click_p = hpss(_clicks(), SR)
    assert float(np.sum(click_p**2)) > float(np.sum(click_h**2))


# --- G-7: transient designer ---------------------------------------------------------
def test_transient_split_sum_exact():
    pluck = _tone() * np.exp(-np.arange(SR) / SR * 6)
    pluck[:80] += np.hanning(80) * 0.8  # sharp attack
    transient, sustain = transient_split(pluck, SR)
    assert np.allclose(transient + sustain, pluck, atol=1e-12)
    # The transient layer concentrates near the onset.
    head = int(0.05 * SR)
    head_share = float(np.sum(transient[:head] ** 2) / (np.sum(transient**2) + 1e-24))
    assert head_share > 0.5


def test_shape_transients_identity_and_direction():
    kick = np.sin(2 * math.pi * (60 + 180 * np.exp(-np.arange(SR) / SR * 25)) * np.arange(SR) / SR)
    kick *= np.exp(-np.arange(SR) / SR * 7)

    identity = shape_transients(kick, SR, attack_gain=1.0, sustain_gain=1.0)
    assert np.allclose(identity, kick, atol=1e-12)

    punchier = shape_transients(kick, SR, attack_gain=2.0)
    softer = shape_transients(kick, SR, attack_gain=0.2)
    head = int(0.04 * SR)
    assert float(np.abs(punchier[:head]).max()) > float(np.abs(kick[:head]).max()) * 1.2
    assert float(np.abs(softer[:head]).max()) < float(np.abs(kick[:head]).max())


# --- R-1..R-5: rhythm ------------------------------------------------------------------
def _drum_loop(bpm=120.0, bars=2, swing_fraction=0.5):
    """Kick on quarters (low), hats on 8ths (high); off-8ths at ``swing_fraction``."""
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
            seg = np.sin(2 * math.pi * 60 * t[:length]) * np.exp(-np.arange(length) / SR * 40)
            x[k : k + length] += seg
            for frac in (0.0, swing_fraction):  # on-8th and off-8th hats
                h = int((beat_start + frac * beat) * SR)
                hat_len = int(0.03 * SR)
                noise = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / SR * 120)
                # High-pass-ish: difference removes low content.
                x[h : h + hat_len] += np.diff(noise, prepend=0.0) * 0.5
    return x / np.abs(x).max()


def test_rhythm_straight_loop():
    profile = analyze_rhythm(_drum_loop(swing_fraction=0.5), SR, bpm=120.0)
    # Kick pattern peaks on the quarters (slots 0, 4, 8, 12).
    low = profile.pattern[0]
    assert set(np.argsort(low)[-4:]) == {0, 4, 8, 12}
    # Straight hats: swing ≈ 50 %.
    assert profile.swing_percent["high"] == pytest.approx(50.0, abs=4.0)
    # Quantized kicks: consistent timing (std tight); the mean carries the
    # documented constant low-band analysis advance (≤ ~12 ms).
    low_devs = profile.deviations_ms["low"]
    assert float(np.std(low_devs)) < 5.0
    assert -14.0 < float(np.mean(low_devs)) < 4.0
    # Median-centered microtiming ≈ 0 for a fully quantized loop.
    assert float(np.abs(profile.microtiming_mean_ms).max()) < 5.0
    # Kicks on strong slots -> low syncopation.
    assert profile.syncopation["low"] < 0.3


def test_rhythm_swung_loop():
    profile = analyze_rhythm(_drum_loop(swing_fraction=2 / 3), SR, bpm=120.0)
    assert profile.swing_percent["high"] == pytest.approx(66.7, abs=4.0)
    # The kick stays consistently placed even while the hats swing.
    assert float(np.std(profile.deviations_ms["low"])) < 5.0


def test_rhythm_profile_shapes():
    profile = analyze_rhythm(_drum_loop(), SR, bpm=120.0)
    assert profile.pattern.shape == (3, 16)
    assert profile.microtiming_mean_ms.shape == (16,)
    assert profile.swing_overall is not None
