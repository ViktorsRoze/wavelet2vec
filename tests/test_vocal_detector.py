"""Vocal detector on synthetic voice parts with known answers.

Test vowels use Peterson & Barney (1952) / Hillenbrand (1995) formants with
random ±6 % shifts and different bandwidths than the templates, so they
never match a template exactly.
"""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from wavelet2vec.tools import detect_vocal_parts, formant_envelope_db

SR = 48000
TEST_FORMANTS = {"i": (270, 2290, 3010), "eh": (530, 1840, 2480), "a": (730, 1090, 2440),
                 "o": (497, 910, 2459), "u": (300, 870, 2240), "e": (476, 2089, 2691)}


def vowel_tone(formants, f0, seconds, *, scale=1.0, seed=0, glide_to=None):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    start = np.array(formants + (3500, 4400)) * scale * rng.uniform(0.94, 1.06, 5)
    end = start if glide_to is None else np.array(glide_to + (3500, 4400)) * scale
    phase = 2 * np.pi * f0 * np.cumsum(1 + 0.004 * np.sin(2 * np.pi * 5 * t)) / SR
    blocks = np.array_split(np.arange(t.size), 20)  # glide: hold 6 blocks, move 8, hold 6
    out = np.zeros_like(t)
    for b, idx in enumerate(blocks):
        f_now = start + (end - start) * float(np.clip((b - 5) / 9, 0.0, 1.0))
        for k in range(1, int(9000 // f0)):
            f = k * f0
            amp = 10 ** ((formant_envelope_db(np.array([f]), f_now, np.array([60, 90, 120, 180, 250]))[0]
                          - 6 * np.log2(f / 100)) / 20)
            out[idx] += amp * np.cos(k * phase[idx] + 0.37 * k)
    out /= np.abs(out).max()
    return 0.5 * out + 1e-3 * rng.standard_normal(t.size)


def band_noise(lo, hi, seconds, seed=1, level=0.2):
    noise = np.random.default_rng(seed).standard_normal(int(SR * seconds))
    filtered = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), noise)
    return level * filtered / np.abs(filtered).max()


def silence(seconds):
    return 1e-5 * np.random.default_rng(9).standard_normal(int(SR * seconds))


def span_parts(track, start, end):
    inside = (track.times_s >= start) & (track.times_s < end)
    names, counts = np.unique(track.part[inside], return_counts=True)
    return dict(zip(names, counts))


@pytest.mark.parametrize("scale", [1.0, 1.17])
@pytest.mark.parametrize("vowel", list(TEST_FORMANTS))
def test_vowels_at_speaking_pitch(vowel, scale):
    audio = vowel_tone(TEST_FORMANTS[vowel], 165.0, 0.8, scale=scale, seed=len(vowel))
    track = detect_vocal_parts(audio, SR)
    summary = track.summarize(0.1, 0.7)
    assert summary["vowel"] == vowel and summary["share"] > 0.6
    assert abs(track.scale - scale) < 0.1


@pytest.mark.parametrize("vowel", ["i", "a", "u"])
def test_vowels_at_singing_pitch(vowel):
    audio = vowel_tone(TEST_FORMANTS[vowel], 330.0, 0.8, scale=1.17, seed=3)
    assert detect_vocal_parts(audio, SR).summarize(0.1, 0.7)["vowel"] == vowel


def test_glide_start_and_end_vowels():
    audio = vowel_tone(TEST_FORMANTS["i"], 140.0, 1.0, glide_to=TEST_FORMANTS["u"])
    track = detect_vocal_parts(audio, SR, scale=1.0)
    assert track.summarize(0.05, 0.25)["vowel"] == "i"
    assert track.summarize(0.75, 0.95)["vowel"] == "u"


@pytest.mark.parametrize("band,kind", [((5000, 10000), "s"), ((2500, 4500), "sh")])
def test_sibilants_and_their_type(band, kind):
    audio = np.concatenate([silence(0.2), band_noise(*band, 0.4), silence(0.2)])
    track = detect_vocal_parts(audio, SR)
    inside = (track.times_s > 0.25) & (track.times_s < 0.55)
    assert (track.part[inside] == "sibilant").mean() > 0.8
    assert (track.sibilant_type[inside & (track.part == "sibilant")] == kind).mean() > 0.8
    assert track.sibilance[inside].min() > 0.5


def test_sibilant_then_vowel_segments_in_order():
    audio = np.concatenate([silence(0.15), band_noise(5000, 10000, 0.25), vowel_tone(TEST_FORMANTS["a"], 165.0, 0.5)])
    parts = [s for s in detect_vocal_parts(audio, SR).segments() if s["end_s"] - s["start_s"] >= 0.05]
    order = [(s["part"], s["detail"]) for s in parts if s["part"] != "other"]
    assert order[:3] == [("silence", ""), ("sibilant", "s"), ("vowel", "a")]


def test_plosive_burst_after_silence():
    burst = np.random.default_rng(4).standard_normal(int(SR * 0.012)) * 0.4
    audio = np.concatenate([silence(0.3), burst, silence(0.03), vowel_tone(TEST_FORMANTS["a"], 165.0, 0.4)])
    track = detect_vocal_parts(audio, SR)
    plosives = track.times_s[track.part == "plosive"]
    assert plosives.size and np.all((plosives > 0.27) & (plosives < 0.36))
    assert span_parts(track, 0.0, 0.25) == {"silence": span_parts(track, 0.0, 0.25).get("silence", -1)}


def test_digital_silence_is_silence():
    track = detect_vocal_parts(np.zeros(SR // 2), SR)
    assert set(track.part) == {"silence"} and not track.vowel.any()
