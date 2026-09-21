"""H-1 f0 tracking + note segmentation, H-2 chroma/key/chords."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import (
    chromagram,
    estimate_key,
    segment_notes,
    track_chords,
    track_f0,
)

SR = 22050

# C=0 pitch-class indices.
A, CS, E, F, G, C = 9, 1, 4, 5, 7, 0


def _tone(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def _notes(freqs, note_s=0.4, sr: int = SR) -> np.ndarray:
    parts = []
    for freq in freqs:
        tone = _tone(freq, note_s, sr)
        fade = int(0.01 * sr)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        parts.append(tone)
    return np.concatenate(parts)


# --- H-1 --------------------------------------------------------------------------
def test_track_f0_sine():
    track = track_f0(_tone(220.0, 1.0), SR)
    voiced = np.isfinite(track.f0_hz)
    assert voiced.mean() > 0.9
    assert np.nanmedian(track.f0_hz) == pytest.approx(220.0, abs=1.0)
    assert track.voicing[voiced].mean() > 0.8


def test_track_f0_noise_is_unvoiced():
    noise = np.random.default_rng(0).standard_normal(SR)
    track = track_f0(noise, SR)
    assert np.isfinite(track.f0_hz).mean() < 0.2


def test_track_f0_vibrato():
    t = np.arange(SR) / SR
    depth = 2 ** (0.3 / 12)  # +-30 cents
    phase = 2 * np.pi * 330.0 * t + (330.0 * np.log(depth) / 5.0) * np.sin(2 * np.pi * 5.0 * t)
    track = track_f0(np.sin(phase), SR)
    assert np.nanmedian(track.f0_hz) == pytest.approx(330.0, abs=3.0)


def test_segment_notes_two_pitches():
    audio = _notes([220.0, 329.63], note_s=0.5)  # A3 then E4
    events = segment_notes(track_f0(audio, SR))
    long_events = [event for event in events if event.end_s - event.start_s > 0.2]
    assert [event.midi for event in long_events] == [57, 64]
    assert long_events[0].f0_hz == pytest.approx(220.0, abs=1.5)
    assert long_events[1].start_s == pytest.approx(0.5, abs=0.08)


# --- H-2 --------------------------------------------------------------------------
def test_chromagram_triad():
    audio = _tone(220.0, 1.0) + _tone(277.18, 1.0) + _tone(329.63, 1.0)  # A major
    chroma, times = chromagram(audio, SR)
    assert chroma.shape[0] == 12 and times.size == chroma.shape[1]
    top3 = set(np.argsort(chroma.mean(axis=1))[-3:])
    assert top3 == {A, CS, E}


def test_estimate_key_minor_and_major():
    # A natural minor with tonic/dominant emphasis.
    a_minor = _notes([220, 246.94, 261.63, 293.66, 329.63, 349.23, 392, 440, 329.63, 220, 220])
    estimate = estimate_key(a_minor, SR)
    assert estimate.name == "A minor"
    assert estimate.confidence > 0.5

    c_major = _notes([261.63, 293.66, 329.63, 349.23, 392, 440, 493.88, 523.25, 392, 261.63, 261.63])
    estimate = estimate_key(c_major, SR)
    assert estimate.name == "C major"


def test_track_chords_two_triads():
    def triad(root, third, fifth, seconds=1.2):
        return _tone(root, seconds) + _tone(third, seconds) + _tone(fifth, seconds)

    audio = np.concatenate(
        [triad(220.0, 261.63, 329.63), triad(174.61, 220.0, 261.63)]  # Am then F
    )
    segments = track_chords(audio, SR)
    labels = [label for _s, _e, label in segments]
    assert labels == ["Am", "F"]
    assert segments[0][1] == pytest.approx(1.2, abs=0.3)
