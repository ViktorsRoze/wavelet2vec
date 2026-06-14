"""CI-safe tests for the experiment suite — synthetic only, no example_audio."""

from __future__ import annotations

import math

import numpy as np

from experiments import metrics
from experiments.dataset import parse_bpm, parse_pitch, sliding_windows
from experiments.mel_baseline import (
    log_mel_spectrogram,
    mel_chroma,
    mel_summary_vector,
    predicted_pitch_class,
)

SR = 22050


def _sine(freq, seconds=1.0, sr=SR):
    return np.sin(2 * math.pi * freq * np.arange(int(sr * seconds)) / sr)


def test_parse_pitch_conventions():
    # (stem, expected pitch class, expected minor)
    cases = [
        ("analog bass A#", 10, False),
        ("RNT_vocal_single_ahh_02_F#1", 6, False),
        ("fhv_vox_aching_C#m", 1, True),
        ("sp_syn100_89_Cm", 0, True),
        ("fhv_vox_accountable_Bb", 10, False),
        ("UAMEE DONK6 PERESTROIKA E", 4, False),
    ]
    for stem, pc, minor in cases:
        got_pc, got_minor, _ = parse_pitch(stem)
        assert got_pc == pc, (stem, got_pc)
        assert got_minor == minor, (stem, got_minor)


def test_parse_pitch_absent():
    assert parse_pitch("nd_perc_808cow") == (None, None, None)


def test_parse_bpm():
    assert parse_bpm("synth_loops_100BPM") == 100.0
    assert parse_bpm("fx_at_120BPM") == 120.0
    assert parse_bpm("percussion") is None


def test_sliding_windows_fixed_and_short():
    assert sliding_windows(SR // 2, SR, window_seconds=2.0) == [(0, SR // 2)]  # shorter than window
    bounds = sliding_windows(5 * SR, SR, window_seconds=2.0, hop_seconds=1.0)
    assert bounds[0] == (0, 2 * SR)
    assert all(end - start == 2 * SR for start, end in bounds)
    assert bounds[-1][1] == 5 * SR  # tail covered


def test_sliding_windows_beat_aligned():
    # 4 bars at 120 BPM = 8 s; 1-bar window (2 s), 1-beat hop (0.5 s).
    bounds = sliding_windows(int(8 * SR), SR, bpm=120.0)
    assert bounds[0] == (0, int(2 * SR))
    assert bounds[1][0] == int(0.5 * SR)  # one beat later
    assert all(end - start == int(2 * SR) for start, end in bounds)


def test_mel_chroma_identifies_pitch_class():
    chroma = mel_chroma(_sine(440.0, 1.0), SR)  # A4
    assert predicted_pitch_class(chroma) == 9


def test_mel_spectrogram_and_summary_shapes():
    log_mel, centers, times = log_mel_spectrogram(_sine(440.0), SR, n_mels=64)
    assert log_mel.shape[0] == 64
    assert centers.shape[0] == 64
    assert log_mel.shape[1] == times.shape[0]
    vector = mel_summary_vector(log_mel)
    assert vector.shape == (128,)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


def test_pitch_scoring_strict_and_tolerant():
    assert metrics.pitch_class_correct(9, 9)
    assert not metrics.pitch_class_correct(10, 9)
    assert metrics.pitch_class_correct(10, 9, tolerant=True)
    assert metrics.pitch_class_correct(0, 11, tolerant=True)  # wraps around

    res = metrics.score_pitch_recovery([0, 1, 2], [0, 1, 5], [0, 9, 2])
    assert abs(res.mel_accuracy - 2 / 3) < 1e-9
    assert abs(res.wavelet_accuracy - 2 / 3) < 1e-9


def test_pca_2d_and_cluster_metrics_separate_groups():
    rng = np.random.default_rng(0)
    group_a = rng.normal(0, 0.05, size=(20, 8)) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
    group_b = rng.normal(0, 0.05, size=(20, 8)) + np.array([0, 5, 0, 0, 0, 0, 0, 0])
    vectors = np.vstack([group_a, group_b])
    labels = ["a"] * 20 + ["b"] * 20

    points = metrics.pca_2d(vectors)
    assert points.shape == (40, 2)
    assert metrics.knn_label_purity(vectors, labels) > 0.95
    assert metrics.silhouette_score(vectors, labels) > 0.5


def test_silhouette_single_label_is_zero():
    assert metrics.silhouette_score(np.random.default_rng(1).normal(size=(5, 4)), ["x"] * 5) == 0.0
