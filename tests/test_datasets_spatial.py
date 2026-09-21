"""G-1 dataset tooling + M-2/M-3 spatial API tests."""

from __future__ import annotations

import json
import math

import numpy as np

from wavelet2vec.datasets import (
    attack_ladder,
    axis_labels,
    filter_ladder,
    find_ladder_files,
    lowpass_pair,
    noise_mix_ladder,
    render_ladder_dataset,
    sine_distortion_ladder,
)
from wavelet2vec.spatial import spatial_matrix, stereo_phase_spectrum

SR = 22050


def _monotone(values):
    return all(b > a for a, b in zip(values, values[1:]))


# --- G-1: ladders supervise their own control axes ------------------------------
def test_filter_ladder_brightness_monotone():
    # 44.1k: keeps every cutoff well below the biquad's Nyquist warp region.
    sr = 44100
    ladder = filter_ladder(110.0, sr, [800.0, 2000.0, 5000.0, 9000.0], seconds=0.6)
    brightness = [axis_labels(audio, sr)["brightness"] for audio, _cut in ladder]
    assert _monotone(brightness), brightness


def test_distortion_ladder_brightness_monotone():
    ladder = sine_distortion_ladder(220.0, SR, [0.5, 2.0, 8.0], seconds=0.6)
    brightness = [axis_labels(audio, SR)["brightness"] for audio, _d in ladder]
    assert _monotone(brightness), brightness


def test_noise_ladder_harmonicity_decreases():
    ladder = noise_mix_ladder(220.0, SR, [0.0, 0.3, 0.6, 0.9], seconds=0.6)
    harmonicity = [axis_labels(audio, SR)["harmonicity"] for audio, _m in ladder]
    assert all(b < a for a, b in zip(harmonicity, harmonicity[1:])), harmonicity


def test_attack_ladder_attack_ms_monotone():
    ladder = attack_ladder(220.0, SR, [1.0, 20.0, 80.0, 200.0], seconds=0.8)
    measured = [axis_labels(audio, SR)["attack_ms"] for audio, _a in ladder]
    assert _monotone(measured), measured


def test_lowpass_pair_removes_highs_only():
    rng = np.random.default_rng(0)
    full = rng.standard_normal(SR)
    limited, target = lowpass_pair(full, SR, cutoff_hz=3000.0)
    assert target is full
    labels_full = axis_labels(full, SR)
    labels_limited = axis_labels(limited, SR)
    assert labels_limited["brightness"] < labels_full["brightness"] * 0.5
    # Low region survives: correlation with a lowpassed reference stays high.
    assert np.corrcoef(limited, full)[0, 1] > 0.5


def test_find_ladder_files_convention(tmp_path):
    for name in ("pluck__cut30.wav", "pluck__cut70.wav", "sine__drive0.5.wav", "plain.wav"):
        (tmp_path / name).write_bytes(b"RIFF")
    ladders = find_ladder_files(tmp_path)
    assert set(ladders) == {"pluck", "sine"}
    assert [value for value, _p in ladders["pluck"]["cut"]] == [30.0, 70.0]
    assert ladders["sine"]["drive"][0][0] == 0.5


def test_render_ladder_dataset_manifest(tmp_path):
    manifest_path = render_ladder_dataset(
        tmp_path, sample_rate=SR, f0s=(220.0,), cutoffs_hz=(1000.0, 6000.0),
        drives=(0.5, 4.0), noise_mixes=(0.0, 0.8), attacks_ms=(1.0, 120.0), seconds=0.5,
    )
    entries = json.loads(manifest_path.read_text())
    assert len(entries) == 8
    kinds = {entry["kind"] for entry in entries}
    assert kinds == {"filter", "distortion", "noise", "attack"}
    for entry in entries:
        assert (tmp_path / entry["file"]).exists()
        assert set(entry["labels"]) == {"brightness", "harmonicity", "attack_ms"}


# --- M-2/M-3: spatial ---------------------------------------------------------------
def _tone(freq=440.0, seconds=0.6):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * math.pi * freq * t)


def test_stereo_phase_spectrum_identical_channels():
    mono = _tone()
    spectrum = stereo_phase_spectrum(np.stack([mono, mono]), SR)
    active = spectrum.coherence[np.abs(spectrum.balance) < 0.9]  # cells with energy in both
    band = np.argmin(np.abs(spectrum.centers_hz - 440.0))
    assert float(spectrum.coherence[band].mean()) > 0.95
    assert abs(float(spectrum.ipd_rad[band].mean())) < 0.05
    assert abs(float(spectrum.balance[band].mean())) < 0.05
    assert spectrum.ipd_rad.shape == spectrum.coherence.shape == spectrum.balance.shape


def test_stereo_phase_spectrum_detects_interchannel_delay():
    mono = _tone(300.0)
    delay = 12  # samples -> IPD = 2*pi*f*delay/SR at each band
    right = np.roll(mono, delay)
    spectrum = stereo_phase_spectrum(np.stack([mono, right]), SR)
    band = np.argmin(np.abs(spectrum.centers_hz - 300.0))
    expected = -2 * math.pi * 300.0 * delay / SR  # L relative to delayed R
    measured = float(np.angle(np.exp(1j * spectrum.ipd_rad[band]).mean()))
    assert abs(abs(measured) - abs(expected)) < 0.15
    assert float(spectrum.coherence[band].mean()) > 0.9


def test_stereo_phase_spectrum_decorrelated_noise():
    rng = np.random.default_rng(1)
    stereo = rng.standard_normal((2, SR))
    spectrum = stereo_phase_spectrum(stereo, SR)
    # Above ~200 Hz the integration window holds many coherence times, so the
    # estimator can genuinely call independent noise incoherent. (The lowest
    # bands of a 1 s clip contain only ~1 independent sample — no estimator
    # can decide there; that region stays honestly uncertain.)
    wide = spectrum.centers_hz >= 200.0
    assert float(spectrum.coherence[wide].mean()) < 0.4
    assert float(spectrum.coherence[wide].mean()) < float(spectrum.coherence[~wide].mean())


def test_spatial_matrix_three_channels():
    base = _tone(500.0)
    audio = np.stack([base, base * 0.5, np.roll(base, 7)])
    result = spatial_matrix(audio, SR, n_bands=32)
    assert result["pairs"] == [(0, 1), (0, 2), (1, 2)]
    assert result["coherence"].shape == (3, 32)
    band = np.argmin(np.abs(result["centers_hz"] - 500.0))
    # Pair (0,1): same signal, different gain -> full coherence, balance < 0 (ch1 quieter).
    assert result["coherence"][0, band] > 0.95
    assert result["balance"][0, band] < -0.3
    # Pair (0,2): delayed copy -> coherent with nonzero IPD.
    assert result["coherence"][1, band] > 0.9
    assert abs(result["ipd_rad"][1, band]) > 0.2
