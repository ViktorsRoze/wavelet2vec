import numpy as np
import pytest

from wavelet2vec.drum_analysis import detect_percussive_events, compare_percussive_events


def test_detector_does_not_invent_zero_event_and_preserves_antiphase_events():
    sr = 8000
    x = np.zeros(sr)
    t = np.arange(1200)/sr
    hit = np.sin(2*np.pi*(180*t+150*t*t))*np.exp(-t*35)
    for start in (1600, 4800):
        x[start:start+len(hit)] += hit
    result = detect_percussive_events(np.stack((x, -x)), sr)
    times = [row["time_seconds"] for row in result["events"]]
    assert len(times) == 2
    np.testing.assert_allclose(times, [.2, .6], atol=.025)
    assert not detect_percussive_events(np.zeros((2, sr)), sr)["events"]


def test_reference_rate_gain_and_timing_fit():
    sr, pre, n = 8000, 160, 2400
    t = np.maximum(np.arange(n)-pre, 0)/sr
    x = (np.sin(2*np.pi*180*t)+.3*np.sin(2*np.pi*517*t))*np.exp(-t*18)
    x[:pre] = 0
    reference = np.stack((x, -.8*x))
    ratio = 2**(3/12)
    target = np.stack([np.interp(pre+(np.arange(n)-pre+.3)*ratio, np.arange(n), channel, left=0, right=0)
                       for channel in reference])*np.array([.7, .4])[:, None]
    result = compare_percussive_events(reference, target, sr, preroll_samples=pre)
    assert result["pitch_shift_semitones"] == pytest.approx(3, abs=.04)
    assert result["waveform_fit"] > .995
    np.testing.assert_allclose(result["channel_gains"], [.7, .4], atol=.03)


def test_unrelated_noise_does_not_get_a_confident_pitch():
    generator = np.random.default_rng(6)
    a, b = generator.normal(size=(2, 2400)), generator.normal(size=(2, 2400))
    result = compare_percussive_events(a, b, 8000, preroll_samples=160)
    assert result["pitch_shift_semitones"] is None and result["waveform_fit"] < .5
    assert compare_percussive_events(np.zeros((2, 2400)), b, 8000)["status"] == "silent"


def test_invalid_and_short_events_are_explicit():
    with pytest.raises(ValueError):
        detect_percussive_events(np.zeros((2, 100)), 8000, sensitivity=2)
    assert compare_percussive_events(np.ones((2, 4)), np.ones((2, 4)), 8000)["status"] == "insufficient_duration"


def test_onset_offset_does_not_become_pitch_or_volume_change():
    sr, pre, n = 8000, 160, 2400
    t = np.maximum(np.arange(n)-pre, 0)/sr
    mono = np.sin(2*np.pi*(65*t+2*(1-np.exp(-20*t))))*np.exp(-15*t)
    mono[:pre] = 0
    target = np.stack((mono, mono))
    reference = np.pad(target[:, 96:], ((0, 0), (0, 96)))
    result = compare_percussive_events(reference, target, sr, preroll_samples=pre)
    assert result["pitch_shift_semitones"] == pytest.approx(0., abs=.02)
    assert result["fitted_level_change_db"] == pytest.approx(0., abs=.02)
    assert result["waveform_fit"] > .999


def test_weighted_source_loss_matches_independent_source_objectives():
    import torch
    from wavelet2vec import SourceExample, collate_sources, separation_loss
    torch.manual_seed(9)
    a, b = torch.randn(2, 65), torch.randn(2, 65)*.1
    batch = collate_sources([SourceExample(a+b, {"a": a, "b": b}, 8000)], source_order=("a", "b"))
    prediction = torch.randn_like(batch.sources)
    weighted = separation_loss(prediction, batch, source_weights=(1., 3.), fft_sizes=(32,))
    singles = []
    for index, (name, target) in enumerate((("a", a), ("b", b))):
        one = collate_sources([SourceExample(a+b, {name: target}, 8000)], source_order=(name,))
        singles.append(separation_loss(prediction[:, index:index+1], one, fft_sizes=(32,)))
    for key in ("loss", "waveform_l1", "complex_spectral_l1"):
        torch.testing.assert_close(weighted[key], (singles[0][key]+3*singles[1][key])/4)
    with pytest.raises(ValueError, match="source_weights"):
        separation_loss(prediction, batch, source_weights=(1., 0.))
