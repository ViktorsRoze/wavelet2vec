"""Lossless stereo wavelet image: exact round trips at 192 kHz, alignment, stereo, file storage, views."""
import numpy as np
import pytest
import torch

from wavelet2vec.wavelet_image import (WaveletImageSpec, band_layout, decode, describe, detail_view, encode,
                                       load_image, magnitude_view, save_image, stereo_view)

SPEC = WaveletImageSpec()  # 192 kHz, 120 x 10 ms frames


def snr_db(a, b):
    return 10 * np.log10(float((a ** 2).sum()) / float(((a - b) ** 2).sum()))


def test_layout_covers_the_spectrum_on_semitones():
    info = describe(SPEC)
    assert info["bands"] == 149 and info["midi_range"] == [12, 160]
    assert info["coverage_min_over_max"] > 0.5
    bands = band_layout(SPEC)
    assert bands[0].start_bin == 0 and bands[-1].stop_bin == SPEC.padded_length // 2 + 1
    assert all(b.samples >= b.stop_bin - b.start_bin for b in bands)  # no aliasing inside a band
    assert all(b.per_frame & (b.per_frame - 1) == 0 for b in bands)    # power-of-two cells


@pytest.mark.parametrize("dtype,floor_db", [(torch.float64, 250.0), (torch.float32, 120.0)])
def test_stereo_round_trip_is_exact(dtype, floor_db):
    rng = np.random.default_rng(0)
    left = rng.standard_normal(SPEC.audio_length) * 0.1
    audio = torch.from_numpy(np.stack([left, 0.6 * left + 0.05 * rng.standard_normal(left.size)]))[None].to(dtype)
    back = decode(encode(audio, SPEC)).double()
    a = audio.double()
    assert snr_db(a[:, 0], back[:, 0]) > floor_db and snr_db(a[:, 1], back[:, 1]) > floor_db
    side, side_back = (a[:, 0] - a[:, 1]) / 2, (back[:, 0] - back[:, 1]) / 2
    assert snr_db(side, side_back) > floor_db - 3


def test_click_lands_in_its_frame_and_tone_in_its_band():
    audio = torch.zeros(1, 2, SPEC.audio_length, dtype=torch.float64)
    audio[:, :, 37 * SPEC.hop + SPEC.hop // 2] = 1.0                       # click in frame 37
    view = magnitude_view(encode(audio, SPEC))[0, 0]
    high = view[100:140]                                                   # bands 112-151 (MIDI): short in time
    assert int(np.argmax(high.mean(0).numpy())) == 37
    t = torch.arange(SPEC.audio_length, dtype=torch.float64) / SPEC.sample_rate
    tone = torch.sin(2 * np.pi * 440.0 * t)
    view = magnitude_view(encode(torch.stack([tone, tone])[None], SPEC))[0, 0]
    assert int(np.argmax(view[:, 40:80].mean(1).numpy())) == 69 - 12      # A4 row


def test_float32_images_on_disk_invert_to_audio(tmp_path):
    rng = np.random.default_rng(1)
    audio = torch.from_numpy(rng.standard_normal((1, 2, SPEC.audio_length)) * 0.1).float()
    image = encode(audio, SPEC)
    save_image(image, tmp_path / "note")
    back = decode(load_image(tmp_path / "note")).double()
    assert snr_db(audio.double(), back) > 120.0
    assert len(list((tmp_path / "note").glob("group_c*.tiff"))) == len(image.groups)


# ----------------------------------------------------------------- screen-resolution views


def _stereo(left, right):
    return encode(torch.from_numpy(np.stack([left, right]))[None], SPEC)


def _rows(m0, m1, height):
    return m0 + (np.arange(height) + 0.5) * (m1 - m0) / height


TONE_MIDI = 69 + 12 * np.log2(1000.0 / 440.0)          # 1 kHz: between B5 and C6


def test_sharp_view_puts_a_tone_on_one_line_at_its_level():
    t = np.arange(SPEC.audio_length) / SPEC.sample_rate
    tone = 0.5 * np.cos(2 * np.pi * 1000.0 * t)
    image = _stereo(tone, tone)
    m0, m1, height = 70.0, 96.0, 208                     # 8 pixel rows per semitone
    midi = _rows(m0, m1, height)
    near = np.abs(midi - TONE_MIDI) <= 0.5
    share = {}
    for mode in ("sharp", "exact", "cells"):
        column = 10 ** (detail_view(image, mode=mode, midi_range=(m0, m1), width=120, height=height)[:, 60] / 10)
        share[mode] = column[near].sum() / column.sum()
        if mode == "sharp":
            assert abs(midi[np.argmax(column)] - TONE_MIDI) < 0.15
            assert abs(10 * np.log10(column.sum()) - 20 * np.log10(0.5)) < 1.0   # power is kept
    assert share["sharp"] > 0.95 and share["cells"] < 0.8 and share["exact"] < 0.85


def test_sharp_view_puts_a_bass_click_back_at_its_time():
    click = np.zeros(SPEC.audio_length)
    click[int(0.5 * SPEC.sample_rate)] = 1.0
    image = _stereo(click, click)
    m0, m1 = 40.0, 70.0                                  # 41 Hz - 466 Hz: bands 30-200 ms long
    times = (np.arange(600) + 0.5) * 1.2 / 600
    near = np.abs(times - 0.5) <= 0.004
    for mode, low, high in (("sharp", 0.9, 1.01), ("exact", 0.0, 0.5)):
        energy = (10 ** (detail_view(image, mode=mode, midi_range=(m0, m1), width=600, height=60) / 10)).sum(0)
        assert low < energy[near].sum() / energy.sum() < high


def test_sharp_view_keeps_noise_smooth_across_band_centres():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(SPEC.audio_length) * 0.05
    view = detail_view(_stereo(noise, noise), mode="sharp", time_range=(0.2, 1.0), midi_range=(84.0, 96.0),
                       width=200, height=96)
    profile = 10 * np.log10((10 ** (view / 10)).mean(1))             # average over time, per pixel row
    trend = np.polyval(np.polyfit(np.arange(96), profile, 1), np.arange(96))
    assert np.abs(profile - trend).max() < 2.0                      # no lines at the band centres


def test_stereo_view_reads_balance_and_phase():
    t = np.arange(SPEC.audio_length) / SPEC.sample_rate
    tone = 0.5 * np.cos(2 * np.pi * 1000.0 * t)
    cases = {"left only": (tone, 0 * tone, -1.0, None), "right only": (0 * tone, tone, 1.0, None),
             "mono": (tone, tone, 0.0, 1.0), "opposite": (tone, -tone, 0.0, -1.0)}
    for name, (left, right, balance, correlation) in cases.items():
        image = _stereo(left, right)
        for mode in ("sharp", "exact", "cells"):
            level, bal, cor = stereo_view(image, mode=mode, midi_range=(70.0, 96.0), width=60, height=104)
            peak = np.unravel_index(np.argmax(level), level.shape)
            assert abs(bal[peak] - balance) < 0.02, (name, mode)
            if correlation is not None:
                assert abs(cor[peak] - correlation) < 0.02, (name, mode)
