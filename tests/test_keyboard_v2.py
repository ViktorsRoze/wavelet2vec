"""KeyboardNet v2: rich features, Swin backbone + UPerNet head (tiny config for speed)."""
import numpy as np
import torch

from wavelet2vec.keyboard import ENV_BANDS, ENV_FRAMES, ENV_RATE, NOTE_SAMPLES
from wavelet2vec.keyboard_v2 import (BASE_PLANE, COND_DIM, IN_PLANES, N_FEATURES, KeyboardV2Config, build_model_v2, load_keyboard_v2,
                                     network_inputs_v2, predict_envelope_v2, rich_note_features, save_keyboard_v2,
                                     shift_features)

TINY = dict(size="tiny", head_dim=64, refine_dims=(32, 16), drop_path=0.0)


def stereo_tone(midi, width=0.0):
    t = np.arange(NOTE_SAMPLES) / ENV_RATE
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    left = np.sin(2 * np.pi * f0 * t) * np.exp(-2 * t)
    right = np.sin(2 * np.pi * f0 * t + width) * np.exp(-2 * t)
    return np.stack([left, right]) * 0.3


def test_features_describe_a_steady_tone_in_its_band():
    feats = rich_note_features(stereo_tone(57))
    assert feats.shape == (N_FEATURES, ENV_BANDS, ENV_FRAMES)
    band = 57 - 12
    assert int(np.argmax(feats[0, :, 20:60].mean(1))) == band
    assert abs(feats[1, band, 20:60].mean()) < 0.01        # IF on the band centre
    assert feats[2, band, 20:60].mean() > 0.95              # steady: coherent
    assert np.hypot(feats[3, band, 30], feats[4, band, 30]) > 0.95
    assert feats[6, band, 20:60].mean() * 40 < -50          # identical channels: no width
    assert rich_note_features(stereo_tone(57, width=1.0))[6, band, 20:60].mean() * 40 > -12


def test_shift_features_and_inputs_layout():
    feats = rich_note_features(stereo_tone(50))
    shifted, valid = shift_features(feats, 5)
    np.testing.assert_array_equal(shifted[:, 5:], feats[:, :-5])
    assert valid[:5].sum() == 0
    planes, cond, base = network_inputs_v2([feats], [50], 55)
    assert planes.shape == (IN_PLANES, 128, 128) and cond.shape == (COND_DIM,) and cond[1] == 0.0
    assert base.shape == (ENV_BANDS, ENV_FRAMES)
    # three examples, given in any order, are sorted nearest-first
    three = [rich_note_features(stereo_tone(m)) for m in (40, 54, 70)]
    planes3, cond3, _ = network_inputs_v2(three, [40, 54, 70], 55)
    assert cond3[1] == 1.0 and np.allclose(cond3[2:5] * 48, [1, 15, -15])
    np.testing.assert_allclose(planes3[BASE_PLANE, :ENV_BANDS, :ENV_FRAMES] * 40,
                               network_inputs_v2(three[::-1], [70, 54, 40], 55)[2], atol=1e-3)


def test_untrained_model_is_the_baseline_and_round_trips(tmp_path):
    model = build_model_v2(KeyboardV2Config(**TINY)).eval()
    feats = [rich_note_features(stereo_tone(48)), rich_note_features(stereo_tone(60)), rich_note_features(stereo_tone(64))]
    pred, base = predict_envelope_v2(model, feats, [48, 60, 64], 55)
    np.testing.assert_allclose(pred, base, atol=1e-4)
    with torch.no_grad():
        model.out.bias.fill_(0.05)
    save_keyboard_v2(tmp_path / "v2.pt", model, extra={"epoch": 3})
    loaded, extra = load_keyboard_v2(tmp_path / "v2.pt")
    np.testing.assert_allclose(predict_envelope_v2(loaded, feats, [48, 60, 64], 55)[0],
                               predict_envelope_v2(model, feats, [48, 60, 64], 55)[0], atol=1e-4)
    assert extra == {"epoch": 3}
