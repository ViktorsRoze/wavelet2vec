"""KeyboardNet v1 building blocks: semitone envelopes, shifts, model, rendering."""
import numpy as np
import torch

from wavelet2vec.keyboard import (ENV_BANDS, ENV_FRAMES, ENV_RATE, NOTE_SAMPLES, KeyboardConfig, build_model,
                                  load_keyboard_model, network_inputs, note_envelope, predict_envelope,
                                  render_note, save_keyboard_model, shift_envelope)


def tone(midi, seconds=1.2, rate=ENV_RATE):
    t = np.arange(int(seconds * rate)) / rate
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    harmonics = [np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8) if k * f0 < 0.45 * rate]
    return 0.3 * np.sum(harmonics, 0) * np.exp(-2.0 * t)


def test_envelope_peaks_on_the_played_semitone():
    env = note_envelope(tone(60))
    assert env.shape == (ENV_BANDS, ENV_FRAMES)
    assert int(np.argmax(env[:, 10:40].mean(1))) == 60 - 12


def test_shift_moves_bands_exactly_and_marks_validity():
    env = note_envelope(tone(57))
    up, valid = shift_envelope(env, 7)
    np.testing.assert_array_equal(up[7:], env[:-7])
    assert valid[:7].sum() == 0 and valid[7:].all()
    down, valid = shift_envelope(env, -5)
    np.testing.assert_array_equal(down[:-5], env[5:])
    assert valid[-5:].sum() == 0


def test_untrained_model_returns_the_baseline():
    model = build_model(KeyboardConfig(channels=(16, 16, 16)))
    examples = [note_envelope(tone(55)), note_envelope(tone(67))]
    pred, base = predict_envelope(model, examples, [55, 67], 60)
    np.testing.assert_allclose(pred, base, atol=1e-4)
    planes, cond, _ = network_inputs(examples[:1], [55], 60)
    assert planes.shape == (7, ENV_BANDS, ENV_FRAMES) and cond[3] == 0.0


def test_render_identity_and_octave_shift():
    audio = tone(48)[None]
    same = render_note(audio, 48, 48, note_envelope(audio[0]))
    assert np.max(np.abs(same - audio)) < 0.02 * np.max(np.abs(audio))
    octave = render_note(audio, 48, 60, shift_envelope(note_envelope(audio[0]), 12)[0])[0]
    spectrum = np.abs(np.fft.rfft(octave[: NOTE_SAMPLES // 2]))
    freqs = np.fft.rfftfreq(NOTE_SAMPLES // 2, 1 / ENV_RATE)
    assert abs(freqs[np.argmax(spectrum)] - 440.0 * 2 ** ((60 - 69) / 12)) < 3.0


def test_save_and_load_round_trip(tmp_path):
    model = build_model(KeyboardConfig(channels=(16, 16, 16)))
    with torch.no_grad():
        model.head.bias.fill_(0.1)
    save_keyboard_model(tmp_path / "k.pt", model, extra={"note": "test"})
    loaded, extra = load_keyboard_model(tmp_path / "k.pt")
    examples = [note_envelope(tone(50))]
    np.testing.assert_allclose(predict_envelope(loaded, examples, [50], 53)[0],
                               predict_envelope(model.eval(), examples, [50], 53)[0], atol=1e-5)
    assert extra == {"note": "test"}
