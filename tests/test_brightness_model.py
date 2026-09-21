"""G-2 learned brightness axis v1: model mechanics + imagine_highs hook."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec.brightness_model import (
    MODEL_BANDS,
    MODEL_FRAMES,
    cutoff_mask,
    envelope_brightness,
    log_envelopes,
)
from wavelet2vec.scenes import render_note

torch = pytest.importorskip("torch")
from wavelet2vec import BrightnessNet, load_brightness_model, save_brightness_model  # noqa: E402
from wavelet2vec.imagine import imagine_highs  # noqa: E402

SR = 22050


def test_log_envelopes_grid_and_brightness_label():
    saw = render_note(60, 1.0, SR, instrument="saw")
    envelopes, centers = log_envelopes(saw, SR)
    assert envelopes.shape == (MODEL_BANDS, MODEL_FRAMES)
    assert centers.shape == (MODEL_BANDS,)
    assert float(envelopes.min()) >= 0.0

    sine = np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    dull, c1 = log_envelopes(sine, SR)
    assert envelope_brightness(envelopes, centers) > envelope_brightness(dull, c1)


def test_cutoff_mask_shape():
    centers = np.geomspace(30.0, 11025.0, 64)
    mask = cutoff_mask(centers, 2000.0)
    assert mask[centers < 1000.0].min() > 0.99  # passband intact
    assert mask[centers > 4000.0].max() < 0.01  # stopband gone
    assert np.all(np.diff(mask) < 1e-6)  # monotone rolloff


def test_net_forward_conditioning_and_roundtrip(tmp_path):
    net = BrightnessNet(width=8)
    envelopes = torch.rand(2, 1, MODEL_BANDS, MODEL_FRAMES)
    dull = net(envelopes, torch.tensor([[0.05], [0.05]]))
    bright = net(envelopes, torch.tensor([[0.6], [0.6]]))
    assert dull.shape == envelopes.shape
    assert float(dull.min()) >= 0.0  # log1p envelopes are non-negative
    assert not torch.allclose(dull, bright)  # FiLM actually conditions

    path = save_brightness_model(net, tmp_path / "b.pt")
    restored = load_brightness_model(path)
    assert torch.allclose(net(envelopes, torch.tensor([[0.3], [0.3]])),
                          restored(envelopes, torch.tensor([[0.3], [0.3]])))


def test_imagine_highs_model_hook_preserves_low_end():
    pluck = render_note(48, 1.0, SR, instrument="pluck").astype(np.float64)
    net = BrightnessNet(width=8)  # untrained: mechanics only
    out = imagine_highs(pluck, SR, 1.0, model=net, brightness=0.3)
    assert out.shape == pluck.shape
    assert np.all(np.isfinite(out))

    # amount=0 stays bit-exact even with a model attached
    silent = imagine_highs(pluck, SR, 0.0, model=net, brightness=0.3)
    assert np.allclose(silent, pluck, atol=1e-6)

    # the low end is the original's (learned path only adds above the split)
    spectrum_in = np.abs(np.fft.rfft(pluck))
    spectrum_out = np.abs(np.fft.rfft(out.astype(np.float64)))
    freqs = np.fft.rfftfreq(pluck.shape[0], 1.0 / SR)
    low = freqs < 2000.0
    ratio = spectrum_out[low].sum() / (spectrum_in[low].sum() + 1e-24)
    assert 0.9 < ratio < 1.1
