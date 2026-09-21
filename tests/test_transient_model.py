"""G-8 transient axis: attack construction mechanics + imagine_transient."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec.frame import analyze
from wavelet2vec.transient_model import (
    MODEL_BANDS,
    default_attack_params,
    fit_attack_params,
    onset_index,
    sharpness_label,
)

torch = pytest.importorskip("torch")
from wavelet2vec import (  # noqa: E402
    TransientNet,
    imagine_transient,
    load_transient_model,
    save_transient_model,
)
from wavelet2vec.scenes import render_note  # noqa: E402

SR = 22050


def _pad(seconds: float = 1.0) -> np.ndarray:
    """A slow-swell pad: rich spectrum, no attack."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * k * 110.0 * t) / k for k in range(1, 10))
    swell = np.minimum(t / 0.4, 1.0)
    pad = tone * swell
    return (pad / np.abs(pad).max()).astype(np.float64)


def test_onset_sharpness_and_fit_labels():
    pluck = render_note(52, 1.0, SR, instrument="pluck")
    pad = _pad()
    assert onset_index(pluck, SR) < SR // 20  # a one-shot attacks early
    assert sharpness_label(pluck, SR) > sharpness_label(pad, SR)

    coeffs, _spec = analyze(pluck, SR, n_bands=MODEL_BANDS, f_min=30.0)
    params = fit_attack_params(coeffs, SR, onset_index(pluck, SR))
    assert params.shape == (2, MODEL_BANDS)
    assert np.all(np.isfinite(params))


def test_deterministic_pluck_adds_attack_and_amount_zero_exact():
    pad = _pad()
    out = imagine_transient(pad, SR, 1.0)
    assert out.shape == pad.shape
    # the pad gained a pluck: measured sharpness rises
    assert sharpness_label(out.astype(np.float64), SR) > sharpness_label(pad, SR)
    # tail (after the attack window) is essentially untouched
    tail = slice(SR // 2, None)
    assert np.abs(out[tail] - pad[tail]).max() < 0.05 * np.abs(pad).max()

    dry = imagine_transient(pad, SR, 0.0)
    assert np.allclose(dry, pad, atol=1e-7)  # amount=0 bit-exact


def test_net_mechanics_and_hook(tmp_path):
    net = TransientNet(width=8)
    envelopes = torch.rand(2, 1, MODEL_BANDS, 128)
    soft = net(envelopes, torch.full((2, 1), 0.1))
    hard = net(envelopes, torch.full((2, 1), 0.9))
    assert soft.shape == (2, 2, MODEL_BANDS)
    assert not torch.allclose(soft, hard)  # FiLM conditions the attack
    gains, taus = soft[:, 0], soft[:, 1]
    assert float(gains.max()) <= np.log(30.0) + 1e-5
    assert float(taus.min()) >= np.log(0.005) - 1e-5

    path = save_transient_model(net, tmp_path / "t.pt")
    restored = load_transient_model(path)
    assert torch.allclose(net(envelopes, torch.full((2, 1), 0.5)),
                          restored(envelopes, torch.full((2, 1), 0.5)))

    pad = _pad()
    out = imagine_transient(pad, SR, 1.0, model=restored, sharpness=0.7)
    assert out.shape == pad.shape and np.all(np.isfinite(out))


def test_default_params_shapes():
    centers = np.geomspace(30.0, 11025.0, MODEL_BANDS)
    params = default_attack_params(centers)
    assert params.shape == (2, MODEL_BANDS)
    gains, taus = np.exp(params[0]), np.exp(params[1])
    assert np.all(gains >= 1.5) and np.all(gains <= 8.0)
    assert taus[0] > taus[-1]  # low bands ring longer than highs
