"""C-3 torch/GPU frame backend: parity, round-trip, batching, interchange."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec.frame import analyze, synthesize

torch = pytest.importorskip("torch")
from wavelet2vec.torch_frame import analyze_batch, analyze_torch, synthesize_torch  # noqa: E402

SR = 22050

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _signal(seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * 440 * t) + 0.3 * rng.standard_normal(t.shape[0])


def _snr(ref, test):
    noise = float(np.sum((ref - test) ** 2))
    return np.inf if noise == 0 else 10 * np.log10(float(np.sum(ref**2)) / noise)


@pytest.mark.parametrize("device", _DEVICES)
def test_matches_numpy_to_machine_precision(device):
    x = _signal()
    coeffs_np, spec_np = analyze(x, SR, n_bands=64)
    coeffs_t, spec_t = analyze_torch(x, SR, n_bands=64, device=device, precision="double")
    assert np.allclose(spec_np.centers, spec_t.centers)
    assert np.abs(coeffs_np - coeffs_t.cpu().numpy()).max() < 1e-11


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("precision,floor_db", [("double", 250.0), ("single", 110.0)])
def test_round_trip(device, precision, floor_db):
    x = _signal(seed=1)
    coeffs, spec = analyze_torch(x, SR, n_bands=128, device=device, precision=precision)
    y = synthesize_torch(coeffs, spec, device=device, precision=precision).cpu().numpy()
    assert _snr(x, y) > floor_db


def test_spec_is_interchangeable_with_numpy():
    x = _signal(seed=2)
    # torch analyze -> numpy synthesize
    coeffs_t, spec = analyze_torch(x, SR, n_bands=96, precision="double")
    y1 = synthesize(coeffs_t.cpu().numpy(), spec)
    assert _snr(x, y1) > 250.0
    # numpy analyze -> torch synthesize
    coeffs_np, spec2 = analyze(x, SR, n_bands=96)
    y2 = synthesize_torch(coeffs_np, spec2, precision="double").cpu().numpy()
    assert _snr(x, y2) > 250.0


def test_batched_equals_individual():
    frags = np.stack([_signal(0.5, seed=s) for s in range(4)])
    coeffs, spec = analyze_batch(frags, SR, n_bands=64, precision="double")
    assert coeffs.shape == (4, 64, frags.shape[1])
    for i in range(4):
        solo, _spec = analyze_torch(frags[i], SR, n_bands=64, precision="double")
        assert np.abs(coeffs[i].cpu().numpy() - solo.cpu().numpy()).max() < 1e-12
    recon = synthesize_torch(coeffs, spec, precision="double").cpu().numpy()
    for i in range(4):
        assert _snr(frags[i], recon[i]) > 250.0


def test_multichannel_leading_dims():
    x = _signal(seed=3)
    stereo = np.stack([x, 0.7 * x])
    coeffs, _spec = analyze_torch(stereo, SR, n_bands=48)
    assert coeffs.shape == (2, 48, x.shape[0])
    # An extra batch dim also passes through: [B, C, N] -> [B, C, bands, N].
    batch = np.stack([stereo, stereo])
    coeffs2, _s = analyze_torch(batch, SR, n_bands=48)
    assert coeffs2.shape == (2, 2, 48, x.shape[0])


def test_accepts_torch_tensor_input():
    x = _signal(seed=4)
    tensor = torch.as_tensor(x)
    coeffs, spec = analyze_torch(tensor, SR, n_bands=32, precision="double")
    y = synthesize_torch(coeffs, spec, precision="double").cpu().numpy()
    assert _snr(x, y) > 250.0


def test_analyze_batch_rejects_ragged():
    with pytest.raises(ValueError):
        analyze_batch(_signal(), SR)  # 1-D, not [B, N]


def test_bad_precision_raises():
    with pytest.raises(ValueError):
        analyze_torch(_signal(), SR, precision="half")
