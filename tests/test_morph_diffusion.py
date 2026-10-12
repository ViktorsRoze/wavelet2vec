"""MorphDiffusion stage 0: feature-image layout, schedule, denoiser, guided sampling, rendering."""
import numpy as np
import torch

from wavelet2vec.morph import SPEC, judge_view
from wavelet2vec.morph_diffusion import (FRAMES, PADDED, PLANES, ROWS, FeatureUNet, alpha_sigma, class_index,
                                         ddim_sample, judge_slice, one_hot, pad_features, predict_noise, predict_x0,
                                         render_features, shift_rows, unpad, v_target)
from wavelet2vec.morph_large import critic_view
from wavelet2vec.wavelet_image import WaveletImage, decode, encode

T = np.arange(SPEC.audio_length) / SPEC.sample_rate


def note(f0=261.63, partials=12):
    x = sum(np.sin(2 * np.pi * f0 * k * T) / k for k in range(1, partials + 1)) * np.exp(-T / 0.7) * 0.1
    return np.stack([x, 0.8 * x])


def tiny():
    torch.manual_seed(0)
    return FeatureUNet(chans=(16, 32, 32, 32), blocks=1, heads=4, emb=64).eval()


def test_feature_image_layout_and_judge_slice():
    image = encode(torch.from_numpy(np.stack([note(), note(220.0)])).float(), SPEC)
    feats = critic_view(image)
    assert feats.shape == (2, PLANES, ROWS, FRAMES)
    assert torch.equal(judge_slice(feats), judge_view(image))       # judges score feature images directly
    padded = pad_features(feats)
    assert padded.shape == (2, PLANES, *PADDED) and torch.equal(unpad(padded), feats)
    up = shift_rows(feats, 2)
    assert torch.equal(up[:, :, 2:], feats[:, :, :-2]) and float(up[:, 0, :2].max()) == -1.0


def test_schedule_and_v_prediction_invert_exactly():
    t = torch.linspace(0, 1, 11)
    a, s = alpha_sigma(t)
    assert torch.allclose(a ** 2 + s ** 2, torch.ones_like(t), atol=1e-6)
    assert float(a[0]) > 0.99 and float(a[-1]) < 0.01
    x0, noise = torch.randn(3, 5, 4, 4), torch.randn(3, 5, 4, 4)
    a, s = alpha_sigma(torch.tensor([0.3, 0.6, 0.9])[:, None, None, None])
    x_t = a * x0 + s * noise
    v = v_target(x0, noise, a, s)
    assert torch.allclose(predict_x0(x_t, v, a, s), x0, atol=1e-5)
    assert torch.allclose(predict_noise(x_t, v, a, s), noise, atol=1e-5)


def test_denoiser_shapes_gradients_and_interpolated_classes():
    model = tiny().train()
    x = torch.randn(2, PLANES, *PADDED)
    weights = torch.stack([one_hot([class_index("vox")])[0], 0.5 * one_hot([0])[0] + 0.5 * one_hot([1])[0]])
    out = model(x, torch.tensor([0.2, 0.8]), torch.tensor([60.0, 72.0]), weights)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_guided_sampling_is_deterministic_composable_and_in_range():
    model = tiny()
    keys = torch.tensor([60.0, 64.0])
    sample = lambda guidance: ddim_sample(model, keys, guidance, steps=4, generator=torch.Generator().manual_seed(5),
                                          device=torch.device("cpu"))
    a = sample({"mode": "label", "label": "vox", "scale": 3.0})
    assert a.shape == (2, PLANES, ROWS, FRAMES)
    assert torch.equal(a, sample({"mode": "label", "label": "vox", "scale": 3.0}))           # same seed, same notes
    assert float(a[:, 0].min()) >= -1.0 and float(a[:, 4].min()) >= 0.0 and float(a[:, 4].max()) <= 1.0
    # "and" with alpha = 1 is pure vox guidance with the same scale
    assert torch.allclose(a, sample({"mode": "and", "alpha": 1.0, "scale": 3.0}), atol=1e-5)
    curve = sample({"mode": "and", "alpha": torch.linspace(0, 1, FRAMES), "scale": 4.0})   # alpha automated in time
    assert curve.shape == a.shape and torch.isfinite(curve).all()
    assert sample({"mode": "interp", "alpha": 0.5, "scale": 3.0}).shape == a.shape


def test_render_features_reproduces_the_carrier_from_its_own_features():
    image = encode(torch.from_numpy(note()[None]).float(), SPEC)
    rendered = render_features(critic_view(image)[0], image)
    x, y = decode(image)[0].double(), decode(rendered)[0].double()
    snr = 10 * torch.log10((x ** 2).sum() / ((x - y) ** 2).sum())
    assert float(snr) > 15.0                          # same levels, same phase: close to the carrier itself


def test_swin_denoiser_shapes_and_sampling():
    from wavelet2vec.morph_diffusion import SwinDenoiser, build_denoiser
    torch.manual_seed(0)
    model = build_denoiser({"arch": "swin", "v2_config": {"size": "tiny"}})
    assert isinstance(model, SwinDenoiser) and model.grid == (160, 128)
    x = torch.randn(2, PLANES, *model.grid)
    out = model(x, torch.tensor([0.3, 0.7]), torch.tensor([60.0, 67.0]), one_hot([0, 1]))
    assert out.shape == x.shape and float(out.abs().max()) == 0.0               # zero-initialized output layer
    out = model.eval()
    feats = ddim_sample(model, torch.tensor([60.0]), {"mode": "and", "alpha": 0.5, "scale": 4.0}, steps=2,
                        generator=torch.Generator().manual_seed(1), device=torch.device("cpu"))
    assert feats.shape == (1, PLANES, ROWS, FRAMES)
