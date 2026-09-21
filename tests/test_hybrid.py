"""Information-preservation, inverse, gradient and alignment contracts."""
from dataclasses import replace
import math

import pytest
import torch

from wavelet2vec import HybridAudioFrontend, HybridConfig, HybridTokenEncoder


def config(backend="wavelet", **kwargs):
    return replace(HybridConfig(sample_rate=8000, backend=backend, n_bands=16,
        f_min=50, q=3, padding_samples=64, hop_length=32, n_fft=128), **kwargs)


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
@pytest.mark.parametrize("n", [1, 127, 128, 129, 1003])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-6), (torch.float64, 2e-14)])
def test_carrier_roundtrip_without_waveform_fallback(backend, n, dtype, tolerance):
    x = torch.randn(2, 2, n, dtype=dtype, generator=torch.Generator().manual_seed(n))
    encoding = HybridAudioFrontend(config(backend))(x)
    # The inverse must use coefficients; zeroing the retained reference must
    # not turn the test into a trivial raw-audio identity shortcut.
    detached_reference = replace(encoding, waveform=torch.zeros_like(x))
    y = detached_reference.synthesize()
    torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)
    assert encoding.waveform is x
    assert encoding.coefficients.real.dtype == dtype


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
@pytest.mark.parametrize("kind", ["dc", "nyquist", "silence", "impulse"])
def test_endpoints_and_degenerate_signals(backend, kind):
    x = torch.zeros(1, 2, 129, dtype=torch.float64)
    if kind == "dc": x[:] = 1
    elif kind == "nyquist": x[:] = (-1.) ** torch.arange(129)
    elif kind == "impulse": x[..., 0] = 1
    encoded = HybridAudioFrontend(config(backend))(x)
    torch.testing.assert_close(encoded.synthesize(), x, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(encoded.context_features()[0]).all()


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_phase_polarity_gain_and_channel_identity_survive(backend):
    x = torch.randn(1, 1, 1024, dtype=torch.float64)
    stereo = torch.cat((x, -x), dim=1)
    frontend = HybridAudioFrontend(config(backend))
    a, b = frontend(stereo), frontend(-stereo)
    assert a.coefficients.abs().max() > 0
    torch.testing.assert_close(a.coefficients[:, 0], -a.coefficients[:, 1])
    torch.testing.assert_close(a.coefficients, -b.coefficients)
    fa, fb = a.context_features()[0], b.context_features()[0]
    assert (fa-fb).abs().max() > 0.01  # unlike legacy polarity-invariant images
    torch.testing.assert_close(frontend(stereo*.01).synthesize(), stereo*.01)
    torch.testing.assert_close(frontend(stereo.flip(1)).synthesize(), stereo.flip(1))


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_coefficients_have_defined_linear_source_synthesis(backend):
    encoded = HybridAudioFrontend(config(backend))(torch.randn(1, 2, 257, dtype=torch.float64))
    # Cancellation example cannot be represented with real masks limited to [0,1].
    sources = torch.stack((encoded.coefficients*10, encoded.coefficients*-9), dim=1)
    decoded = encoded.synthesize(sources)
    torch.testing.assert_close(decoded.sum(1), encoded.waveform, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(decoded[:, 0], encoded.waveform*10)


def test_wavelet_density_does_not_change_bandwidth_at_shared_centers():
    x = torch.randn(1, 1, 2048, dtype=torch.float64)
    coarse = HybridAudioFrontend(config(n_bands=9, q=2))(x)
    dense = HybridAudioFrontend(config(n_bands=17, q=2))(x)
    torch.testing.assert_close(coarse.centers_hz, dense.centers_hz[::2])
    torch.testing.assert_close(coarse.windows, dense.windows[::2])
    torch.testing.assert_close(coarse.coefficients, dense.coefficients[..., ::2, :])


def test_canonical_wavelet_projection_without_padding_is_orthogonal():
    frontend = HybridAudioFrontend(config(padding_samples=0))
    carrier = frontend(torch.randn(1, 1, 257, dtype=torch.float64))
    a, b = torch.randn_like(carrier.coefficients), torch.randn_like(carrier.coefficients)
    pa = frontend(carrier.synthesize(a)).coefficients
    pb = frontend(carrier.synthesize(b)).coefficients
    ppa = frontend(carrier.synthesize(pa)).coefficients
    torch.testing.assert_close(pa, ppa, rtol=1e-12, atol=1e-12)
    lhs, rhs = (pa.conj()*b).sum().real, (a.conj()*pb).sum().real
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_padding_suppresses_opposite_edge_impulse_for_interior_band():
    cfg = config(n_bands=17, f_min=64, f_max=2048, q=3, padding_samples=0)
    x = torch.zeros(1, 1, 2048, dtype=torch.float64); x[..., -1] = 1
    unpadded = HybridAudioFrontend(cfg)(x)
    padded = HybridAudioFrontend(replace(cfg, padding_samples=1024))(x)
    band = int(torch.argmin(abs(padded.centers_hz-256)))
    before = unpadded.coefficients[..., band, 0].abs()
    after = padded.coefficients[..., band, 1024].abs()
    assert float(after/before) < 1e-4  # this case, not a universal apron guarantee


def test_demodulated_features_keep_stationary_carrier_phase():
    cfg = HybridConfig(sample_rate=8192, n_bands=6, f_min=64, f_max=2048,
                       q=2, padding_samples=0, hop_length=128)
    phase = 0.4
    x = torch.cos(2*torch.pi*256*torch.arange(8192, dtype=torch.float64)/8192+phase)[None,None]
    encoded = HybridAudioFrontend(cfg)(x)
    features, _, _ = encoded.context_features()
    band_features = features.reshape(1, 64, 1, 6, 3)[0, :, 0, 2]
    torch.testing.assert_close(band_features[:, 0], torch.full((64,), math.cos(phase)/2, dtype=x.dtype))
    torch.testing.assert_close(band_features[:, 1], torch.full((64,), math.sin(phase)/2, dtype=x.dtype))


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_tokens_align_partial_patch_and_train_all_branches(backend):
    cfg = config(backend)
    model = HybridTokenEncoder(cfg, channels=2, dim=12).double()
    x = torch.randn(2, 2, 101, dtype=torch.float64, requires_grad=True)
    output = model(x)
    assert output.tokens.shape == (2, 4, 12)
    assert output.valid_samples.tolist() == [32, 32, 32, 5]
    torch.testing.assert_close(output.times_s*8000, torch.tensor([16.,48.,80.,98.5],dtype=x.dtype))
    output.tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for branch in ("waveform", "spectral"):
        assert model(output.encoding, branches=branch).tokens.shape == output.tokens.shape


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_silence_gradients_and_inverse_gradient(backend):
    model = HybridTokenEncoder(config(backend), channels=2, dim=8).double()
    x = torch.zeros(1, 2, 127, dtype=torch.float64, requires_grad=True)
    output = model(x)
    output.tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    x = torch.randn(1, 1, 127, dtype=torch.float64, requires_grad=True)
    y = model.frontend(x).synthesize()
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2*x, rtol=1e-12, atol=1e-12)


def test_numerical_gradient_through_coefficients_and_context():
    frontend = HybridAudioFrontend(config(n_bands=8, q=1, padding_samples=3, hop_length=4))
    x = torch.randn(1, 1, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: frontend(z).context_features()[0], (x,))


@pytest.mark.parametrize("kwargs", [dict(q=0), dict(f_min=0), dict(f_max=5000),
    dict(n_bands=1), dict(padding_samples=-1), dict(bandwidth_floor_hz=-1),
    dict(backend="invalid"), dict(backend="stft", n_fft=127),
    dict(backend="stft", hop_length=100)])
def test_bad_config(kwargs):
    with pytest.raises(ValueError): config(**kwargs)


def test_invalid_input_conditioning_and_allocation_are_rejected():
    frontend = HybridAudioFrontend(config())
    for x in (torch.zeros(10), torch.zeros(1,1,0), torch.zeros(1,1,10,dtype=torch.int16),
              torch.full((1,1,10), float('nan'))):
        with pytest.raises(ValueError): frontend(x)
    with pytest.raises(ValueError, match="ill-conditioned"):
        HybridAudioFrontend(config(q=100))(torch.randn(1,1,1024))
    tiny = HybridAudioFrontend(config(max_coefficient_bytes=16))
    with pytest.raises(ValueError, match="bytes"):
        tiny(torch.randn(1,1,10))
    encoded = frontend(torch.randn(1,1,33))
    assert frontend.coefficient_bytes((1,1,33)) == encoded.coefficients.numel()*encoded.coefficients.element_size()
    with pytest.raises(ValueError): encoded.synthesize(encoded.coefficients.real)


def test_checkpoint_roundtrip_and_configuration_guard():
    cfg = config()
    assert HybridConfig.from_dict(cfg.to_dict()) == cfg
    a, b = HybridTokenEncoder(cfg, channels=1, dim=8), HybridTokenEncoder(cfg, channels=1, dim=8)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1,1,71)
    torch.testing.assert_close(a(x).tokens, b(x).tokens)
    mismatch = HybridTokenEncoder(replace(cfg,q=2), channels=1, dim=8)
    with pytest.raises(ValueError, match="config"):
        mismatch.load_state_dict(a.state_dict())


@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_phase_ablation_changes_features_only(backend):
    x = torch.randn(1,2,311, dtype=torch.float64)
    cfg = config(backend)
    complex_encoding = HybridAudioFrontend(cfg)(x)
    magnitude_encoding = HybridAudioFrontend(replace(cfg, context_mode="magnitude"))(x)
    torch.testing.assert_close(complex_encoding.coefficients, magnitude_encoding.coefficients)
    torch.testing.assert_close(magnitude_encoding.synthesize(), x)
    full = complex_encoding.context_features()[0]
    mag = magnitude_encoding.context_features()[0]
    torch.testing.assert_close(full[..., 2::3], mag[..., 2::3])
    assert mag[..., 0::3].count_nonzero() == mag[..., 1::3].count_nonzero() == 0
    negative = HybridAudioFrontend(magnitude_encoding.config)(-x).context_features()[0]
    torch.testing.assert_close(mag, negative)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["wavelet", "stft"])
def test_cuda_carriers_and_training_match_cpu(backend):
    cfg = config(backend)
    model = HybridTokenEncoder(cfg, channels=2, dim=8)
    gpu_model = HybridTokenEncoder(cfg, channels=2, dim=8).cuda()
    gpu_model.load_state_dict(model.state_dict())
    x = torch.randn(1,2,511)
    cpu = model(x)
    cuda_x = x.cuda().requires_grad_()
    gpu = gpu_model(cuda_x)
    torch.testing.assert_close(gpu.encoding.synthesize().cpu(), x, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(cpu.tokens, gpu.tokens.cpu(), rtol=2e-5, atol=2e-5)
    gpu.tokens.square().mean().backward()
    assert torch.isfinite(cuda_x.grad).all()
