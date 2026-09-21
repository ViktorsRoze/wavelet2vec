"""Stereo statistics, resolution ablations, synthesis and sliding integration."""
from dataclasses import replace
import math

import pytest
import torch

from wavelet2vec import (HybridAudioFrontend, HybridConfig, HybridTokenEncoder,
    MultiResolutionTokenEncoder, StereoFeatureConfig, StereoTokenEncoder,
    hybrid_stereo_features, SlidingHybridEncoder, SlidingWindowConfig)


def config(backend="compact_wavelet", **kwargs):
    return replace(HybridConfig(sample_rate=8192, backend=backend, n_bands=8, f_min=64,
        f_max=2048, q=2, hop_length=32, n_fft=128, padding_samples=64), **kwargs)


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
@pytest.mark.parametrize("gain", [1., -1., .5, 0.])
def test_stereo_relationships(backend, gain):
    x = torch.randn(1, 1, 1024, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    stereo = torch.cat((x, gain*x), 1)
    encoding = HybridAudioFrontend(config(backend))(stereo)
    f = hybrid_stereo_features(encoding)
    torch.testing.assert_close(encoding.synthesize(), stereo)
    assert f.values.shape == (1, 32, encoding.config.bands, 7)
    assert torch.isfinite(f.values).all()
    active = f.values[..., 5] > 1e-8
    if gain:
        torch.testing.assert_close(f.values[..., 0][f.energy_valid],
                                   torch.full_like(f.values[..., 0][f.energy_valid], math.copysign(1., gain)))
        torch.testing.assert_close(f.values[..., 2][f.energy_valid], torch.ones_like(f.values[..., 2][f.energy_valid]))
    else:
        assert not f.energy_valid.any()
        assert f.values[..., :3].count_nonzero() == 0
    balance = (gain**2-1)/(gain**2+1)
    side = (1-gain)**2/(2*(1+gain**2))
    torch.testing.assert_close(f.values[..., 3][active], torch.full_like(f.values[..., 3][active], balance))
    torch.testing.assert_close(f.values[..., 4][active], torch.full_like(f.values[..., 4][active], side))


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_delay_phase_sign_and_channel_swap(backend):
    cfg = config(backend, n_bands=6, n_fft=256, padding_samples=0)
    t = torch.arange(8192, dtype=torch.float64)/8192
    x = torch.stack((torch.cos(2*torch.pi*256*t), torch.cos(2*torch.pi*256*(t-3/8192))))[None]
    frontend = HybridAudioFrontend(cfg)
    f = hybrid_stereo_features(frontend(x))
    swapped = hybrid_stereo_features(frontend(x.flip(1)))
    k = (f.centers_hz-256).abs().argmin()
    center = f.values[0, 64:192, k]
    torch.testing.assert_close(center[:, 0], torch.full_like(center[:, 0], math.cos(2*math.pi*256*3/8192)), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(center[:, 1], torch.full_like(center[:, 1], math.sin(2*math.pi*256*3/8192)), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(f.values[..., 0], swapped.values[..., 0])
    torch.testing.assert_close(f.values[..., 1], -swapped.values[..., 1])
    torch.testing.assert_close(f.values[..., 3], -swapped.values[..., 3])


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_silence_gradients_and_short_clip_support(backend):
    x = torch.zeros(1, 2, 17, dtype=torch.float64, requires_grad=True)
    f = hybrid_stereo_features(HybridAudioFrontend(config(backend))(x))
    assert f.values[..., :6].count_nonzero() == 0
    assert not f.energy_valid.any()
    assert f.support_fraction.max() < 1
    f.values.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_independent_noise_and_bandwidth_aware_integration():
    cfg = config(n_bands=24, q=4)
    x = torch.randn(1, 2, 8192, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    f = hybrid_stereo_features(HybridAudioFrontend(cfg)(x), StereoFeatureConfig(minimum_seconds=.05))
    high = f.centers_hz > 400
    assert float(f.values[:, 64:192, high, 2].mean()) < .65
    assert f.required_seconds[0] > f.required_seconds[-1]
    assert f.support_fraction[0, 0] < f.support_fraction[128, 0]
    assert f.support_fraction.min() >= 0 and f.support_fraction.max() <= 1


def test_stereo_origin_invariance_and_numeric_gradient():
    cfg = config(n_bands=5, q=1, padding_samples=3, hop_length=4)
    frontend = HybridAudioFrontend(cfg)
    x = torch.randn(1, 2, 11, dtype=torch.float64, requires_grad=True)
    a, b = hybrid_stereo_features(frontend(x)), hybrid_stereo_features(frontend(x, sample_origin=8192))
    torch.testing.assert_close(a.values, b.values)
    torch.testing.assert_close(b.times_s, a.times_s+1)
    assert torch.autograd.gradcheck(lambda y: hybrid_stereo_features(frontend(y)).values, (x,))


def test_arbitrary_channel_pair_keeps_pair_metadata():
    frontend=HybridAudioFrontend(config())
    x=torch.randn(1,3,251,dtype=torch.float64)
    pair=StereoFeatureConfig(pair=(0,2))
    selected=hybrid_stereo_features(frontend(x),pair)
    explicit=hybrid_stereo_features(frontend(x[:,[0,2]]))
    assert selected.pair==(0,2)
    torch.testing.assert_close(selected.values,explicit.values)


def configs():
    return (config(), config("stft", n_fft=64), config("stft", n_fft=256))


def test_resolutions_share_timeline_reconstruct_and_train():
    model = MultiResolutionTokenEncoder(configs(), channels=2, dim=12).double()
    x = torch.randn(2, 2, 301, dtype=torch.float64, requires_grad=True)
    encoded = model.frontend(x, sample_origin=999)
    output = model(encoded, retain_carriers=True)
    assert output.tokens.shape == (2, 10, 12)
    assert output.valid_samples[-1] == 13
    torch.testing.assert_close(output.times_s[0], torch.tensor((999+16)/8192, dtype=torch.float64))
    for carrier in output.carriers:
        torch.testing.assert_close(carrier.synthesize(), x)
        torch.testing.assert_close(carrier.context_features()[1], output.times_s)
    torch.testing.assert_close(output.resolution_weights.sum(-1), torch.ones(2,10,dtype=x.dtype))
    output.tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_disabled_resolution_is_not_analyzed_and_raw_ablation_is_exact():
    model = MultiResolutionTokenEncoder(configs(), channels=2, dim=8).double()
    calls = []
    handle = model.frontends[2].register_forward_hook(lambda *args: calls.append(1))
    x = torch.randn(1,2,65,dtype=torch.float64)
    output = model(x,resolution_mask=(True,True,False))
    assert not calls and output.resolution_weights[..., 3].count_nonzero() == 0
    assert output.carriers[1:] == (None,None)
    raw = model(x,branches="waveform")
    patches = torch.nn.functional.pad(x,(0,31)).unfold(-1,32,32).permute(0,2,1,3).flatten(2)
    torch.testing.assert_close(raw.tokens, model.waveform_projection(patches))
    assert raw.resolution_weights[..., 1:].count_nonzero() == 0
    handle.remove()
    with pytest.raises(ValueError):model(x,branches="spectral",resolution_mask=(False,False,False))


def test_joint_memory_guard_runs_before_primary_analysis():
    model = MultiResolutionTokenEncoder(configs(),channels=2,dim=8,max_total_coefficient_bytes=1)
    calls = []
    handle = model.frontend.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError,match="joint coefficient"):model(torch.randn(1,2,100))
    assert not calls
    handle.remove()


def test_stereo_multiresolution_composition_and_sliding():
    base = MultiResolutionTokenEncoder(configs(),channels=2,dim=8).double()
    model = StereoTokenEncoder(base).double()
    x = torch.randn(1,2,301,dtype=torch.float64,requires_grad=True)
    result = model(x)
    assert result.stereo is not None and result.base.resolution_weights.shape[-1] == 4
    result.tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.stereo_projection.weight.grad.abs().sum() > 0
    torch.testing.assert_close(model(x,stereo=False).tokens,base(x).tokens)
    sliding = SlidingHybridEncoder(model,SlidingWindowConfig(4,2))
    batches = list(sliding.iter_encode(x.detach(),retain_carrier=True))
    torch.testing.assert_close(torch.cat([b.synthesize_core() for b in batches],-1),x)
    assert sum(b.tokens.shape[1] for b in batches) == 10


def test_stereo_local_adapter_checkpoint_and_pair_contract():
    base = HybridTokenEncoder(config(),channels=2,dim=8)
    model = StereoTokenEncoder(base)
    other = StereoTokenEncoder(HybridTokenEncoder(config(),channels=2,dim=8))
    other.load_state_dict(model.state_dict())
    x = torch.randn(1,2,127)
    torch.testing.assert_close(model(x).tokens,other(x).tokens)
    reversed_pair = StereoTokenEncoder(HybridTokenEncoder(config(),channels=2,dim=8),StereoFeatureConfig(pair=(1,0)))
    with pytest.raises(ValueError,match="config"):reversed_pair.load_state_dict(model.state_dict())
    with pytest.raises(ValueError):StereoTokenEncoder(HybridTokenEncoder(config(),channels=1))
    assert StereoFeatureConfig.from_dict(StereoFeatureConfig().to_dict()) == StereoFeatureConfig()


def test_resolution_config_and_checkpoint_guards():
    with pytest.raises(ValueError):MultiResolutionTokenEncoder([config(),config(hop_length=16)])
    with pytest.raises(ValueError):MultiResolutionTokenEncoder([config(),config(sample_rate=16000)])
    model = MultiResolutionTokenEncoder(configs(),channels=2,dim=8)
    other = MultiResolutionTokenEncoder(configs(),channels=2,dim=8)
    other.load_state_dict(model.state_dict())
    changed = (config(q=3),)+configs()[1:]
    with pytest.raises(ValueError,match="config"):
        MultiResolutionTokenEncoder(changed,channels=2,dim=8).load_state_dict(model.state_dict())


def test_sliding_preserves_runtime_adapter_ablations():
    base=MultiResolutionTokenEncoder(configs(),channels=2,dim=8)
    model=StereoTokenEncoder(base)
    sliding=SlidingHybridEncoder(model,SlidingWindowConfig(4,2))
    x=torch.randn(1,2,201)
    calls=[]
    hook=base.frontends[2].register_forward_hook(lambda *args:calls.append(1))
    batches=list(sliding.iter_encode(x,encoder_options={'stereo':False,'resolution_mask':(True,True,False)}))
    assert not calls
    assert batches[0].as_sequence().sample_rate==8192
    hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_combined_cuda_parity_and_silence_backward():
    cpu = StereoTokenEncoder(MultiResolutionTokenEncoder(configs(),channels=2,dim=8))
    gpu = StereoTokenEncoder(MultiResolutionTokenEncoder(configs(),channels=2,dim=8)).cuda()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(1,2,257)
    torch.testing.assert_close(cpu(x).tokens,gpu(x.cuda()).tokens.cpu(),rtol=3e-5,atol=3e-5)
    silence = torch.zeros(1,2,99,device="cuda",requires_grad=True)
    gpu(silence).tokens.square().mean().backward()
    assert torch.isfinite(silence.grad).all()
