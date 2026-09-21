"""Compact carrier invertibility, sampling, learning and allocation contracts."""
from dataclasses import replace
import math

import pytest
import torch

from wavelet2vec import HybridAudioFrontend, HybridConfig, HybridTokenEncoder


def config(**kwargs):
    return replace(HybridConfig(backend="compact_wavelet",sample_rate=8000,
        n_bands=16,f_min=50,q=3,padding_samples=64,hop_length=32),**kwargs)


@pytest.mark.parametrize("n", [1,2,127,128,129,1003])
@pytest.mark.parametrize("dtype,tol", [(torch.float32,2e-6),(torch.float64,2e-13)])
@pytest.mark.parametrize("mode", ["fast","power2"])
def test_roundtrip_and_ragged_allocation(n,dtype,tol,mode):
    x=torch.randn(2,2,n,dtype=dtype,generator=torch.Generator().manual_seed(n))
    frontend=HybridAudioFrontend(config(compact_fft_mode=mode))
    encoding=frontend(x)
    assert isinstance(encoding.coefficients,tuple)
    assert frontend.coefficient_bytes(tuple(x.shape),dtype)==encoding.coefficient_nbytes
    no_raw=replace(encoding,waveform=torch.zeros_like(x))
    torch.testing.assert_close(no_raw.synthesize(),x,rtol=tol,atol=tol)
    assert all(b.samples>=b.stop_bin-b.start_bin for b in encoding.band_specs)


@pytest.mark.parametrize("kind",["silence","dc","nyquist","impulse","antiphase"])
def test_endpoints_and_stereo(kind):
    x=torch.zeros(1,2,256,dtype=torch.float64)
    if kind=="dc": x[:]=1
    if kind=="nyquist": x[:]=(-1.)**torch.arange(256)
    if kind=="impulse": x[...,0]=1
    if kind=="antiphase":
        x[:,0]=torch.randn(256);x[:,1]=-x[:,0]
    e=HybridAudioFrontend(config())(x)
    torch.testing.assert_close(e.synthesize(),x,rtol=1e-12,atol=1e-12)
    assert torch.isfinite(e.context_features()[0]).all()


def test_compact_samples_equal_independent_fourier_evaluation():
    cfg=config(n_bands=8,q=1.5,padding_samples=7)
    x=torch.randn(1,1,127,dtype=torch.float64)
    e=HybridAudioFrontend(cfg)(x)
    n=x.shape[-1]+2*cfg.padding
    fft=torch.fft.rfft(torch.nn.functional.pad(x,(cfg.padding,cfg.padding)))
    a=torch.full((n//2+1,),2.,dtype=x.dtype);a[0]=1
    if n%2==0: a[-1]=1
    for index in [0,3,7]:
        b=e.band_specs[index]
        bins=torch.arange(b.start_bin,b.stop_bin)
        j=torch.arange(b.samples,dtype=x.dtype)
        angles=2*torch.pi*(bins[:,None].to(x.dtype)-b.carrier_bin)*j/b.samples
        basis=torch.complex(angles.cos(),angles.sin())
        expected=(fft[...,bins]*a[bins]*e.windows[index])@basis/n
        torch.testing.assert_close(e.coefficients[index],expected,rtol=1e-11,atol=1e-11)
        times=e.band_times_s(index)
        torch.testing.assert_close(times,(j*n/b.samples-cfg.padding)/cfg.sample_rate)


def test_modified_bands_source_axis_and_reanalysis_consistency():
    frontend=HybridAudioFrontend(config())
    x=torch.randn(1,2,257,dtype=torch.float64)
    e=frontend(x)
    source_bands=tuple(torch.stack((10*b,-9*b),dim=1) for b in e.coefficients)
    sources=e.synthesize(source_bands)
    torch.testing.assert_close(sources[:,0],10*x)
    torch.testing.assert_close(sources.sum(1),x,rtol=1e-12,atol=1e-12)
    random_bands=tuple(torch.randn_like(b) for b in e.coefficients)
    audio=e.synthesize(random_bands)
    torch.testing.assert_close(frontend(audio).synthesize(),audio)
    zero=tuple(torch.zeros_like(b) for b in e.coefficients)
    assert e.synthesize(zero).count_nonzero()==0


def test_center_tone_phase_and_integrated_patch_alignment():
    cfg=config(sample_rate=8192,n_bands=6,f_min=64,f_max=2048,q=2,
               padding_samples=0,hop_length=127)
    phase=.4
    x=torch.cos(2*torch.pi*256*torch.arange(8192,dtype=torch.float64)/8192+phase)[None,None]
    e=HybridAudioFrontend(cfg)(x)
    features,times,lengths=e.context_features()
    band=features.reshape(1,len(lengths),1,6,3)[0,:,0,2]
    torch.testing.assert_close(band[:,0],torch.full_like(band[:,0],math.cos(phase)/2),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(band[:,1],torch.full_like(band[:,1],math.sin(phase)/2),rtol=1e-10,atol=1e-10)
    assert lengths.sum()==8192 and lengths[-1]==64
    torch.testing.assert_close(times[-1],torch.tensor((8128+32)/8192,dtype=times.dtype))
    inverted=HybridAudioFrontend(cfg)(-x)
    assert (features-inverted.context_features()[0]).abs().max()>.1


def test_gradcheck_inverse_and_context():
    frontend=HybridAudioFrontend(config(n_bands=5,q=1,padding_samples=3,hop_length=4))
    x=torch.randn(1,1,9,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:frontend(z).context_features()[0],(x,))
    y=frontend(x).synthesize()
    y.square().sum().backward()
    torch.testing.assert_close(x.grad,2*x,rtol=1e-12,atol=1e-12)


def test_learning_and_silence_gradients():
    model=HybridTokenEncoder(config(),channels=2,dim=12).double()
    for silent in (False,True):
        x=(torch.zeros(1,2,127) if silent else torch.randn(1,2,127)).double().requires_grad_()
        model.zero_grad()
        output=model(x)
        assert output.tokens.shape==(1,4,12)
        output.tokens.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_compression_and_guard_before_dense_allocation():
    cfg=config(sample_rate=48000,n_bands=64,f_min=20,q=8,padding_samples=12000)
    compact=HybridAudioFrontend(cfg)
    dense=HybridAudioFrontend(replace(cfg,backend="wavelet"))
    shape=(1,2,48000)
    assert dense.coefficient_bytes(shape)/compact.coefficient_bytes(shape)>8
    limit=compact.coefficient_bytes(shape)
    x=torch.zeros(shape)
    with pytest.raises(ValueError,match="bytes"):
        HybridAudioFrontend(replace(cfg,backend="wavelet",max_coefficient_bytes=limit))(x)
    result=HybridAudioFrontend(replace(cfg,max_coefficient_bytes=limit))(x)
    assert result.coefficient_nbytes==limit
    assert sum(w.numel() for w in result.windows) < cfg.n_bands*(shape[-1]+2*cfg.padding)/4


def test_invalid_replacements_and_conditioning():
    e=HybridAudioFrontend(config())(torch.randn(1,2,128))
    for invalid in (e.coefficients[:-1],list(e.coefficients),
                    (e.coefficients[0].real,)+e.coefficients[1:],
                    (e.coefficients[0][...,:1],)+e.coefficients[1:]):
        with pytest.raises(ValueError):e.synthesize(invalid)
    with pytest.raises(ValueError,match="ill-conditioned"):
        HybridAudioFrontend(config(q=100,compact_truncate_sigma=1))(torch.randn(1,1,1024))
    with pytest.raises(ValueError):config(compact_truncate_sigma=0)


def test_v1_checkpoint_config_migration():
    data=config(backend="wavelet").to_dict()
    data['version']='hybrid-v1'
    for key in ('compact_truncate_sigma','compact_oversampling','compact_fft_mode'):data.pop(key)
    assert HybridConfig.from_dict(data)==config(backend="wavelet")
    data['backend']='compact_wavelet'
    with pytest.raises(ValueError):HybridConfig.from_dict(data)


def test_oversampling_preserves_filters_and_audio():
    x=torch.randn(1,2,511,dtype=torch.float64)
    minimal=HybridAudioFrontend(config())(x)
    oversampled=HybridAudioFrontend(config(compact_oversampling=2,compact_fft_mode='power2'))(x)
    for a,b in zip(minimal.windows,oversampled.windows):torch.testing.assert_close(a,b)
    torch.testing.assert_close(oversampled.synthesize(),x)
    assert oversampled.coefficient_nbytes>minimal.coefficient_nbytes
    assert torch.isfinite(oversampled.context_features()[0]).all()


def test_supervised_compact_head_backpropagates():
    from experiments.hybrid_foundation import _Separator, scenes
    mixture,target=scenes(2,91,n=256)
    cfg=config(n_bands=8,q=1.5,hop_length=32)
    model=_Separator(cfg,'both',dim=8)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(2):
        output=model(model.encoder.frontend(mixture))
        loss=(output-target).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert model.encoder.waveform_projection.weight.grad.abs().sum()>0
    assert model.encoder.spectral_projection.weight.grad.abs().sum()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["fast","power2"])
def test_cuda_parity_and_gradient(mode):
    cfg=config(compact_fft_mode=mode)
    cpu=HybridTokenEncoder(cfg,channels=2,dim=8)
    gpu=HybridTokenEncoder(cfg,channels=2,dim=8).cuda()
    gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(1,2,511)
    reference=cpu(x)
    x_cuda=x.cuda().requires_grad_()
    actual=gpu(x_cuda)
    torch.testing.assert_close(actual.encoding.synthesize().cpu(),x,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(actual.tokens.cpu(),reference.tokens,rtol=2e-5,atol=2e-5)
    actual.tokens.square().mean().backward()
    assert torch.isfinite(x_cuda.grad).all()
