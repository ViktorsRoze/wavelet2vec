"""Window ownership, global phase/time coordinates and bounded extraction."""
from dataclasses import replace
import math

import pytest
import torch

from wavelet2vec import (HybridAudioFrontend,HybridConfig,HybridTokenEncoder,
                        SlidingHybridEncoder,SlidingWindowConfig)


def encoder(backend="compact_wavelet"):
    cfg=HybridConfig(sample_rate=8192,backend=backend,n_bands=8,f_min=64,
                     f_max=2048,q=2,hop_length=16,n_fft=64,padding_samples=32)
    return HybridTokenEncoder(cfg,channels=2,dim=8).double()


@pytest.mark.parametrize("backend",["wavelet","compact_wavelet","stft"])
@pytest.mark.parametrize("n",[1,129,509])
def test_window_ownership_timestamps_and_core_reconstruction(backend,n):
    model=SlidingHybridEncoder(encoder(backend),SlidingWindowConfig(4,2))
    audio=torch.randn(1,2,n,dtype=torch.float64)
    batches=list(model.iter_encode(audio,sample_origin=123,retain_carrier=True))
    starts=torch.cat([b.sample_starts for b in batches])
    lengths=torch.cat([b.valid_samples for b in batches])
    times=torch.cat([b.times_s for b in batches])
    torch.testing.assert_close(starts,torch.arange(0,n,16)+123)
    assert lengths.sum()==n and lengths[-1]==((n-1)%16+1)
    torch.testing.assert_close(times,(starts.double()+lengths/2)/8192)
    reconstruction=torch.cat([b.synthesize_core() for b in batches],-1)
    torch.testing.assert_close(reconstruction,audio,rtol=1e-11,atol=1e-11)
    assert all(b.analysis_stop-b.analysis_start<=128 for b in batches)
    assert not batches[0].has_requested_context[0]
    assert not batches[-1].has_requested_context[-1]
    if n==509:assert batches[1].has_requested_context.all()
    if n==129:assert batches[1].has_requested_context.tolist()==[True,True,False,False]


@pytest.mark.parametrize("backend",["wavelet","compact_wavelet","stft"])
def test_raw_patches_are_not_duplicated_or_changed(backend):
    core=encoder(backend)
    sliding=SlidingHybridEncoder(core,SlidingWindowConfig(4,2))
    x=torch.randn(2,2,301,dtype=torch.float64)
    with torch.no_grad():
        a=core(x,branches="waveform")
        b=sliding(x,branches="waveform")
    torch.testing.assert_close(a.tokens,b.tokens)
    torch.testing.assert_close(a.times_s,b.times_s)


@pytest.mark.parametrize("backend",["wavelet","compact_wavelet","stft"])
def test_global_origin_changes_features_not_coefficients(backend):
    cfg=encoder(backend).frontend.config
    frontend=HybridAudioFrontend(cfg)
    x=torch.randn(1,2,99,dtype=torch.float64)
    a=frontend(x);b=frontend(x,sample_origin=4099)
    torch.testing.assert_close(a.coefficients,b.coefficients)
    torch.testing.assert_close(b.synthesize(),x)
    torch.testing.assert_close(b.context_features()[1]-a.context_features()[1],torch.full_like(a.context_features()[1],4099/8192))
    assert (a.context_features()[0]-b.context_features()[0]).abs().max()>.01
    if backend=="compact_wavelet":
        torch.testing.assert_close(b.band_times_s(0)-a.band_times_s(0),torch.full_like(a.band_times_s(0),4099/8192))


@pytest.mark.parametrize("backend",["wavelet","compact_wavelet"])
def test_stationary_tone_phase_matches_across_fragment_origins(backend):
    cfg=HybridConfig(sample_rate=8192,backend=backend,n_bands=6,f_min=64,f_max=2048,
                     q=2,padding_samples=0,hop_length=64)
    frontend=HybridAudioFrontend(cfg)
    features=[]
    for origin in (0,37,1000000037):
        # Reduce sample coordinate modulo the tone period before evaluation.
        phase=2*math.pi*256*((torch.arange(1024,dtype=torch.float64)+origin)%32)/8192+.4
        x=phase.cos()[None,None]
        f=frontend(x,sample_origin=origin).context_features()[0]
        features.append(f.reshape(1,16,1,6,3)[:,:,0,2])
    for f in features[1:]:torch.testing.assert_close(f,features[0],rtol=1e-7,atol=1e-7)


def test_reader_bounds_no_retained_carriers_and_grad_mode_is_restored():
    model=SlidingHybridEncoder(encoder(),SlidingWindowConfig(4,2))
    requests=[]
    def read(lo,hi):
        requests.append((lo,hi))
        return torch.zeros(1,2,hi-lo,dtype=torch.float64)
    iterator=model.iter_from_reader(read,100000)
    assert torch.is_grad_enabled()
    first=next(iterator)
    assert torch.is_grad_enabled()  # no-grad must not leak through yield
    assert len(requests)==1 and requests[0]==(0,96)
    assert first.encoding is None and first.tokens.grad_fn is None
    with pytest.raises(ValueError):first.synthesize_core()
    next(iterator)
    assert requests[-1]==(32,160)
    iterator.close()


def test_training_gradients_and_weighted_multiscale_pooling():
    model=SlidingHybridEncoder(encoder(),SlidingWindowConfig(4,2))
    x=torch.randn(1,2,151,dtype=torch.float64,requires_grad=True)
    fine=model(x)
    coarse=fine.pooled(4)
    assert coarse.tokens.shape==(1,3,8)
    assert coarse.valid_samples.tolist()==[64,64,23]
    expected=(fine.tokens[:,8:]*fine.valid_samples[8:][None,:,None]).sum(1)/23
    torch.testing.assert_close(coarse.tokens[:,-1],expected)
    assert coarse.valid_samples.sum()==151
    coarse.tokens.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()


def test_reader_contract_errors_and_checkpoint_metadata():
    model=SlidingHybridEncoder(encoder(),SlidingWindowConfig(4,2))
    for reader in (lambda a,b:torch.zeros(1,2,b-a),
                   lambda a,b:torch.zeros(1,2,b-a+1,dtype=torch.float64)):
        with pytest.raises(ValueError):list(model.iter_from_reader(reader,100))
    with pytest.raises(ValueError):list(model.iter_from_reader(lambda a,b:None,0))
    other=SlidingHybridEncoder(encoder(),SlidingWindowConfig(3,2))
    with pytest.raises(ValueError,match="config"):other.load_state_dict(model.state_dict())
    same=SlidingHybridEncoder(encoder(),SlidingWindowConfig(4,2))
    same.load_state_dict(model.state_dict())


def test_file_extraction_matches_tensor_and_rejects_rate(tmp_path):
    import soundfile as sf
    x=torch.randn(1,2,301,dtype=torch.float64)*.1
    path=tmp_path/'sample.wav'
    sf.write(path,x[0].T.numpy(),8192,subtype='DOUBLE')
    model=SlidingHybridEncoder(encoder(),SlidingWindowConfig(4,2))
    actual=list(model.iter_file(path))
    expected=list(model.iter_encode(x))
    for a,b in zip(actual,expected):torch.testing.assert_close(a.tokens,b.tokens)
    sf.write(path,x[0].T.numpy(),16000,subtype='DOUBLE')
    with pytest.raises(ValueError,match="sample rate"):list(model.iter_file(path))


def test_seconds_round_to_whole_tokens():
    cfg=SlidingWindowConfig.from_seconds(2,.5,sample_rate=44100,hop_length=256)
    assert cfg.core_tokens==345 and cfg.context_tokens==87
    assert SlidingWindowConfig.from_dict(cfg.to_dict())==cfg
    for args in ((0,0),(1,-1),(True,1)):
        with pytest.raises(ValueError):SlidingWindowConfig(*args)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cpu_reader_to_gpu_windows():
    model=SlidingHybridEncoder(encoder().float().cuda(),SlidingWindowConfig(4,2))
    x=torch.randn(1,2,141)
    batches=list(model.iter_encode(x,retain_carrier=True))
    assert x.device.type=='cpu'
    assert all(b.tokens.device.type=='cuda' for b in batches)
    torch.testing.assert_close(torch.cat([b.synthesize_core() for b in batches],-1).cpu(),x,rtol=2e-5,atol=2e-5)
