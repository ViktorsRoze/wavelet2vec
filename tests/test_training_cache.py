"""Source supervision and inference-cache validity, including padded tails."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from wavelet2vec import (SourceExample,collate_sources,separation_loss,FeatureCache,FeatureCacheKey,
    model_fingerprint,source_fingerprint,HybridConfig,HybridTokenEncoder,SlidingHybridEncoder,
    SlidingWindowConfig,HybridSequence)


def source_batch():
    a=torch.linspace(-.5,.5,65,dtype=torch.float64)[None]
    b=torch.sin(torch.arange(33,dtype=torch.float64))[None]*.2
    examples=[SourceExample(a,{'bass':a,'drums':torch.zeros_like(a)},8192,True),
              SourceExample(b,{'bass':b},8192,False)]
    return collate_sources(examples,source_order=('bass','drums'))


def test_padding_missing_and_explicit_absent_sources():
    batch=source_batch()
    assert batch.sources.shape==(2,2,1,65)
    assert batch.lengths.tolist()==[65,33]
    assert batch.source_valid.tolist()==[[True,True],[True,False]]
    assert batch.sample_mask.sum()==98
    assert batch.mixture[1,:,33:].count_nonzero()==0
    assert batch.to('cpu').source_order==('bass','drums')


def test_loss_ignores_padding_and_missing_annotations_but_detects_phase():
    batch=source_batch()
    estimate=batch.sources.clone()
    estimate[1,1]=float('nan')
    estimate[1,0,:,33:]=float('nan')
    result=separation_loss(estimate,batch,fft_sizes=(16,64))
    assert result['loss']==0
    inverted=separation_loss(-batch.sources,batch,fft_sizes=(16,64))
    assert inverted['waveform_l1']>0 and inverted['complex_spectral_l1']>0
    known_absent=batch.sources.clone();known_absent[0,1]=.1
    assert separation_loss(known_absent,batch,fft_sizes=(16,))['loss']>0


def test_reconstruction_loss_gradients_and_padding_mask():
    batch=source_batch()
    estimate=(batch.sources+.02).requires_grad_()
    loss=separation_loss(estimate,batch,fft_sizes=(16,64))['loss']
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
    assert estimate.grad[1,1].count_nonzero()==0
    assert estimate.grad[1,0,:,33:].count_nonzero()==0
    assert estimate.grad[0,0].abs().sum()>0


def test_stereo_loss_normalization_and_silent_gradient():
    x=torch.zeros(2,1,dtype=torch.float64)
    batch=collate_sources([SourceExample(x,{'a':x},8192,True)],source_order=('a',))
    one=torch.ones_like(batch.sources)
    assert separation_loss(one,batch,spectral_weight=0)['waveform_l1']==1
    exact=batch.sources.clone().requires_grad_()
    separation_loss(exact,batch,fft_sizes=(16,64))['loss'].backward()
    assert torch.isfinite(exact.grad).all() and exact.grad.count_nonzero()==0


def test_additive_consistency_is_not_applied_to_dry_or_incomplete_targets():
    batch=source_batch()
    with pytest.raises(ValueError,match='complete additive'):
        separation_loss(batch.sources,batch,mixture_weight=1)
    x=torch.randn(1,31,dtype=torch.float64)
    additive=collate_sources([SourceExample(x,{'a':x*.4,'b':x*.6},8192,True)],source_order=('a','b'))
    result=separation_loss(additive.sources,additive,fft_sizes=(16,),mixture_weight=1)
    assert result['loss']<1e-12
    with pytest.raises(ValueError,match='sum'):
        collate_sources([SourceExample(x,{'a':x*2},8192,True)],source_order=('a',))


def test_source_contract_validation():
    x=torch.zeros(1,5)
    for example in (SourceExample(x,{},8192),SourceExample(x,{'extra':x},8192),
                    SourceExample(x,{'a':x[:,:4]},8192),SourceExample(x,{'a':x},8192,True)):
        with pytest.raises(ValueError):collate_sources([example],source_order=('a','b'))
    with pytest.raises(ValueError):
        collate_sources([SourceExample(x,{'a':x},8192),SourceExample(x,{'a':x},16000)],source_order=('a',))


def cached_sequence():
    cfg=HybridConfig(sample_rate=8192,backend='compact_wavelet',n_bands=8,f_min=64,
                     f_max=2048,q=2,hop_length=16,padding_samples=32)
    model=SlidingHybridEncoder(HybridTokenEncoder(cfg,channels=2,dim=8),SlidingWindowConfig(4,2)).eval()
    x=torch.randn(1,2,131)
    with torch.no_grad():sequence=model(x,sample_origin=100)
    key=FeatureCacheKey.for_model(hashlib.sha256(b'source').hexdigest(),model,
        sample_rate=8192,channels=2,sample_origin=100,processing={'branches':'both','range':[0,131]})
    return model,sequence,key


def test_feature_cache_roundtrip_and_no_pickle(tmp_path):
    model,sequence,key=cached_sequence()
    cache=FeatureCache(tmp_path)
    assert cache.load(key) is None
    path=cache.save(key,sequence)
    result=cache.load(key)
    torch.testing.assert_close(result.tokens,sequence.tokens)
    torch.testing.assert_close(result.times_s,sequence.times_s)
    assert not result.tokens.requires_grad
    assert path.name==key.digest+'.npz'
    with np.load(path,allow_pickle=False) as stored:
        assert json.loads(stored['metadata'].item())==key.to_dict()
    assert not list(tmp_path.glob('*.tmp'))


def test_cache_key_changes_for_model_source_runtime_choices_and_origin():
    model,sequence,key=cached_sequence()
    before=model_fingerprint(model)
    with torch.no_grad():next(model.parameters()).add_(.001)
    assert model_fingerprint(model)!=before
    assert replace(key,source_hash='0'*64).digest!=key.digest
    assert replace(key,sample_origin=101).digest!=key.digest
    assert replace(key,library_version='future').digest!=key.digest
    assert replace(key,processing_json='{"branches":"waveform"}').digest!=key.digest
    model.train()
    with pytest.raises(ValueError,match='eval'):
        FeatureCacheKey.for_model(key.source_hash,model,sample_rate=8192,channels=2)


def test_corrupt_and_mismatched_cache_is_a_miss(tmp_path):
    _,sequence,key=cached_sequence()
    cache=FeatureCache(tmp_path)
    path=cache.save(key,sequence)
    path.write_bytes(b'not an array')
    assert cache.load(key) is None
    path=cache.save(key,sequence)
    other=replace(key,processing_json='{"branches":"spectral"}')
    (tmp_path/(other.digest+'.npz')).write_bytes(path.read_bytes())
    assert cache.load(other) is None


def test_cache_rejects_training_graph_and_wrong_coordinates(tmp_path):
    _,sequence,key=cached_sequence()
    cache=FeatureCache(tmp_path)
    with pytest.raises(ValueError,match='training'):
        cache.save(key,replace(sequence,tokens=sequence.tokens.clone().requires_grad_()))
    with pytest.raises(ValueError,match='time'):
        cache.save(key,replace(sequence,times_s=sequence.times_s+1))
    with pytest.raises(ValueError,match='origin'):
        cache.save(replace(key,sample_origin=0),sequence)


def test_file_hash_and_canonical_parameters(tmp_path):
    source=tmp_path/'source.bin';source.write_bytes(b'audio bytes')
    assert source_fingerprint(source)==hashlib.sha256(b'audio bytes').hexdigest()
    a=FeatureCacheKey('0'*64,'1'*64,44100,2,processing_json='{"b":2,"a":1}')
    b=replace(a,processing_json='{"a":1, "b":2}')
    assert a.digest==b.digest
    with pytest.raises(ValueError):replace(a,processing_json='{"x":NaN}')


def test_editable_package_code_changes_invalidate_model_identity(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('wavelet2vec.feature_cache')
    source=tmp_path/'helper.py'
    source.write_text('value = 1\n',encoding='utf-8')
    monkeypatch.setattr(module,'__file__',str(tmp_path/'feature_cache.py'))
    model=torch.nn.Linear(2,2).eval()
    before=model_fingerprint(model)
    source.write_text('value = 2\n',encoding='utf-8')
    assert model_fingerprint(model)!=before


def test_long_coarse_interval_keeps_half_sample_precision(tmp_path):
    # No long audio allocation: metadata alone exposes float32 length/2 loss.
    length=torch.tensor([2**25+1],dtype=torch.int64)
    starts=torch.tensor([123],dtype=torch.int64)
    sequence=HybridSequence(torch.ones(1,1,2),starts,length,
        (starts.double()+length.double()/2)/44100,torch.tensor([True]),44100)
    pooled=sequence.pooled(1)
    torch.testing.assert_close(pooled.times_s,sequence.times_s,rtol=0,atol=1e-12)
    key=FeatureCacheKey('0'*64,'1'*64,44100,2,sample_origin=123)
    cache=FeatureCache(tmp_path);cache.save(key,pooled)
    torch.testing.assert_close(cache.load(key).times_s,sequence.times_s)
