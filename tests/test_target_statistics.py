"""Frozen training-only target calibration and student isolation."""
from dataclasses import replace
import hashlib

import pytest
import torch

from wavelet2vec import (FixedTargetStatistics,HybridConfig,HybridTokenEncoder,MusicContextConfig,
    MaskedPredictionConfig,MaskedMusicModel,collate_audio,MusicPretrainer,PretrainingConfig)


def model(standardize=True):
    analysis=HybridConfig(sample_rate=8192,backend='compact_wavelet',n_bands=8,
                          f_min=64,f_max=2048,q=2,hop_length=16,padding_samples=32)
    return MaskedMusicModel(HybridTokenEncoder(analysis,channels=2,dim=8),
        context_config=MusicContextConfig(dim=8,heads=2,local_layers=1,global_layers=1,local_tokens=4,dropout=0),
        prediction_config=MaskedPredictionConfig(span_tokens=2,standardize_targets=standardize)).double()


def batch():return collate_audio([torch.randn(2,128,dtype=torch.float64)*.1],sample_rate=8192)


def test_weighted_streaming_statistics_and_inverse():
    statistics=FixedTargetStatistics(3,.01).double()
    x=torch.tensor([[1.,2.,7.],[3.,4.,7.]],dtype=torch.float64,requires_grad=True)
    weight=torch.tensor([1,3])
    summary=statistics.fit([(x[:1],weight[:1]),(x[1:],weight[1:])],calibration_id='training-v1')
    mean=(x.detach()*weight[:,None]).sum(0)/4
    variance=((x.detach()-mean).square()*weight[:,None]).sum(0)/4
    torch.testing.assert_close(statistics.mean,mean)
    torch.testing.assert_close(statistics.scale,variance.sqrt().clamp_min(.01))
    torch.testing.assert_close(statistics.inverse(statistics(x)),x)
    assert not statistics.mean.requires_grad and statistics.total_samples==4
    assert summary['calibration_hash']==hashlib.sha256(b'training-v1').hexdigest()
    with pytest.raises(ValueError,match='frozen'):statistics.fit([(x,weight)],calibration_id='validation')


def test_failed_fit_is_atomic_and_restores_gradient_mode():
    network=model();training=batch()
    bad=replace(training,waveform=torch.full_like(training.waveform,float('nan')))
    with pytest.raises(ValueError):network.fit_target_statistics([training,bad],calibration_id='train')
    assert torch.is_grad_enabled()
    assert not network.target_statistics.fitted
    assert network.target_statistics.total_samples==0


def test_unrepresentable_scale_does_not_commit():
    statistics=FixedTargetStatistics(2,scale_floor=1e-60)
    with pytest.raises(ValueError,match='representable'):
        statistics.fit([(torch.zeros(2,2),torch.ones(2,dtype=torch.int64))],calibration_id='train')
    assert not statistics.fitted and statistics.total_samples==0


def test_calibrated_objective_requires_fit_and_does_not_refit_on_queries():
    network=model().eval();training=batch()
    with pytest.raises(ValueError,match='fit target'):network(training)
    network.fit_target_statistics([training],calibration_id='train-only')
    mean=network.target_statistics.mean.clone();scale=network.target_statistics.scale.clone()
    a=batch();x=a.waveform.clone();x[:,:,32:64]+=10
    b=replace(a,waveform=x)
    mask=torch.zeros(1,8,dtype=torch.bool);mask[:,2:4]=True
    first=network(a,mask=mask);second=network(b,mask=mask)
    torch.testing.assert_close(first.predictions,second.predictions,rtol=0,atol=0)
    assert (first.targets-second.targets).abs().max()>.1
    torch.testing.assert_close(network.target_statistics.mean,mean,rtol=0,atol=0)
    torch.testing.assert_close(network.target_statistics.scale,scale,rtol=0,atol=0)
    raw=network.teacher(a.waveform).context_features()[0]
    torch.testing.assert_close(network.from_target_space(first.targets),raw)


def test_calibration_and_training_checkpoint_roundtrip(tmp_path):
    network=model();data=batch()
    network.fit_target_statistics([data],calibration_id='train')
    cfg=PretrainingConfig(warmup_steps=0,total_steps=3)
    trainer=MusicPretrainer(network,cfg);trainer.step(data)
    path=tmp_path/'calibrated.pt';trainer.save_checkpoint(path)
    resumed=MusicPretrainer(model(),cfg)
    assert not resumed.model.target_statistics.fitted
    resumed.load_checkpoint(path)
    assert resumed.model.target_statistics.fitted
    torch.testing.assert_close(resumed.model.target_statistics.mean,network.target_statistics.mean)
    assert resumed.model.target_statistics.summary()==network.target_statistics.summary()


def test_raw_v1_model_state_migrates_without_new_buffers():
    original=model(False)
    state=original.state_dict()
    prediction=dict(state['_extra_state']['prediction'])
    prediction.pop('standardize_targets');prediction.pop('target_scale_floor')
    prediction['version']='masked-prediction-v1'
    state['_extra_state']=dict(state['_extra_state'],prediction=prediction)
    restored=model(False);restored.load_state_dict(state)
    assert restored.target_statistics is None
    assert MaskedPredictionConfig.from_dict(prediction).standardize_targets is False


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_calibrated_cuda_backward_and_masked_gradient():
    network=model().float().cuda()
    train=collate_audio([torch.randn(2,128,device='cuda')*.1],sample_rate=8192)
    network.fit_target_statistics([train],calibration_id='train')
    x=torch.randn(2,128,device='cuda',requires_grad=True)
    query=collate_audio([x],sample_rate=8192)
    mask=torch.zeros(1,8,dtype=torch.bool,device='cuda');mask[:,2:4]=True
    result=network(query,mask=mask);result.loss.backward()
    assert torch.isfinite(result.loss) and torch.isfinite(x.grad).all()
    assert x.grad[:,32:64].count_nonzero()==0
