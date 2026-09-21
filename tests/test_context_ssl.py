"""Learned context, raw-mask isolation and exact trainer resume contracts."""
from dataclasses import replace

import pytest
import torch

from wavelet2vec import (HybridAudioFrontend,HybridConfig,HybridTokenEncoder,MultiResolutionTokenEncoder,
    StereoTokenEncoder,MusicContextConfig,MusicContextEncoder,MaskedPredictionConfig,MaskedMusicModel,
    AudioBatch,collate_audio,sample_span_mask,PretrainingConfig,MusicPretrainer,LengthBucketSampler)


def context_config(**kwargs):
    return replace(MusicContextConfig(dim=8,heads=2,local_layers=1,global_layers=1,
                   local_tokens=4,dropout=0.,max_groups=32),**kwargs)


def model(kind="compact_wavelet",dropout=0.):
    cfg=HybridConfig(sample_rate=8192,backend="compact_wavelet" if kind=="multi_stereo" else kind,
                     n_bands=8,f_min=64,f_max=2048,q=2,hop_length=16,n_fft=64,padding_samples=32)
    if kind=="multi_stereo":
        encoder=StereoTokenEncoder(MultiResolutionTokenEncoder([cfg,replace(cfg,backend='stft')],channels=2,dim=8))
    else:encoder=HybridTokenEncoder(cfg,channels=2,dim=8)
    return MaskedMusicModel(encoder,context_config=context_config(dropout=dropout),
                           prediction_config=MaskedPredictionConfig(span_tokens=2))


def test_global_context_changes_distant_tokens_while_local_states_stay_local():
    torch.manual_seed(3)
    encoder=MusicContextEncoder(context_config()).double().eval()
    x=torch.randn(1,12,8,dtype=torch.float64)
    times=torch.arange(12,dtype=torch.float64)/32
    changed=x.clone();changed[:,9:,0]+=4
    a=encoder(x,times_s=times,return_hidden_states=True)
    b=encoder(changed,times_s=times)
    torch.testing.assert_close(a.local_tokens[:,:4],b.local_tokens[:,:4])
    assert (a.tokens[:,:4]-b.tokens[:,:4]).abs().max()>1e-5
    assert len(a.hidden_states)==2
    assert a.hidden_states[0].shape==(1,12,8) and a.hidden_states[1].shape==(1,3,8)
    shifted=encoder(x,times_s=times+2**20)
    torch.testing.assert_close(a.tokens,shifted.tokens)


def test_padding_empty_blocks_and_gradients():
    encoder=MusicContextEncoder(context_config()).double()
    valid=torch.arange(12)[None]<torch.tensor([7,2])[:,None]
    x=torch.randn(2,12,8,dtype=torch.float64)
    x[~valid]=float('nan');x.requires_grad_()
    times=torch.arange(12,dtype=torch.float64)[None].expand(2,-1).clone()/32
    times[~valid]=float('nan')
    output=encoder(x,times_s=times,valid_mask=valid)
    assert torch.isfinite(output.tokens).all()
    assert output.tokens[~valid].count_nonzero()==0
    assert output.coarse_tokens[~output.coarse_valid_mask].count_nonzero()==0
    (output.tokens*torch.randn_like(output.tokens)).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[~valid].count_nonzero()==0


def test_attention_guard_before_attention_and_invalid_metadata():
    encoder=MusicContextEncoder(context_config(max_attention_elements=1))
    calls=[]
    handle=encoder.local_layers[0].register_forward_hook(lambda *args:calls.append(1))
    with pytest.raises(ValueError,match='budget'):encoder(torch.randn(1,8,8),times_s=torch.arange(8.))
    assert not calls;handle.remove()
    encoder=MusicContextEncoder(context_config())
    with pytest.raises(ValueError):encoder(torch.randn(1,8,8),times_s=torch.arange(8.),valid_mask=torch.zeros(1,8,dtype=torch.bool))
    with pytest.raises(ValueError):encoder(torch.randn(1,8,8),times_s=torch.zeros(8))


def test_coarse_time_uses_real_duration_weights():
    encoder=MusicContextEncoder(context_config()).double()
    samples=torch.tensor([16,16,16,1])
    times=torch.tensor([8.,24.,40.,48.5],dtype=torch.float64)/8192
    output=encoder(torch.randn(1,4,8,dtype=torch.float64),times_s=times,valid_samples=samples)
    torch.testing.assert_close(output.coarse_times_s,torch.tensor([[24.5/8192]],dtype=torch.float64))


def test_seeded_span_masks_respect_padding_and_visible_context():
    valid=torch.arange(10)[None]<torch.tensor([10,5,1])[:,None]
    cfg=MaskedPredictionConfig(mask_fraction=.6,span_tokens=3)
    a=sample_span_mask(valid,cfg,generator=torch.Generator().manual_seed(7))
    b=sample_span_mask(valid,cfg,generator=torch.Generator().manual_seed(7))
    assert torch.equal(a,b) and not (a&~valid).any()
    assert a.sum(1).tolist()==[6,3,0]
    assert ((valid&~a).sum(1)>0).all()


@pytest.mark.parametrize("kind",["wavelet","compact_wavelet","stft","multi_stereo"])
def test_hidden_audio_cannot_leak_through_any_student_view(kind):
    torch.manual_seed(4)
    network=model(kind).double().eval()
    a=torch.randn(2,128,dtype=torch.float64)
    b=a.clone();b[:,32:64]=20*torch.randn_like(b[:,32:64])
    mask=torch.tensor([[False,False,True,True,False,False,False,False]])
    ba,bb=collate_audio([a],sample_rate=8192),collate_audio([b],sample_rate=8192)
    teacher_calls=[]
    hook=network.teacher.register_forward_hook(lambda *args:teacher_calls.append(1))
    sa=network.student_forward(ba,mask);sb=network.student_forward(bb,mask)
    assert not teacher_calls  # student path never accesses clean teacher
    torch.testing.assert_close(sa.predictions,sb.predictions,rtol=0,atol=0)
    ta,tb=network(ba,mask=mask).targets,network(bb,mask=mask).targets
    assert (ta-tb).abs().max()>.01
    assert not ta.requires_grad
    hook.remove()


def test_masked_sample_gradient_is_zero_with_clean_teacher_targets():
    network=model('multi_stereo').double()
    x=torch.randn(2,128,dtype=torch.float64,requires_grad=True)
    batch=collate_audio([x],sample_rate=8192,sample_origins=[1003])
    mask=torch.tensor([[False,False,True,True,False,False,False,False]])
    output=network(batch,mask=mask)
    output.loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:,32:64].count_nonzero()==0
    assert x.grad[:,:32].abs().sum()>0
    assert not output.targets.requires_grad
    assert network.mask_embedding.grad.abs().sum()>0
    assert all(p.grad is None for p in network.teacher.parameters())


def test_ragged_audio_padding_origins_and_short_clip_masks():
    network=model().double().eval()
    batch=collate_audio([torch.randn(2,77,dtype=torch.float64),torch.randn(2,128,dtype=torch.float64)],
                        sample_rate=8192,sample_origins=[999,3000])
    mask=torch.zeros(2,8,dtype=torch.bool);mask[0,4]=True;mask[1,2:4]=True
    output=network(batch,mask=mask)
    assert output.student.valid_samples[0,4]==13
    assert output.metrics['masked_samples']==45
    poisoned=batch.waveform.clone();poisoned[0,:,77:]=float('nan')
    other=network(replace(batch,waveform=poisoned),mask=mask)
    torch.testing.assert_close(output.predictions,other.predictions)
    torch.testing.assert_close(output.targets,other.targets)
    assert output.predictions[0,5:].count_nonzero()==0
    with pytest.raises(ValueError,match='padded'):network(batch,mask=mask|~output.student.valid_mask)
    with pytest.raises(ValueError,match='no masked'):
        network(collate_audio([torch.zeros(2,1,dtype=torch.float64)],sample_rate=8192))


def test_rejects_clean_carriers_as_student_input_and_empty_context():
    network=model()
    batch=collate_audio([torch.randn(2,64)],sample_rate=8192)
    with pytest.raises(ValueError,match='AudioBatch'):network(network.teacher(batch.waveform))
    with pytest.raises(ValueError,match='visible'):network(batch,mask=torch.ones(1,4,dtype=torch.bool))
    assert network.encode(batch).tokens.shape==(1,4,8)


def test_resume_reproduces_next_mask_dropout_loss_and_update(tmp_path):
    torch.manual_seed(11)
    batch=collate_audio([torch.randn(2,128),torch.randn(2,111)],sample_rate=8192)
    cfg=PretrainingConfig(warmup_steps=1,total_steps=4,mask_seed=91)
    trainer=MusicPretrainer(model(dropout=.2),cfg)
    trainer.step(batch)
    path=tmp_path/'resume.pt'
    trainer.save_checkpoint(path,data_state={'next_batch':1})
    expected=trainer.step(batch)
    saved={k:v.clone() for k,v in trainer.model.state_dict().items() if isinstance(v,torch.Tensor)}
    resumed=MusicPretrainer(model(dropout=.2),cfg)
    assert resumed.load_checkpoint(path)=={'next_batch':1}
    actual=resumed.step(batch)
    assert actual['loss']==expected['loss'] and actual['update_step']==2
    for key,value in saved.items():torch.testing.assert_close(resumed.model.state_dict()[key],value,rtol=0,atol=0)


def test_accumulation_checkpoint_boundary_and_training_budget(tmp_path):
    batch=collate_audio([torch.randn(2,96)],sample_rate=8192)
    cfg=PretrainingConfig(warmup_steps=0,total_steps=1,accumulation_steps=2)
    trainer=MusicPretrainer(model(),cfg)
    assert not trainer.step(batch)['updated']
    with pytest.raises(ValueError,match='boundary'):trainer.save_checkpoint(tmp_path/'pending.pt')
    assert trainer.step(batch)['updated']
    trainer.save_checkpoint(tmp_path/'finished.pt')
    with pytest.raises(ValueError,match='budget'):trainer.step(batch)
    mismatch=MusicPretrainer(model(),replace(cfg,total_steps=2))
    with pytest.raises(ValueError,match='configuration'):mismatch.load_checkpoint(tmp_path/'finished.pt')


def test_failed_batch_discards_pending_gradients_without_updating():
    batch=collate_audio([torch.randn(2,96)],sample_rate=8192)
    trainer=MusicPretrainer(model(),PretrainingConfig(warmup_steps=0,total_steps=2,accumulation_steps=2))
    before=[p.detach().clone() for p in trainer.model.parameters()]
    trainer.step(batch)
    bad=batch.waveform.clone();bad[...,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):trainer.step(replace(batch,waveform=bad))
    assert trainer.pending==0 and trainer.update_step==0
    for a,p in zip(before,trainer.model.parameters()):
        torch.testing.assert_close(a,p,rtol=0,atol=0)
        assert p.grad is None


def test_checkpoint_model_mismatch_is_checked_before_weight_loading(tmp_path):
    cfg=PretrainingConfig(warmup_steps=0,total_steps=2)
    trainer=MusicPretrainer(model(),cfg)
    path=tmp_path/'state.pt';trainer.save_checkpoint(path)
    altered=model()
    altered.context=MusicContextEncoder(context_config(local_tokens=8))
    other=MusicPretrainer(altered,cfg)
    before=[p.detach().clone() for p in other.model.parameters()]
    with pytest.raises(ValueError,match='model configuration'):other.load_checkpoint(path)
    for a,p in zip(before,other.model.parameters()):torch.testing.assert_close(a,p,rtol=0,atol=0)


def test_checkpoint_rejects_changed_implementation(tmp_path,monkeypatch):
    import wavelet2vec.pretraining as module
    trainer=MusicPretrainer(model(),PretrainingConfig(warmup_steps=0,total_steps=2))
    path=tmp_path/'state.pt';trainer.save_checkpoint(path)
    monkeypatch.setattr(module,'implementation_fingerprint',lambda:'different')
    with pytest.raises(ValueError,match='implementation'):trainer.load_checkpoint(path)


def test_silent_targets_have_finite_gradients_and_zero_target_variance():
    network=model().double()
    result=network(collate_audio([torch.zeros(2,128,dtype=torch.float64)],sample_rate=8192),
                   generator=torch.Generator().manual_seed(5))
    assert torch.isfinite(result.loss) and result.metrics['target_std']==0
    result.loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())


def test_length_buckets_cover_dataset_and_are_epoch_reproducible():
    sampler=LengthBucketSampler([30,10,55,20,90,11,17,9,8,40,70],3,bucket_multiplier=2,seed=4)
    a=list(sampler)
    assert a==list(sampler)
    assert sorted(i for batch in a for i in batch)==list(range(11))
    assert len(a)==len(sampler)==4
    sampler.set_epoch(1)
    assert a!=list(sampler)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_ssl_backward_and_mask_isolation():
    network=model('multi_stereo').cuda().eval()
    x=torch.randn(2,128,device='cuda',requires_grad=True)
    batch=collate_audio([x],sample_rate=8192)
    mask=torch.zeros(1,8,dtype=torch.bool,device='cuda');mask[:,2:4]=True
    result=network(batch,mask=mask)
    result.loss.backward()
    assert torch.isfinite(result.loss) and torch.isfinite(x.grad).all()
    assert x.grad[:,32:64].count_nonzero()==0
