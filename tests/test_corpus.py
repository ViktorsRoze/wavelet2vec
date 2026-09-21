"""Data provenance, deterministic decoding and resumable corpus evaluation."""
from dataclasses import replace
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import (AudioRecord,AudioManifest,AudioCropConfig,AudioWindowDataset,collate_windows,
    HybridConfig,HybridTokenEncoder,MusicContextConfig,MaskedPredictionConfig,MaskedMusicModel,
    PretrainingConfig,MusicPretrainer,CorpusPretrainingRun,CorpusRunConfig,
    fit_training_target_mean,evaluate_masked_corpus,visible_neighbor_prediction,collate_audio)


def manifest(tmp_path):
    records=[]
    for i,(split,group) in enumerate([('train','a'),('train','b'),('validation','c'),('validation','d')]):
        x=np.random.default_rng(i).normal(0,.1,(512+32*i,2)).astype(np.float32)
        path=tmp_path/f'{i}.wav';sf.write(path,x,8192,subtype='FLOAT')
        records.append(AudioRecord.inspect(path,group_id=group,split=split))
    return AudioManifest(records)


def network():
    cfg=HybridConfig(sample_rate=8192,backend='compact_wavelet',n_bands=8,f_min=64,f_max=2048,
                     q=2,hop_length=16,padding_samples=32)
    return MaskedMusicModel(HybridTokenEncoder(cfg,channels=2,dim=8),
        context_config=MusicContextConfig(dim=8,heads=2,local_layers=1,global_layers=1,local_tokens=4,dropout=.1),
        prediction_config=MaskedPredictionConfig(span_tokens=2,standardize_targets=True))


def datasets(m):
    crop=AudioCropConfig(8192,2,128,hop_length=16,crops_per_record=2,seed=7)
    return AudioWindowDataset(m,'train',crop),AudioWindowDataset(m,'validation',replace(crop,mode='fixed'))


def run(m):
    train,val=datasets(m)
    trainer=MusicPretrainer(network(),PretrainingConfig(total_steps=4,warmup_steps=0,mask_seed=11))
    return CorpusPretrainingRun(trainer,train,val,CorpusRunConfig(batch_size=2,shuffle_seed=9,evaluation_mask_seed=3))


def test_manifest_integrity_and_relative_roundtrip(tmp_path):
    m=manifest(tmp_path)
    path=tmp_path/'metadata'/'corpus.json';m.save(path)
    loaded=AudioManifest.load(path)
    assert loaded.identity==m.identity and loaded.records==m.records
    with pytest.raises(ValueError,match='crosses'):
        AudioManifest([m.records[0],replace(m.records[2],group_id='a')])
    with pytest.raises(ValueError,match='crosses'):
        AudioManifest([m.records[0],replace(m.records[2],group_id=' a ')])
    with pytest.raises(ValueError,match='duplicate'):
        AudioManifest([m.records[0],replace(m.records[0],path=str(tmp_path/'else.wav'),group_id='e')])


def test_pcm_duplicates_across_containers(tmp_path):
    x=(np.arange(256,dtype=np.float64)%31-15)/128
    wav=tmp_path/'x.wav';flac=tmp_path/'x.flac'
    sf.write(wav,x,8192,subtype='FLOAT');sf.write(flac,x,8192,subtype='PCM_16')
    a=AudioRecord.inspect(wav,group_id='a',split='train')
    b=AudioRecord.inspect(flac,group_id='b',split='validation')
    assert a.file_hash!=b.file_hash and a.pcm_hash==b.pcm_hash
    with pytest.raises(ValueError,match='decoded'):AudioManifest([a,b])


def test_crops_are_deterministic_native_and_epoch_aware(tmp_path):
    train,val=datasets(manifest(tmp_path))
    first=train[0];torch.manual_seed(1234);again=train[0]
    assert first.window_id==again.window_id and torch.equal(first.waveform,again.waveform)
    assert first.waveform.shape==(2,128) and first.sample_origin%16==0
    p0=[train.plan(i) for i in range(len(train))];train.set_epoch(1)
    assert p0!=[train.plan(i) for i in range(len(train))]
    pv=[val.plan(i) for i in range(len(val))];val.set_epoch(7)
    assert pv==[val.plan(i) for i in range(len(val))]
    batch=collate_windows([first,again]);assert batch.sample_origins.tolist()==[first.sample_origin]*2


def test_fixed_short_crops_are_deduplicated_and_window_ids_ignore_other_records(tmp_path):
    m=manifest(tmp_path)
    cfg=AudioCropConfig(8192,2,1024,hop_length=16,mode='fixed',crops_per_record=4)
    a=AudioWindowDataset(m,'validation',cfg)
    b=AudioWindowDataset(AudioManifest(tuple(reversed(m.records))),'validation',cfg)
    assert len(a)==2
    assert {a.plan(i)[-1] for i in range(len(a))}=={b.plan(i)[-1] for i in range(len(b))}


def test_changed_files_and_wrong_format_are_rejected(tmp_path):
    m=manifest(tmp_path);train,_=datasets(m)
    p=m.records[0].path
    sf.write(p,np.zeros((512,2),dtype=np.float32),8192,subtype='FLOAT')
    stat=os.stat(p);os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns+10000000))
    with pytest.raises(ValueError,match='changed'):train[0]
    with pytest.raises(ValueError,match='matches'):datasets(m)
    with pytest.raises(ValueError,match='rate/channels'):
        AudioWindowDataset(m,'validation',AudioCropConfig(16000,2,128,hop_length=16,mode='fixed'))


def test_complete_run_resume_restores_crop_cursor_mask_and_dropout(tmp_path):
    torch.manual_seed(4);m=manifest(tmp_path)
    a=run(m);a.step();path=tmp_path/'resume.pt';a.save_checkpoint(path)
    expected=a.step()
    parameters=[p.detach().clone() for p in a.trainer.model.parameters()]
    b=run(m);b.load_checkpoint(path);actual=b.step()
    assert actual['window_ids']==expected['window_ids']
    assert actual['loss']==expected['loss'] and actual['update_step']==2
    for x,y in zip(parameters,b.trainer.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_changed_corpus_guard_precedes_weight_load(tmp_path):
    m=manifest(tmp_path);a=run(m);a.step();path=tmp_path/'resume.pt';a.save_checkpoint(path)
    modified=AudioManifest([replace(r,group_id=r.group_id+'-changed') for r in m.records])
    b=run(modified);before=[p.detach().clone() for p in b.trainer.model.parameters()]
    with pytest.raises(ValueError,match='provenance'):b.load_checkpoint(path)
    for x,y in zip(before,b.trainer.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_evaluation_is_repeatable_and_restores_modes_and_reports_groups(tmp_path):
    r=run(manifest(tmp_path));r.step()
    modes=[m.training for m in r.trainer.model.modules()]
    first=r.evaluate();second=r.evaluate()
    assert first==second and modes==[m.training for m in r.trainer.model.modules()]
    assert set(first['per_group_loss'])=={'c','d'}
    assert set(first['micro_loss'])=={'model','training_mean','zero_visible_audio','global_path_disabled','visible_neighbor'}
    expected=sum(w['losses']['model']*w['masked_samples'] for w in first['windows'])/sum(w['masked_samples'] for w in first['windows'])
    assert first['micro_loss']['model']==pytest.approx(expected)
    with pytest.raises(ValueError,match='training split'):fit_training_target_mean(r.trainer.model,r.validation)
    with pytest.raises(ValueError,match='held-out'):evaluate_masked_corpus(r.trainer.model,r.train,r.baseline)


def test_neighbor_baseline_cannot_see_hidden_audio():
    model=network().eval()
    a=collate_audio([torch.randn(2,128)],sample_rate=8192)
    x=a.waveform.clone();x[:,:,32:64]+=100
    b=replace(a,waveform=x)
    mask=torch.zeros(1,8,dtype=torch.bool);mask[:,2:4]=True
    torch.testing.assert_close(visible_neighbor_prediction(model,a,mask),visible_neighbor_prediction(model,b,mask),rtol=0,atol=0)


def test_global_ablation_skips_global_layers():
    model=network().eval();batch=collate_audio([torch.randn(2,128)],sample_rate=8192)
    mask=torch.zeros(1,8,dtype=torch.bool);mask[:,2:4]=True
    calls=[];hook=model.context.global_layers[0].register_forward_hook(lambda *args:calls.append(1))
    output=model.student_forward(batch,mask,use_global_context=False)
    assert not calls and output.context.coarse_tokens.count_nonzero()==0
    model.student_forward(batch,mask);assert calls
    hook.remove()


def test_unverified_data_is_not_used_for_training_statistics(tmp_path):
    m=manifest(tmp_path)
    dataset=AudioWindowDataset(m,'train',AudioCropConfig(8192,2,128,hop_length=16),verify=False)
    with pytest.raises(ValueError,match='verified'):fit_training_target_mean(network(),dataset)
