"""Manifest-backed pretraining and fixed, group-aware diagnostic evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict,dataclass,replace
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from wavelet2vec.corpus import AudioWindowDataset,collate_windows
from wavelet2vec.feature_cache import model_fingerprint
from wavelet2vec.pretraining import MusicPretrainer,LengthBucketSampler
from wavelet2vec.self_supervised import sample_span_mask


@dataclass(frozen=True)
class TrainingTargetMean:
    value:torch.Tensor
    dataset_identity:str
    total_samples:int
    manifest_identity:str
    teacher_config:dict


def _check_dataset(model,dataset):
    if not dataset.verified:raise ValueError("training/evaluation requires verified source files")
    cfg=model.encoder.frontend.config
    parameter=next(model.parameters())
    if parameter.dtype not in (torch.float32,torch.float64):raise ValueError("corpus runner supports float32/float64 models")
    dtype='float64' if parameter.dtype==torch.float64 else 'float32'
    if (dataset.config.sample_rate!=cfg.sample_rate or dataset.config.hop_length!=cfg.hop_length
        or dataset.config.channels!=model.encoder.channels or dataset.config.dtype!=dtype):
        raise ValueError("dataset rate, channels, precision or hop differs from model")
    if min(dataset.lengths)<=cfg.hop_length:raise ValueError("all crops need at least two tokens for masked prediction")


@torch.no_grad()
def fit_training_target_mean(model,dataset:AudioWindowDataset) -> TrainingTargetMean:
    if dataset.split!='train':raise ValueError("target mean must be fitted on the training split")
    _check_dataset(model,dataset)
    parameter=next(model.parameters())
    total=0;value=torch.zeros(model.target_dim,device=parameter.device,dtype=torch.float64)
    epoch=dataset.epoch;dataset.set_epoch(0)
    try:
        for i in range(len(dataset)):
            window=dataset[i]
            encoding=model.teacher(window.waveform[None].to(parameter.device),sample_origin=window.sample_origin)
            features,_,lengths=encoding.context_features()
            value+=(features[0].double()*lengths[:,None]).sum(0)
            total+=int(lengths.sum())
    finally:dataset.set_epoch(epoch)
    return TrainingTargetMean((value/total).to(parameter.dtype),dataset.identity,total,dataset.manifest.identity,model.teacher.config.to_dict())


def _raw_error(prediction,target,mask,lengths,config):
    error=F.smooth_l1_loss(prediction,target,reduction='none').reshape(*mask.shape,-1,3)
    per_token=config.magnitude_weight*error[...,2].mean(-1)+config.phase_weight*error[...,:2].mean((-1,-2))
    weights=mask*lengths.to(per_token.dtype)
    return float((per_token*weights).sum()),float(weights.sum())


@torch.no_grad()
def visible_neighbor_prediction(model,batch,mask):
    """Interpolate features of CORRUPTED audio at visible token neighbors.

    Clean teacher features outside a mask would leak hidden samples through
    wavelet/STFT support. This baseline erases the waveform first, just like
    the student. Endpoint missing spans use the nearest visible feature.
    """
    cfg=model.teacher.config
    results=[]
    for i,(n,origin) in enumerate(zip(batch.lengths.tolist(),batch.sample_origins.tolist())):
        sample_mask=mask[i].repeat_interleave(cfg.hop_length)[:n]
        corrupted=torch.where(sample_mask[None,None],0.,batch.waveform[i:i+1,:,:n])
        features,times,_=model.teacher(corrupted,sample_origin=origin).context_features()
        visible=(~mask[i,:features.shape[1]]).nonzero(as_tuple=True)[0]
        if not len(visible):raise ValueError("neighbor baseline requires visible tokens")
        positions=torch.arange(features.shape[1],device=features.device)
        insertion=torch.searchsorted(visible,positions)
        left=visible[(insertion-1).clamp(0,len(visible)-1)]
        right=visible[insertion.clamp(0,len(visible)-1)]
        denominator=(times[right]-times[left]).clamp_min(torch.finfo(times.dtype).eps)
        mix=((times-times[left])/denominator).clamp(0,1).to(features.dtype)
        prediction=features[:,left]*(1-mix[None,:,None])+features[:,right]*mix[None,:,None]
        results.append(F.pad(prediction,(0,0,0,mask.shape[1]-features.shape[1])))
    return torch.cat(results,0)


@torch.no_grad()
def evaluate_masked_corpus(model,dataset:AudioWindowDataset,baseline:TrainingTargetMean,*,mask_seed=0):
    if isinstance(mask_seed,bool) or not isinstance(mask_seed,int) or mask_seed<0:raise ValueError("mask_seed must be nonnegative integer")
    if dataset.split not in ('validation','test') or dataset.config.mode!='fixed':
        raise ValueError("evaluation requires a held-out split with fixed crops")
    _check_dataset(model,dataset)
    parameter=next(model.parameters())
    if (baseline.value.shape!=(model.target_dim,) or baseline.value.dtype!=parameter.dtype or not torch.isfinite(baseline.value).all()
        or baseline.manifest_identity!=dataset.manifest.identity or baseline.teacher_config!=model.teacher.config.to_dict()):
        raise ValueError("constant baseline identity, teacher configuration or precision differs")
    modes=[(module,module.training) for module in model.modules()]
    model.eval()
    grouped=defaultdict(lambda:defaultdict(lambda:[0.,0.]))
    windows=[]
    try:
        fingerprint=model_fingerprint(model)
        for index in range(len(dataset)):
            window=dataset[index];batch=collate_windows([window]).to(parameter.device)
            valid,_,_=model._layout(batch)
            seed=int(hashlib.sha256(f'{mask_seed}:{window.window_id}'.encode()).hexdigest()[:16],16)
            mask=sample_span_mask(valid,model.config,generator=torch.Generator().manual_seed(seed))
            output=model(batch,mask=mask)
            raw_target=model.from_target_space(output.targets)
            zero=model.student_forward(replace(batch,waveform=torch.zeros_like(batch.waveform)),mask)
            local=model.student_forward(batch,mask,use_global_context=False)
            predictions={
                'model':model.from_target_space(output.predictions),
                'training_mean':baseline.value.to(parameter.device).expand_as(raw_target),
                'zero_visible_audio':model.from_target_space(zero.predictions),
                'global_path_disabled':model.from_target_space(local.predictions),
                'visible_neighbor':visible_neighbor_prediction(model,batch,mask),
            }
            row={'window_id':window.window_id,'group_id':window.group_id,'source_hash':window.source_hash,
                 'start_sample':window.sample_origin,'samples':window.waveform.shape[-1],'mask_seed':seed,
                 'audio_rms':float(batch.waveform.square().mean().sqrt()),
                 'raw_target_std':float(raw_target[mask].std(0,unbiased=False).mean()),'losses':{}}
            for name,prediction in predictions.items():
                numerator,weight=_raw_error(prediction,raw_target,mask,output.student.valid_samples,model.config)
                grouped[window.group_id][name][0]+=numerator;grouped[window.group_id][name][1]+=weight
                row['losses'][name]=numerator/weight
            row['masked_samples']=weight;windows.append(row)
    finally:
        for module,mode in modes:module.training=mode
    per_group={group:{name:numerator/weight for name,(numerator,weight) in methods.items()} for group,methods in grouped.items()}
    names=next(iter(per_group.values())).keys()
    micro={name:sum(g[name][0] for g in grouped.values())/sum(g[name][1] for g in grouped.values()) for name in names}
    macro={name:sum(g[name] for g in per_group.values())/len(per_group) for name in names}
    return {'schema':'masked-corpus-evaluation-v1','dataset_identity':dataset.identity,'baseline_train_identity':baseline.dataset_identity,
            'crop_config':dataset.config.to_dict(),'prediction_config':model.config.to_dict(),'teacher_config':model.teacher.config.to_dict(),
            'model_fingerprint':fingerprint,'mask_seed':mask_seed,'micro_loss':micro,'macro_group_loss':macro,
            'per_group_loss':per_group,'windows':windows,
            'scope':'Raw acoustic teacher-feature errors. Group validity depends on declared provenance; ablations are not semantic scores.'}


@dataclass(frozen=True)
class CorpusRunConfig:
    batch_size:int=2
    shuffle_seed:int=0
    evaluation_mask_seed:int=0

    def __post_init__(self):
        for value,minimum in ((self.batch_size,1),(self.shuffle_seed,0),(self.evaluation_mask_seed,0)):
            if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError("invalid run configuration")


class CorpusPretrainingRun:
    """Connect verified crops, calibration, sampler cursor and model checkpoints."""
    def __init__(self,trainer:MusicPretrainer,train:AudioWindowDataset,validation:AudioWindowDataset,config:CorpusRunConfig|None=None):
        self.trainer=trainer;self.train=train;self.validation=validation;self.config=config or CorpusRunConfig()
        if train.split!='train' or validation.split!='validation' or validation.config.mode!='fixed':
            raise ValueError("run needs training data and fixed validation crops")
        if train.manifest.identity!=validation.manifest.identity:raise ValueError("use one validated manifest for both splits")
        _check_dataset(trainer.model,train);_check_dataset(trainer.model,validation)
        self.sampler=LengthBucketSampler(train.lengths,self.config.batch_size,seed=self.config.shuffle_seed)
        self.epoch=0;self.cursor=0;self.baseline=None
        teacher_identity=hashlib.sha256(json.dumps(trainer.model.teacher.config.to_dict(),sort_keys=True).encode()).hexdigest()
        self.calibration_id='training-crops-epoch0:'+train.identity+':'+teacher_identity

    def _identity(self):
        return {'schema':'corpus-run-v1','train_identity':self.train.identity,'validation_identity':self.validation.identity,
                'run_config':asdict(self.config),'calibration_id':self.calibration_id}

    def prepare(self):
        if self.baseline is not None:return
        model=self.trainer.model;parameter=next(model.parameters())
        if model.target_statistics is not None:
            expected=hashlib.sha256(self.calibration_id.encode()).hexdigest()
            if model.target_statistics.fitted:
                if model.target_statistics.summary()['calibration_hash']!=expected:raise ValueError("calibration provenance differs from training crops")
            else:
                epoch=self.train.epoch;self.train.set_epoch(0)
                try:
                    model.fit_target_statistics((collate_windows([self.train[i]]).to(parameter.device) for i in range(len(self.train))),
                                                calibration_id=self.calibration_id)
                finally:self.train.set_epoch(epoch)
        self.baseline=fit_training_target_mean(model,self.train)

    def step(self):
        self.prepare()
        self.sampler.set_epoch(self.epoch);batches=list(self.sampler)
        if self.cursor==len(batches):
            self.epoch+=1;self.cursor=0;self.sampler.set_epoch(self.epoch);batches=list(self.sampler)
        if not 0<=self.cursor<len(batches):raise ValueError("invalid data cursor")
        self.train.set_epoch(self.epoch)
        windows=[self.train[i] for i in batches[self.cursor]]
        batch=collate_windows(windows).to(next(self.trainer.model.parameters()).device)
        result=self.trainer.step(batch)
        self.cursor+=1
        return {**result,'epoch':self.epoch,'next_batch':self.cursor,'window_ids':[w.window_id for w in windows]}

    def evaluate(self):
        self.prepare()
        return evaluate_masked_corpus(self.trainer.model,self.validation,self.baseline,mask_seed=self.config.evaluation_mask_seed)

    def save_checkpoint(self,path):
        self.prepare()
        state={**self._identity(),'epoch':self.epoch,'cursor':self.cursor,
               'baseline':self.baseline.value.detach().cpu().tolist(),'baseline_samples':self.baseline.total_samples}
        self.trainer.save_checkpoint(path,data_state=state)

    def load_checkpoint(self,path):
        state=self.trainer.load_checkpoint(path,expected_data_state=self._identity())
        self.epoch=int(state['epoch']);self.cursor=int(state['cursor'])
        parameter=next(self.trainer.model.parameters())
        self.baseline=TrainingTargetMean(torch.tensor(state['baseline'],dtype=parameter.dtype,device=parameter.device),
                                         self.train.identity,int(state['baseline_samples']),self.train.manifest.identity,
                                         self.trainer.model.teacher.config.to_dict())
