"""Masked acoustic prediction with a clean-target / corrupted-student boundary.

No pretrained teacher or quantizer is required. Targets are fixed hybrid
features; this is an initial acoustic pretext task, not established musical
semantics. All official student views originate from the corrupted waveform.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,replace
import json
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from wavelet2vec.context import MusicContextConfig,MusicContextEncoder,MusicContextOutput
from wavelet2vec.hybrid import HybridAudioFrontend,HybridConfig
from wavelet2vec.target_statistics import FixedTargetStatistics


@dataclass(frozen=True)
class AudioBatch:
    waveform:torch.Tensor  # B,C,Nmax
    lengths:torch.Tensor  # B, real sample counts
    sample_origins:torch.Tensor  # B, origins may differ between clips
    sample_rate:int

    def to(self,device):
        return AudioBatch(self.waveform.to(device),self.lengths.to(device),self.sample_origins.to(device),self.sample_rate)


def collate_audio(clips:Sequence[torch.Tensor],*,sample_rate:int,sample_origins:Sequence[int]|None=None) -> AudioBatch:
    if not clips:raise ValueError("provide at least one [C,N] audio clip")
    if isinstance(sample_rate,bool) or not isinstance(sample_rate,int) or sample_rate<=0:raise ValueError("sample_rate must be a positive integer")
    first=clips[0]
    if not isinstance(first,torch.Tensor) or first.ndim!=2 or first.shape[0]==0 or first.dtype not in (torch.float32,torch.float64):
        raise ValueError("clips must be [C,N] float32/float64 tensors")
    for clip in clips:
        if (not isinstance(clip,torch.Tensor) or clip.ndim!=2 or clip.shape[0]!=first.shape[0] or clip.shape[-1]==0
            or clip.dtype!=first.dtype or clip.device!=first.device or not torch.isfinite(clip).all()):
            raise ValueError("clips must be finite, nonempty, and share channels/dtype/device")
    origins=tuple(sample_origins) if sample_origins is not None else (0,)*len(clips)
    if len(origins)!=len(clips) or any(isinstance(v,bool) or not isinstance(v,int) for v in origins):
        raise ValueError("supply one integer sample origin per clip")
    n=max(c.shape[-1] for c in clips)
    return AudioBatch(torch.stack([F.pad(c,(0,n-c.shape[-1])) for c in clips]),
                      torch.tensor([c.shape[-1] for c in clips],device=first.device),
                      torch.tensor(origins,device=first.device),sample_rate)


@dataclass(frozen=True)
class MaskedPredictionConfig:
    mask_fraction:float=.5
    span_tokens:int=8
    minimum_visible_tokens:int=1
    magnitude_weight:float=1.
    phase_weight:float=.1
    standardize_targets:bool=False
    target_scale_floor:float=.01

    def __post_init__(self):
        if not math.isfinite(self.mask_fraction) or not 0<self.mask_fraction<1:raise ValueError("mask_fraction must lie in (0,1)")
        for name in ("span_tokens","minimum_visible_tokens"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError(f"{name} must be a positive integer")
        weights=(self.magnitude_weight,self.phase_weight)
        if any(not math.isfinite(v) or v<0 for v in weights) or sum(weights)==0:raise ValueError("loss weights must be nonnegative and not all zero")
        if not isinstance(self.standardize_targets,bool):raise ValueError("standardize_targets must be a bool")
        if not math.isfinite(self.target_scale_floor) or self.target_scale_floor<=0:raise ValueError("target_scale_floor must be finite and positive")

    def to_dict(self):return {"version":"masked-prediction-v2",**asdict(self)}

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        version=data.pop("version",None)
        if version not in ("masked-prediction-v1","masked-prediction-v2"):raise ValueError("unsupported masked prediction config")
        if version=="masked-prediction-v1" and data.get("standardize_targets",False):
            raise ValueError("target standardization requires masked-prediction-v2")
        return cls(**data)


def sample_span_mask(valid_mask:torch.Tensor,config:MaskedPredictionConfig|None=None,*,generator=None):
    """Reproducible span selection on prefix-valid sequences using a CPU RNG.

    The last span can be shortened to meet the requested count. Very short
    sequences retain visible context and may have no masked target.
    """
    cfg=config or MaskedPredictionConfig()
    if valid_mask.ndim!=2 or valid_mask.dtype!=torch.bool:raise ValueError("valid_mask must be [B,T] bool")
    valid=valid_mask.cpu()
    counts=valid.sum(1)
    if not torch.equal(valid,torch.arange(valid.shape[1])[None]<counts[:,None]):raise ValueError("span sampling requires prefix-valid sequences")
    if generator is not None and generator.device.type!="cpu":raise ValueError("mask generator must use CPU")
    mask=torch.zeros_like(valid)
    for row,n in enumerate(counts.tolist()):
        target=min(math.ceil(n*cfg.mask_fraction),max(0,n-cfg.minimum_visible_tokens))
        remaining=target
        for start in torch.randperm(n,generator=generator).tolist():
            if remaining==0:break
            new=(~mask[row,start:min(start+cfg.span_tokens,n)]).nonzero(as_tuple=True)[0]+start
            new=new[:remaining]
            mask[row,new]=True
            remaining-=len(new)
    return mask.to(valid_mask.device)


@dataclass(frozen=True)
class StudentPrediction:
    predictions:torch.Tensor
    context:MusicContextOutput
    valid_mask:torch.Tensor
    valid_samples:torch.Tensor
    times_s:torch.Tensor


@dataclass(frozen=True)
class MaskedMusicOutput:
    loss:torch.Tensor
    magnitude_loss:torch.Tensor
    phase_loss:torch.Tensor
    predictions:torch.Tensor
    targets:torch.Tensor
    mask:torch.Tensor
    student:StudentPrediction
    metrics:dict[str,torch.Tensor]


class MaskedMusicModel(nn.Module):
    """Train local hybrid + learned context to predict fixed clean features.

    The student API accepts AudioBatch, never precomputed clean carriers or
    cached clean features. It zeroes every channel of masked waveform patches
    BEFORE local front ends, then substitutes a learned latent mask token.
    Clean teacher values are computed separately under no_grad and are never
    arguments to the student/context/prediction head.
    """
    def __init__(self,encoder:nn.Module,*,context_config:MusicContextConfig|None=None,
                 prediction_config:MaskedPredictionConfig|None=None,teacher_config:HybridConfig|None=None,
                 encoder_options:dict|None=None):
        super().__init__()
        self.encoder=encoder
        dim=encoder.waveform_projection.out_features
        self.context=MusicContextEncoder(context_config or MusicContextConfig(dim=dim))
        if self.context.config.dim!=dim:raise ValueError("context dimension differs from local encoder output")
        self.config=prediction_config or MaskedPredictionConfig()
        primary=encoder.frontend.config
        teacher=teacher_config or replace(primary,context_mode="complex")
        if teacher.sample_rate!=primary.sample_rate or teacher.hop_length!=primary.hop_length:
            raise ValueError("teacher and student must share sample rate and token hop")
        if teacher.context_mode!="complex":raise ValueError("teacher targets require complex context; use phase_weight=0 for a magnitude-only objective")
        options=dict(encoder_options or {})
        if set(options)&{"audio","branches","sample_origin"}:raise ValueError("encoder_options contains reserved arguments")
        self._options_json=json.dumps(options,sort_keys=True,allow_nan=False)
        self.teacher=HybridAudioFrontend(teacher)
        self.target_dim=encoder.channels*teacher.bands*3
        self.target_statistics=(FixedTargetStatistics(self.target_dim,self.config.target_scale_floor)
                                if self.config.standardize_targets else None)
        self.mask_embedding=nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_embedding,std=.02)
        self.predictor=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,self.target_dim))

    def get_extra_state(self):
        return {"version":"masked-music-v1","prediction":self.config.to_dict(),"options":self._options_json,
                "channels":self.encoder.channels,"target_dim":self.target_dim,
                "encoder_type":type(self.encoder).__module__+"."+type(self.encoder).__qualname__}

    def set_extra_state(self,state):
        state=dict(state)
        state['prediction']=MaskedPredictionConfig.from_dict(state['prediction']).to_dict()
        if state!=self.get_extra_state():raise ValueError("checkpoint masked music configuration differs")

    def fit_target_statistics(self,batches,*,calibration_id:str):
        """Freeze corpus statistics using training batches only, before training.

        No validation/test data should be supplied. The ID records caller
        provenance, not proof of a correct split. Student views never receive
        per-example clean statistics. Refitting a calibrated model is rejected.
        """
        if self.target_statistics is None:raise ValueError("enable standardize_targets in prediction config first")
        def chunks():
            for batch in batches:
                self._layout(batch)
                for i,(n,origin) in enumerate(zip(batch.lengths.tolist(),batch.sample_origins.tolist())):
                    with torch.no_grad():
                        values,_,lengths=self.teacher(batch.waveform[i:i+1,:,:n],sample_origin=origin).context_features()
                    yield values,lengths[None]
        return self.target_statistics.fit(chunks(),calibration_id=calibration_id)

    def to_target_space(self,features):
        return self.target_statistics(features) if self.target_statistics is not None else features

    def from_target_space(self,features):
        return self.target_statistics.inverse(features) if self.target_statistics is not None else features

    def _layout(self,batch):
        if not isinstance(batch,AudioBatch):raise ValueError("masked learning requires AudioBatch, not clean cached features/carriers")
        x=batch.waveform
        parameter=self.encoder.waveform_projection.weight
        if (x.ndim!=3 or x.shape[0]==0 or x.shape[1]!=self.encoder.channels or x.shape[-1]==0
            or x.dtype!=parameter.dtype or x.device!=parameter.device):raise ValueError("audio shape, channels, precision and device must match encoder")
        if batch.sample_rate!=self.encoder.frontend.config.sample_rate:raise ValueError("audio sample rate differs from encoder")
        for value in (batch.lengths,batch.sample_origins):
            if value.shape!=(x.shape[0],) or value.dtype!=torch.int64 or value.device!=x.device:raise ValueError("lengths/origins must be [B] int64 on the audio device")
        if not ((batch.lengths>0)&(batch.lengths<=x.shape[-1])).all():raise ValueError("invalid audio lengths")
        real=torch.arange(x.shape[-1],device=x.device)[None]<batch.lengths[:,None]
        if not torch.isfinite(torch.where(real[:,None],x,0.)).all():raise ValueError("real audio samples must be finite")
        hop=self.encoder.frontend.config.hop_length
        count=(x.shape[-1]+hop-1)//hop
        starts=torch.arange(count,device=x.device)*hop
        lengths=(batch.lengths[:,None]-starts).clamp(0,hop)
        valid=lengths>0
        times=(starts.double()[None]+lengths.double()/2+batch.sample_origins[:,None])/batch.sample_rate
        return valid,lengths,times

    def _check_mask(self,mask,valid):
        if not isinstance(mask,torch.Tensor) or mask.shape!=valid.shape or mask.dtype!=torch.bool or mask.device!=valid.device:
            raise ValueError("mask must be [B,T] bool on the audio device")
        if (mask&~valid).any():raise ValueError("cannot mask padded tokens")
        if ((mask.any(1)) & ((valid&~mask).sum(1)<self.config.minimum_visible_tokens)).any():
            raise ValueError("masked sequences need the configured visible context")

    def student_forward(self,batch:AudioBatch,mask:torch.Tensor,*,return_hidden_states=False,use_global_context=True) -> StudentPrediction:
        valid,lengths,times=self._layout(batch)
        self._check_mask(mask,valid)
        hop=self.encoder.frontend.config.hop_length
        tokens=[]
        for i,(n,origin) in enumerate(zip(batch.lengths.tolist(),batch.sample_origins.tolist())):
            patch_count=(n+hop-1)//hop
            sample_mask=mask[i].repeat_interleave(hop)[:n]
            corrupted=torch.where(sample_mask[None,None],0.,batch.waveform[i:i+1,:,:n])
            carrier=self.encoder.frontend(corrupted,sample_origin=origin)
            local=self.encoder(carrier,**json.loads(self._options_json)).tokens
            local=torch.where(mask[i:i+1,:patch_count,None],self.mask_embedding[None,None],local)
            tokens.append(F.pad(local,(0,0,0,valid.shape[1]-patch_count)))
        context=self.context(torch.cat(tokens,0),times_s=times,valid_mask=valid,valid_samples=lengths,
                             return_hidden_states=return_hidden_states,use_global_context=use_global_context)
        predictions=torch.where(valid[...,None],self.predictor(context.tokens),0.)
        return StudentPrediction(predictions,context,valid,lengths,times)

    def encode(self,batch:AudioBatch,*,return_hidden_states=False) -> MusicContextOutput:
        valid,_,_=self._layout(batch)
        return self.student_forward(batch,torch.zeros_like(valid),return_hidden_states=return_hidden_states).context

    def forward(self,batch:AudioBatch,*,mask:torch.Tensor|None=None,generator=None,return_hidden_states=False) -> MaskedMusicOutput:
        if self.target_statistics is not None:self.target_statistics.require_fitted()
        valid,_,_=self._layout(batch)
        mask=sample_span_mask(valid,self.config,generator=generator) if mask is None else mask
        self._check_mask(mask,valid)
        if not mask.any():raise ValueError("batch has no masked targets; use clips with enough visible and masked tokens")
        student=self.student_forward(batch,mask,return_hidden_states=return_hidden_states)
        targets=[]
        with torch.no_grad():
            for i,(n,origin) in enumerate(zip(batch.lengths.tolist(),batch.sample_origins.tolist())):
                target=self.teacher(batch.waveform[i:i+1,:,:n],sample_origin=origin).context_features()[0]
                target=self.to_target_space(target)
                targets.append(F.pad(target,(0,0,0,valid.shape[1]-target.shape[1])))
            targets=torch.cat(targets,0)
        error=F.smooth_l1_loss(student.predictions,targets,reduction="none").reshape(*valid.shape,-1,3)
        weight=mask*student.valid_samples.to(error.dtype)
        magnitude=(error[...,2].mean(-1)*weight).sum()/weight.sum()
        phase=(error[...,:2].mean((-1,-2))*weight).sum()/weight.sum()
        loss=self.config.magnitude_weight*magnitude+self.config.phase_weight*phase
        selected=student.predictions.detach()[mask]
        target_std=targets[mask].std(0,unbiased=False).mean()
        metrics={"masked_fraction":(mask.sum()/valid.sum()).detach(),
                 "prediction_std":selected.std(0,unbiased=False).mean(),"target_std":target_std,
                 "masked_samples":weight.sum().detach()}
        return MaskedMusicOutput(loss,magnitude,phase,student.predictions,targets,mask,student,metrics)
