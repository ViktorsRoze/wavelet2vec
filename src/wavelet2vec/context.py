"""Learned local/global context over physically timed music tokens."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MusicContextConfig:
    dim: int = 128
    heads: int = 4
    local_layers: int = 2
    global_layers: int = 2
    local_tokens: int = 128
    feedforward_multiplier: int = 4
    dropout: float = .1
    min_period_seconds: float = .04
    max_period_seconds: float = 600.
    max_groups: int = 512
    max_attention_elements: int = 64*1024**2

    def __post_init__(self):
        for name in ("dim","heads","local_layers","global_layers","local_tokens",
                     "feedforward_multiplier","max_groups","max_attention_elements"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim%2 or self.dim%self.heads:raise ValueError("dim must be even and divisible by heads")
        if not math.isfinite(self.dropout) or not 0<=self.dropout<1:raise ValueError("dropout must be in [0,1)")
        if not (math.isfinite(self.min_period_seconds) and math.isfinite(self.max_period_seconds)
                and 0<self.min_period_seconds<self.max_period_seconds):
            raise ValueError("position periods must be finite, positive and increasing")

    def to_dict(self):return {"version":"music-context-v1",**asdict(self)}

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if data.pop("version",None)!="music-context-v1":raise ValueError("unsupported context configuration")
        return cls(**data)


@dataclass(frozen=True)
class MusicContextOutput:
    tokens: torch.Tensor  # B,T,D: local states fused with global context
    local_tokens: torch.Tensor  # before global fusion
    coarse_tokens: torch.Tensor  # B,G,D: learned contextual group summaries
    valid_mask: torch.Tensor  # B,T
    coarse_valid_mask: torch.Tensor  # B,G
    times_s: torch.Tensor  # B,T, absolute input metadata; padding set to first real time
    coarse_times_s: torch.Tensor  # B,G
    hidden_states: tuple[torch.Tensor,...] = ()  # optional fine local layers then coarse global layers


class MusicContextEncoder(nn.Module):
    """Hierarchical bidirectional Transformer; not a causal/live streaming model.

    Local attention is limited to token blocks; global attention operates on
    duration-weighted block summaries. Its global result is fused back into
    fine states. Positions use seconds relative to each sequence's first real
    token. The attention guard is a workload estimate, not total RAM/VRAM.
    """
    def __init__(self,config: MusicContextConfig | None=None):
        super().__init__()
        self.config=config or MusicContextConfig()
        cfg=self.config
        def layer():
            return nn.TransformerEncoderLayer(cfg.dim,cfg.heads,cfg.dim*cfg.feedforward_multiplier,
                                               dropout=cfg.dropout,activation="gelu",batch_first=True,norm_first=True)
        self.input_norm=nn.LayerNorm(cfg.dim)
        self.local_layers=nn.ModuleList([layer() for _ in range(cfg.local_layers)])
        self.global_layers=nn.ModuleList([layer() for _ in range(cfg.global_layers)])
        self.fusion=nn.Linear(2*cfg.dim,cfg.dim)
        self.output_norm=nn.LayerNorm(cfg.dim)

    def get_extra_state(self):return self.config.to_dict()

    def set_extra_state(self,state):
        if MusicContextConfig.from_dict(state)!=self.config:raise ValueError("checkpoint context config differs")

    def _position(self,relative_seconds,dtype):
        periods=torch.logspace(math.log10(self.config.min_period_seconds),math.log10(self.config.max_period_seconds),
                               self.config.dim//2,device=relative_seconds.device,dtype=torch.float64)
        phase=2*math.pi*torch.remainder(relative_seconds[...,None]/periods,1.)
        return torch.cat((phase.sin(),phase.cos()),-1).to(dtype)

    def forward(self,tokens:torch.Tensor,*,times_s:torch.Tensor,valid_mask:torch.Tensor|None=None,
                valid_samples:torch.Tensor|None=None,return_hidden_states=False,use_global_context=True) -> MusicContextOutput:
        if not isinstance(use_global_context,bool):raise ValueError("use_global_context must be a bool")
        cfg=self.config
        if (not isinstance(tokens,torch.Tensor) or tokens.ndim!=3 or tokens.shape[-1]!=cfg.dim
            or tokens.shape[0]==0 or tokens.shape[1]==0):raise ValueError("tokens must be nonempty [B,T,dim]")
        if tokens.dtype!=self.input_norm.weight.dtype or tokens.device!=self.input_norm.weight.device:
            raise ValueError("context model and tokens must share dtype and device")
        b,t,d=tokens.shape
        valid=torch.ones(b,t,dtype=torch.bool,device=tokens.device) if valid_mask is None else valid_mask
        if valid.shape!=(b,t) or valid.dtype!=torch.bool or valid.device!=tokens.device or not valid.any(1).all():
            raise ValueError("valid_mask must be [B,T] bool with a real token in every sequence")
        if times_s.device!=tokens.device or times_s.shape not in ((t,),(b,t)):
            raise ValueError("times_s must share the device and have shape [T] or [B,T]")
        times=times_s.double().expand(b,t)
        if not torch.isfinite(times[valid]).all():raise ValueError("real token times must be finite")
        for row,mask in zip(times,valid):
            if not (row[mask].diff()>0).all():raise ValueError("real token times must increase")
        first=torch.where(valid,times,torch.inf).amin(1,keepdim=True)
        times=torch.where(valid,times,first)
        if not torch.isfinite(torch.where(valid[...,None],tokens,0.)).all():raise ValueError("real tokens must be finite")
        if valid_samples is None:
            durations=valid.double()
        else:
            if valid_samples.device!=tokens.device or valid_samples.shape not in ((t,),(b,t)):
                raise ValueError("valid_samples must share the device and have shape [T] or [B,T]")
            durations=valid_samples.double().expand(b,t)
            if not torch.isfinite(durations[valid]).all() or not (durations[valid]>0).all():
                raise ValueError("real token durations must be finite and positive")
            durations=torch.where(valid,durations,0.)
        groups=(t+cfg.local_tokens-1)//cfg.local_tokens
        work=b*cfg.heads*(groups*cfg.local_tokens**2+groups**2)
        if groups>cfg.max_groups or work>cfg.max_attention_elements:
            raise ValueError("context attention budget exceeded; use shorter sequences/coarser tokens or explicit larger limits")
        x=self.input_norm(torch.where(valid[...,None],tokens,0.))+self._position(times-first,tokens.dtype)
        pad=groups*cfg.local_tokens-t
        block_valid=F.pad(valid,(0,pad)).reshape(b*groups,cfg.local_tokens)
        x=F.pad(x,(0,0,0,pad)).reshape(b*groups,cfg.local_tokens,d)
        active=block_valid.any(1).nonzero(as_tuple=True)[0]
        local=x[active]
        hidden=[]
        for layer in self.local_layers:
            local=layer(local,src_key_padding_mask=~block_valid[active])
            local=torch.where(block_valid[active,...,None],local,0.)
            if return_hidden_states:
                restored=torch.zeros_like(x).index_copy(0,active,local).reshape(b,-1,d)[:,:t]
                hidden.append(restored)
        local_all=torch.zeros_like(x).index_copy(0,active,local).reshape(b,groups,cfg.local_tokens,d)
        weights=F.pad(durations,(0,pad)).reshape(b,groups,cfg.local_tokens)
        denominator=weights.sum(-1)
        coarse_valid=denominator>0
        coarse=(local_all*weights.to(tokens.dtype)[...,None]).sum(-2)/denominator.clamp_min(1).to(tokens.dtype)[...,None]
        coarse_times=(F.pad(times,(0,pad)).reshape(b,groups,cfg.local_tokens)*weights).sum(-1)/denominator.clamp_min(1)
        coarse_times=torch.where(coarse_valid,coarse_times,first)
        coarse=coarse+self._position(coarse_times-first,tokens.dtype)
        for layer in self.global_layers if use_global_context else ():
            coarse=layer(coarse,src_key_padding_mask=~coarse_valid)
            coarse=torch.where(coarse_valid[...,None],coarse,0.)
            if return_hidden_states:hidden.append(coarse)
        if not use_global_context:coarse=torch.zeros_like(coarse)
        fine=local_all.reshape(b,-1,d)[:,:t]
        expanded=coarse.repeat_interleave(cfg.local_tokens,dim=1)[:,:t]
        gate=torch.sigmoid(self.fusion(torch.cat((fine,expanded),-1)))
        result=self.output_norm(fine+gate*expanded)
        result=torch.where(valid[...,None],result,0.)
        return MusicContextOutput(result,fine,coarse,valid,coarse_valid,times,coarse_times,tuple(hidden))
