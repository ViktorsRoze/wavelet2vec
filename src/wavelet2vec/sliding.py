"""Sliding hybrid extraction with explicit context and global coordinates.

This is a token transport/alignment layer, not a trained model of musical
structure. Inference iterators release each window's carrier by default.
Collecting all outputs or retaining training graphs necessarily uses memory
proportional to the collected sequence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn

from wavelet2vec.hybrid import HybridEncoding, HybridTokenEncoder


@dataclass(frozen=True)
class SlidingWindowConfig:
    core_tokens: int = 256
    context_tokens: int = 64

    def __post_init__(self):
        for name, minimum in (("core_tokens",1),("context_tokens",0)):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @classmethod
    def from_seconds(cls, core_seconds: float, context_seconds: float, *, sample_rate: int, hop_length: int):
        if (not math.isfinite(core_seconds) or core_seconds<=0
            or not math.isfinite(context_seconds) or context_seconds<0):
            raise ValueError("core duration must be positive and context duration nonnegative")
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (sample_rate,hop_length)):
            raise ValueError("sample_rate and hop_length must be positive integers")
        return cls(math.ceil(core_seconds*sample_rate/hop_length),
                   math.ceil(context_seconds*sample_rate/hop_length))

    def to_dict(self):
        return {"version":"sliding-v1",**asdict(self)}

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if data.pop("version",None)!="sliding-v1":raise ValueError("unsupported sliding config version")
        return cls(**data)


@dataclass(frozen=True)
class SlidingTokenBatch:
    tokens: torch.Tensor  # [B,T,D], owned core tokens only
    sample_starts: torch.Tensor  # [T], global sample coordinates
    valid_samples: torch.Tensor  # [T], actual lengths, including partial final patch
    times_s: torch.Tensor  # [T], global patch midpoint times
    left_context_samples: torch.Tensor  # real context supplied to analysis, not a DSP validity guarantee
    right_context_samples: torch.Tensor
    has_requested_context: torch.Tensor
    analysis_start: int
    analysis_stop: int
    core_start: int
    core_stop: int
    encoding: HybridEncoding | None = None
    sample_rate: int = 0

    def as_sequence(self) -> "HybridSequence":
        """Token-only view for caching or a downstream context model."""
        if self.sample_rate <= 0:raise ValueError("this batch has no sample-rate metadata")
        return HybridSequence(self.tokens,self.sample_starts,self.valid_samples,self.times_s,
                              self.has_requested_context,self.sample_rate)

    def synthesize_core(self, coefficients=None):
        """Reconstruct the owned source interval if retain_carrier was enabled."""
        if self.encoding is None:raise ValueError("enable retain_carrier to synthesize this window")
        audio=self.encoding.synthesize(coefficients)
        return audio[...,self.core_start-self.analysis_start:self.core_stop-self.analysis_start]


@dataclass(frozen=True)
class HybridSequence:
    tokens: torch.Tensor
    sample_starts: torch.Tensor
    valid_samples: torch.Tensor
    times_s: torch.Tensor
    has_requested_context: torch.Tensor
    sample_rate: int

    def pooled(self, group_tokens: int) -> "HybridSequence":
        """Duration-weighted coarse features; pooling does not infer phrases/beats.

        Fine tokens are kept in the original object. Returned features are
        lossy summaries on a fixed grouping grid, with true interval lengths.
        """
        if isinstance(group_tokens,bool) or not isinstance(group_tokens,int) or group_tokens<=0:
            raise ValueError("group_tokens must be a positive integer")
        tokens,lengths,starts,complete=[],[],[],[]
        for i in range(0,self.tokens.shape[1],group_tokens):
            weight=self.valid_samples[i:i+group_tokens]
            tokens.append((self.tokens[:,i:i+group_tokens]*weight.to(self.tokens.dtype)[None,:,None]).sum(1)
                          /weight.sum().to(self.tokens.dtype))
            lengths.append(weight.sum());starts.append(self.sample_starts[i])
            complete.append(self.has_requested_context[i:i+group_tokens].all())
        lengths=torch.stack(lengths);starts=torch.stack(starts)
        return HybridSequence(torch.stack(tokens,1),starts,lengths,
                              (starts.double()+lengths.double()/2)/self.sample_rate,
                              torch.stack(complete),self.sample_rate)


class SlidingHybridEncoder(nn.Module):
    """Overlapping analysis windows with disjoint ownership of output tokens.

    Iterators default to no-grad and no retained carriers. ``forward`` collects
    the complete token sequence and respects the caller's gradient mode.
    Input tensors may remain on CPU while only each window moves to the model
    device. Dtype conversion/resampling/downmixing is never implicit.
    """
    def __init__(self, encoder: nn.Module, windows: SlidingWindowConfig | None = None):
        super().__init__()
        self.encoder=encoder
        self.windows=windows or SlidingWindowConfig()

    def get_extra_state(self):
        return self.windows.to_dict()

    def set_extra_state(self,state):
        if SlidingWindowConfig.from_dict(state)!=self.windows:
            raise ValueError("checkpoint sliding config differs from this extractor")

    def iter_from_reader(self, read: Callable[[int,int],torch.Tensor], n_samples: int, *,
                         sample_origin: int=0, branches: str="both", track_grad: bool=False,
                         retain_carrier: bool=False, encoder_options: dict | None=None) -> Iterator[SlidingTokenBatch]:
        """read(start,stop) returns exactly [B,C,stop-start] samples, local indices.

        Inputs across calls must have a consistent batch size and model dtype.
        The origin is used for global timestamps and spectral demodulation.
        Available real context shrinks at file edges; no context is invented.
        Caller-owned reader resources must stay open until iteration finishes.
        """
        if isinstance(n_samples,bool) or not isinstance(n_samples,int) or n_samples<=0:
            raise ValueError("n_samples must be a positive integer")
        if isinstance(sample_origin,bool) or not isinstance(sample_origin,int):
            raise ValueError("sample_origin must be an integer")
        if branches not in ("both","waveform","spectral"):raise ValueError("invalid branches")
        options=dict(encoder_options or {})
        if set(options)&{"audio","branches","sample_origin"}:raise ValueError("encoder_options contains a reserved argument")
        cfg=self.encoder.frontend.config
        hop=cfg.hop_length
        core=self.windows.core_tokens*hop
        context=self.windows.context_tokens*hop
        parameter=self.encoder.waveform_projection.weight
        batch_size=None
        for start in range(0,n_samples,core):
            stop=min(start+core,n_samples)
            lo,hi=max(0,start-context),min(n_samples,stop+context)
            # The context manager must end BEFORE yield, otherwise generator
            # suspension would change the consumer's thread-local grad mode.
            with torch.set_grad_enabled(track_grad):
                audio=read(lo,hi)
                if (not isinstance(audio,torch.Tensor) or audio.ndim!=3 or audio.shape[-1]!=hi-lo
                    or audio.shape[1]!=self.encoder.channels or audio.shape[0]==0):
                    raise ValueError("reader must return nonempty [B,model_channels,stop-start] audio")
                if audio.dtype!=parameter.dtype:raise ValueError("reader dtype must match encoder dtype explicitly")
                if batch_size is not None and audio.shape[0]!=batch_size:raise ValueError("reader batch size changed")
                batch_size=audio.shape[0]
                audio=audio.to(device=parameter.device)
                encoded=self.encoder.frontend(audio,sample_origin=sample_origin+lo)
                latent=self.encoder(encoded,branches=branches,**options)
                first=(start-lo)//hop
                count=(stop-start+hop-1)//hop
                tokens=latent.tokens[:,first:first+count].clone()
                local_starts=torch.arange(count,device=tokens.device)*hop+start
                lengths=(n_samples-local_starts).clamp(max=hop)
                global_starts=local_starts+sample_origin
                left=local_starts-lo;right=hi-(local_starts+lengths)
                output=SlidingTokenBatch(tokens,global_starts,lengths,
                    (global_starts.double()+lengths.double()/2)/cfg.sample_rate,left,right,
                    (left>=context)&(right>=context),lo+sample_origin,hi+sample_origin,
                    start+sample_origin,stop+sample_origin,encoded if retain_carrier else None,
                    sample_rate=cfg.sample_rate)
                # Avoid keeping full carrier references in suspended generator
                # frames in no-grad extraction when only tokens are requested.
                del latent,encoded,audio
            yield output

    def iter_encode(self, audio: torch.Tensor, **kwargs) -> Iterator[SlidingTokenBatch]:
        if not isinstance(audio,torch.Tensor) or audio.ndim!=3:
            raise ValueError("audio must have explicit [batch,channels,samples] axes")
        yield from self.iter_from_reader(lambda lo,hi:audio[...,lo:hi],audio.shape[-1],**kwargs)

    def iter_file(self, path: str | Path, **kwargs) -> Iterator[SlidingTokenBatch]:
        """Seek/read only required windows; closes file on exhaustion or close().

        Decodes in the model's explicit float32/float64 precision. File sample
        rate and channel count must match; no resampling or downmixing occurs.
        """
        import soundfile as sf
        parameter=self.encoder.waveform_projection.weight
        if parameter.dtype not in (torch.float32,torch.float64):raise ValueError("model must use float32 or float64")
        with sf.SoundFile(path) as source:
            if source.samplerate!=self.encoder.frontend.config.sample_rate:raise ValueError("file sample rate differs from encoder")
            if source.channels!=self.encoder.channels:raise ValueError("file channels differ from encoder")
            def read(lo,hi):
                source.seek(lo)
                array=source.read(hi-lo,dtype="float32" if parameter.dtype==torch.float32 else "float64",always_2d=True)
                return torch.from_numpy(array.T.copy())[None]
            yield from self.iter_from_reader(read,len(source),**kwargs)

    def forward(self, audio: torch.Tensor, *, sample_origin: int=0, branches: str="both",
                encoder_options: dict | None=None) -> HybridSequence:
        batches=list(self.iter_encode(audio,sample_origin=sample_origin,branches=branches,
                                      track_grad=torch.is_grad_enabled(),encoder_options=encoder_options))
        return HybridSequence(torch.cat([b.tokens for b in batches],1),
            torch.cat([b.sample_starts for b in batches]),torch.cat([b.valid_samples for b in batches]),
            torch.cat([b.times_s for b in batches]),torch.cat([b.has_requested_context for b in batches]),
            self.encoder.frontend.config.sample_rate)
