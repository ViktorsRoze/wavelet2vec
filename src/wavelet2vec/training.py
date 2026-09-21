"""Explicit source-waveform supervision, padding masks and reconstruction losses.

Missing source annotations are ignored, not assumed silent. Use an explicit
zero target for a known absent source. Mixture consistency is only allowed
for complete, validated additive targets; dry restoration needs a different
production/rendering objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceExample:
    mixture: torch.Tensor  # [C,N]
    sources: Mapping[str, torch.Tensor]  # each annotated target is [C,N]
    sample_rate: int
    additive: bool = False


@dataclass(frozen=True)
class SourceBatch:
    mixture: torch.Tensor  # [B,C,Nmax]
    sources: torch.Tensor  # [B,S,C,Nmax]
    lengths: torch.Tensor  # [B]
    source_valid: torch.Tensor  # [B,S]
    additive: torch.Tensor  # [B]
    source_order: tuple[str, ...]
    sample_rate: int

    @property
    def sample_mask(self):
        return torch.arange(self.mixture.shape[-1], device=self.mixture.device)[None] < self.lengths[:, None]

    def to(self, device) -> "SourceBatch":
        """Move tensors without changing signal precision."""
        return SourceBatch(self.mixture.to(device), self.sources.to(device), self.lengths.to(device),
                           self.source_valid.to(device), self.additive.to(device), self.source_order, self.sample_rate)


def collate_sources(examples: Sequence[SourceExample], *, source_order: Sequence[str]) -> SourceBatch:
    order = tuple(source_order)
    if not examples or not order or any(not isinstance(s,str) or not s for s in order) or len(set(order)) != len(order):
        raise ValueError("provide examples and a unique nonempty source_order")
    first = examples[0].mixture
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise ValueError("mixtures must be [channels,samples] tensors")
    channels, dtype, device, rate = first.shape[0], first.dtype, first.device, examples[0].sample_rate
    if channels == 0 or dtype not in (torch.float32, torch.float64):
        raise ValueError("audio must have channels and use float32 or float64")
    if isinstance(rate,bool) or not isinstance(rate,int) or rate<=0:raise ValueError("sample rate must be a positive integer")
    for example in examples:
        x = example.mixture
        if not isinstance(example.sources,Mapping):raise ValueError("sources must map names to waveform targets")
        if isinstance(example.sample_rate,bool) or not isinstance(example.sample_rate,int):raise ValueError("sample rates must be integers")
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != channels or x.shape[-1] == 0
            or x.dtype != dtype or x.device != device or example.sample_rate != rate or not torch.isfinite(x).all()):
            raise ValueError("mixture shapes, rates, precision, device and finite samples must agree")
        if not isinstance(example.additive, bool):raise ValueError("additive must be explicit bool")
        if set(example.sources)-set(order):raise ValueError("source annotation is outside source_order")
        for target in example.sources.values():
            if (not isinstance(target,torch.Tensor) or target.shape != x.shape or target.dtype != dtype
                or target.device != device or not torch.isfinite(target).all()):
                raise ValueError("source targets must match their mixture's shape, dtype and device and be finite")
        if example.additive:
            if set(example.sources) != set(order):raise ValueError("additive supervision needs all source targets, including explicit zeros")
            total = torch.stack([example.sources[name] for name in order]).sum(0)
            if not torch.allclose(total, x, rtol=1e-5, atol=1e-6):raise ValueError("additive source targets do not sum to the mixture")
    n = max(e.mixture.shape[-1] for e in examples)
    mixtures, targets, valid = [], [], []
    for e in examples:
        pad = (0, n-e.mixture.shape[-1])
        mixtures.append(F.pad(e.mixture, pad))
        targets.append(torch.stack([F.pad(e.sources[name], pad) if name in e.sources else torch.zeros_like(mixtures[-1]) for name in order]))
        valid.append([name in e.sources for name in order])
    if not any(any(row) for row in valid):raise ValueError("batch has no supervised sources")
    return SourceBatch(torch.stack(mixtures), torch.stack(targets),
        torch.tensor([e.mixture.shape[-1] for e in examples], device=device),
        torch.tensor(valid, device=device, dtype=torch.bool),
        torch.tensor([e.additive for e in examples], device=device, dtype=torch.bool), order, rate)


def separation_loss(estimate: torch.Tensor, batch: SourceBatch, *, fft_sizes=(256,1024),
                    waveform_weight=1., spectral_weight=1., mixture_weight=0.) -> dict[str, torch.Tensor]:
    """Masked waveform L1 plus amplitude-calibrated complex-STFT L1.

    Spectral transforms use each example's real length (no padded-batch tail),
    and each FFT-size loss is weighted by supervised source sample count.
    This reference implementation loops over examples; it is not a high-
    throughput distributed training pipeline or a perceptual quality metric.
    """
    if estimate.shape != batch.sources.shape or estimate.dtype != batch.sources.dtype or estimate.device != batch.sources.device:
        raise ValueError("estimate must match [B,S,C,N] target shape, dtype and device")
    weights = (waveform_weight, spectral_weight, mixture_weight)
    if any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)==0:
        raise ValueError("loss weights must be finite, nonnegative and not all zero")
    if mixture_weight and (not batch.additive.all() or not batch.source_valid.all()):
        raise ValueError("mixture consistency requires complete additive targets, not dry or missing-source targets")
    if spectral_weight and (not fft_sizes or any(isinstance(k,bool) or not isinstance(k,int) or k<4 or k%2 for k in fft_sizes)):
        raise ValueError("fft_sizes must contain even integers >= 4")
    mask = batch.source_valid[:,:,None,None] & batch.sample_mask[:,None,None,:]
    # Sanitize ignored positions before arithmetic so NaNs in an unused model
    # tail or unannotated source cannot contaminate an otherwise valid loss.
    prediction = torch.where(mask, estimate, 0.)
    target = torch.where(mask, batch.sources, 0.)
    count = mask.sum()*estimate.shape[2]
    if not count:raise ValueError("batch has no valid supervised samples")
    waveform = (prediction-target).abs().sum()/count
    spectral = prediction.sum()*0.
    if spectral_weight:
        for i, length in enumerate(batch.lengths.tolist()):
            selected = batch.source_valid[i]
            if not selected.any():continue
            p = prediction[i,selected,:,:length].reshape(-1,length)
            y = target[i,selected,:,:length].reshape(-1,length)
            contribution = p.numel()/count.to(estimate.dtype)
            for size in fft_sizes:
                window = torch.hann_window(size, device=p.device, dtype=p.dtype)
                options = dict(n_fft=size, hop_length=max(size//4,1), window=window, center=True,
                               pad_mode="constant", return_complex=True)
                scale = p.new_full((size//2+1,),2.)
                scale[0] = scale[-1] = 1.
                error = ((torch.stft(p,**options)-torch.stft(y,**options)).abs()*scale[:,None]/window.sum()).mean()
                spectral = spectral+error*contribution/len(fft_sizes)
    mixture = prediction.sum()*0.
    if mixture_weight:
        mixture_error = prediction.sum(1)-batch.mixture
        mixture = torch.where(batch.sample_mask[:,None,:], mixture_error, 0.).abs().sum()/(batch.lengths.sum()*estimate.shape[2])
    return {"loss": waveform_weight*waveform+spectral_weight*spectral+mixture_weight*mixture,
            "waveform_l1": waveform, "complex_spectral_l1": spectral, "mixture_l1": mixture}
