"""Frozen, training-only target scaling for masked acoustic prediction."""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn


class FixedTargetStatistics(nn.Module):
    """Duration-weighted corpus mean/std; never fitted from the current query.

    fit consumes (features [...,D], integer sample weights [...]) chunks under
    no_grad. Float64 Welford accumulation commits only after the iterator
    succeeds. Statistics are frozen, checkpointed buffers, not trainable
    teacher parameters. The floor limits amplification of nearly constant
    targets; it is a tunable objective choice, not perceptual calibration.
    """
    def __init__(self,dim:int,scale_floor:float=.01):
        super().__init__()
        if isinstance(dim,bool) or not isinstance(dim,int) or dim<=0:raise ValueError("dim must be positive integer")
        if not math.isfinite(scale_floor) or scale_floor<=0:raise ValueError("scale_floor must be finite and positive")
        self.dim=dim;self.scale_floor=scale_floor
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('scale',torch.ones(dim))
        self.register_buffer('observed_std',torch.zeros(dim))
        self.register_buffer('total_samples',torch.tensor(0,dtype=torch.int64))
        self.register_buffer('fitted',torch.tensor(False))
        self.register_buffer('calibration_hash',torch.zeros(32,dtype=torch.uint8))

    def get_extra_state(self):return {'version':'fixed-target-statistics-v1','dim':self.dim,'scale_floor':self.scale_floor}

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError("checkpoint target statistics configuration differs")

    def require_fitted(self):
        if not self.fitted:raise ValueError("fit target statistics on training data before using this objective")

    @torch.no_grad()
    def fit(self,chunks,*,calibration_id:str):
        if self.fitted:raise ValueError("target statistics are already frozen; create a new model for recalibration")
        if not isinstance(calibration_id,str) or not calibration_id:raise ValueError("provide a nonempty training calibration_id")
        total=0
        mean=torch.zeros(self.dim,device=self.mean.device,dtype=torch.float64)
        m2=torch.zeros_like(mean)
        for features,weights in chunks:
            if (not isinstance(features,torch.Tensor) or not isinstance(weights,torch.Tensor)
                or features.ndim<2 or features.dtype!=self.mean.dtype
                or features.shape[-1]!=self.dim or features.shape[:-1]!=weights.shape or weights.dtype!=torch.int64
                or features.device!=self.mean.device or weights.device!=features.device or (weights<0).any()):
                raise ValueError("calibration feature dimensions, device and nonnegative int64 weights must agree")
            active=weights>0
            if not torch.isfinite(features[active]).all():raise ValueError("calibration targets must be finite")
            amount=int(weights.sum())
            if amount==0:continue
            x=features[active].double();w=weights[active].double()
            chunk_mean=(x*w[:,None]).sum(0)/amount
            chunk_m2=((x-chunk_mean).square()*w[:,None]).sum(0)
            delta=chunk_mean-mean
            combined=total+amount
            m2=m2+chunk_m2+delta.square()*(total*amount/combined)
            mean=mean+delta*(amount/combined)
            total=combined
        if total==0:raise ValueError("calibration has no real samples")
        std=(m2/total).clamp_min(0).sqrt()
        scale=std.clamp_min(self.scale_floor)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():raise ValueError("nonfinite target statistics")
        stored_mean=mean.to(self.mean.dtype);stored_scale=scale.to(self.scale.dtype)
        if not torch.isfinite(stored_mean).all() or not torch.isfinite(stored_scale).all() or not (stored_scale>0).all():
            raise ValueError("target scale is not representable in the model precision")
        digest=hashlib.sha256(calibration_id.encode('utf-8')).digest()
        self.mean.copy_(stored_mean);self.scale.copy_(stored_scale);self.observed_std.copy_(std)
        self.total_samples.fill_(total)
        self.calibration_hash.copy_(torch.tensor(list(digest),device=self.mean.device,dtype=torch.uint8))
        self.fitted.fill_(True)
        return self.summary()

    def summary(self):
        self.require_fitted()
        return {'total_samples':int(self.total_samples),'features':self.dim,
                'scale_min':float(self.scale.min()),'scale_max':float(self.scale.max()),
                'floored_fraction':float((self.observed_std<self.scale_floor).float().mean()),
                'calibration_hash':bytes(self.calibration_hash.cpu().tolist()).hex()}

    def forward(self,features):
        self.require_fitted()
        return (features-self.mean)/self.scale

    def inverse(self,features):
        self.require_fitted()
        return features*self.scale+self.mean
