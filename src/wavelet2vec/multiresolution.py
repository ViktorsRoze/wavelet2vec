"""Aligned multi-resolution hybrid tokens with one retained synthesis carrier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from wavelet2vec.hybrid import HybridAudioFrontend, HybridConfig, HybridEncoding


@dataclass(frozen=True)
class MultiResolutionLatents:
    encoding: HybridEncoding  # primary carrier, independent of fusion weights
    tokens: torch.Tensor
    times_s: torch.Tensor
    valid_samples: torch.Tensor
    resolution_weights: torch.Tensor  # [B,T,1+R]: raw, then ordered spectral views
    carriers: tuple[HybridEncoding | None, ...]  # secondary carriers retained on request


class MultiResolutionTokenEncoder(nn.Module):
    """Share raw patches and learn a mixture of distinct spectral views.

    All configs must have identical sample rate and token hop. FFT lengths,
    wavelet bandwidth/density, and backends may differ. No colored images or
    hidden time resampling are used. Fusion tokens are lossy; the primary
    carrier retains synthesis information. This is not a trained separator.
    """
    def __init__(self, configs: Sequence[HybridConfig], *, channels=2, dim=128,
                 max_total_coefficient_bytes=256*1024**2):
        super().__init__()
        configs = tuple(configs)
        if not configs or not all(isinstance(c, HybridConfig) for c in configs):
            raise ValueError("provide at least one HybridConfig")
        if any(c.sample_rate != configs[0].sample_rate or c.hop_length != configs[0].hop_length for c in configs):
            raise ValueError("resolution configs must share sample rate and token hop")
        for name, value in (("channels", channels), ("dim", dim), ("max_total_coefficient_bytes", max_total_coefficient_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.frontends = nn.ModuleList([HybridAudioFrontend(c) for c in configs])
        self.channels = channels
        self.max_total_coefficient_bytes = max_total_coefficient_bytes
        self.waveform_projection = nn.Linear(channels*configs[0].hop_length, dim)
        self.spectral_projections = nn.ModuleList([nn.Linear(channels*c.bands*3, dim) for c in configs])
        self.resolution_gate = nn.Linear((len(configs)+1)*dim, len(configs)+1)

    @property
    def frontend(self): return self.frontends[0]

    def get_extra_state(self):
        return {"version": "multiresolution-v1", "max_total_coefficient_bytes": self.max_total_coefficient_bytes}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("checkpoint multi-resolution config differs")

    def _enabled(self, resolution_mask):
        mask = tuple(resolution_mask) if resolution_mask is not None else (True,)*len(self.frontends)
        if len(mask) != len(self.frontends) or not all(isinstance(v, bool) for v in mask):
            raise ValueError("resolution_mask must contain one bool per spectral view")
        return mask

    def coefficient_bytes(self, shape, dtype=torch.float32, *, resolution_mask=None):
        mask = self._enabled(resolution_mask)
        # Primary always exists for reconstruction, even when excluded from fusion.
        return sum(f.coefficient_bytes(shape, dtype) for i, f in enumerate(self.frontends) if i == 0 or mask[i])

    def forward(self, audio: torch.Tensor | HybridEncoding, *, branches="both",
                resolution_mask=None, retain_carriers=False) -> MultiResolutionLatents:
        if branches not in ("both", "waveform", "spectral"):
            raise ValueError("branches must be 'both', 'waveform' or 'spectral'")
        enabled = self._enabled(resolution_mask)
        if branches == "waveform": enabled = (False,)*len(enabled)
        if branches == "spectral" and not any(enabled):
            raise ValueError("spectral mode needs at least one enabled resolution")
        x = audio.waveform if isinstance(audio, HybridEncoding) else audio
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError("audio must be [B,model_channels,N]")
        if x.dtype != self.waveform_projection.weight.dtype or x.device != self.waveform_projection.weight.device:
            raise ValueError("move/cast the encoder and audio to the same dtype and device explicitly")
        required = self.coefficient_bytes(tuple(x.shape), x.dtype, resolution_mask=enabled)
        if required > self.max_total_coefficient_bytes:
            raise ValueError(f"joint coefficient payload {required} bytes exceeds max_total_coefficient_bytes")
        primary = audio if isinstance(audio, HybridEncoding) else self.frontend(x)
        if primary.config != self.frontend.config:
            raise ValueError("primary carrier config differs from encoder config")
        hop, n = primary.config.hop_length, x.shape[-1]
        count = (n+hop-1)//hop
        starts = torch.arange(count, device=x.device)*hop
        lengths = (n-starts).clamp(max=hop)
        times = (starts.double()+lengths.double()/2+primary.sample_origin)/primary.config.sample_rate
        patches = F.pad(x, (0, count*hop-n)).unfold(-1, hop, hop).permute(0,2,1,3).flatten(2)
        zero = x.new_zeros(x.shape[0], count, self.waveform_projection.out_features)
        views = [self.waveform_projection(patches) if branches != "spectral" else zero]
        carriers = [primary]+[None]*(len(self.frontends)-1)
        for i, (frontend, projection, active) in enumerate(zip(self.frontends, self.spectral_projections, enabled)):
            if not active:
                views.append(zero)
                continue
            carrier = primary if i == 0 else frontend(x, sample_origin=primary.sample_origin)
            features, _, _ = carrier.context_features()
            views.append(projection(features))
            if retain_carriers: carriers[i] = carrier
        active = torch.tensor((branches != "spectral",)+enabled, device=x.device)
        scores = self.resolution_gate(torch.cat(views, -1)).masked_fill(~active, -torch.inf)
        weights = torch.softmax(scores, -1)
        tokens = (torch.stack(views, -2)*weights[..., None]).sum(-2)
        return MultiResolutionLatents(primary, tokens, times, lengths, weights, tuple(carriers))
