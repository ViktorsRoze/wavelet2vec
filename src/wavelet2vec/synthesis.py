"""Trainable, carrier-conditioned source synthesis on explicit sample grids.

This is a separation decoder, not a carrier-free note generator. Complex
masks are unbounded; a waveform residual can recover content absent from a
mixture coefficient. Output source additivity is not imposed implicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from wavelet2vec.context import MusicContextConfig, MusicContextEncoder
from wavelet2vec.hybrid import HybridAudioFrontend, HybridConfig, HybridEncoding, HybridTokenEncoder


@dataclass(frozen=True)
class SynthesisConfig:
    source_names: tuple[str, ...] = ("target", "remainder")
    channels: int = 2
    dim: int = 128
    waveform_residual: bool = True
    max_output_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if isinstance(self.source_names, str):
            raise ValueError("source_names must be a sequence of unique names")
        names = tuple(self.source_names)
        if (not names or any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in names)
                or len(set(names)) != len(names)):
            raise ValueError("source_names must contain unique nonempty names without surrounding whitespace")
        object.__setattr__(self, "source_names", names)
        for key in ("channels", "dim", "max_output_bytes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.waveform_residual, bool):
            raise ValueError("waveform_residual must be bool")

    def to_dict(self):
        return {"version": "source-synthesis-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("version", None) != "source-synthesis-v1":
            raise ValueError("unsupported synthesis configuration")
        return cls(**data)


@dataclass(frozen=True)
class SourceSynthesis:
    waveform: torch.Tensor  # B,S,C,N; zero outside lengths
    lengths: torch.Tensor  # B, original sample counts
    source_names: tuple[str, ...]
    sample_rate: int
    sample_origin: int
    # Complete offline regions [origin, origin + length); no causal latency claim.
    latency_samples: int | None = None
    coefficients: torch.Tensor | tuple[torch.Tensor, ...] | None = None


def _interpolate(values, centers, positions):
    """Linear interpolation on physical sample coordinates, edge-held outside."""
    if centers.numel() == 1:
        return values[..., :1].expand(*values.shape[:-1], positions.numel())
    upper = torch.searchsorted(centers, positions).clamp(1, centers.numel() - 1)
    lower = upper - 1
    fraction = ((positions - centers[lower]) / (centers[upper] - centers[lower])).clamp(0, 1)
    return values[..., lower] + (values[..., upper] - values[..., lower]) * fraction.to(values.real.dtype)


class HybridSourceDecoder(nn.Module):
    """Token-conditioned complex masks plus optional overlapping waveform patches.

    Tokens must use the frontend's raw-patch grid, including a shortened final
    patch. Masks interpolate at actual coefficient coordinates, not by resizing
    band arrays. Initialization returns equal shares of the mixture through
    coefficient synthesis; the residual starts at zero.
    """
    def __init__(self, frontend: HybridConfig, config: SynthesisConfig | None = None):
        super().__init__()
        self.frontend_config = frontend
        self.config = config or SynthesisConfig()
        cfg = self.config
        sources = len(cfg.source_names)
        self.mask_head = nn.Linear(cfg.dim, sources * cfg.channels * frontend.bands * 2)
        self.residual_head = (nn.Linear(cfg.dim, sources * cfg.channels * frontend.hop_length * 2)
                              if cfg.waveform_residual else None)
        for head in (self.mask_head, self.residual_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def get_extra_state(self):
        return {"frontend": self.frontend_config.to_dict(), "synthesis": self.config.to_dict()}

    def set_extra_state(self, state):
        if (HybridConfig.from_dict(state["frontend"]) != self.frontend_config
                or SynthesisConfig.from_dict(state["synthesis"]) != self.config):
            raise ValueError("checkpoint synthesis configuration differs")

    def _budget(self, encoding):
        b, c, n = encoding.waveform.shape
        s, hop = len(self.config.source_names), self.frontend_config.hop_length
        t = (n + hop - 1) // hop
        element = encoding.waveform.element_size()
        # Explicit payload estimate, NOT a bound on FFT/autograd/allocator RAM.
        size = s * encoding.coefficient_nbytes + b*s*c*(n + 2*self.frontend_config.bands*t)*element
        if self.residual_head is not None:
            size += b*s*c*(2*hop*t + 2*(t+1)*hop)*element
        if size > self.config.max_output_bytes:
            raise ValueError(f"synthesis payload estimate {size} exceeds max_output_bytes")

    def forward(self, tokens: torch.Tensor, encoding: HybridEncoding, *,
                return_coefficients: bool = False) -> SourceSynthesis:
        cfg, front = self.config, self.frontend_config
        if encoding.config != front:
            raise ValueError("decoder and carrier configurations differ")
        b, c, n = encoding.waveform.shape
        hop, sources = front.hop_length, len(cfg.source_names)
        count = (n + hop - 1) // hop
        if c != cfg.channels or tokens.shape != (b, count, cfg.dim):
            raise ValueError("tokens must match batch, channels and the carrier's raw-patch grid")
        if (tokens.dtype != encoding.waveform.dtype or tokens.device != encoding.waveform.device
                or tokens.dtype != self.mask_head.weight.dtype or tokens.device != self.mask_head.weight.device):
            raise ValueError("tokens, carrier and decoder must share dtype and device")
        if not torch.isfinite(tokens).all():
            raise ValueError("synthesis tokens must be finite")
        self._budget(encoding)
        starts = torch.arange(count, device=tokens.device, dtype=torch.float64) * hop
        centers = starts + (n - starts).clamp(max=hop) / 2
        raw = self.mask_head(tokens).reshape(b, count, sources, c, front.bands, 2)
        masks = torch.complex(raw[..., 0], raw[..., 1]).permute(0, 2, 3, 4, 1) + 1 / sources
        if front.backend == "compact_wavelet":
            total = n + 2*front.padding
            predicted = []
            for k, coefficient in enumerate(encoding.coefficients):
                # Relative coordinates avoid subtracting large absolute origins.
                positions = (torch.arange(coefficient.shape[-1], device=tokens.device, dtype=torch.float64)
                             * (total / coefficient.shape[-1]) - front.padding)
                mask = _interpolate(masks[..., k, :], centers, positions)
                predicted.append(coefficient.unsqueeze(1) * mask)
            predicted = tuple(predicted)
        else:
            positions = torch.arange(encoding.coefficients.shape[-1], device=tokens.device, dtype=torch.float64)
            positions = positions * hop if front.backend == "stft" else positions - front.padding
            predicted = encoding.coefficients.unsqueeze(1) * _interpolate(masks, centers, positions)
        waveform = encoding.synthesize(predicted)
        if self.residual_head is not None:
            # Each patch spans [j*hop-hop, j*hop+hop). Positive midpoint Hann
            # weights avoid a zero denominator even for one-sample fragments.
            patches = self.residual_head(tokens).reshape(b, count, sources*c, 2*hop)
            patches = patches.permute(0, 2, 3, 1).reshape(b*sources*c, 2*hop, count)
            phase = (torch.arange(2*hop, device=tokens.device, dtype=tokens.dtype) + .5) / (2*hop)
            window = torch.sin(torch.pi * phase).square()
            options = dict(output_size=(1, (count+1)*hop), kernel_size=(1, 2*hop), stride=(1, hop))
            residual = F.fold(patches * window[None, :, None], **options).reshape(b, sources, c, -1)
            weights = F.fold(window[None, :, None].expand(1, -1, count), **options).flatten()
            waveform = waveform + (residual / weights)[..., hop:hop+n]
        return SourceSynthesis(waveform, torch.full((b,), n, device=tokens.device, dtype=torch.long),
                               cfg.source_names, front.sample_rate, encoding.sample_origin,
                               coefficients=predicted if return_coefficients else None)


def project_source_coefficients(encoding: HybridEncoding, coefficients):
    """Project predicted coefficients onto analysis of their synthesized audio.

    Covers the spectral component only, before any waveform residual. Preserves
    synthesized audio to numerical tolerance. Analysis uses the same boundary
    padding; this also removes inconsistent content outside the real fragment.
    The extra analysis obeys the frontend's coefficient allocation limit.
    """
    waveform = encoding.synthesize(coefficients)
    if waveform.ndim != 4 or waveform.shape[0] != encoding.waveform.shape[0] or waveform.shape[2:] != encoding.waveform.shape[1:]:
        raise ValueError("source coefficients must synthesize to [B,S,C,N]")
    b, s, c, n = waveform.shape
    analyzed = HybridAudioFrontend(encoding.config)(waveform.reshape(b*s, c, n), sample_origin=encoding.sample_origin)
    if isinstance(analyzed.coefficients, tuple):
        return tuple(value.reshape(b, s, c, -1) for value in analyzed.coefficients)
    return analyzed.coefficients.reshape(b, s, c, *analyzed.coefficients.shape[-2:])


class HybridSeparator(nn.Module):
    """Reference supervised separator: hybrid tokens, optional context, synthesis.

    Ragged examples run at their own lengths; batch padding never becomes
    acoustic context. This favors a clear training contract over throughput.
    """
    def __init__(self, frontend: HybridConfig, synthesis: SynthesisConfig | None = None,
                 context: MusicContextConfig | None = None):
        super().__init__()
        self.config = synthesis or SynthesisConfig()
        if context is not None and context.dim != self.config.dim:
            raise ValueError("context and synthesis dimensions must match")
        self.encoder = HybridTokenEncoder(frontend, channels=self.config.channels, dim=self.config.dim)
        self.context = MusicContextEncoder(context) if context is not None else None
        self.decoder = HybridSourceDecoder(frontend, self.config)

    def _forward(self, audio, origin, branches):
        encoding = self.encoder.frontend(audio, sample_origin=origin)
        latent = self.encoder(encoding, branches=branches)
        tokens = latent.tokens
        if self.context is not None:
            tokens = self.context(tokens, times_s=latent.times_s, valid_samples=latent.valid_samples).tokens
        return self.decoder(tokens, encoding)

    def forward(self, audio: torch.Tensor, *, lengths: torch.Tensor | None = None,
                sample_origin: int = 0, branches: str = "both") -> SourceSynthesis:
        if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or min(audio.shape) <= 0:
            raise ValueError("audio must be nonempty [B,C,N]")
        if lengths is None:
            return self._forward(audio, sample_origin, branches)
        if (lengths.shape != (audio.shape[0],) or lengths.dtype != torch.long or lengths.device != audio.device
                or (lengths <= 0).any() or (lengths > audio.shape[-1]).any()):
            raise ValueError("lengths must be int64 [B] on the audio device, in [1,N]")
        pieces = [self._forward(audio[i:i+1, :, :n], sample_origin, branches).waveform
                  for i, n in enumerate(lengths.tolist())]
        waveform = torch.cat([F.pad(piece, (0, audio.shape[-1] - piece.shape[-1])) for piece in pieces])
        return SourceSynthesis(waveform, lengths.clone(), self.config.source_names,
                               self.encoder.frontend.config.sample_rate, sample_origin)


@torch.no_grad()
def separate_audio(model: HybridSeparator, audio: torch.Tensor, *, window_samples: int,
                   overlap_samples: int, context_samples: int = 0, sample_origin: int = 0,
                   branches: str = "both") -> SourceSynthesis:
    """Offline overlap-add inference with bounded model windows and real context.

    Input and output waveforms remain resident. Context/overlap reduce seams
    but do not guarantee chunk invariance for learned models. Source slots are
    fixed by the model; arbitrary instrument-instance tracking is not provided.
    Individual module training modes are restored, including on exceptions.
    """
    from wavelet2vec.rendering import SourceRenderConfig, iter_separate_reader
    config = SourceRenderConfig(window_samples, overlap_samples, context_samples)
    if isinstance(sample_origin, bool) or not isinstance(sample_origin, int):
        raise ValueError("sample_origin must be an integer")
    if (not isinstance(audio, torch.Tensor) or audio.ndim != 3 or min(audio.shape) <= 0
            or audio.shape[1] != model.config.channels):
        raise ValueError("audio must be nonempty [B,model channels,N]")
    if audio.dtype != model.decoder.mask_head.weight.dtype or audio.device != model.decoder.mask_head.weight.device:
        raise ValueError("model and audio must share dtype and device")
    b, c, n = audio.shape
    result = audio.new_zeros(b, len(model.config.source_names), c, n)
    for chunk in iter_separate_reader(model, lambda lo, hi: audio[..., lo:hi], n, config,
                                      sample_origin=sample_origin, branches=branches):
        start = chunk.sample_origin-sample_origin
        result[..., start:start+chunk.waveform.shape[-1]] = chunk.waveform
    return SourceSynthesis(result, torch.full((b,), n, device=audio.device, dtype=torch.long),
                           model.config.source_names, model.encoder.frontend.config.sample_rate, sample_origin)
