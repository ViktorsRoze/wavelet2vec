"""Phase-preserving signal carriers and trainable waveform/spectral fusion.

The retained waveform and complex coefficients are signal carriers. Tokens
are lossy learned features, never an advertised replacement for the carriers.
This is a short-fragment research frontend, not a source separator or codec.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wavelet2vec.multirate import CompactBand, CompactGroup

HYBRID_VERSION = "hybrid-v2"


@dataclass(frozen=True)
class HybridConfig:
    sample_rate: int = 44100
    backend: str = "wavelet"
    n_bands: int = 64
    f_min: float = 20.0
    f_max: float | None = None
    q: float = 8.0  # center / Gaussian frequency standard deviation
    bandwidth_floor_hz: float = 0.0
    padding_samples: int | None = None  # None: 250 ms zero context per side
    hop_length: int = 256  # raw patch and context-token spacing
    n_fft: int = 2048  # STFT comparator only
    context_mode: str = "complex"  # "magnitude" ablates signed baseband features
    compact_truncate_sigma: float = 4.0  # finite support, rounded outwards to FFT bins
    compact_oversampling: int = 1  # extra complex time samples per finite band
    compact_fft_mode: str = "power2"  # "fast" minimizes lengths; power2 batches FFTs
    max_coefficient_bytes: int = 256 * 1024**2

    def __post_init__(self):
        for name in ("sample_rate", "n_bands", "hop_length", "n_fft", "max_coefficient_bytes", "compact_oversampling"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.backend not in ("wavelet", "compact_wavelet", "stft"):
            raise ValueError("backend must be 'wavelet', 'compact_wavelet' or 'stft'")
        if not math.isfinite(self.compact_truncate_sigma) or self.compact_truncate_sigma <= 0:
            raise ValueError("compact_truncate_sigma must be finite and positive")
        if self.compact_fft_mode not in ("fast", "power2"):
            raise ValueError("compact_fft_mode must be 'fast' or 'power2'")
        if self.context_mode not in ("complex", "magnitude"):
            raise ValueError("context_mode must be 'complex' or 'magnitude'")
        if self.n_bands < 2:
            raise ValueError("n_bands must be at least 2")
        upper = self.sample_rate / 2 if self.f_max is None else self.f_max
        if not (math.isfinite(self.f_min) and math.isfinite(upper)
                and 0 < self.f_min < upper <= self.sample_rate / 2):
            raise ValueError("require 0 < f_min < f_max <= Nyquist")
        if not math.isfinite(self.q) or self.q <= 0:
            raise ValueError("q must be finite and positive")
        if not math.isfinite(self.bandwidth_floor_hz) or self.bandwidth_floor_hz < 0:
            raise ValueError("bandwidth_floor_hz must be finite and nonnegative")
        if self.padding_samples is not None and (
            isinstance(self.padding_samples, bool) or not isinstance(self.padding_samples, int)
            or self.padding_samples < 0
        ):
            raise ValueError("padding_samples must be a nonnegative integer or None")
        if self.backend == "stft" and (self.n_fft < 4 or self.n_fft % 2
                                      or self.hop_length > self.n_fft // 2):
            raise ValueError("STFT requires even n_fft >= 4 and hop_length <= n_fft / 2")

    @property
    def bands(self) -> int:
        return self.n_fft // 2 + 1 if self.backend == "stft" else self.n_bands

    @property
    def padding(self) -> int:
        return self.sample_rate // 4 if self.padding_samples is None else self.padding_samples

    def to_dict(self) -> dict:
        return {"version": HYBRID_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "HybridConfig":
        data = dict(data)
        version = data.pop("version", None)
        if version not in ("hybrid-v1", HYBRID_VERSION):
            raise ValueError("unsupported hybrid configuration version")
        if version == "hybrid-v1" and data.get("backend") == "compact_wavelet":
            raise ValueError("compact_wavelet requires hybrid-v2 metadata")
        return cls(**data)


@dataclass(frozen=True)
class HybridEncoding:
    """Carriers for explicit ``[batch, channels, samples]`` input.

    Dense wavelet: [B,C,K,N+2*pad]; STFT: [B,C,F,T]; compact wavelet:
    tuple of K tensors [B,C,M_k] with explicit band/sample-grid metadata.
    ``waveform``
    references the input tensor without copying, mixing, scaling or detaching.
    Treat all tensors as immutable. Metadata/windows must travel with the
    coefficients. Do not combine these coefficients with the legacy inverse.
    """
    waveform: torch.Tensor
    coefficients: torch.Tensor | tuple[torch.Tensor, ...]
    config: HybridConfig
    centers_hz: torch.Tensor
    windows: torch.Tensor | tuple[torch.Tensor, ...]
    coverage: torch.Tensor | None
    band_specs: tuple[CompactBand, ...] | None = None
    compact_groups: tuple[CompactGroup, ...] | None = None
    sample_origin: int = 0  # sample coordinate of this fragment in its parent timeline

    @property
    def coefficient_nbytes(self) -> int:
        bands = self.coefficients if isinstance(self.coefficients, tuple) else (self.coefficients,)
        return sum(b.numel()*b.element_size() for b in bands)

    def band_times_s(self, index: int) -> torch.Tensor:
        """Compact band's actual sample coordinates relative to original audio."""
        if self.config.backend != "compact_wavelet":
            raise ValueError("band_times_s is for compact_wavelet carriers")
        band = self.band_specs[index]
        total = self.waveform.shape[-1] + 2*self.config.padding
        return (torch.arange(band.samples, device=self.waveform.device, dtype=torch.float64)
                * total/band.samples-self.config.padding+self.sample_origin) / self.config.sample_rate

    def synthesize(self, coefficients: torch.Tensor | tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        """Synthesize from coefficients, never from the retained waveform.

        Optional source/batch leading dimensions are supported. The last two
        dimensions, dtype and device must match the carrier. Compact input
        instead takes a tuple matching each band's final sample dimension.
        Predicted complex
        coefficients are projected to a real signal by this defined inverse;
        arbitrary predictions need not be consistent analysis coefficients.
        """
        if self.config.backend == "compact_wavelet":
            from wavelet2vec.multirate import synthesize_compact
            return synthesize_compact(self, coefficients)
        c = self.coefficients if coefficients is None else coefficients
        if (c.ndim < 2 or c.shape[-2:] != self.coefficients.shape[-2:]
                or c.dtype != self.coefficients.dtype or c.device != self.coefficients.device):
            raise ValueError("coefficient shape, complex dtype and device must match the carrier")
        n = self.waveform.shape[-1]
        if self.config.backend == "stft":
            shape = c.shape[:-2]
            y = torch.istft(c.reshape(-1, *c.shape[-2:]), n_fft=self.config.n_fft,
                            hop_length=self.config.hop_length, window=self.windows,
                            center=True, length=n)
            return y.reshape(*shape, n)
        padded_n = c.shape[-1]
        half = padded_n // 2 + 1
        # Canonical weighted inverse on analytic positive-frequency bands.
        band_spectra = torch.fft.fft(c, dim=-1)[..., :half]
        weighted = (band_spectra * self.windows).sum(dim=-2) / self.coverage
        weights = torch.full((half,), 0.5, dtype=c.real.dtype, device=c.device)
        weights[0] = 1.0
        if padded_n % 2 == 0:
            weights[-1] = 1.0
        y = torch.fft.irfft(weighted * weights, n=padded_n, dim=-1)
        pad = self.config.padding
        return y[..., pad:pad + n]

    def context_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lossy [B,T,C*K*3] features and [T] times/counts for raw patches.

        Each band supplies signed real/imaginary baseband averages and log RMS.
        Demodulation removes its center carrier before time pooling, avoiding
        cancellation of a stationary center-frequency tone. No claim of full
        phase retention is made for these pooled features. Carrier tensors are
        retained separately. STFT features interpolate demodulated frames at
        raw-patch centers; wavelet features average over each patch.
        """
        if self.config.backend == "compact_wavelet":
            from wavelet2vec.multirate import compact_context
            return compact_context(self)
        c = self.coefficients
        n, hop = self.waveform.shape[-1], self.config.hop_length
        count = (n + hop - 1) // hop
        starts = torch.arange(count, device=c.device) * hop
        lengths = (n - starts).clamp(max=hop)
        times = (starts.to(torch.float64) + lengths.double() / 2 + self.sample_origin) / self.config.sample_rate
        if self.config.backend == "wavelet":
            pad = self.config.padding
            local = c[..., pad:pad + n]
            sample_times = torch.arange(n, dtype=torch.float64, device=c.device) / self.config.sample_rate
        else:
            scale = torch.full((c.shape[-2],), 2., dtype=c.real.dtype, device=c.device)
            scale[0] = scale[-1] = 1.
            # Match a bin-centered sinusoid's amplitude to analytic wavelets.
            # The retained STFT remains untouched for exact synthesis.
            local = c * (scale / self.windows.sum())[:, None]
            sample_times = (torch.arange(c.shape[-1], dtype=torch.float64, device=c.device)
                            * hop - self.config.n_fft / 2) / self.config.sample_rate
        # Double-precision phase reduction also avoids large float32 angles on
        # longer fragments; the carrier dtype itself is not changed.
        centers = self.centers_hz.double()[:, None]
        origin_turns = torch.remainder(centers * (self.sample_origin / self.config.sample_rate), 1.0)
        turns = torch.remainder(centers * sample_times + origin_turns, 1.0)
        phase = -2 * math.pi * turns
        oscillator = torch.complex(phase.cos().to(c.real.dtype), phase.sin().to(c.real.dtype))
        baseband = local * oscillator
        power = local.real.square() + local.imag.square()
        if self.config.backend == "wavelet":
            right = count * hop - n
            pooled = F.pad(baseband, (0, right)).reshape(*baseband.shape[:-1], count, hop).sum(-1)
            pooled = pooled / lengths.to(c.real.dtype)
            energy = F.pad(power, (0, right)).reshape(*power.shape[:-1], count, hop).sum(-1)
            energy = energy / lengths.to(c.real.dtype)
        else:
            i0 = torch.arange(count, device=c.device).clamp(max=local.shape[-1] - 1)
            i1 = (i0 + 1).clamp(max=local.shape[-1] - 1)
            fraction = lengths.to(c.real.dtype) / (2 * hop)
            pooled = baseband[..., i0] * (1 - fraction) + baseband[..., i1] * fraction
            energy = power[..., i0] * (1 - fraction) + power[..., i1] * fraction
        # Smooth at silence; no angle(0), sqrt(0) or per-clip peak division.
        eps = torch.finfo(c.real.dtype).eps
        rms = torch.sqrt(energy + eps**2)
        features = torch.stack((pooled.real / (1 + rms), pooled.imag / (1 + rms),
                                torch.log1p(rms - eps)), dim=-1)  # B,C,K,T,3
        if self.config.context_mode == "magnitude":
            # Keep dimensionality and full carriers identical for ablations.
            features = features * features.new_tensor([0., 0., 1.])
        features = features.permute(0, 3, 1, 2, 4).flatten(2)
        return features, times, lengths


class HybridAudioFrontend(nn.Module):
    """Native-rate stereo waveform plus one invertible spectral carrier.

    Float32/float64 input is preserved. Compact wavelets use different
    complex sample rates per band; dense wavelets remain the reference.
    Wavelet Q and center density are
    independent. Zero padding reduces circular boundary interaction but is
    not a universal error/apron guarantee. The byte guard limits the final
    coefficient tensor only, not all FFT/autograd working memory.
    """
    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()

    def get_extra_state(self):
        return self.config.to_dict()

    def set_extra_state(self, state):
        config = HybridConfig.from_dict(state)
        if config != self.config:
            raise ValueError("checkpoint hybrid config differs; construct the frontend with its saved config")

    def coefficient_bytes(self, shape: tuple[int, int, int], dtype=torch.float32) -> int:
        """Final coefficient allocation only; call before allocating batches."""
        if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
            raise ValueError("expected nonempty [batch, channels, samples] shape")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("only float32 and float64 are supported")
        b, channels, n = shape
        if self.config.backend == "compact_wavelet":
            from wavelet2vec.multirate import compact_layout
            layout = compact_layout(self.config, n+2*self.config.padding)
            return b*channels*sum(band.samples for band in layout)*(8 if dtype == torch.float32 else 16)
        steps = n + 2 * self.config.padding if self.config.backend == "wavelet" else n // self.config.hop_length + 1
        return b * channels * self.config.bands * steps * (8 if dtype == torch.float32 else 16)

    def forward(self, audio: torch.Tensor, *, sample_origin: int = 0) -> HybridEncoding:
        if isinstance(sample_origin, bool) or not isinstance(sample_origin, int):
            raise ValueError("sample_origin must be an integer sample coordinate")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 3:
            raise ValueError("audio must be a torch tensor with explicit [batch, channels, samples] axes")
        size = self.coefficient_bytes(tuple(audio.shape), audio.dtype)
        if size > self.config.max_coefficient_bytes:
            raise ValueError(f"coefficient tensor requires {size} bytes, exceeding max_coefficient_bytes; "
                             "reduce batch, channels, duration or bands (FFT/gradient memory is additional)")
        if not bool(torch.isfinite(audio).all()):
            raise ValueError("audio must contain only finite values")
        cfg = self.config
        if cfg.backend == "compact_wavelet":
            from wavelet2vec.multirate import analyze_compact
            return replace(analyze_compact(audio, cfg), sample_origin=sample_origin)
        if cfg.backend == "stft":
            window = torch.hann_window(cfg.n_fft, dtype=audio.dtype, device=audio.device)
            c = torch.stft(audio.reshape(-1, audio.shape[-1]), n_fft=cfg.n_fft,
                           hop_length=cfg.hop_length, window=window, center=True,
                           pad_mode="constant", return_complex=True)
            c = c.reshape(*audio.shape[:2], *c.shape[-2:])
            centers = torch.fft.rfftfreq(cfg.n_fft, 1 / cfg.sample_rate, device=audio.device, dtype=audio.dtype)
            return HybridEncoding(audio, c, cfg, centers, window, None, sample_origin=sample_origin)

        n = audio.shape[-1] + 2 * cfg.padding
        # Design in float64, reject badly covered banks before tensor FFTs.
        centers = np.geomspace(cfg.f_min, cfg.f_max or cfg.sample_rate / 2, cfg.n_bands)
        freqs = np.fft.rfftfreq(n, 1 / cfg.sample_rate)
        sigmas = np.sqrt((centers / cfg.q)**2 + cfg.bandwidth_floor_hz**2)
        windows = np.exp(-0.5 * ((freqs[None, :] - centers[:, None]) / sigmas[:, None])**2)
        windows[0, freqs <= centers[0]] = 1
        windows[-1, freqs >= centers[-1]] = 1
        coverage = np.sum(windows**2, axis=0)
        if not np.isfinite(coverage).all() or coverage.min() / coverage.max() < 1e-6:
            raise ValueError("ill-conditioned wavelet bank: increase bands or bandwidth (reduce q)")
        w = torch.as_tensor(windows, dtype=audio.dtype, device=audio.device)
        cov = w.square().sum(0)
        padded = F.pad(audio, (cfg.padding, cfg.padding))
        spectrum = torch.fft.rfft(padded, dim=-1)
        analytic_weights = torch.full((spectrum.shape[-1],), 2., dtype=audio.dtype, device=audio.device)
        analytic_weights[0] = 1
        if n % 2 == 0:
            analytic_weights[-1] = 1
        positive = spectrum.unsqueeze(-2) * w * analytic_weights
        c = torch.fft.ifft(F.pad(positive, (0, n - spectrum.shape[-1])), dim=-1)
        return HybridEncoding(audio, c, cfg,
                              torch.as_tensor(centers, dtype=audio.dtype, device=audio.device), w, cov,
                              sample_origin=sample_origin)


@dataclass(frozen=True)
class HybridLatents:
    encoding: HybridEncoding
    tokens: torch.Tensor  # [B, ceil(N/hop), D], lossy
    times_s: torch.Tensor
    valid_samples: torch.Tensor


class HybridTokenEncoder(nn.Module):
    """Trainable aligned raw-patch / phase-aware spectral feature fusion.

    Branch ablations share the output dimension. This simple baseline uses
    linear branch projections and a learned per-feature gate; it does not
    implement long-context attention or demonstrate novel separation quality.
    Save config alongside state_dict; mismatched config is rejected on load.
    """
    def __init__(self, config: HybridConfig | None = None, *, channels: int = 2, dim: int = 128):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0 or not isinstance(dim, int) or dim <= 0:
            raise ValueError("channels and dim must be positive integers")
        self.frontend = HybridAudioFrontend(config)
        self.channels = channels
        cfg = self.frontend.config
        self.waveform_projection = nn.Linear(channels * cfg.hop_length, dim)
        self.spectral_projection = nn.Linear(channels * cfg.bands * 3, dim)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, audio: torch.Tensor | HybridEncoding, *, branches: str = "both") -> HybridLatents:
        if branches not in ("both", "waveform", "spectral"):
            raise ValueError("branches must be 'both', 'waveform' or 'spectral'")
        encoded = audio if isinstance(audio, HybridEncoding) else self.frontend(audio)
        if encoded.config != self.frontend.config:
            raise ValueError("encoding config differs from encoder config")
        x = encoded.waveform
        if x.shape[1] != self.channels:
            raise ValueError("input channels differ from encoder channels")
        if x.dtype != self.waveform_projection.weight.dtype or x.device != self.waveform_projection.weight.device:
            raise ValueError("move/cast the encoder and audio to the same dtype and device explicitly")
        features, times, lengths = encoded.context_features()
        hop = encoded.config.hop_length
        patches = F.pad(x, (0, len(lengths) * hop - x.shape[-1])).unfold(-1, hop, hop)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        if branches == "waveform":
            tokens = self.waveform_projection(patches)
        elif branches == "spectral":
            tokens = self.spectral_projection(features)
        else:
            raw = self.waveform_projection(patches)
            spectral = self.spectral_projection(features)
            gate = torch.sigmoid(self.gate(torch.cat((raw, spectral), -1)))
            tokens = gate * raw + (1 - gate) * spectral
        return HybridLatents(encoded, tokens, times, lengths)
