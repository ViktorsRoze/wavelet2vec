"""Differentiable stereo relationships on hybrid carriers.

Temporal support is reported explicitly. It is not a calibrated statistical
confidence or a guarantee of independent observations. All cues are lossy;
the supplied waveform/complex carrier remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from wavelet2vec.hybrid import HybridEncoding

STEREO_FEATURE_NAMES = ("cross_real", "cross_imag", "coherence", "balance",
                        "side_fraction", "log_energy", "support_fraction")


@dataclass(frozen=True)
class StereoFeatureConfig:
    pair: tuple[int, int] = (0, 1)
    coherence_cycles: float = 1.5
    minimum_seconds: float = 0.02
    power_floor: float = 1e-12

    def __post_init__(self):
        pair = tuple(self.pair)
        if (len(pair) != 2 or pair[0] == pair[1]
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair)):
            raise ValueError("pair must contain two distinct nonnegative channel indices")
        object.__setattr__(self, "pair", pair)
        for name in ("coherence_cycles", "power_floor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.minimum_seconds) or self.minimum_seconds < 0:
            raise ValueError("minimum_seconds must be finite and nonnegative")

    def to_dict(self):
        return {"version": "stereo-features-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("version", None) != "stereo-features-v1":
            raise ValueError("unsupported stereo feature version")
        return cls(**data)


@dataclass(frozen=True)
class StereoFeatures:
    values: torch.Tensor  # [B,T,K,7], named by STEREO_FEATURE_NAMES
    energy_valid: torch.Tensor  # [B,T,K], both channels above the power floor
    support_fraction: torch.Tensor  # [T,K], real interval / requested interval
    required_seconds: torch.Tensor  # [K], nominal bandwidth-based integration
    times_s: torch.Tensor  # [T], global patch midpoints
    centers_hz: torch.Tensor
    pair: tuple[int, int]


def _interval_mean(values: torch.Tensor, begin: torch.Tensor, end: torch.Tensor):
    """Piecewise-linear interval means with broadcast per-band coordinates.

    values [..., M+1] includes the last endpoint. Coordinate arrays broadcast
    against leading axes and carry their own final time axis. Double prefix
    sums avoid cancellation when a short interval lies late in a long buffer.
    """
    dtype = values.dtype
    v = values.double()
    prefix = F.pad(((v[..., :-1]+v[..., 1:])*.5).cumsum(-1), (1, 0))
    coordinates = torch.cat((begin, end), -1)
    index = coordinates.floor().long().clamp(0, v.shape[-1]-2)
    fraction = coordinates-index
    index = index.expand(*v.shape[:-1], index.shape[-1])
    left = v.gather(-1, index)
    integral = (prefix.gather(-1, index)+left*fraction
                +.5*(v.gather(-1, index+1)-left)*fraction.square())
    count = begin.shape[-1]
    return ((integral[..., count:]-integral[..., :count])/(end-begin)).to(dtype)


def hybrid_stereo_features(encoding: HybridEncoding, config: StereoFeatureConfig | None = None) -> StereoFeatures:
    """Cross-spectrum L*conj(R), amplitude coherence and energy relationships.

    Positive phase means L leads R. Coherence is |mean(L*conj(R))| divided
    by sqrt(mean|L|² mean|R|²), not magnitude-squared Welch coherence.
    The primary carrier's nominal bandwidth determines integration duration.
    Dense, compact and STFT grids are integrated in physical sample time.
    """
    cfg = config or StereoFeatureConfig()
    if max(cfg.pair) >= encoding.waveform.shape[1]:
        raise ValueError("stereo pair is outside the carrier's channel count")
    analysis = encoding.config
    device, dtype = encoding.waveform.device, encoding.waveform.dtype
    floor = max(cfg.power_floor, torch.finfo(dtype).tiny)
    n, hop = encoding.waveform.shape[-1], analysis.hop_length
    count = (n+hop-1)//hop
    starts = torch.arange(count, device=device)*hop
    lengths = (n-starts).clamp(max=hop)
    centers_sample = starts.double()+lengths.double()/2
    times = (centers_sample+encoding.sample_origin)/analysis.sample_rate
    if analysis.backend == "stft":
        window = encoding.windows.double()
        bandwidth = (analysis.sample_rate*window.square().sum()/window.sum().square()).expand(analysis.bands)
        scale = torch.full((analysis.bands,), 2., device=device, dtype=dtype)
        scale[0] = scale[-1] = 1
        values = encoding.coefficients*(scale/encoding.windows.sum())[:, None]
        groups = [(tuple(range(analysis.bands)), values, float(hop), 0., False)]
    elif analysis.backend == "compact_wavelet":
        bandwidth = torch.tensor([b.sigma_hz for b in encoding.band_specs], device=device, dtype=torch.float64)
        total = n+2*analysis.padding
        groups = ((g.band_indices, torch.stack([encoding.coefficients[i] for i in g.band_indices], -2),
                   total/g.windows.shape[-1], float(analysis.padding), True) for g in encoding.compact_groups)
    else:
        bandwidth = torch.sqrt((encoding.centers_hz.double()/analysis.q)**2+analysis.bandwidth_floor_hz**2)
        groups = [(tuple(range(analysis.bands)), encoding.coefficients, 1., float(analysis.padding), True)]
    minimum = max(cfg.minimum_seconds, hop/analysis.sample_rate)
    required = (cfg.coherence_cycles/bandwidth).clamp_min(minimum)
    features, validity, support = [None]*analysis.bands, [None]*analysis.bands, [None]*analysis.bands
    for indices, coefficients, step, offset, periodic in groups:
        left, right = coefficients[:, cfg.pair[0]], coefficients[:, cfg.pair[1]]
        cross = left*right.conj()
        moments = torch.stack((left.real.square()+left.imag.square(), right.real.square()+right.imag.square(),
                               cross.real, cross.imag), -2)  # B,K,4,M
        # The STFT has a final sampled frame, extended constantly up to EOF;
        # wavelet carriers are periodic on their padded transform interval.
        endpoint = moments[..., :1] if periodic else moments[..., -1:]
        moments = torch.cat((moments, endpoint), -1)
        duration = required[list(indices), None]*analysis.sample_rate
        begin = (centers_sample[None]-duration/2).clamp(0, n)
        end = (centers_sample[None]+duration/2).clamp(0, n)
        fraction = ((end-begin)/duration).clamp(0, 1)
        mean = _interval_mean(moments, ((begin+offset)/step)[:, None], ((end+offset)/step)[:, None])
        pl, pr, real, imag = mean.unbind(-2)
        pl, pr = pl.clamp_min(0), pr.clamp_min(0)
        active = (pl > floor)&(pr > floor)
        denominator = pl.clamp_min(floor).sqrt()*pr.clamp_min(floor).sqrt()
        cr = torch.where(active, real/denominator, 0.)
        ci = torch.where(active, imag/denominator, 0.)
        coherence = torch.linalg.vector_norm(torch.stack((cr, ci), -1), dim=-1)
        # Preserve the cross-vector direction while correcting roundoff > 1.
        normalizer = coherence.clamp_min(1.)
        cr, ci, coherence = cr/normalizer, ci/normalizer, coherence.clamp(max=1.)
        total_power = pl+pr
        balance = ((pr-pl)/total_power.clamp_min(floor)).clamp(-1, 1)
        side = torch.where(total_power > floor, (.5-real/total_power.clamp_min(floor)).clamp(0, 1), 0.)
        available = fraction.to(dtype)[None].expand_as(pl)
        f = torch.stack((cr, ci, coherence, balance, side, torch.log1p(total_power/2), available), -1)
        for local, index in enumerate(indices):
            features[index] = f[:, local]
            validity[index] = active[:, local]
            support[index] = fraction[local]
    return StereoFeatures(torch.stack(features, 2), torch.stack(validity, 2), torch.stack(support, 1),
                          required, times, encoding.centers_hz, cfg.pair)


@dataclass(frozen=True)
class StereoLatents:
    encoding: HybridEncoding
    tokens: torch.Tensor
    times_s: torch.Tensor
    valid_samples: torch.Tensor
    stereo: StereoFeatures | None
    base: object  # retains the wrapped model's diagnostics, including resolution weights


class StereoTokenEncoder(nn.Module):
    """Opt-in trainable stereo adapter for local or multi-resolution encoders."""
    def __init__(self, encoder: nn.Module, config: StereoFeatureConfig | None = None):
        super().__init__()
        self.encoder = encoder
        self.config = config or StereoFeatureConfig()
        if max(self.config.pair) >= self.channels:
            raise ValueError("stereo pair is outside the encoder's channel count")
        dim = self.waveform_projection.out_features
        self.stereo_projection = nn.Linear(self.frontend.config.bands*len(STEREO_FEATURE_NAMES), dim)
        self.stereo_gate = nn.Linear(2*dim, dim)

    @property
    def frontend(self): return self.encoder.frontend

    @property
    def waveform_projection(self): return self.encoder.waveform_projection

    @property
    def channels(self): return self.encoder.channels

    def get_extra_state(self): return self.config.to_dict()

    def set_extra_state(self, state):
        if StereoFeatureConfig.from_dict(state) != self.config:
            raise ValueError("checkpoint stereo feature config differs")

    def forward(self, audio, *, branches="both", stereo=True, **kwargs):
        base = self.encoder(audio, branches=branches, **kwargs)
        if not stereo:
            return StereoLatents(base.encoding, base.tokens, base.times_s, base.valid_samples, None, base)
        features = hybrid_stereo_features(base.encoding, self.config)
        spatial = self.stereo_projection(features.values.flatten(2))
        gate = torch.sigmoid(self.stereo_gate(torch.cat((base.tokens, spatial), -1)))
        tokens = gate*base.tokens+(1-gate)*spatial
        return StereoLatents(base.encoding, tokens, base.times_s, base.valid_samples, features, base)
