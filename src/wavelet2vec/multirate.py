"""Compact analytic Gaussian filter bank for the hybrid carrier.

Each finite frequency interval is translated and represented by its own
complex sample grid. There is no stride-decimation of a full-rate carrier.
Full spectral coverage and a weighted inverse retain the input waveform.
This uses established nonuniform filter-bank ideas, not a novel transform.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import next_fast_len
import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from wavelet2vec.hybrid import HybridConfig, HybridEncoding


@dataclass(frozen=True)
class CompactBand:
    center_hz: float
    sigma_hz: float
    start_bin: int
    stop_bin: int  # exclusive, on padded input's positive FFT grid
    carrier_bin: int
    samples: int  # complex sample count across the entire padded duration


@dataclass(frozen=True)
class CompactGroup:
    """Batch layout for bands sharing a complex FFT length."""
    band_indices: tuple[int, ...]
    bin_indices: torch.Tensor  # [group_bands, band_samples] input rFFT indices
    windows: torch.Tensor  # packed windows; zero outside each band's interval


def compact_layout(config: HybridConfig, padded_n: int) -> tuple[CompactBand, ...]:
    """Band layout without signal/window tensors; DFT intervals round outwards."""
    centers = np.geomspace(config.f_min, config.f_max or config.sample_rate / 2, config.n_bands)
    half, spacing = padded_n // 2 + 1, config.sample_rate / padded_n
    bands = []
    for i, center in enumerate(centers):
        sigma = math.hypot(center / config.q, config.bandwidth_floor_hz)
        radius = config.compact_truncate_sigma * sigma
        lo = max(0, min(half - 1, math.floor((center - radius) / spacing)))
        hi = max(lo + 1, min(half, math.ceil((center + radius) / spacing) + 1))
        if i == 0: lo = 0
        if i == len(centers) - 1: hi = half
        carrier = max(lo, min(hi - 1, round(center / spacing)))
        needed = (hi-lo)*config.compact_oversampling
        samples = next_fast_len(needed) if config.compact_fft_mode == "fast" else 1 << (needed-1).bit_length()
        bands.append(CompactBand(float(center), sigma, lo, hi, carrier, samples))
    return tuple(bands)


def analyze_compact(audio: torch.Tensor, config: HybridConfig):
    """No [bands, full_length] analysis windows or coefficient intermediates."""
    from wavelet2vec.hybrid import HybridEncoding

    n = audio.shape[-1] + 2 * config.padding
    layout = compact_layout(config, n)
    coverage = np.zeros(n // 2 + 1)
    windows = []
    for i, band in enumerate(layout):
        frequencies = np.arange(band.start_bin, band.stop_bin) * config.sample_rate / n
        window = np.exp(-0.5 * ((frequencies - band.center_hz) / band.sigma_hz)**2)
        if i == 0: window[frequencies <= band.center_hz] = 1
        if i == len(layout)-1: window[frequencies >= band.center_hz] = 1
        coverage[band.start_bin:band.stop_bin] += window**2
        windows.append(window)
    if not np.isfinite(coverage).all() or coverage.min() / coverage.max() < 1e-6:
        raise ValueError("ill-conditioned compact wavelet bank: increase bands, bandwidth or truncation support")
    # Group by FFT length. Power-of-two planning trades some extra samples for
    # many fewer CUDA kernels. Packing tables are O(sum band lengths), not K*N.
    by_length = {}
    for i, band in enumerate(layout):
        by_length.setdefault(band.samples, []).append(i)
    groups = []
    for samples, indices in by_length.items():
        packed_indices, packed_windows = [], []
        for index in indices:
            band = layout[index]
            local = (np.arange(samples) - (band.start_bin-band.carrier_bin)) % samples
            valid = local < band.stop_bin-band.start_bin
            safe = np.minimum(local, band.stop_bin-band.start_bin-1)
            packed_indices.append(safe+band.start_bin)
            packed_windows.append(np.where(valid,windows[index][safe],0))
        groups.append(CompactGroup(tuple(indices),
            torch.as_tensor(np.stack(packed_indices),device=audio.device,dtype=torch.int64),
            torch.as_tensor(np.stack(packed_windows),device=audio.device,dtype=audio.dtype)))
    windows = tuple(torch.as_tensor(w, dtype=audio.dtype, device=audio.device) for w in windows)
    cov = torch.as_tensor(coverage, dtype=audio.dtype, device=audio.device)
    spectrum = torch.fft.rfft(F.pad(audio, (config.padding, config.padding)), dim=-1)
    analytic = torch.full((spectrum.shape[-1],), 2., dtype=audio.dtype, device=audio.device)
    analytic[0] = 1
    if n % 2 == 0: analytic[-1] = 1
    coefficients = [None] * len(layout)
    for group in groups:
        packed = spectrum[..., group.bin_indices] * group.windows * analytic[group.bin_indices]
        values = torch.fft.ifft(packed,dim=-1) * (packed.shape[-1]/n)
        for index, value in zip(group.band_indices,values.unbind(-2)):
            coefficients[index] = value
    return HybridEncoding(audio, tuple(coefficients), config,
        torch.tensor([b.center_hz for b in layout], dtype=audio.dtype, device=audio.device),
        windows, cov, layout, tuple(groups))


def synthesize_compact(encoding: HybridEncoding, coefficients=None) -> torch.Tensor:
    c = encoding.coefficients if coefficients is None else coefficients
    if not isinstance(c, tuple) or len(c) != len(encoding.band_specs):
        raise ValueError("compact coefficients must be a tuple with one tensor per band")
    reference = encoding.coefficients[0]
    leading = c[0].shape[:-1] if isinstance(c[0], torch.Tensor) else None
    for value, band in zip(c, encoding.band_specs):
        if (not isinstance(value, torch.Tensor) or value.ndim < 1
            or value.shape != (*leading, band.samples) or value.dtype != reference.dtype
            or value.device != reference.device):
            raise ValueError("compact band lengths, shared leading axes, complex dtype and device must match")
    n = encoding.waveform.shape[-1] + 2 * encoding.config.padding
    summed = torch.zeros(*leading, n//2+1, dtype=reference.dtype, device=reference.device)
    for group in encoding.compact_groups:
        values = torch.stack([c[i] for i in group.band_indices],dim=-2)
        spectrum = torch.fft.fft(values,dim=-1)*(n/values.shape[-1])*group.windows
        summed.index_add_(-1,group.bin_indices.flatten(),spectrum.flatten(-2))
    weights = torch.full((n//2+1,), .5, dtype=reference.real.dtype, device=reference.device)
    weights[0] = 1
    if n % 2 == 0: weights[-1] = 1
    audio = torch.fft.irfft(summed * weights / encoding.coverage, n=n, dim=-1)
    pad = encoding.config.padding
    return audio[..., pad:pad+encoding.waveform.shape[-1]]


def _integrated_average(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Average a piecewise-linear curve over consecutive fractional intervals.

    values includes the final endpoint; accumulate in double precision to
    avoid float32 prefix-sum cancellation for short patches late in a clip.
    This operates on compact grids and never expands to full-rate bands.
    """
    dtype = values.dtype
    values = values.to(torch.complex128 if values.is_complex() else torch.float64)
    segment_area = (values[..., :-1] + values[..., 1:]) * .5
    prefix = F.pad(segment_area.cumsum(-1), (1, 0))
    index = boundaries.floor().long().clamp(0, values.shape[-1] - 2)
    fraction = boundaries - index
    integral = (prefix[..., index] + values[..., index] * fraction
                + .5 * (values[..., index+1] - values[..., index]) * fraction.square())
    return ((integral[..., 1:] - integral[..., :-1]) / boundaries.diff()).to(dtype)


def compact_context(encoding: HybridEncoding):
    """Lossy signed baseband averages and energy on aligned raw-patch intervals.

    Fractional integration of interpolated band samples avoids assigning
    physical time by coefficient index or expanding all bands to full rate.
    Energy is interpolated from the sampled squared magnitudes, an estimate
    rather than an exact integral of continuous-band power.
    """
    cfg = encoding.config
    n = encoding.waveform.shape[-1]
    total = n + 2 * cfg.padding
    reference = encoding.coefficients[0]
    count = (n + cfg.hop_length - 1) // cfg.hop_length
    starts = torch.arange(count, device=reference.device) * cfg.hop_length
    lengths = (n-starts).clamp(max=cfg.hop_length)
    edges = torch.cat((starts, starts.new_tensor([n]))).double()
    times = (starts.double()+lengths.double()/2+encoding.sample_origin) / cfg.sample_rate
    features = [None] * len(encoding.band_specs)
    for group in encoding.compact_groups:
        value = torch.stack([encoding.coefficients[i] for i in group.band_indices],dim=-2)
        specs = [encoding.band_specs[i] for i in group.band_indices]
        samples = value.shape[-1]
        # Original-time phase convention, including the zero-context offset.
        grid = torch.arange(samples+1, device=value.device, dtype=torch.float64)
        t = grid * total / (samples * cfg.sample_rate)
        centers = torch.tensor([b.center_hz for b in specs],device=value.device,dtype=torch.float64)[:,None]
        carriers = torch.tensor([b.carrier_bin for b in specs],device=value.device,dtype=torch.float64)[:,None]
        delta = centers-carriers*cfg.sample_rate/total
        origin_turns = torch.remainder(centers*(encoding.sample_origin/cfg.sample_rate),1.)
        turns = delta*t-centers*cfg.padding/cfg.sample_rate+origin_turns
        phase = -2*math.pi*torch.remainder(turns,1.)
        oscillator = torch.complex(phase.cos().to(value.real.dtype), phase.sin().to(value.real.dtype))
        extended = torch.cat((value, value[..., :1]), -1)
        boundaries = (edges+cfg.padding) * (samples/total)
        mean = _integrated_average(extended*oscillator, boundaries)
        power = _integrated_average(extended.real.square()+extended.imag.square(), boundaries).clamp_min(0)
        eps = torch.finfo(value.real.dtype).eps
        rms = torch.sqrt(power+eps**2)
        feature = torch.stack((mean.real/(1+rms), mean.imag/(1+rms), torch.log1p(rms-eps)), -1)
        for index, band_feature in zip(group.band_indices,feature.unbind(-3)):
            features[index] = band_feature
    features = torch.stack(features, 2)  # B,C,K,T,3
    if cfg.context_mode == "magnitude":
        features = features * features.new_tensor([0.,0.,1.])
    return features.permute(0,3,1,2,4).flatten(2), times, lengths
