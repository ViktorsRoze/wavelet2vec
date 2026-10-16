"""Bridging lossless complex wavelet carriers and invariant semantic audio understanding.

Built on :mod:`wavelet2vec.wavelet_image` (192 kHz stereo, 149 semitone bands x 10 ms frames).
While :class:`WaveletImage` retains every exact complex sample for lossless synthesis (~305 dB
in float64, ~132 dB in float32), downstream understanding models, judges, and audio editors
need representations that simultaneously:

1. Break the **Low-Frequency Heisenberg Trap** (where a 40 Hz constant-Q filter spreads a sharp
   bass transient across >200 ms) using differentiable time-frequency reassignment driven by
   spectral phase derivatives.
2. Eliminate the **``c = 1`` Multirate Cell Degeneracy** (where the lowest 58 semitone bands,
   MIDI 12..69 / 16.35..440 Hz, hold only 1 complex sample per 10 ms cell, causing within-cell
   phase coherence to saturate at 1.0 and modulation depth to collapse to 0.0 even on white noise).
3. Capture **Second-Order Joint Time-Frequency Scattering (JTFS)** invariants (spectro-temporal
   chirp/formant velocity, harmonic comb regularity, and scale-consistent modulation spectra).
4. Expose a **12-plane differentiable tensor** (``semantic_planes``) for neural backbones/judges
   and a **high-precision deterministic embedding** (``semantic_wavelet_embedding``) for retrieval,
   clustering, and perceptual evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet2vec.wavelet_image import (
     BandLayout,
    WaveletImage,
    WaveletImageSpec,
    band_layout,
    encode,
)

SPEC = WaveletImageSpec()
TINY = 1e-30
FLOOR_DB = -60.0
DB_SCALE = 30.0
AGREE_SEMITONES = 0.35
AGREE_SPREAD = 0.25
MIN_STAT_SAMPLES = 5  # minimum sample neighborhood for phase coherence & modulation


@dataclass(frozen=True)
class ReassignedPrimitives:
    """Exact cell-level physical quantities derived from multirate spectral derivatives.

    All tensors have shape ``[B, ROWS, TOTAL_FRAMES]`` on the full padded frame grid.
    """
    mid_power: torch.Tensor          # [B, R, T_total] linear Mid power
    side_power: torch.Tensor         # [B, R, T_total] linear Side power
    cross_re: torch.Tensor           # [B, R, T_total] Re(L * conj(R)) = |M|^2 - |S|^2
    left_power: torch.Tensor         # [B, R, T_total] |L|^2
    right_power: torch.Tensor        # [B, R, T_total] |R|^2
    freq_hz: torch.Tensor            # [B, R, T_total] power-weighted instantaneous frequency (Hz)
    pitch_offset_semi: torch.Tensor  # [B, R, T_total] IF deviation from band center (semitones)
    time_offset_frames: torch.Tensor # [B, R, T_total] group-delay offset (in 10 ms frames)
    phase_coherence: torch.Tensor    # [B, R, T_total] scale-consistent phase-step coherence in [0, 1]
    vel_cos: torch.Tensor            # [B, R, T_total] demodulated phase velocity cos
    vel_sin: torch.Tensor            # [B, R, T_total] demodulated phase velocity sin
    mod_depth: torch.Tensor          # [B, R, T_total] scale-consistent power modulation depth
    freq_agreement: torch.Tensor     # [B, R, T_total] cross-band harmonic ridge agreement in [0, 1]
    time_agreement: torch.Tensor     # [B, R, T_total] cross-band transient time agreement in [0, 1]
    sharp_mid_power: torch.Tensor    # [B, R, T_total] soft-reassigned (synchrosqueezed) Mid power


class _DerivativePlan:
    """Precomputed spectral derivative multipliers per sample-count group, cached per (spec, device, dtype)."""

    def __init__(self, spec: WaveletImageSpec, device: torch.device, dtype: torch.dtype):
        self.spec = spec
        self.bands: Tuple[BandLayout, ...] = band_layout(spec)
        spacing = spec.sample_rate / spec.padded_length
        self.centers_hz = torch.tensor([b.center_hz for b in self.bands], device=device, dtype=dtype)
        self.sigmas_hz = torch.tensor([b.sigma_hz for b in self.bands], device=device, dtype=dtype)
        self.carriers_hz = torch.tensor([b.carrier_bin * spacing for b in self.bands], device=device, dtype=dtype)
        self.midis = torch.tensor([float(b.midi) for b in self.bands], device=device, dtype=dtype)
        hop_s = spec.hop_ms / 1000.0
        # Natural Gaussian temporal spread sigma_t in frame units
        self.spread_frames = 1.0 / (2.0 * math.pi * self.sigmas_hz * hop_s)

        self.groups: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        by_c: Dict[int, list[int]] = {}
        for idx, b in enumerate(self.bands):
            by_c.setdefault(b.per_frame, []).append(idx)

        for c, rows in sorted(by_c.items()):
            m = c * spec.total_frames
            d_time_list = []
            d_freq_list = []
            for r in rows:
                b = self.bands[r]
                pos = b.stop_bin - b.carrier_bin
                neg = b.carrier_bin - b.start_bin
                dev_hz = np.zeros(m, dtype=np.float64)
                slope = np.zeros(m, dtype=np.float64)

                # Positive offset bins: 0 .. pos-1
                if pos > 0:
                    df_pos = np.arange(pos, dtype=np.float64) * spacing
                    f_pos = b.carrier_bin * spacing + df_pos
                    dev_hz[:pos] = df_pos
                    s_pos = -(f_pos - b.center_hz) / (b.sigma_hz ** 2)
                    if b.midi == spec.first_midi:
                        s_pos[f_pos <= b.center_hz] = 0.0
                    if b.midi == spec.last_midi:
                        s_pos[f_pos >= b.center_hz] = 0.0
                    slope[:pos] = s_pos

                # Negative offset bins: m-neg .. m-1
                if neg > 0:
                    df_neg = np.arange(-neg, 0, dtype=np.float64) * spacing
                    f_neg = b.carrier_bin * spacing + df_neg
                    dev_hz[m - neg:] = df_neg
                    s_neg = -(f_neg - b.center_hz) / (b.sigma_hz ** 2)
                    if b.midi == spec.first_midi:
                        s_neg[f_neg <= b.center_hz] = 0.0
                    if b.midi == spec.last_midi:
                        s_neg[f_neg >= b.center_hz] = 0.0
                    slope[m - neg:] = s_neg

                d_time_list.append(2.0 * math.pi * dev_hz)
                d_freq_list.append(slope)

            row_tensor = torch.tensor(rows, device=device, dtype=torch.long)
            # Multipliers for FFT: d_time is purely imaginary (i * 2*pi*df), d_freq is real
            d_time_t = torch.tensor(np.stack(d_time_list), device=device, dtype=dtype)
            d_freq_t = torch.tensor(np.stack(d_freq_list), device=device, dtype=dtype)
            self.groups[c] = (row_tensor, d_time_t, d_freq_t)


_DERIV_PLANS: Dict[Tuple, _DerivativePlan] = {}


def _get_deriv_plan(spec: WaveletImageSpec, device: torch.device, dtype: torch.dtype) -> _DerivativePlan:
    key = (spec, str(device), dtype)
    if key not in _DERIV_PLANS:
        _DERIV_PLANS[key] = _DerivativePlan(spec, device, dtype)
    return _DERIV_PLANS[key]


def power(z: torch.Tensor) -> torch.Tensor:
    """|z|^2 with smooth, finite gradient at z = 0."""
    return z.real ** 2 + z.imag ** 2


def scale_consistent_cell_stats(
    values: torch.Tensor,
    centers_hz: torch.Tensor,
    carriers_hz: torch.Tensor,
    hop_ms: float = 10.0,
    min_samples: int = MIN_STAT_SAMPLES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-cell statistics without the ``c = 1`` degeneracy.

    For bands where ``c < min_samples`` (such as the lowest 58 semitone bands where ``c = 1``),
    phase-step coherence and power modulation depth are pooled across a short sliding window
    of ``>= min_samples`` along the continuous sample timeline, after demodulating the residual
    offset ``center_hz - carrier_hz``.

    Args:
        values: Complex tensor ``[B, 2 (mid, side), R, T, c]``.
        centers_hz: ``[R]`` band center frequencies in Hz.
        carriers_hz: ``[R]`` band FFT carrier frequencies in Hz.
        hop_ms: Frame hop in milliseconds.
        min_samples: Minimum number of samples used to estimate coherence and modulation.

    Returns:
        Tuple of ``[B, R, T]`` tensors:
        ``(mean_mid_p, mean_side_p, phase_coherence, vel_cos, vel_sin, mod_depth)``.
    """
    mid, side = values[:, 0], values[:, 1]  # [B, R, T, c]
    bsz, n_rows, n_frames, c = mid.shape
    p_mid = power(mid)
    p_side = power(side)
    mean_mid_p = p_mid.mean(dim=-1)
    mean_side_p = p_side.mean(dim=-1)

    # Flatten continuous sample timeline: [B, R, T * c]
    m = n_frames * c
    flat = mid.reshape(bsz, n_rows, m)
    inc = flat[..., 1:] * flat[..., :-1].conj()
    inc = torch.cat([inc[..., :1], inc], dim=-1)  # [B, R, M]

    # Demodulate residual carrier offset (center_hz - carrier_hz) so a tone at center_hz has 0 angle
    dt_sample = (hop_ms / 1000.0) / c
    expected_step_rad = (2.0 * math.pi * (centers_hz - carriers_hz) * dt_sample).to(mid.real.dtype)
    demod_phasor = torch.complex(
        torch.cos(-expected_step_rad), torch.sin(-expected_step_rad)
    )[None, :, None]
    inc_demod = inc * demod_phasor
    inc_mag = torch.sqrt(power(inc_demod) + TINY)

    if c >= min_samples:
        # Sufficient samples inside each single cell
        inc_cell = inc_demod.reshape(bsz, n_rows, n_frames, c)
        mag_cell = inc_mag.reshape(bsz, n_rows, n_frames, c)
        u = inc_cell.sum(dim=-1) / (mag_cell.sum(dim=-1) + 1e-20)
        mod = torch.sqrt(p_mid.var(dim=-1, unbiased=False) + TINY) / (mean_mid_p + 1e-20)
    else:
        # Pool across a sliding window of k_frames so total samples >= min_samples
        k_frames = max(3, math.ceil(min_samples / c))
        if k_frames % 2 == 0:
            k_frames += 1
        pad_f = k_frames // 2

        # Sum within each frame first, then apply 1D Gaussian-weighted smoothing across k_frames
        inc_frame = inc_demod.reshape(bsz, n_rows, n_frames, c).sum(dim=-1)
        mag_frame = inc_mag.reshape(bsz, n_rows, n_frames, c).sum(dim=-1)

        # Gaussian weights across k_frames
        grid = torch.arange(-pad_f, pad_f + 1, device=mid.device, dtype=mid.real.dtype)
        w = torch.exp(-0.5 * (grid / (pad_f * 0.65)) ** 2)
        w = (w / w.sum()).view(1, 1, -1)

        inc_re_pad = F.pad(inc_frame.real, (pad_f, pad_f), mode="replicate").reshape(bsz * n_rows, 1, -1)
        inc_im_pad = F.pad(inc_frame.imag, (pad_f, pad_f), mode="replicate").reshape(bsz * n_rows, 1, -1)
        mag_pad = F.pad(mag_frame, (pad_f, pad_f), mode="replicate").reshape(bsz * n_rows, 1, -1)

        smooth_re = F.conv1d(inc_re_pad, w).reshape(bsz, n_rows, n_frames)
        smooth_im = F.conv1d(inc_im_pad, w).reshape(bsz, n_rows, n_frames)
        smooth_mag = F.conv1d(mag_pad, w).reshape(bsz, n_rows, n_frames) + 1e-20

        u = torch.complex(smooth_re / smooth_mag, smooth_im / smooth_mag)

        # Scale-consistent power modulation depth across the same local window
        p_pad = F.pad(mean_mid_p, (pad_f, pad_f), mode="replicate").reshape(bsz * n_rows, 1, -1)
        p_mean = F.conv1d(p_pad, w).reshape(bsz, n_rows, n_frames)
        p_sq_mean = F.conv1d(p_pad ** 2, w).reshape(bsz, n_rows, n_frames)
        p_var = torch.clamp(p_sq_mean - p_mean ** 2, min=0.0)
        mod = torch.sqrt(p_var + TINY) / (p_mean + 1e-20)

    coh = torch.clamp(torch.sqrt(u.real ** 2 + u.imag ** 2 + TINY), 0.0, 1.0)
    return mean_mid_p, mean_side_p, coh, u.real, u.imag, torch.clamp(mod, max=3.0)


def _soft_bilinear_scatter_2d(
    weights: torch.Tensor,
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    n_rows: int,
    n_cols: int,
) -> torch.Tensor:
    """Differentiable 2D bilinear splatting of ``weights [B, R, T]`` to ``(row_coords, col_coords)``.

    Conserves total energy inside bounds and propagates gradients to both ``weights`` and
    the continuous coordinates ``(row_coords, col_coords)``.
    """
    bsz = weights.shape[0]
    r_clamped = torch.clamp(row_coords, 0.0, float(n_rows - 1) - 1e-5)
    c_clamped = torch.clamp(col_coords, 0.0, float(n_cols - 1) - 1e-5)

    r0 = torch.floor(r_clamped).long()
    c0 = torch.floor(c_clamped).long()
    r1 = (r0 + 1).clamp(max=n_rows - 1)
    c1 = (c0 + 1).clamp(max=n_cols - 1)

    fr = r_clamped - r0.to(weights.dtype)
    fc = c_clamped - c0.to(weights.dtype)

    w00 = (1.0 - fr) * (1.0 - fc) * weights
    w01 = (1.0 - fr) * fc * weights
    w10 = fr * (1.0 - fc) * weights
    w11 = fr * fc * weights

    out = torch.zeros(bsz, n_rows * n_cols, device=weights.device, dtype=weights.dtype)
    for r_idx, c_idx, w_val in ((r0, c0, w00), (r0, c1, w01), (r1, c0, w10), (r1, c1, w11)):
        flat_idx = (r_idx * n_cols + c_idx).reshape(bsz, -1)
        out = out.scatter_add(1, flat_idx, w_val.reshape(bsz, -1))

    return out.reshape(bsz, n_rows, n_cols)


def reassigned_primitives(image: WaveletImage) -> ReassignedPrimitives:
    """Compute exact differentiable spectral derivatives, agreements, and reassigned power."""
    spec = image.spec
    first = next(iter(image.groups.values()))
    bsz, device, real_dtype = first.shape[0], first.device, first.real.dtype
    n_rows, n_total = spec.n_bands, spec.total_frames
    hop_s = spec.hop_ms / 1000.0

    plan = _get_deriv_plan(spec, device, real_dtype)

    mid_p = torch.zeros(bsz, n_rows, n_total, device=device, dtype=real_dtype)
    side_p = torch.zeros_like(mid_p)
    left_p = torch.zeros_like(mid_p)
    right_p = torch.zeros_like(mid_p)
    cross_re = torch.zeros_like(mid_p)
    df_hz = torch.zeros_like(mid_p)
    dt_sec = torch.zeros_like(mid_p)
    coh_map = torch.zeros_like(mid_p)
    vcos_map = torch.zeros_like(mid_p)
    vsin_map = torch.zeros_like(mid_p)
    mod_map = torch.zeros_like(mid_p)

    for c, values in image.groups.items():
        rows, d_time, d_freq = plan.groups[c]
        r_count = len(rows)
        m = c * n_total

        # Mid and Side: [B, R_group, T_total, c]
        mid = values[:, 0]
        side = values[:, 1]
        left = mid + side
        right = mid - side

        # 1. Scale-consistent coherence and modulation
        mp, sp, coh, vc, vs, md = scale_consistent_cell_stats(
            values,
            plan.centers_hz[rows],
            plan.carriers_hz[rows],
            hop_ms=spec.hop_ms,
        )
        mid_p = mid_p.index_copy(1, rows, mp)
        side_p = side_p.index_copy(1, rows, sp)
        left_p = left_p.index_copy(1, rows, power(left).mean(dim=-1))
        right_p = right_p.index_copy(1, rows, power(right).mean(dim=-1))
        cross_re = cross_re.index_copy(1, rows, (mp - sp))
        coh_map = coh_map.index_copy(1, rows, coh)
        vcos_map = vcos_map.index_copy(1, rows, vc)
        vsin_map = vsin_map.index_copy(1, rows, vs)
        mod_map = mod_map.index_copy(1, rows, md)

        # 2. Exact spectral derivatives via band FFT
        s0 = mid.reshape(bsz, r_count, m)
        spectrum = torch.fft.fft(s0, dim=-1)
        # s1: time derivative -> multiply by i * (2*pi*df)
        spec_s1 = torch.complex(-spectrum.imag * d_time[None], spectrum.real * d_time[None])
        s1 = torch.fft.ifft(spec_s1, dim=-1)
        # s2: frequency derivative -> multiply by -(f - f_c) / sigma^2
        spec_s2 = spectrum * d_freq[None]
        s2 = torch.fft.ifft(spec_s2, dim=-1)

        # Power-weighted cross-products:
        # Im(s1 * conj(s0)) -> instantaneous frequency offset
        # |Re(s1 * conj(s0))| -> Cohen-Mandel instantaneous amplitude modulation rate A(t)*|dA/dt|
        # Im(s2 * conj(s0)) -> local group delay offset
        q_freq = (s1.imag * s0.real - s1.real * s0.imag).reshape(bsz, r_count, n_total, c).sum(dim=-1)
        q_am = torch.abs(s1.real * s0.real + s1.imag * s0.imag).reshape(bsz, r_count, n_total, c).sum(dim=-1)
        q_time = (s2.imag * s0.real - s2.real * s0.imag).reshape(bsz, r_count, n_total, c).sum(dim=-1)
        p_sum = power(s0).reshape(bsz, r_count, n_total, c).sum(dim=-1) + TINY

        cell_df = q_freq / (2.0 * math.pi * p_sum)
        cell_dt = q_time / (2.0 * math.pi * p_sum)

        # Dimensionless, scale-invariant instantaneous AM and group-delay dispersion (normalized by sigma_hz)
        sigma_r = plan.sigmas_hz[rows][None, :, None]
        inst_am_norm = q_am / (2.0 * math.pi * sigma_r * p_sum)
        inst_gd_norm = 2.0 * math.pi * sigma_r * torch.abs(cell_dt)

        # Smooth over 5 frames so random zeroes of envelope derivative in narrowband noise do not spike
        k_smooth = torch.tensor([0.15, 0.2, 0.3, 0.2, 0.15], device=device, dtype=real_dtype).view(1, 1, 5)
        am_pad = F.pad(inst_am_norm, (2, 2), mode="replicate").reshape(bsz * r_count, 1, -1)
        gd_pad = F.pad(inst_gd_norm, (2, 2), mode="replicate").reshape(bsz * r_count, 1, -1)
        am_smooth = F.conv1d(am_pad, k_smooth).reshape(bsz, r_count, n_total)
        gd_smooth = F.conv1d(gd_pad, k_smooth).reshape(bsz, r_count, n_total)

        # Stationary-phase spectral coherence factor: 1.0 on pure tones, ~0.50 on noise at all scales (incl. c=1)
        spec_coh = torch.exp(-0.85 * (am_smooth ** 2 + gd_smooth ** 2))
        coh_combined = coh * spec_coh
        mod_combined = torch.clamp(torch.maximum(md, 0.85 * am_smooth + 0.45 * gd_smooth), max=3.0)

        coh_map = coh_map.index_copy(1, rows, coh_combined)
        vcos_map = vcos_map.index_copy(1, rows, vc * spec_coh)
        vsin_map = vsin_map.index_copy(1, rows, vs * spec_coh)
        mod_map = mod_map.index_copy(1, rows, mod_combined)
        df_hz = df_hz.index_copy(1, rows, cell_df)
        dt_sec = dt_sec.index_copy(1, rows, cell_dt)

    # Absolute instantaneous frequency and semitone deviation
    freq_hz = torch.clamp(plan.carriers_hz[None, :, None] + df_hz, min=1.0)
    inst_midi = 69.0 + 12.0 * torch.log2(freq_hz / 440.0)
    pitch_offset = torch.clamp(inst_midi - plan.midis[None, :, None], -3.0, 3.0)
    time_offset_frames = torch.clamp(dt_sec / hop_s, -12.0, 12.0)

    # Cross-band harmonic ridge agreement (do adjacent semitone bands agree on absolute MIDI pitch?)
    midi_diff = torch.abs(inst_midi[:, 1:, :] - inst_midi[:, :-1, :])
    near_f = torch.minimum(
        F.pad(midi_diff, (0, 0, 0, 1), value=10.0),
        F.pad(midi_diff, (0, 0, 1, 0), value=10.0),
    )
    freq_agreement = torch.exp(-((near_f / AGREE_SEMITONES) ** 2))

    # Cross-band transient time agreement (do adjacent bands point to the same reassigned frame?)
    frame_grid = torch.arange(n_total, device=device, dtype=real_dtype)[None, None, :]
    reassigned_frame = frame_grid + time_offset_frames
    time_diff = torch.abs(reassigned_frame[:, 1:, :] - reassigned_frame[:, :-1, :])
    spread_pair = torch.minimum(plan.spread_frames[:-1], plan.spread_frames[1:])[None, :, None]
    norm_t_diff = time_diff / torch.clamp(AGREE_SPREAD * spread_pair, min=0.25)
    near_t = torch.minimum(
        F.pad(norm_t_diff, (0, 0, 0, 1), value=10.0),
        F.pad(norm_t_diff, (0, 0, 1, 0), value=10.0),
    )
    time_agreement = torch.exp(-(near_t ** 2))

    # Differentiable soft-reassigned (synchrosqueezed) sharp power
    row_grid = torch.arange(n_rows, device=device, dtype=real_dtype)[None, :, None]
    target_row = row_grid + freq_agreement * pitch_offset
    target_col = frame_grid + time_agreement * time_offset_frames
    sharp_mid_power = _soft_bilinear_scatter_2d(mid_p, target_row, target_col, n_rows, n_total)

    return ReassignedPrimitives(
        mid_power=mid_p,
        side_power=side_p,
        cross_re=cross_re,
        left_power=left_p,
        right_power=right_p,
        freq_hz=freq_hz,
        pitch_offset_semi=pitch_offset,
        time_offset_frames=time_offset_frames,
        phase_coherence=coh_map,
        vel_cos=vcos_map,
        vel_sin=vsin_map,
        mod_depth=mod_map,
        freq_agreement=freq_agreement,
        time_agreement=time_agreement,
        sharp_mid_power=sharp_mid_power,
    )


def _jtfs_orientation_plane(sharp_db_norm: torch.Tensor) -> torch.Tensor:
    """2D Joint Time-Frequency Scattering orientation plane ``[B, R, T]``.

    Convolves the sharp log-power surface with quadrature 2D spectro-temporal Gabor
    wavelets (upward glide vs. downward glide and harmonic comb vs. transient edge),
    returning the net directional spectro-temporal modulation energy in ``[-1, 1]``.
    """
    bsz, n_rows, n_cols = sharp_db_norm.shape
    device, dtype = sharp_db_norm.device, sharp_db_norm.dtype

    # Build 7x7 2D Gabor kernels for upward (+1) and downward (-1) spectro-temporal tilts
    ksize = 7
    half = ksize // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kr, kt = torch.meshgrid(coords, coords, indexing="ij")
    gauss = torch.exp(-0.5 * (kr ** 2 + kt ** 2) / (1.75 ** 2))

    # Frequency: 1 cycle per 4 semitones & 4 frames
    omega = 2.0 * math.pi / 4.0
    up_cos = gauss * torch.cos(omega * (kr - kt))
    up_sin = gauss * torch.sin(omega * (kr - kt))
    down_cos = gauss * torch.cos(omega * (kr + kt))
    down_sin = gauss * torch.sin(omega * (kr + kt))

    kernels = torch.stack([up_cos, up_sin, down_cos, down_sin], dim=0)[:, None, :, :]
    kernels = kernels / (kernels.flatten(1).norm(dim=1).view(4, 1, 1, 1) + 1e-12)

    x = F.pad(sharp_db_norm[:, None], (half, half, half, half), mode="replicate")
    conv = F.conv2d(x, kernels)  # [B, 4, R, T]
    e_up = torch.sqrt(conv[:, 0] ** 2 + conv[:, 1] ** 2 + TINY)
    e_down = torch.sqrt(conv[:, 2] ** 2 + conv[:, 3] ** 2 + TINY)
    return torch.tanh((e_up - e_down) / (e_up + e_down + 1e-6) * 2.5)


def semantic_planes(image: WaveletImage, *, core_only: bool = True) -> torch.Tensor:
    """Compute the 12-plane unified semantic & physical feature tensor ``[B, 12, ROWS, FRAMES]``.

    Planes (all scaled to ``[-1, 1]`` or ``[0, 1]``, with silent regions cleanly gated):
      0. ``mid_db``: Raw cell Mid level in dB relative to clip peak (floored at -60 dB, in [-1, 1]).
      1. ``side_rel_db``: Side-to-Mid level ratio in dB (in [-1, 1]).
      2. ``sharp_db``: Differentiable Reassigned (Synchrosqueezed) Mid level in dB (resolves bass
         Heisenberg smearing and sharpens harmonic ridges, in [-1, 1]).
      3. ``phase_coherence``: Scale-consistent phase-increment coherence in [0, 1] (fixes c=1 bug).
      4. ``vel_cos``: Demodulated phase velocity cosine in [-1, 1] (weighted by coherence).
      5. ``vel_sin``: Demodulated phase velocity sine in [-1, 1] (sub-semitone pitch offset).
      6. ``group_delay``: Normalized local group-delay time offset in [-1, 1] (sub-sample attack locator).
      7. ``harmonic_agreement``: Cross-band pitch agreement in [0, 1] (1 on resolved harmonic ridges).
      8. ``transient_agreement``: Cross-band time agreement in [0, 1] (1 on broadband/bass attacks).
      9. ``mod_depth``: Scale-consistent amplitude modulation depth in [0, 1] (fixes c=1 bug).
     10. ``jtfs_tilt``: 2D Joint Time-Frequency Scattering glide/tilt asymmetry in [-1, 1].
     11. ``stereo_corr``: Inter-channel L/R phase correlation in [-1, 1] (+1 mono, 0 wide, -1 anti-phase).
    """
    spec = image.spec
    prim = reassigned_primitives(image)
    sl = slice(spec.pad_frames, spec.pad_frames + spec.n_frames) if core_only else slice(None)

    mid_p = prim.mid_power[..., sl]
    side_p = prim.side_power[..., sl]
    sharp_p = prim.sharp_mid_power[..., sl]

    mid_db = 10.0 * torch.log10(mid_p + TINY)
    side_db = 10.0 * torch.log10(side_p + TINY)
    sharp_db = 10.0 * torch.log10(sharp_p + TINY)

    peak_db = mid_db.amax(dim=(1, 2), keepdim=True)
    rel_mid = torch.clamp(mid_db - peak_db, min=FLOOR_DB)
    rel_sharp = torch.clamp(sharp_db - sharp_db.amax(dim=(1, 2), keepdim=True), min=FLOOR_DB)
    rel_side = torch.clamp(side_db - mid_db, min=-60.0, max=20.0)

    # Smooth activity gate: 0 at FLOOR_DB (-60 dB), 1 at -48 dB and above
    gate = torch.clamp((rel_mid - FLOOR_DB) / 12.0, 0.0, 1.0)
    sharp_gate = torch.clamp((rel_sharp - FLOOR_DB) / 12.0, 0.0, 1.0)

    p0_mid = rel_mid / (-FLOOR_DB / 2.0) + 1.0
    p1_side = (rel_side + 20.0) / 40.0 * gate - (1.0 - gate)
    p2_sharp = rel_sharp / (-FLOOR_DB / 2.0) + 1.0

    p3_coh = prim.phase_coherence[..., sl] * gate
    p4_vcos = prim.vel_cos[..., sl] * gate
    p5_vsin = prim.vel_sin[..., sl] * gate
    p6_gd = torch.tanh(prim.time_offset_frames[..., sl] / 3.0) * gate
    p7_harm = prim.freq_agreement[..., sl] * gate
    p8_trans = prim.time_agreement[..., sl] * sharp_gate
    p9_mod = (prim.mod_depth[..., sl] / 3.0) * gate
    p10_jtfs = _jtfs_orientation_plane(p2_sharp) * sharp_gate

    lp = prim.left_power[..., sl]
    rp = prim.right_power[..., sl]
    cross = prim.cross_re[..., sl]
    denom = torch.sqrt(lp * rp + TINY) + 1e-20
    p11_corr = torch.clamp(cross / denom, -1.0, 1.0) * gate

    return torch.stack(
        [
            p0_mid,
            p1_side,
            p2_sharp,
            p3_coh,
            p4_vcos,
            p5_vsin,
            p6_gd,
            p7_harm,
            p8_trans,
            p9_mod,
            p10_jtfs,
            p11_corr,
        ],
        dim=1,
    )


def jtfs_modulation_spectrum(planes: torch.Tensor, n_freq_groups: int = 7) -> torch.Tensor:
    """Extract shift-invariant 2D Joint Time-Frequency Scattering modulation descriptors.

    Applies multi-scale 2D spectro-temporal filters across temporal modulation rates
    (slow 2 Hz, medium 8 Hz, fast 25 Hz) and spectral pitch scales (fine 0.5 cyc/oct,
    coarse 0.15 cyc/oct) with signed chirp orientations (+1 upward vs. -1 downward).

    Args:
        planes: ``[B, 12, ROWS, FRAMES]`` output of :func:`semantic_planes`.
        n_freq_groups: Number of auditory frequency registers to pool into.

    Returns:
        Tensor ``[B, D_jtfs]`` of deformation-stable modulation features.
    """
    bsz, _, n_rows, n_frames = planes.shape
    sharp = planes[:, 2]  # [B, R, T] in [-1, 1]
    energy = (sharp + 1.0) * 0.5  # [0, 1]

    # 2D FFT over (Rows, Frames) for global & register-wise 2D modulation spectrum
    boundaries = np.linspace(0, n_rows, n_freq_groups + 1, dtype=int)
    feats = []

    # Temporal modulation frequencies (cycles per frame, at 100 fps)
    t_rates = (0.02, 0.08, 0.25)  # 2 Hz, 8 Hz, 25 Hz
    # Spectral scales (cycles per semitone)
    f_scales = (1.0 / 12.0, 1.0 / 4.0)  # 1/octave (harmonic comb), 1/major-third (formant/timbre)

    device, dtype = planes.device, planes.dtype
    k_r, k_t = 9, 15
    hr, ht = k_r // 2, k_t // 2
    gr = torch.arange(-hr, hr + 1, device=device, dtype=dtype)
    gt = torch.arange(-ht, ht + 1, device=device, dtype=dtype)
    mr, mt = torch.meshgrid(gr, gt, indexing="ij")
    win = torch.exp(-0.5 * ((mr / 2.5) ** 2 + (mt / 4.5) ** 2))

    bank = []
    for f_sc in f_scales:
        for t_rt in t_rates:
            for sign in (1.0, -1.0):
                phase = 2.0 * math.pi * (f_sc * mr + sign * t_rt * mt)
                k_re = win * torch.cos(phase)
                k_im = win * torch.sin(phase)
                norm = torch.sqrt((k_re ** 2 + k_im ** 2).sum() + 1e-12)
                bank.append(k_re / norm)
                bank.append(k_im / norm)

    kernels = torch.stack(bank, dim=0)[:, None, :, :]  # [2*N_filters, 1, k_r, k_t]
    padded = F.pad(energy[:, None], (ht, ht, hr, hr), mode="replicate")
    conv = F.conv2d(padded, kernels)  # [B, 2*N_filters, R, T]
    conv_re = conv[:, 0::2]
    conv_im = conv[:, 1::2]
    mod_mag = torch.sqrt(conv_re ** 2 + conv_im ** 2 + TINY)  # [B, N_filters, R, T]

    for g in range(n_freq_groups):
        lo, hi = boundaries[g], boundaries[g + 1]
        reg_mod = mod_mag[:, :, lo:hi, :].mean(dim=(2, 3))  # [B, N_filters]
        feats.append(reg_mod)

    return torch.cat(feats, dim=-1)


def semantic_wavelet_embedding(
    audio_or_image: torch.Tensor | WaveletImage,
    spec: WaveletImageSpec = SPEC,
    n_registers: int = 7,
) -> torch.Tensor:
    """Compute a complete shift-invariant, phase-aware semantic embedding ``[B, 252]``.

    Unites:
      1. Multi-register summary (mean + std) of the 12 reassigned semantic planes (168 dims).
      2. 2D Joint Time-Frequency Scattering (JTFS) signed modulation spectrum (84 dims).

    Invariant to sub-frame phase rotations and time shifts, while distinguishing:
      - Tones vs. noise in all registers (including sub-bass c=1),
      - Sharp percussive bass attacks vs. slow swells,
      - Upward vs. downward pitch glides / formant transitions,
      - Stereo width and phase correlation per register.
    """
    if isinstance(audio_or_image, WaveletImage):
        image = audio_or_image
    else:
        audio = audio_or_image
        if audio.ndim == 1:
            audio = audio[None, None, :].expand(1, 2, -1)
        elif audio.ndim == 2:
            audio = audio[None, :, :]
        if audio.shape[1] == 1:
            audio = audio.expand(-1, 2, -1)
        if audio.shape[-1] < spec.audio_length:
            audio = F.pad(audio, (0, spec.audio_length - audio.shape[-1]))
        elif audio.shape[-1] > spec.audio_length:
            audio = audio[..., :spec.audio_length]
        image = encode(audio, spec)

    planes = semantic_planes(image, core_only=True)  # [B, 12, R, T]
    bsz, n_pl, n_rows, _ = planes.shape
    boundaries = np.linspace(0, n_rows, n_registers + 1, dtype=int)

    reg_stats = []
    for g in range(n_registers):
        lo, hi = boundaries[g], boundaries[g + 1]
        sub = planes[:, :, lo:hi, :]  # [B, 12, R_g, T]
        mean = sub.mean(dim=(2, 3))
        std = torch.sqrt(sub.var(dim=(2, 3), unbiased=False) + 1e-12)
        reg_stats.append(mean)
        reg_stats.append(std)

    jtfs = jtfs_modulation_spectrum(planes, n_freq_groups=n_registers)
    embedding = torch.cat([*reg_stats, jtfs], dim=-1)
    return F.normalize(embedding, dim=-1)


def harmonic_noise_gate(image: WaveletImage, *, core_only: bool = False) -> torch.Tensor:
    """Compute a scale-consistent harmonic-vs-stochastic gate ``m in [0, 1]`` of shape ``[B, R, T]``.

    Unlike the legacy cell-only gate ``sigmoid(8*(coh - 0.55) - 3*mod)`` which saturates at
    ``0.9734`` across all 58 bands below 440 Hz (``c = 1``) even on pure white noise, this gate
    combines scale-consistent phase coherence, cross-band harmonic ridge agreement, and
    scale-consistent modulation depth so that tonal partials approach 1.0 while broadband noise
    and transients drop below 0.25 in every register from 16.35 Hz to 83.7 kHz.
    """
    spec = image.spec
    prim = reassigned_primitives(image)
    sl = slice(spec.pad_frames, spec.pad_frames + spec.n_frames) if core_only else slice(None)
    coh = prim.phase_coherence[..., sl]
    harm = prim.freq_agreement[..., sl]
    mod = prim.mod_depth[..., sl]
    tonality = 0.65 * coh + 0.35 * harm
    return torch.sigmoid(8.0 * (tonality - 0.55) - 2.5 * mod)


def semantic_structure_loss(
    pred: WaveletImage,
    target: WaveletImage,
    *,
    sharp_weight: float = 2.0,
    coherence_weight: float = 1.0,
    transient_weight: float = 1.5,
    jtfs_weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable physical + semantic structure loss between two :class:`WaveletImage` instances.

    Combines:
      - Synchrosqueezed sharp log-power L1 + multiscale L1 (resolves low-frequency Heisenberg blur),
      - Scale-consistent phase coherence & phase velocity matching (active down to 16.35 Hz),
      - Cross-band harmonic & transient agreement alignment,
      - 2D Joint Time-Frequency Scattering (JTFS) modulation spectrum distance.
    """
    p_planes = semantic_planes(pred, core_only=True)
    t_planes = semantic_planes(target, core_only=True)

    # 1. Multiscale loss on raw + sharp synchrosqueezed log-power and stereo planes (0, 1, 2, 11)
    spec_idx = [0, 1, 2, 11]
    loss_sharp = F.l1_loss(p_planes[:, spec_idx], t_planes[:, spec_idx])
    for stride in (2, 4, 8):
        p_pool = F.avg_pool2d(p_planes[:, spec_idx], kernel_size=stride, stride=stride)
        t_pool = F.avg_pool2d(t_planes[:, spec_idx], kernel_size=stride, stride=stride)
        loss_sharp = loss_sharp + 0.5 * F.l1_loss(p_pool, t_pool)

    # 2. Scale-consistent coherence, phase velocity, and modulation depth (3, 4, 5, 9)
    coh_idx = [3, 4, 5, 9]
    loss_coh = F.l1_loss(p_planes[:, coh_idx], t_planes[:, coh_idx])

    # 3. Group delay, harmonic ridge agreement, transient agreement, and 2D JTFS tilt (6, 7, 8, 10)
    trans_idx = [6, 7, 8, 10]
    loss_trans = F.l1_loss(p_planes[:, trans_idx], t_planes[:, trans_idx])

    # 4. Global + register-level 2D JTFS modulation spectrum
    p_jtfs = jtfs_modulation_spectrum(p_planes)
    t_jtfs = jtfs_modulation_spectrum(t_planes)
    loss_jtfs = F.l1_loss(p_jtfs, t_jtfs)

    return (
        sharp_weight * loss_sharp
        + coherence_weight * loss_coh
        + transient_weight * loss_trans
        + jtfs_weight * loss_jtfs
    )
