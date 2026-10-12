"""KeyboardNet v3: generate notes in the lossless stereo wavelet image, warm-started from v2.

The task is unchanged: from one, two or three notes of a sound, generate
another note. v3 works in :mod:`wavelet2vec.wavelet_image`: 192 kHz, 149
semitone rows (16 Hz - 96 kHz) x 10 ms frames, exact complex mid/side cells.

- Features: v2's seven planes (log energy, IF deviation, phase coherence,
  baseband re/im, modulation depth, stereo width) are computed from the exact
  cells, in v2's normalization. The v2 Swin-L backbone and head, warm-started
  from a v2 checkpoint, read them on a 160 x 128 grid. Rows 0-123 are v2's
  rows; rows 124-148 (20-96 kHz) are new.
- Carrier: the nearest example is moved to the target key inside the image.
  Each row shifts by the interval with the exact heterodyne phase ramp, and
  its cells are resampled band-limitedly to the destination row's cell size.
- Lossless path (new, zero-initialized): the carrier's exact cells, per
  octave group, are embedded and joined into the full-resolution decoder.
- Output: per cell, a mid gain (from v2's envelope prediction), a side gain
  (from v2's predicted stereo width), a phase rotation (new), and an additive
  residual per octave group (new), applied to the carrier's exact cells.
  Decoding the result with the exact inverse gives 192 kHz stereo audio.

At initialization the new parts are zero, so v3 starts from v2's envelope
behaviour, now applied exactly in the lossless domain.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wavelet2vec.keyboard import DB_SCALE, FLOOR_DB
from wavelet2vec.keyboard_v2 import (AUX_TARGETS, BASE_PLANE, IN_PLANES, MAX_EXAMPLES, N_FEATURES, KeyboardV2Config,
                                     build_model_v2, shift_features)
from wavelet2vec.wavelet_image import WaveletImage, WaveletImageSpec, _plan, band_layout

SPEC = WaveletImageSpec()
ROWS = SPEC.n_bands            # 149
FRAMES = SPEC.n_frames         # 120
GRID = (160, 128)
CORE = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)


def power(z: torch.Tensor) -> torch.Tensor:
    """|z|^2 with a defined gradient at z = 0 (complex abs() has none there)."""
    return z.real ** 2 + z.imag ** 2


def cell_levels_db(image: WaveletImage) -> torch.Tensor:
    """``[B, 2 (mid, side), ROWS, FRAMES]`` mean cell power in dB (core frames); differentiable."""
    first = next(iter(image.groups.values()))
    out = torch.full((first.shape[0], 2, image.spec.n_bands, image.spec.n_frames), FLOOR_DB, dtype=first.real.dtype)
    for c, values in image.groups.items():
        level = 10 * torch.log10(power(values[:, :, :, CORE]).mean(-1) + 1e-30)
        out = out.index_copy(2, torch.as_tensor(image.rows[c]), torch.clamp(level, min=FLOOR_DB))
    return out


def _row_tables(spec=SPEC):
    bands = band_layout(spec)
    spacing = spec.sample_rate / spec.padded_length
    centers = torch.tensor([b.center_hz for b in bands], dtype=torch.float64)
    carriers = torch.tensor([b.carrier_bin * spacing for b in bands], dtype=torch.float64)
    per_frame = [b.per_frame for b in bands]
    return bands, centers, carriers, per_frame


BANDS, CENTERS, CARRIERS, PER_FRAME = _row_tables()


def row_nyquist_mask(source_rates, spec=SPEC, margin=0.95):
    """[B, ROWS] 1 where the row's band lies below each source's Nyquist (scored), else 0."""
    rates = torch.as_tensor(source_rates, dtype=torch.float64)[:, None]
    upper = CENTERS[None] + 3 * torch.tensor([b.sigma_hz for b in BANDS])[None]
    return (upper <= margin * rates / 2).float()


def cell_features(image: WaveletImage) -> torch.Tensor:
    """v2's seven feature planes ``[B, 7, ROWS, FRAMES]`` from exact cells (core frames), v2 normalization."""
    spec = image.spec
    first = next(iter(image.groups.values()))
    b = first.shape[0]
    out = torch.zeros(b, N_FEATURES, spec.n_bands, spec.n_frames, dtype=first.real.dtype)
    rate_per_frame = 1000.0 / spec.hop_ms
    for c, values in image.groups.items():
        rows = torch.as_tensor(image.rows[c])
        mid, side = values[:, 0], values[:, 1]                       # [B, R, F_total, c]
        flat = mid.reshape(*mid.shape[:-2], -1)
        inc = flat[..., 1:] * flat[..., :-1].conj()
        inc = torch.cat([inc[..., :1], inc], -1).reshape(mid.shape)
        band_if = torch.angle(inc) * (c * rate_per_frame) / (2 * math.pi)
        absolute = band_if + CARRIERS[rows].to(band_if.dtype)[None, :, None, None]
        centre = CENTERS[rows].to(band_if.dtype)[None, :, None, None]
        deviation = torch.tanh((absolute - centre) / centre)
        mid, side, deviation = mid[:, :, CORE], side[:, :, CORE], deviation[:, :, CORE]
        power = mid.real ** 2 + mid.imag ** 2
        energy = power.mean(-1)
        out[:, 0, rows] = torch.clamp(10 * torch.log10(energy + 1e-30), min=FLOOR_DB) / DB_SCALE
        out[:, 1, rows] = (power * deviation).sum(-1) / (power.sum(-1) + 1e-30)
        unit = mid / (mid.abs() + 1e-12)
        out[:, 2, rows] = unit.mean(-1).abs()
        rms = torch.sqrt(energy) + 1e-12
        mean = mid.mean(-1)
        out[:, 3, rows] = mean.real / rms
        out[:, 4, rows] = mean.imag / rms
        magnitude = mid.abs()
        out[:, 5, rows] = torch.clamp(magnitude.std(-1, unbiased=False) / (magnitude.mean(-1) + 1e-12), max=3.0)
        side_energy = (side.real ** 2 + side.imag ** 2).mean(-1)
        out[:, 6, rows] = torch.clamp(10 * torch.log10((side_energy + 1e-30) / (energy + 1e-30)), -60.0, 20.0) / DB_SCALE
    quiet = out[:, 0] * DB_SCALE <= FLOOR_DB + 1e-3
    out[:, 1:6] = out[:, 1:6].masked_fill(quiet[:, None], 0.0)
    out[:, 6] = out[:, 6].masked_fill(quiet, -60.0 / DB_SCALE)
    return out


def _row_tensors(image: WaveletImage, item: int):
    """Per row: complex samples ``[2, total_frames * c]`` of one batch item."""
    rows = [None] * image.spec.n_bands
    for c, values in image.groups.items():
        for k, r in enumerate(image.rows[c]):
            rows[r] = values[item, :, k].reshape(2, -1)
    return rows


def shift_image(image: WaveletImage, semitones) -> WaveletImage:
    """Move every row by ``semitones[i]`` (per batch item) with the heterodyne phase ramp and
    band-limited resampling to the destination row's cell size. Rows with no source are zero."""
    spec = image.spec
    t_total = spec.padded_length / spec.sample_rate
    groups = {c: torch.zeros_like(v) for c, v in image.groups.items()}
    position = {r: (c, k) for c, rows in image.rows.items() for k, r in enumerate(rows)}
    for item, d in enumerate(semitones):
        d = int(d)
        rows = _row_tensors(image, item)
        for target in range(spec.n_bands):
            source = target - d
            if not 0 <= source < spec.n_bands:
                continue
            x = rows[source]
            m_src = x.shape[-1]
            c_dst, k_dst = position[target]
            m_dst = c_dst * spec.total_frames
            delta = float((CENTERS[target] - CENTERS[source]) - (CARRIERS[target] - CARRIERS[source]))
            t = torch.arange(m_src, dtype=torch.float64) * (t_total / m_src)
            ramp = torch.exp(2j * math.pi * delta * t).to(x.dtype)
            spectrum = torch.fft.fft(x * ramp, dim=-1)
            out = torch.zeros(2, m_dst, dtype=x.dtype)
            keep = min(m_src, m_dst) // 2
            out[:, :keep] = spectrum[:, :keep]
            out[:, -keep:] = spectrum[:, -keep:]
            y = torch.fft.ifft(out, dim=-1) * (m_dst / m_src)
            groups[c_dst][item, :, k_dst] = y.reshape(2, spec.total_frames, c_dst)
    return WaveletImage(spec, groups, dict(image.rows))


def inputs_v3(example_features, example_midis, target_midi):
    """v2-layout planes on the v3 grid: ``[IN_PLANES, 160, 128]``, conditioning ``[COND_DIM]``,
    baseline envelope (dB) and the nearest example's index. ``example_features``: ``[7, ROWS, FRAMES]`` each."""
    order = sorted(range(len(example_features)), key=lambda i: abs(int(target_midi) - int(example_midis[i])))
    feats = [example_features[i].numpy() if torch.is_tensor(example_features[i]) else example_features[i]
             for i in order]
    distances = [int(target_midi) - int(example_midis[i]) for i in order]
    shifted, valid = zip(*(shift_features(f, d) for f, d in zip(feats, distances)))
    envelopes = np.stack([s[0] * DB_SCALE for s in shifted])
    weights = np.array([1.0 / (1.0 + abs(d)) for d in distances], dtype=np.float32)[:, None, None] * np.stack(valid)
    total = weights.sum(0)
    base = np.where(total > 0, (weights * envelopes).sum(0) / np.maximum(total, 1e-9), envelopes[0]).astype(np.float32)
    planes = np.zeros((IN_PLANES, *GRID), dtype=np.float32)
    stride = N_FEATURES + 1
    for slot in range(MAX_EXAMPLES):
        if slot < len(feats):
            planes[slot * stride:slot * stride + N_FEATURES, :ROWS, :FRAMES] = shifted[slot]
            planes[slot * stride + N_FEATURES, :ROWS, :FRAMES] = valid[slot]
        else:
            planes[slot * stride, :ROWS, :FRAMES] = FLOOR_DB / DB_SCALE
    planes[BASE_PLANE, :ROWS, :FRAMES] = base / DB_SCALE
    planes[BASE_PLANE + 1, :ROWS, :FRAMES] = np.linspace(0, 1, ROWS, dtype=np.float32)[:, None]
    planes[BASE_PLANE + 2, :ROWS, :FRAMES] = np.linspace(0, 1, FRAMES, dtype=np.float32)[None, :]
    if len(feats) >= 2:
        planes[BASE_PLANE + 3, :ROWS, :FRAMES] = (shifted[0][0] - shifted[1][0]) * valid[0] * valid[1]
    padded = distances + [0] * (MAX_EXAMPLES - len(distances))
    cond = np.array([target_midi / 127.0, (len(feats) - 1) / (MAX_EXAMPLES - 1)]
                    + [d / 48.0 for d in padded] + [abs(d) / 48.0 for d in padded], dtype=np.float32)
    return planes, cond, base, order[0]


class KeyboardNetV3(nn.Module):
    def __init__(self, v2_state=None, v2_config: KeyboardV2Config | None = None, detail_dim: int = 32):
        super().__init__()
        base = dict((v2_config or KeyboardV2Config()).__dict__)
        base.update(grid=GRID, rows=ROWS, frames=FRAMES)
        self.net = build_model_v2(KeyboardV2Config(**base))
        if v2_state is not None:
            self.net.load_state_dict(v2_state)
        cfg = self.net.config
        self.detail_dim = detail_dim
        groups = sorted(set(PER_FRAME))
        self.detail_embed = nn.ModuleDict({str(c): nn.Linear(4 * c + 2, detail_dim) for c in groups})
        old = self.net.refine_full[0]
        r_half, r_full = cfg.refine_dims
        new = type(old)(r_half + 48 + detail_dim, r_full)
        with torch.no_grad():  # copy v2 weights, zero the new input channels: starts as v2
            for name, p in new.named_parameters():
                src = dict(old.named_parameters())[name]
                p.zero_()
                if p.shape == src.shape:
                    p.copy_(src)
                else:
                    p[:, :src.shape[1]] = src
        self.net.refine_full[0] = new
        self.phase = nn.Conv2d(r_full, 1, 1)
        nn.init.zeros_(self.phase.weight)
        nn.init.zeros_(self.phase.bias)
        self.residual = nn.ModuleDict({str(c): nn.Linear(r_full, 4 * c) for c in groups})
        for layer in self.residual.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def detail_map(self, carrier: WaveletImage):
        first = next(iter(carrier.groups.values()))
        b = first.shape[0]
        out = torch.zeros(b, self.detail_dim, *GRID, dtype=first.real.dtype)
        for c, values in carrier.groups.items():
            rows = torch.as_tensor(carrier.rows[c])
            cells = values[:, :, :, CORE]                                         # [B, 2, R, F, c]
            mid_energy = power(cells[:, 0]).mean(-1)                             # [B, R, F]
            side_energy = power(cells[:, 1]).mean(-1)
            scale = torch.sqrt(mid_energy + side_energy + 1e-20)[..., None]
            norm = cells / scale[:, None]
            parts = torch.cat([norm.real, norm.imag], -1)                         # [B, 2, R, F, 2c]
            vector = parts.permute(0, 2, 3, 1, 4).reshape(b, len(rows), FRAMES, 4 * c)
            levels = torch.stack([torch.clamp(10 * torch.log10(mid_energy + 1e-30), min=FLOOR_DB),
                                  torch.clamp(10 * torch.log10(side_energy + 1e-30), min=FLOOR_DB)], -1) / DB_SCALE
            embedded = self.detail_embed[str(c)](torch.cat([vector, levels], -1).float())  # [B, R, F, D]
            out[:, :, rows, :FRAMES] = embedded.permute(0, 3, 1, 2).to(out.dtype)
        return out

    def forward(self, planes, cond, carrier: WaveletImage):
        net = self.net
        cfg = net.config
        c = net.cond(cond)
        full = net.stem_full(planes)
        half = net.stem_half(full)
        x = net.patch(planes)
        b = x.shape[0]
        x = net.patch_norm(x.flatten(2).transpose(1, 2))
        stages = []
        for i, blocks in enumerate(net.stages):
            for block in blocks:
                x = block(x, c)
            stages.append(net.stage_norms[i](x).transpose(1, 2).reshape(b, -1, *net.resolutions[i]))
            if i < 3:
                x = net.merges[i](x)
        top = stages[3]
        pooled = [F.interpolate(p(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for p in net.ppm]
        levels = [None, None, None, net.ppm_fuse(torch.cat([top] + pooled, 1))]
        for i in (2, 1, 0):
            up = F.interpolate(levels[i + 1], size=stages[i].shape[-2:], mode="bilinear", align_corners=False)
            levels[i] = net.fpn[i](net.lateral[i](stages[i]) + up)
        size = levels[0].shape[-2:]
        h = net.fuse(torch.cat([levels[0]] + [F.interpolate(l, size=size, mode="bilinear", align_corners=False)
                                              for l in levels[1:]], 1))
        h = net.fuse_film(h, c)
        h = F.interpolate(h, size=half.shape[-2:], mode="bilinear", align_corners=False)
        h = net.refine_half[1](net.refine_half[0](torch.cat([h, half], 1), c), c)
        h = F.interpolate(h, size=full.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail_map(carrier)
        h = net.refine_full[1](net.refine_full[0](torch.cat([h, full, detail], 1), c), c)
        out = net.out(h)[:, :, :ROWS, :FRAMES]
        limit = cfg.max_correction_db / DB_SCALE
        envelope = planes[:, BASE_PLANE, :ROWS, :FRAMES] + limit * torch.tanh(out[:, 0] / limit)
        aux = out[:, 1:]                                    # IF deviation, coherence, width (dB/40)
        phase = math.pi * torch.tanh(self.phase(h)[:, 0, :ROWS, :FRAMES])
        return self.apply_to_carrier(carrier, envelope, aux, phase, h), envelope, aux

    def apply_to_carrier(self, carrier: WaveletImage, envelope, aux, phase, features):
        """Carrier cells x (mid gain, side gain) x phase rotation + residual, on core frames."""
        width_index = AUX_TARGETS.index(6)
        groups = {}
        for c, values in carrier.groups.items():
            rows = torch.as_tensor(carrier.rows[c])
            cells = values[:, :, :, CORE]
            mid_db = torch.clamp(10 * torch.log10(power(cells[:, 0]).mean(-1) + 1e-30), min=FLOOR_DB)
            side_db = torch.clamp(10 * torch.log10(power(cells[:, 1]).mean(-1) + 1e-30), min=FLOOR_DB)
            target_mid = envelope[:, rows] * DB_SCALE
            target_side = target_mid + aux[:, width_index, rows] * DB_SCALE
            g_mid = 10 ** (torch.clamp(target_mid - mid_db, -40.0, 40.0) / 20)
            g_side = 10 ** (torch.clamp(target_side - side_db, -40.0, 40.0) / 20)
            rotation = torch.polar(torch.ones_like(phase[:, rows]), phase[:, rows])
            gains = torch.stack([g_mid, g_side], 1) * rotation[:, None]           # [B, 2, R, F]
            shaped = cells * gains[..., None].to(cells.dtype)
            f = features[:, :, rows, :FRAMES].permute(0, 2, 3, 1)                 # [B, R, F, C]
            res = self.residual[str(c)](f).reshape(*f.shape[:3], 2, 2, c)        # (mid/side, re/im)
            level = (10 ** (target_mid / 20))[..., None, None]                    # [B, R, F, 1, 1]
            res = torch.complex(res[..., 0, :], res[..., 1, :]) * level           # [B, R, F, 2, c]
            shaped = shaped + res.permute(0, 3, 1, 2, 4).to(cells.dtype)
            full = values.clone()
            full[:, :, :, CORE] = shaped
            groups[c] = full
        return WaveletImage(carrier.spec, groups, dict(carrier.rows))


def count_parameters_v3(model) -> dict:
    groups = {"backbone (from v2)": 0, "head (from v2)": 0, "new lossless path": 0}
    for name, p in model.named_parameters():
        if name.startswith(("detail_embed", "phase", "residual")):
            groups["new lossless path"] += p.numel()
        elif name.startswith(("net.patch", "net.stages", "net.merges", "net.stage_norms")):
            groups["backbone (from v2)"] += p.numel()
        else:
            groups["head (from v2)"] += p.numel()
    groups["total"] = sum(groups.values())
    return groups
