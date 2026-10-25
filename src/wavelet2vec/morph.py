"""Vox-synth morphing in the lossless stereo wavelet image (MorphNet).

Second purpose of note_dataset (note_dataset/README.md, "Second purpose"): new
short notes heard as partly vocal and partly synth, rated about 50 % by
vox/synth judges, that are neither a plain mix nor a synth filtered by the
voice's envelope (a vocoder), nor the reverse.

- ``judge_view``: the judges' confound-controlled view of a note (README,
  "Shortcuts the classifier would learn"): mid channel only, rows below
  18 kHz, levels relative to the note's peak and floored at -60 dB, plus
  phase-increment coherence (re, im) and modulation depth per 10 ms x
  semitone cell. Differentiable (no abs/angle at zero), so it scores
  generated notes too. ``stats="scale"`` (opt-in, for new runs) fixes the
  one-sample cells below 440 Hz, see ``VIEW_STATS``.
- ``JudgeCNN`` / ``JudgeTCN``: two differently built judges with two sigmoid
  heads, "vox present" and "synth present". Trained on clean vox, synth and
  noise (noise: both absent), so "50/50" cannot be reached by garbage.
- ``MorphNet``: one encoder for both notes (feature planes plus an embedding
  of every exact sample of every cell), a fusion transformer where vocal and
  synth tokens attend to each other, conditioned on the vocal amount alpha,
  and a decoder back to full resolution. It writes the hybrid's exact complex
  cells as a learned, per-cell, time-varying combination of four materials
  plus new content:

  1. the synth's cells;
  2. the voice's cells;
  3. the synth's fine structure carrying the voice's sample-level magnitude
     (articulation, formant motion);
  4. the voice's fine structure (vibrato, glide, breath phase) carrying the
     synth's magnitude;
  5. a residual synthesized per octave group.

  Complex gains (level and phase) are interpolated inside each cell, so there
  are no 10 ms steps. Decoding is the exact inverse: 192 kHz stereo audio.
- Baselines the hybrid must not collapse into: the plain mix and channel
  vocoders in both directions (``mix_baseline``, ``vocoder_baseline``).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wavelet2vec.wavelet_image import WaveletImage, WaveletImageSpec, band_layout

SPEC = WaveletImageSpec()
ROWS, FRAMES, TOTAL = SPEC.n_bands, SPEC.n_frames, SPEC.total_frames        # 149, 120, 136
CORE = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)
BANDS = band_layout(SPEC)
GROUPS = sorted({b.per_frame for b in BANDS})                                  # 1, 2, 4, ... 256
JUDGE_TOP_HZ = 18000.0
JUDGE_ROWS = sum(1 for b in BANDS if b.center_hz + 3 * b.sigma_hz <= JUDGE_TOP_HZ)
JUDGE_FLOOR_DB = -60.0
JUDGE_PLANES = 4
NOTE_PLANES = 5
GRID = (160, 144)            # generator grid: rows, frames (all 136 frames incl. context), padded
TERMS = ("synth", "vocal", "synth_with_vocal_magnitude", "vocal_with_synth_magnitude")
TINY = 1e-30
# Per-cell statistics of the views. "cell": within-cell statistics, what every trained judge, critic and cache
# saw. In the 58 rows below 440 Hz a cell holds one sample (c = 1), so there the coherence magnitude is
# always 1 and the modulation depth always 0 (the phase-step direction still varies). "scale": in those rows,
# coherence magnitude and modulation depth from wavelet2vec.semantic_bridge (amplitude-modulation rate and
# group-delay dispersion pooled over the band's ring time). Real notes, harmonic vs noise cells below 440 Hz:
# AUC 0.54 (chance: constant) -> 0.84 (docs/BRIDGING_SIGNAL_AND_SEMANTICS.md). Other rows are identical in both.
VIEW_STATS = ("cell", "scale")


def power(z: torch.Tensor) -> torch.Tensor:
    """|z|^2 with a defined gradient at 0."""
    return z.real ** 2 + z.imag ** 2


def _cell_stats(values: torch.Tensor):
    """Per cell of ``values [B, 2 (mid, side), R, T, c]``: mean mid power, mean side power, phase-increment
    coherence (re, im: the power-weighted mean phase step relative to the band carrier, as a unit vector
    scaled by its consistency) and modulation depth (std / mean of the sample powers)."""
    mid, side = values[:, 0], values[:, 1]
    p = power(mid)
    mean_p = p.mean(-1)
    flat = mid.reshape(*mid.shape[:-2], -1)
    inc = flat[..., 1:] * flat[..., :-1].conj()
    inc = torch.cat([inc[..., :1], inc], -1).reshape(mid.shape)
    u = inc.sum(-1) / (torch.sqrt(power(inc) + TINY).sum(-1) + 1e-20)
    mod = torch.sqrt(p.var(-1, unbiased=False) + TINY) / (mean_p + 1e-20)
    return mean_p, power(side).mean(-1), u.real, u.imag, torch.clamp(mod, max=3.0)


def _view_primitives(image: WaveletImage, stats: str):
    """The semantic-bridge primitives a ``stats="scale"`` view needs, or None for ``"cell"``."""
    if stats not in VIEW_STATS:
        raise ValueError(f"stats must be one of {VIEW_STATS}, not {stats!r}")
    if stats == "cell":
        return None
    from wavelet2vec.semantic_bridge import reassigned_primitives
    return reassigned_primitives(image)


def _group_stats(values: torch.Tensor, c: int, rows: torch.Tensor, prim):
    """``_cell_stats`` of one sample-count group; with ``prim`` (``stats="scale"``), the one-sample rows take
    their coherence magnitude (the direction stays) and modulation depth from the semantic bridge."""
    mean_p, side_p, re, im, md = _cell_stats(values)
    if prim is not None and c == 1:
        coherence = prim.phase_coherence[:, rows].to(re.dtype)
        re, im, md = re * coherence, im * coherence, prim.mod_depth[:, rows].to(md.dtype)
    return mean_p, side_p, re, im, md


def _rows_index(image, c, device):
    return torch.as_tensor(list(image.rows[c]), device=device, dtype=torch.long)


def judge_view(image: WaveletImage, stats: str = "cell") -> torch.Tensor:
    """``[B, 4, JUDGE_ROWS, FRAMES]``: level re the note's peak (floored at -60 dB, scaled to [-1, 1]),
    phase-increment coherence re/im and modulation depth (both faded out near the floor). Mid only.
    ``stats``: see ``VIEW_STATS`` (judges trained on one must be fed the same)."""
    prim = _view_primitives(image, stats)
    first = next(iter(image.groups.values()))
    b, device, dtype = first.shape[0], first.device, first.real.dtype
    level = torch.full((b, JUDGE_ROWS, TOTAL), -300.0, device=device, dtype=dtype)
    ure, uim, mod = (torch.zeros(b, JUDGE_ROWS, TOTAL, device=device, dtype=dtype) for _ in range(3))
    for c, values in image.groups.items():
        rows = _rows_index(image, c, device)
        keep = rows < JUDGE_ROWS
        if not bool(keep.any()):
            continue
        r = rows[keep]
        mean_p, _, re, im, md = _group_stats(values[:, :, keep], c, r, prim)
        level = level.index_copy(1, r, 10 * torch.log10(mean_p + TINY))
        ure, uim, mod = ure.index_copy(1, r, re), uim.index_copy(1, r, im), mod.index_copy(1, r, md)
    level, ure, uim, mod = (x[..., CORE] for x in (level, ure, uim, mod))
    rel = torch.clamp(level - level.amax(dim=(1, 2), keepdim=True), min=JUDGE_FLOOR_DB)
    weight = torch.clamp((rel - JUDGE_FLOOR_DB) / 12.0, 0.0, 1.0)     # exactly 0 at the floor
    return torch.stack([rel / (-JUDGE_FLOOR_DB / 2) + 1.0, ure * weight, uim * weight, mod / 3.0 * weight], 1)


def augment_view(view: torch.Tensor, generator: torch.Generator | None = None, max_shift=6, max_cut=24):
    """Expectation over transformation for judge inputs: a random time shift (filled with the floor) and a
    random low-pass (rows above a random cut set to the floor), per batch item."""
    b, _, rows, frames = view.shape
    out = view.clone()
    floor = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=view.device, dtype=view.dtype)[:, None, None]
    shifts = torch.randint(-max_shift // 2, max_shift + 1, (b,), generator=generator).tolist()
    cuts = torch.randint(rows - max_cut, rows + 1, (b,), generator=generator).tolist()
    for i, (s, cut) in enumerate(zip(shifts, cuts)):
        if s > 0:
            out[i, :, :, s:] = view[i, :, :, :-s]
            out[i, :, :, :s] = floor
        elif s < 0:
            out[i, :, :, :s] = view[i, :, :, -s:]
            out[i, :, :, s:] = floor
        if cut < rows:
            out[i, :, cut:] = floor
    return out


# ------------------------------------------------------------------------------------------- judges


class _Res2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, ch), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class _AttentionPool(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                          # [B, C, N] -> [B, C]
        return (x * torch.softmax(self.score(x), -1)).sum(-1)


class JudgeCNN(nn.Module):
    """2-D judge on the judge view (+ a row-coordinate plane): a thin full-resolution stem, then a residual
    CNN from half resolution, attention pooling, a 128-dim embedding and two logits (vox present, synth
    present). Sized for a 4 GB GPU."""

    def __init__(self, width=24, embedding=128):
        super().__init__()
        chans = [width, 2 * width, 4 * width, 6 * width]
        layers = [nn.Conv2d(JUDGE_PLANES + 1, 16, 3, padding=1), nn.GELU(),
                  nn.Conv2d(16, chans[0], 3, stride=2, padding=1)]
        for i, ch in enumerate(chans):
            layers += [_Res2d(ch), _Res2d(ch)]
            if i + 1 < len(chans):
                layers.append(nn.Conv2d(ch, chans[i + 1], 3, stride=2, padding=1))
        self.body = nn.Sequential(*layers, nn.GroupNorm(8, chans[-1]), nn.GELU())
        self.pool = _AttentionPool(chans[-1])
        self.embed = nn.Sequential(nn.Linear(chans[-1], embedding), nn.LayerNorm(embedding))
        self.head = nn.Linear(embedding, 2)

    def forward(self, view):
        rows = torch.linspace(-1, 1, view.shape[2], device=view.device, dtype=view.dtype)
        x = torch.cat([view, rows[None, None, :, None].expand(view.shape[0], 1, -1, view.shape[3])], 1)
        e = self.embed(self.pool(self.body(x).flatten(2)))
        return self.head(e), e


class JudgeTCN(nn.Module):
    """1-D judge: every frame's full spectrum view is one vector; dilated temporal convolutions, attention
    pooling over time. A different inductive bias from JudgeCNN, so the two do not share blind spots."""

    def __init__(self, width=192, embedding=128, dilations=(1, 2, 4, 8, 16, 1)):
        super().__init__()
        self.inp = nn.Conv1d(JUDGE_PLANES * JUDGE_ROWS, width, 1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GroupNorm(8, width), nn.GELU(),
                                                  nn.Conv1d(width, width, 3, padding=d, dilation=d))
                                    for d in dilations)
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU())
        self.pool = _AttentionPool(width)
        self.embed = nn.Sequential(nn.Linear(width, embedding), nn.LayerNorm(embedding))
        self.head = nn.Linear(embedding, 2)

    def forward(self, view):
        x = self.inp(view.flatten(1, 2))
        for block in self.blocks:
            x = x + block(x)
        e = self.embed(self.pool(self.out(x)))
        return self.head(e), e


class Critic(nn.Module):
    """Realism critic (hinge GAN) on the judge view: spectrally normalized CNN, one score per note."""

    def __init__(self, width=32):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        chans = [width, 2 * width, 4 * width, 8 * width]
        layers, cin = [], JUDGE_PLANES + 1
        for ch in chans:
            layers += [sn(nn.Conv2d(cin, ch, 3, padding=1)), nn.LeakyReLU(0.2),
                       sn(nn.Conv2d(ch, ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)]
            cin = ch
        self.body = nn.Sequential(*layers)
        self.out = sn(nn.Linear(chans[-1], 1))

    def forward(self, view):
        rows = torch.linspace(-1, 1, view.shape[2], device=view.device, dtype=view.dtype)
        x = torch.cat([view, rows[None, None, :, None].expand(view.shape[0], 1, -1, view.shape[3])], 1)
        return self.out(self.body(x).mean((2, 3)))[:, 0]


def vox_probability(logits: torch.Tensor) -> torch.Tensor:
    """P(vox) of the two-way decision: softmax over (vox present, synth present)."""
    return torch.sigmoid(logits[:, 0] - logits[:, 1])


# ------------------------------------------------------------------------------------------- baselines


def _scaled(image: WaveletImage, gains: dict) -> WaveletImage:
    return WaveletImage(image.spec, {c: image.groups[c] * gains[c] for c in image.groups}, dict(image.rows))


def mix_baseline(vocal: WaveletImage, synth: WaveletImage, alpha) -> WaveletImage:
    """alpha * vocal + (1 - alpha) * synth: the plain mix (inputs are loudness-matched)."""
    a = torch.as_tensor(alpha, dtype=next(iter(vocal.groups.values())).real.dtype,
                        device=next(iter(vocal.groups.values())).device).reshape(-1, 1, 1, 1, 1)
    return WaveletImage(vocal.spec, {c: a * vocal.groups[c] + (1 - a) * synth.groups[c] for c in vocal.groups},
                        dict(vocal.rows))


def vocoder_baseline(carrier: WaveletImage, envelope: WaveletImage, amount=1.0) -> WaveletImage:
    """Channel vocoder: the carrier's cells scaled, per cell, toward the envelope note's cell level
    (semitone channels, 10 ms frames); ``amount`` 1 imposes it fully."""
    gains = {}
    for c in carrier.groups:
        pc = power(carrier.groups[c]).mean(-1, keepdim=True).sum(1, keepdim=True)
        pe = power(envelope.groups[c]).mean(-1, keepdim=True).sum(1, keepdim=True)
        peak = pc.amax(dim=(2, 3, 4), keepdim=True)
        gains[c] = ((pe + 1e-8 * peak) / (pc + 1e-8 * peak)) ** (0.5 * amount)
    return _scaled(carrier, gains)


# ------------------------------------------------------------------------------------------- generator


@dataclass
class MorphConfig:
    dims: tuple = (48, 96, 160, 256)
    detail: int = 24
    depth: int = 6
    heads: int = 8
    max_gain_db: float = 30.0
    residual_scale: float = 0.5

    def to_dict(self):
        return asdict(self)


def note_planes(image: WaveletImage) -> torch.Tensor:
    """``[B, 5, ROWS, TOTAL]`` generator input planes (all frames, both channels): mid level re the note's
    peak (/40, floored at -100 dB), side level re mid (/40), phase-increment coherence re/im, modulation."""
    first = next(iter(image.groups.values()))
    b, device, dtype = first.shape[0], first.device, first.real.dtype
    out = torch.zeros(b, NOTE_PLANES, ROWS, TOTAL, device=device, dtype=dtype)
    level = torch.full((b, ROWS, TOTAL), -300.0, device=device, dtype=dtype)
    for c, values in image.groups.items():
        rows = _rows_index(image, c, device)
        mean_p, side_p, re, im, md = _cell_stats(values)
        level[:, rows] = 10 * torch.log10(mean_p + TINY)
        out[:, 1, rows] = torch.clamp(10 * torch.log10((side_p + TINY) / (mean_p + TINY)), -60, 20) / 40
        out[:, 2, rows], out[:, 3, rows], out[:, 4, rows] = re, im, md / 3
    rel = torch.clamp(level - level.amax(dim=(1, 2), keepdim=True), min=-100.0)
    out[:, 0] = rel / 40
    quiet = (rel <= -99.0)[:, None]
    out[:, 1:] = out[:, 1:].masked_fill(quiet, 0.0)
    return out


class _FusionBlock(nn.Module):
    """Joint self-attention over vocal + synth tokens with adaLN-Zero conditioning on alpha."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(cond))[:, None].chunk(6, -1)
        h = self.norm1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).reshape(x.shape[0], x.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape)
        x = x + g1 * self.proj(a)
        return x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)


class _Up(nn.Module):
    def __init__(self, cin, cout, skip, cond):
        super().__init__()
        self.reduce = nn.Conv2d(cin + skip, cout, 3, padding=1)
        self.res = _Res2d(cout)
        self.film = nn.Linear(cond, 2 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, skips, cond):
        x = F.interpolate(x, size=skips[0].shape[-2:], mode="bilinear", align_corners=False)
        x = self.res(self.reduce(torch.cat([x, *skips], 1)))
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, 1)
        return x * (1 + scale) + shift


def _fourier(alpha, n=8):
    freqs = torch.arange(1, n + 1, device=alpha.device, dtype=alpha.dtype) * math.pi
    return torch.cat([torch.sin(alpha[:, None] * freqs), torch.cos(alpha[:, None] * freqs)], -1)


class MorphNet(nn.Module):
    def __init__(self, config: MorphConfig | None = None):
        super().__init__()
        self.config = cfg = config or MorphConfig()
        d0, d1, d2, d3 = cfg.dims
        self.detail = nn.ModuleDict({str(c): nn.Linear(4 * c, cfg.detail) for c in GROUPS})
        self.stem = nn.Sequential(nn.Conv2d(NOTE_PLANES + 2 + cfg.detail, d0, 3, padding=1), _Res2d(d0))
        self.enc1 = nn.Sequential(nn.Conv2d(d0, d1, 3, stride=2, padding=1), _Res2d(d1))
        self.enc2 = nn.Sequential(nn.Conv2d(d1, d2, 3, stride=2, padding=1), _Res2d(d2))
        self.enc3 = nn.Sequential(nn.Conv2d(d2, d3, 3, stride=2, padding=1), _Res2d(d3))
        tokens = (GRID[0] // 8) * (GRID[1] // 8)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d3))
        self.kind = nn.Parameter(torch.zeros(2, d3))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.kind, std=0.02)
        self.cond = nn.Sequential(nn.Linear(16, d3), nn.SiLU(), nn.Linear(d3, d3))
        self.blocks = nn.ModuleList(_FusionBlock(d3, cfg.heads) for _ in range(cfg.depth))
        self.merge = nn.Linear(2 * d3, d3)
        self.up3 = _Up(d3, d2, 2 * d2, d3)
        self.up2 = _Up(d2, d1, 2 * d1, d3)
        self.up1 = _Up(d1, d0, 2 * d0, d3)
        self.gain = nn.Conv2d(d0, len(TERMS) * 2 * 2, 1)      # term x (mid, side) x (log level, phase)
        nn.init.zeros_(self.gain.weight)
        with torch.no_grad():
            bias = torch.zeros(len(TERMS), 2, 2)
            limit = cfg.max_gain_db / 20 * math.log(10)
            bias[:, :, 0] = limit * math.atanh(math.log(0.25) / limit)   # start: every material at -12 dB
            self.gain.bias.copy_(bias.flatten())
        self.residual = nn.ModuleDict({str(c): nn.Linear(d0, 4 * c) for c in GROUPS})
        for layer in self.residual.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _inputs(self, image: WaveletImage) -> torch.Tensor:
        first = next(iter(image.groups.values()))
        b, device, dtype = first.shape[0], first.device, first.real.dtype
        planes = note_planes(image)
        detail = torch.zeros(b, ROWS, TOTAL, self.config.detail, device=device, dtype=dtype)
        peak = max(float(power(v[:, 0]).mean(-1).amax()) for v in image.groups.values())
        for c, values in image.groups.items():
            rms = torch.sqrt(power(values[:, 0]).mean(-1, keepdim=True) + 1e-6 * peak)   # mid cell rms
            unit = values / rms[:, None]
            flat = torch.stack([unit.real, unit.imag], -2).permute(0, 2, 3, 1, 4, 5).flatten(3)  # [B, R, T, 4c]
            detail = detail.index_copy(1, _rows_index(image, c, device), self.detail[str(c)](flat))
        rows = torch.linspace(-1, 1, ROWS, device=device, dtype=dtype)[None, None, :, None].expand(b, 1, ROWS, TOTAL)
        frames = torch.linspace(-1, 1, TOTAL, device=device, dtype=dtype)[None, None, None, :].expand(b, 1, ROWS, TOTAL)
        x = torch.cat([planes, rows, frames, detail.permute(0, 3, 1, 2)], 1)
        return F.pad(x, (0, GRID[1] - TOTAL, 0, GRID[0] - ROWS))

    def _encode(self, image):
        f0 = self.stem(self._inputs(image))
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        return [f0, f1, f2, self.enc3(f2)]

    def forward(self, vocal: WaveletImage, synth: WaveletImage, alpha: torch.Tensor):
        """Hybrid of loudness-matched ``vocal`` and ``synth`` notes (same key), ``alpha`` = vocal amount
        in [0, 1] per item. Returns the hybrid image and extras for regularizers: ``gain_level`` (log gain per
        term, channel, row, frame), ``residual_energy`` ([B], core frames) and the complex ``gains``."""
        fv, fs = self._encode(vocal), self._encode(synth)
        cond = self.cond(_fourier(alpha))
        b, d, h, w = fv[3].shape
        tv = fv[3].flatten(2).transpose(1, 2) + self.pos + self.kind[0]
        ts = fs[3].flatten(2).transpose(1, 2) + self.pos + self.kind[1]
        x = torch.cat([tv, ts], 1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.merge(torch.cat([x[:, :h * w], x[:, h * w:]], -1)).transpose(1, 2).reshape(b, d, h, w)
        x = self.up3(x, [fv[2], fs[2]], cond)
        x = self.up2(x, [fv[1], fs[1]], cond)
        x = self.up1(x, [fv[0], fs[0]], cond)[:, :, :ROWS, :TOTAL]
        return self._render(x, vocal, synth)

    def _render(self, features, vocal: WaveletImage, synth: WaveletImage):
        cfg = self.config
        raw = self.gain(features).reshape(features.shape[0], len(TERMS), 2, 2, ROWS, TOTAL)
        limit = cfg.max_gain_db / 20 * math.log(10)
        level = limit * torch.tanh(raw[:, :, :, 0] / limit)
        phase = math.pi * torch.tanh(raw[:, :, :, 1])
        gains = torch.stack([torch.exp(level) * torch.cos(phase), torch.exp(level) * torch.sin(phase)], -1)
        peak_v = max(float(power(v).mean(-1).amax()) for v in vocal.groups.values())
        peak_s = max(float(power(v).mean(-1).amax()) for v in synth.groups.values())
        groups, residual_energy = {}, 0.0
        for c in vocal.groups:
            rows = _rows_index(vocal, c, features.device)
            v, s = vocal.groups[c], synth.groups[c]                                   # [B, 2, R, T, c]
            g = gains[:, :, :, rows]                                                  # [B, K, 2, R, T, 2]
            g = g.permute(0, 1, 2, 3, 5, 4).reshape(g.shape[0], -1, TOTAL)
            g = F.interpolate(g, size=TOTAL * c, mode="linear", align_corners=False)  # smooth inside cells
            g = g.reshape(features.shape[0], len(TERMS), 2, len(rows), 2, TOTAL, c)
            g = torch.complex(g[:, :, :, :, 0], g[:, :, :, :, 1])                     # [B, K, 2, R, T, c]
            pv, ps = power(v), power(s)
            mag_v, mag_s = torch.sqrt(pv + 1e-8 * peak_v), torch.sqrt(ps + 1e-8 * peak_s)
            terms = torch.stack([s, v, s / mag_s * mag_v, v / mag_v * mag_s], 1)
            out = (g * terms).sum(1)
            res = torch.tanh(self.residual[str(c)](features[:, :, rows].permute(0, 2, 3, 1)))
            res = res.reshape(*res.shape[:3], 2, 2, c)                                 # [B, R, T, ch, re/im, c]
            res = torch.complex(res[..., 0, :], res[..., 1, :]).permute(0, 3, 1, 2, 4)  # [B, 2, R, T, c]
            ref = torch.sqrt(torch.maximum(pv.sum(1).mean(-1), ps.sum(1).mean(-1))
                             + 1e-4 * max(peak_v, peak_s))[:, None, :, :, None]
            res = cfg.residual_scale * ref * res
            residual_energy = residual_energy + power(res[:, :, :, CORE]).sum((1, 2, 3, 4)) / c
            groups[c] = out + res
        extras = {"gains": gains, "gain_level": level, "residual_energy": residual_energy}
        return WaveletImage(vocal.spec, groups, dict(vocal.rows)), extras


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
