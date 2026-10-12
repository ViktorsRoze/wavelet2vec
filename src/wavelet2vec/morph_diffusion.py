"""MorphDiffusion, stage 0: a diffusion prior over note feature images, morphing by composition.

The evaluation experiment for a diffusion approach to vox-synth hybrids
(docs/MORPH_DIFFUSION.md). It generates a note's **feature image**: the
5-plane critic view of :mod:`wavelet2vec.morph_large`. The planes are mid
and side level re the note's peak, phase-increment coherence (re, im) and
modulation depth, over 122 semitone rows up to 20 kHz x 120 frames of 10 ms.
The judges' view is an exact slice of it (planes 0, 2, 3, 4; rows below
18 kHz), so judges score generated notes without rendering audio.

- ``SwinDenoiser`` (default, ~245 M parameters): KeyboardNet v3's Swin-L
  backbone and UPerNet head, warm-started from its checkpoint, used as the
  denoiser. A 1x1 adapter maps the feature image to v3's 28 input planes.
  Diffusion time and class are added to v3's adaLN conditioning vector; the
  key enters through v3's own key embedding. A new 5-plane output layer
  reads the full-resolution head. Grid: 160 x 128.
- ``FeatureUNet`` (small, ~23-45 M, for the GPU): a UNet with
  self-attention at the coarse levels, conditioned by FiLM on the diffusion
  time, the class and the key (MIDI). Grid: 128 x 128.
  Classes: vox, synth, processed_vox (Exhale), synth_vox (Harmor vox), and
  null for classifier-free guidance. The class enters as weights over the
  class table, so embeddings can be interpolated.
- Diffusion: continuous time with the cosine schedule, v-prediction, and a
  deterministic DDIM sampler. Predictions are clamped to each plane's valid
  range at every step.
- Morphing at sampling time (no paired training):
  - ``"and"``: v = v_null + g*alpha*(v_vox - v_null) + g*(1 - alpha)*(v_synth - v_null).
    This samples where both classes are likely (a product of experts).
    alpha may be a value per frame: the vocal amount automated within a note.
  - ``"interp"``: the class embedding interpolated between vox and synth,
    ordinary guidance.
  - ``"label"``: one class, for fidelity checks.
- ``render_features``: for listening. It imposes a generated feature image
  on a carrier note at the same key. The carrier keeps its phase (stable
  pitch); cells the image marks noise-like use a random-phase copy of the
  carrier.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from wavelet2vec.morph import CORE, TOTAL, power
from wavelet2vec.morph_large import CRITIC_FLOOR_DB, CRITIC_PLANES, CRITIC_ROWS, _within
from wavelet2vec.wavelet_image import WaveletImage

CLASSES = ("vox", "synth", "processed_vox", "synth_vox")
NULL = len(CLASSES)
PLANES = CRITIC_PLANES                    # mid level, side level, coherence re, coherence im, modulation
ROWS, FRAMES = CRITIC_ROWS, 120           # 122 rows (<= 20 kHz) x 120 frames
PADDED = (128, 128)
FLOOR = (-1.0, -1.0, 0.0, 0.0, 0.0)       # value of each plane in silence (also the padding)
RANGES = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
JUDGE_SLICE = ((0, 2, 3, 4), 120)         # judge view = features[:, planes][:, :, :120]


def _floor(device=None, dtype=torch.float32):
    return torch.tensor(FLOOR, device=device, dtype=dtype)[None, :, None, None]


def pad_features(x: torch.Tensor, grid=PADDED) -> torch.Tensor:
    """``[B, 5, 122, 120]`` -> ``[B, 5, *grid]``, padded with each plane's silence value."""
    out = _floor(x.device, x.dtype).expand(x.shape[0], PLANES, *grid).clone()
    out[:, :, :ROWS, :FRAMES] = x
    return out


def unpad(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :ROWS, :FRAMES]


def judge_slice(features: torch.Tensor) -> torch.Tensor:
    """The judges' view ``[B, 4, 120, 120]`` of feature images ``[B, 5, 122, 120]``."""
    planes, rows = JUDGE_SLICE
    return features[:, list(planes), :rows]


def clamp_features(x: torch.Tensor) -> torch.Tensor:
    lo = torch.tensor([r[0] for r in RANGES], device=x.device, dtype=x.dtype)[None, :, None, None]
    hi = torch.tensor([r[1] for r in RANGES], device=x.device, dtype=x.dtype)[None, :, None, None]
    return torch.maximum(torch.minimum(x, hi), lo)


def shift_rows(x: torch.Tensor, semitones: int) -> torch.Tensor:
    """Move a feature image up (positive) or down by whole semitones; vacated rows become silence."""
    if semitones == 0:
        return x
    out = _floor(x.device, x.dtype).expand_as(x).clone()
    if semitones > 0:
        out[:, :, semitones:] = x[:, :, :-semitones]
    else:
        out[:, :, :semitones] = x[:, :, -semitones:]
    return out


# ----------------------------------------------------------------------------------------- schedule


def alpha_sigma(t: torch.Tensor, s: float = 0.008):
    """Cosine schedule (Nichol & Dhariwal) in continuous time t in [0, 1]: signal and noise scales."""
    f = torch.cos((t + s) / (1 + s) * math.pi / 2)
    f0 = math.cos(s / (1 + s) * math.pi / 2)
    alpha_bar = (f / f0) ** 2
    alpha = torch.sqrt(alpha_bar.clamp(1e-8, 1.0))
    return alpha, torch.sqrt((1 - alpha_bar).clamp(1e-8, 1.0))


def v_target(x0, noise, alpha, sigma):
    return alpha * noise - sigma * x0


def predict_x0(x_t, v, alpha, sigma):
    return alpha * x_t - sigma * v


def predict_noise(x_t, v, alpha, sigma):
    return sigma * x_t + alpha * v


# ----------------------------------------------------------------------------------------- model


def _sinusoid(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    arg = x.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(arg), torch.cos(arg)], -1)


def _groups(ch):
    return 32 if ch % 32 == 0 else 16 if ch % 16 == 0 else 8


class _Res(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(_groups(cin), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(_groups(cout), cout), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return self.skip(x) + h


class _Attn(nn.Module):
    def __init__(self, ch, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).permute(1, 0, 2, 4, 3)
        a = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(a)


class FeatureUNet(nn.Module):
    """Denoiser for padded feature images ``[B, 5, 128, 128]`` (v-prediction)."""

    def __init__(self, chans=(64, 128, 192, 256), blocks=2, attn_levels=(3,), heads=8, emb=512):
        super().__init__()
        self.chans, self.emb_dim = chans, emb
        self.time = nn.Sequential(nn.Linear(256, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.key = nn.Sequential(nn.Linear(64, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.classes = nn.Parameter(torch.randn(len(CLASSES) + 1, emb) * 0.02)
        self.inp = nn.Conv2d(PLANES + 2, chans[0], 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skips, ch = [chans[0]], chans[0]
        for level, out in enumerate(chans):
            for _ in range(blocks):
                self.down.append(nn.ModuleList([_Res(ch, out, emb)] + ([_Attn(out, heads)] if level in attn_levels else [])))
                ch = out
                skips.append(ch)
            if level < len(chans) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)
        self.mid = nn.ModuleList([_Res(ch, ch, emb), _Attn(ch, heads), _Res(ch, ch, emb)])
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for level, out in reversed(list(enumerate(chans))):
            for _ in range(blocks + 1):
                self.up.append(nn.ModuleList([_Res(ch + skips.pop(), out, emb)]
                                             + ([_Attn(out, heads)] if level in attn_levels else [])))
                ch = out
            if level > 0:
                self.upsample.append(nn.Conv2d(ch, ch, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(_groups(ch), ch), nn.SiLU(), nn.Conv2d(ch, PLANES, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.blocks = blocks

    def embed(self, t, midi, class_weights):
        """``class_weights`` ``[B, len(CLASSES) + 1]`` (one-hot, or a mixture for interpolated classes)."""
        return (self.time(_sinusoid(t * 1000.0, 256)) + self.key(_sinusoid(midi.float(), 64))
                + class_weights.to(self.classes.dtype) @ self.classes)

    def forward(self, x, t, midi, class_weights):
        e = self.embed(t, midi, class_weights)
        b, _, h, w = x.shape
        rows = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)[None, None, :, None].expand(b, 1, h, w)
        frames = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)[None, None, None, :].expand(b, 1, h, w)
        hcur = self.inp(torch.cat([x, rows, frames], 1))
        skips = [hcur]
        di = 0
        for level in range(len(self.chans)):
            for _ in range(self.blocks):
                block = self.down[di]
                di += 1
                hcur = block[0](hcur, e)
                for extra in block[1:]:
                    hcur = extra(hcur)
                skips.append(hcur)
            if level < len(self.chans) - 1:
                hcur = self.downsample[level](hcur)
                skips.append(hcur)
        hcur = self.mid[2](self.mid[1](self.mid[0](hcur, e)), e)
        ui = 0
        for k, level in enumerate(reversed(range(len(self.chans)))):
            for _ in range(self.blocks + 1):
                block = self.up[ui]
                ui += 1
                hcur = block[0](torch.cat([hcur, skips.pop()], 1), e)
                for extra in block[1:]:
                    hcur = extra(hcur)
            if level > 0:
                hcur = self.upsample[k](F.interpolate(hcur, scale_factor=2, mode="nearest"))
        return self.out(hcur)


class SwinDenoiser(nn.Module):
    """KeyboardNet v3's Swin-L backbone + UPerNet head as a feature-image denoiser (v-prediction), ~245 M
    parameters, warm-started from a v3 checkpoint. New parts: input adapter, time / class embeddings
    (added to v3's conditioning, zero-initialized), detail input and the output layer."""

    def __init__(self, v3_state=None, v2_config: dict | None = None):
        super().__init__()
        from wavelet2vec.keyboard_v2 import COND_DIM, IN_PLANES, KeyboardV2Config
        from wavelet2vec.keyboard_v3 import GRID, KeyboardNetV3
        self.grid = GRID                                                           # (160, 128)
        self.cond_dim = COND_DIM
        self.v3 = KeyboardNetV3(None, KeyboardV2Config(**(v2_config or {"size": "large"})))
        if v3_state is not None:
            self.v3.load_state_dict(v3_state)
        net = self.v3.net
        ce = net.config.cond_embed
        r_full = net.config.refine_dims[1]
        self.adapter = nn.Conv2d(PLANES + 2, IN_PLANES, 1)
        self.detail = nn.Conv2d(PLANES + 2, self.v3.detail_dim, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(256, ce), nn.SiLU(), nn.Linear(ce, ce))
        self.classes = nn.Parameter(torch.zeros(len(CLASSES) + 1, ce))
        nn.init.zeros_(self.time[2].weight)
        nn.init.zeros_(self.time[2].bias)
        self.out = nn.Conv2d(r_full, PLANES, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def pretrained_parameters(self):
        return list(self.v3.parameters())

    def new_parameters(self):
        ids = {id(p) for p in self.pretrained_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def forward(self, x, t, midi, class_weights):
        net = self.v3.net
        b, _, h, w = x.shape
        rows = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)[None, None, :, None].expand(b, 1, h, w)
        frames = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)[None, None, None, :].expand(b, 1, h, w)
        inp = torch.cat([x, rows, frames], 1)
        planes = self.adapter(inp)
        cond = torch.zeros(b, self.cond_dim, device=x.device, dtype=x.dtype)
        cond[:, 0] = midi.to(x.dtype) / 127.0                                      # v3 layout: one example at the key
        c = (net.cond(cond) + self.time(_sinusoid(t * 1000.0, 256)).to(x.dtype)
             + class_weights.to(x.dtype) @ self.classes)
        full = net.stem_full(planes)
        half = net.stem_half(full)
        y = net.patch(planes)
        y = net.patch_norm(y.flatten(2).transpose(1, 2))
        stages = []
        for i, blocks in enumerate(net.stages):
            for block in blocks:
                y = block(y, c)
            stages.append(net.stage_norms[i](y).transpose(1, 2).reshape(b, -1, *net.resolutions[i]))
            if i < 3:
                y = net.merges[i](y)
        top = stages[3]
        pooled = [F.interpolate(p(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for p in net.ppm]
        levels = [None, None, None, net.ppm_fuse(torch.cat([top] + pooled, 1))]
        for i in (2, 1, 0):
            up = F.interpolate(levels[i + 1], size=stages[i].shape[-2:], mode="bilinear", align_corners=False)
            levels[i] = net.fpn[i](net.lateral[i](stages[i]) + up)
        size = levels[0].shape[-2:]
        hh = net.fuse(torch.cat([levels[0]] + [F.interpolate(l, size=size, mode="bilinear", align_corners=False)
                                               for l in levels[1:]], 1))
        hh = net.fuse_film(hh, c)
        hh = F.interpolate(hh, size=half.shape[-2:], mode="bilinear", align_corners=False)
        hh = net.refine_half[1](net.refine_half[0](torch.cat([hh, half], 1), c), c)
        hh = F.interpolate(hh, size=full.shape[-2:], mode="bilinear", align_corners=False)
        hh = net.refine_full[1](net.refine_full[0](torch.cat([hh, full, self.detail(inp)], 1), c), c)
        return self.out(hh)


def build_denoiser(meta: dict, v3_state=None):
    """The denoiser described by a checkpoint's ``meta`` ({"arch": "swin", "v2_config": ...} or
    {"arch": "unet", "chans": [...]})."""
    if meta.get("arch", "unet") == "swin":
        return SwinDenoiser(v3_state, meta.get("v2_config"))
    return FeatureUNet(chans=tuple(meta["chans"]))


def one_hot(labels, device=None) -> torch.Tensor:
    labels = torch.as_tensor(labels, device=device, dtype=torch.long)
    return F.one_hot(labels, len(CLASSES) + 1).float()


def class_index(name: str) -> int:
    return CLASSES.index(name)


# ----------------------------------------------------------------------------------------- sampling


@torch.no_grad()
def ddim_sample(model, midi, guidance: dict, steps: int = 40, generator=None, device=None,
                max_batch: int = 16) -> torch.Tensor:
    """Feature images ``[n, 5, 122, 120]`` for keys ``midi`` (``[n]``), deterministic DDIM.

    ``guidance``:
    - ``{"mode": "label", "label": "vox", "scale": 3.0}``;
    - ``{"mode": "and", "alpha": 0.5 or [120] per frame, "scale": 4.0}``: vox and synth together,
      vox weight scale*alpha, synth weight scale*(1 - alpha), per frame if alpha is a curve;
    - ``{"mode": "interp", "alpha": 0.5, "scale": 3.0}``: the class embedding alpha*vox + (1 - alpha)*synth.

    The denoiser runs in chunks of ``max_batch`` (memory on small GPUs)."""
    device = device or next(model.parameters()).device
    midi = torch.as_tensor(midi, device=device, dtype=torch.float32)
    n = midi.shape[0]
    grid = getattr(model, "grid", PADDED)
    x = torch.randn(n, PLANES, *grid, device=device, generator=generator)
    null = one_hot([NULL] * n, device)
    mode, scale = guidance.get("mode", "label"), float(guidance.get("scale", 3.0))
    if mode == "label":
        conds = [(one_hot([class_index(guidance["label"])] * n, device), scale)]
    elif mode == "interp":
        a = float(guidance.get("alpha", 0.5))
        mix = a * one_hot([class_index("vox")] * n, device) + (1 - a) * one_hot([class_index("synth")] * n, device)
        conds = [(mix, scale)]
    elif mode == "and":
        a = torch.as_tensor(guidance.get("alpha", 0.5), device=device, dtype=torch.float32)
        if a.ndim == 1:                              # per-frame curve: padded frames hold the last value
            a = torch.cat([a, a[-1:].expand(grid[1] - a.shape[0])])[None, None, None, :]
        conds = [(one_hot([class_index("vox")] * n, device), scale * a),
                 (one_hot([class_index("synth")] * n, device), scale * (1 - a))]
    else:
        raise ValueError(f"unknown guidance mode {mode}")
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t, t_next = times[i], times[i + 1]
        a_t, s_t = alpha_sigma(t)
        a_n, s_n = alpha_sigma(t_next)
        tt = t.expand(n)
        batch_x = torch.cat([x] * (1 + len(conds)))
        batch_c = torch.cat([null] + [c for c, _ in conds])
        batch_t, batch_k = tt.repeat(1 + len(conds)), midi.repeat(1 + len(conds))
        v_all = torch.cat([model(batch_x[i:i + max_batch], batch_t[i:i + max_batch], batch_k[i:i + max_batch],
                                 batch_c[i:i + max_batch]) for i in range(0, len(batch_x), max_batch)]).chunk(1 + len(conds))
        v = v_all[0] + sum(w * (vc - v_all[0]) for (_, w), vc in zip(conds, v_all[1:]))
        x0 = pad_features(clamp_features(unpad(predict_x0(x, v, a_t, s_t))), grid)
        eps = (x - a_t * x0) / s_t
        x = a_n * x0 + s_n * eps
    return clamp_features(unpad(x))


# ----------------------------------------------------------------------------------------- listening


@torch.no_grad()
def render_features(features: torch.Tensor, carrier: WaveletImage, seed: int = 0, max_gain_db: float = 30.0):
    """Impose a feature image ``[5, 122, 120]`` on a one-note ``carrier`` (same key). Mid and side levels
    follow the image (re the carrier's mid peak). The carrier keeps its own phase, so the pitch is stable.
    Where the image is noise-like (coherence below 0.4), a random-phase copy of the carrier is used. Rows
    above 20 kHz keep the carrier, scaled like the top rows. Returns a WaveletImage."""
    f = features.float()
    scale = -CRITIC_FLOOR_DB / 2
    target = torch.stack([(f[0] - 1) * scale, (f[1] - 1) * scale])                 # dB re peak, [2, 122, 120]
    coherence = torch.sqrt(f[2] ** 2 + f[3] ** 2)
    noisy = torch.clamp((0.75 - coherence) / 0.35, 0.0, 1.0)
    noisy = torch.where(f[0] > -0.95, noisy, torch.zeros_like(noisy))
    gen = torch.Generator().manual_seed(seed)
    first = next(iter(carrier.groups.values()))
    peak = max(float(power(v[0, 0]).mean(-1)[:, CORE].amax()) for v in carrier.groups.values())
    peak_db = 10 * math.log10(peak + 1e-30)
    groups = {}
    for c, values in carrier.groups.items():
        rows = list(map(int, carrier.rows[c]))
        cells = values[0]                                                            # [2, R, T, c]
        level = 10 * torch.log10(power(cells).mean(-1) + 1e-30) - peak_db            # [2, R, T]
        gain_db = torch.zeros(2, len(rows), TOTAL, dtype=torch.float32)
        mask = torch.zeros(1, len(rows), TOTAL, dtype=torch.float32)
        for k, r in enumerate(rows):
            src = min(r, ROWS - 1)
            want = target[:, src]                                                    # [2, 120]
            have = level[:, k, CORE].float()
            g = torch.clamp(want - have, -max_gain_db, max_gain_db)
            g = torch.where(want <= CRITIC_FLOOR_DB + 0.5, torch.full_like(g, -max_gain_db), g)
            gain_db[:, k, CORE] = g
            gain_db[:, k, :CORE.start], gain_db[:, k, CORE.stop:] = g[:, :1], g[:, -1:]
            mask[0, k, CORE] = noisy[src]
        gains = 10 ** (_within(gain_db[None], c)[0] / 20)                          # [2, R, T, c]
        m = _within(mask[None], c)[0]                                                # [1, R, T, c]
        phase = torch.rand(cells.shape, generator=gen) * 2 * math.pi
        noise = cells.abs() * torch.polar(torch.ones_like(phase), phase).to(cells.dtype)
        groups[c] = ((1 - m) * gains * cells + m * gains * noise)[None].to(first.dtype)
    return WaveletImage(carrier.spec, groups, dict(carrier.rows))
