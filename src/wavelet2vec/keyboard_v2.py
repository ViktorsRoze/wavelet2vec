"""KeyboardNet v2: rich wavelet features, a Swin backbone and a deep dense head.

Same task and carrier as v1 (:mod:`wavelet2vec.keyboard`): from one, two or
three notes of a sound, predict the envelope of another note. That envelope reshapes
the pitch-shifted nearest example. v2 changes what the network sees and how
large it is.

Rich note features, ``[7, 124 semitone bands, 120 x 10 ms]`` at 48 kHz.
Spectral, phase and wave information together, using the library's encodings:

0. log energy (dB / 40), mid channel;
1. instantaneous-frequency deviation from the band centre (tanh of the
   relative deviation, energy-weighted per frame), as in ``vision``;
2. phase coherence: |mean demodulated unit phasor| per frame (``vision``'s
   coherence, taken after removing the band carrier so steady partials read
   ~1 and noise reads low);
3-4. demodulated baseband real / imaginary (band signal times the conjugate
   carrier, frame mean / frame RMS), as in ``hybrid`` context features.
   Exactly unchanged by the heterodyne pitch shift, so shifted examples stay
   consistent;
5. modulation depth: std / mean of |c| inside each frame (beating, tremolo);
6. stereo width: side / mid energy (dB / 40).

Backbone: a Swin Transformer written here (no timm dependency). It works on
the 128 x 128 padded input with 4 x 4 patches and 8 x 8 windows. Every block
is adaLN-Zero conditioned on the target note and intervals, so a deep stack
starts as the identity. ``SWIN_SIZES["large"]`` is Swin-L
(192, depths 2/2/18/2, heads 6/12/24/48).

Head: UPerNet-style pyramid pooling + FPN over all four stages, fused, then a
refinement decoder to full resolution with skips from a high-resolution stem.
It outputs 4 maps: envelope correction (main) plus the target's IF deviation,
phase coherence and stereo width (auxiliary targets).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from wavelet2vec.keyboard import (DB_SCALE, ENV_BANDS, ENV_FRAMES, ENV_HOP, ENV_RATE, FLOOR_DB, NOTE_SAMPLES,
                                  baseline_envelope, note_frame)

FEATURES = ("log_energy", "if_deviation", "phase_coherence", "baseband_re", "baseband_im",
            "modulation_depth", "stereo_width")
N_FEATURES = len(FEATURES)
AUX_TARGETS = (1, 2, 6)  # predicted besides the envelope: IF deviation, coherence, width
PAD = 128


def rich_note_features(note_48k: np.ndarray) -> np.ndarray:
    """``[N_FEATURES, ENV_BANDS, ENV_FRAMES]`` float32 from a 1.2 s note at 48 kHz (``[2, N]`` or ``[N]``)."""
    audio = np.atleast_2d(np.asarray(note_48k, dtype=np.float64))
    stereo = np.zeros((2, NOTE_SAMPLES))
    n = min(NOTE_SAMPLES, audio.shape[1])
    stereo[0, :n] = audio[0, :n]
    stereo[1, :n] = audio[-1, :n]
    mid_c, spec = note_frame((stereo[0] + stereo[1]) / 2, ENV_RATE, ENV_BANDS)
    side_c, _ = note_frame((stereo[0] - stereo[1]) / 2, ENV_RATE, ENV_BANDS)
    centers = spec.centers[:, None, None]
    frames = mid_c.reshape(ENV_BANDS, ENV_FRAMES, ENV_HOP)
    magnitude = np.abs(frames)
    power = magnitude ** 2
    energy = power.mean(-1)
    out = np.zeros((N_FEATURES, ENV_BANDS, ENV_FRAMES), dtype=np.float32)
    out[0] = np.maximum(10 * np.log10(energy + 1e-30), FLOOR_DB) / DB_SCALE

    increment = mid_c[:, 1:] * np.conj(mid_c[:, :-1])
    inst = np.angle(np.concatenate([increment[:, :1], increment], axis=1)) * ENV_RATE / (2 * np.pi)
    deviation = np.tanh((inst.reshape(ENV_BANDS, ENV_FRAMES, ENV_HOP) - centers) / centers)
    out[1] = (power * deviation).sum(-1) / (power.sum(-1) + 1e-30)

    # Coherence of the demodulated unit phasor: ~1 for a steady partial in the band,
    # low for noise. (Undemodulated, a steady tone's phasor spins at the band
    # frequency and averages out within a 10 ms frame.)
    t = np.arange(NOTE_SAMPLES) / ENV_RATE
    carrier = np.exp(-2j * np.pi * spec.centers[:, None] * t[None, :])
    demodulated = (mid_c * carrier).reshape(ENV_BANDS, ENV_FRAMES, ENV_HOP)
    out[2] = np.abs((demodulated / (magnitude + 1e-12)).mean(-1))

    baseband = demodulated.mean(-1)
    rms = np.sqrt(energy) + 1e-12
    out[3] = baseband.real / rms
    out[4] = baseband.imag / rms

    out[5] = np.minimum(magnitude.std(-1) / (magnitude.mean(-1) + 1e-12), 3.0)
    side = (np.abs(side_c.reshape(ENV_BANDS, ENV_FRAMES, ENV_HOP)) ** 2).mean(-1)
    out[6] = np.clip(10 * np.log10((side + 1e-30) / (energy + 1e-30)), -60.0, 20.0) / DB_SCALE
    quiet = out[0] * DB_SCALE <= FLOOR_DB + 1e-3
    out[1:6][:, quiet] = 0.0
    out[6][quiet] = -60.0 / DB_SCALE
    return out


def shift_features(features, semitones):
    """Move all feature planes by whole semitones; vacated bands get neutral values."""
    shifted = np.zeros_like(features)
    shifted[0] = FLOOR_DB / DB_SCALE
    shifted[6] = -60.0 / DB_SCALE
    valid = np.zeros(features.shape[1:], dtype=np.float32)
    d = int(semitones)
    if d >= 0:
        shifted[:, d:] = features[:, :features.shape[1] - d]
        valid[d:] = 1.0
    else:
        shifted[:, :d] = features[:, -d:]
        valid[:d] = 1.0
    return shifted, valid


MAX_EXAMPLES = 3
IN_PLANES = MAX_EXAMPLES * (N_FEATURES + 1) + 4
COND_DIM = 2 + 2 * MAX_EXAMPLES
BASE_PLANE = MAX_EXAMPLES * (N_FEATURES + 1)  # index of the baseline envelope plane


def network_inputs_v2(examples, example_midis, target_midi):
    """Planes ``[IN_PLANES, 128, 128]`` (zero-padded), conditioning ``[COND_DIM]`` and the baseline (dB).

    Takes 1-3 examples, ordered nearest-first. Per example: 7 shifted feature planes and a
    validity mask. Then: baseline envelope, band and time coordinates, and the nearest-minus-
    second envelope difference (the sound's own trend). Conditioning: target note, the
    example count, and each signed / absolute interval (0 for absent examples).
    """
    if not 1 <= len(examples) <= MAX_EXAMPLES:
        raise ValueError(f"1-{MAX_EXAMPLES} examples")
    order = sorted(range(len(examples)), key=lambda i: abs(int(target_midi) - int(example_midis[i])))
    examples = [examples[i] for i in order]
    distances = [int(target_midi) - int(example_midis[i]) for i in order]
    shifted, valid = zip(*(shift_features(f, d) for f, d in zip(examples, distances)))
    envelopes = np.stack([s[0] * DB_SCALE for s in shifted])
    base = baseline_envelope(envelopes, np.stack(valid), distances)
    planes = np.zeros((IN_PLANES, PAD, PAD), dtype=np.float32)
    stride = N_FEATURES + 1
    for slot in range(MAX_EXAMPLES):
        if slot < len(examples):
            planes[slot * stride:slot * stride + N_FEATURES, :ENV_BANDS, :ENV_FRAMES] = shifted[slot]
            planes[slot * stride + N_FEATURES, :ENV_BANDS, :ENV_FRAMES] = valid[slot]
        else:
            planes[slot * stride, :ENV_BANDS, :ENV_FRAMES] = FLOOR_DB / DB_SCALE
    planes[BASE_PLANE, :ENV_BANDS, :ENV_FRAMES] = base / DB_SCALE
    planes[BASE_PLANE + 1, :ENV_BANDS, :ENV_FRAMES] = np.linspace(0, 1, ENV_BANDS, dtype=np.float32)[:, None]
    planes[BASE_PLANE + 2, :ENV_BANDS, :ENV_FRAMES] = np.linspace(0, 1, ENV_FRAMES, dtype=np.float32)[None, :]
    if len(examples) >= 2:
        planes[BASE_PLANE + 3, :ENV_BANDS, :ENV_FRAMES] = (shifted[0][0] - shifted[1][0]) * valid[0] * valid[1]
    padded = distances + [0] * (MAX_EXAMPLES - len(distances))
    cond = np.array([target_midi / 127.0, (len(examples) - 1) / (MAX_EXAMPLES - 1)]
                    + [d / 48.0 for d in padded] + [abs(d) / 48.0 for d in padded], dtype=np.float32)
    return planes, cond, base.astype(np.float32)


SWIN_SIZES = {
    "tiny": {"embed": 96, "depths": (2, 2, 6, 2), "heads": (3, 6, 12, 24)},
    "base": {"embed": 128, "depths": (2, 2, 18, 2), "heads": (4, 8, 16, 32)},
    "large": {"embed": 192, "depths": (2, 2, 18, 2), "heads": (6, 12, 24, 48)},
}


@dataclass
class KeyboardV2Config:
    size: str = "large"
    window: int = 8
    patch: int = 4
    mlp_ratio: float = 4.0
    drop_path: float = 0.2
    head_dim: int = 384          # UPerNet / FPN width
    refine_dims: tuple[int, int] = (160, 64)  # decoder widths at 1/2 and full resolution
    cond_embed: int = 256
    max_correction_db: float = 40.0
    version: str = "keyboard-v2"
    swin: dict = field(default_factory=dict)
    grid: tuple[int, int] = (PAD, PAD)        # input planes (rows, frames), multiples of patch * 2**3
    rows: int = ENV_BANDS                     # output crop: rows (semitone bands) ...
    frames: int = ENV_FRAMES                  # ... and frames
    in_planes: int = 0                        # 0: IN_PLANES

    def __post_init__(self):
        if not self.swin:
            self.swin = dict(SWIN_SIZES[self.size])


def build_model_v2(config: KeyboardV2Config | None = None):
    import torch
    from torch import nn
    import torch.nn.functional as F

    config = config or KeyboardV2Config()
    sw = config.swin
    depths, heads, embed = tuple(sw["depths"]), tuple(sw["heads"]), int(sw["embed"])
    dims = [embed * 2 ** i for i in range(4)]
    total_blocks = sum(depths)

    def window_partition(x, w):
        b, h, wd, c = x.shape
        x = x.view(b, h // w, w, wd // w, w, c)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)

    def window_merge(windows, w, h, wd):
        b = windows.shape[0] // ((h // w) * (wd // w))
        x = windows.view(b, h // w, wd // w, w, w, -1)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, wd, -1)

    class WindowAttention(nn.Module):
        def __init__(self, dim, n_heads, w):
            super().__init__()
            self.heads, self.scale = n_heads, (dim // n_heads) ** -0.5
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.bias_table = nn.Parameter(torch.zeros((2 * w - 1) ** 2, n_heads))
            nn.init.trunc_normal_(self.bias_table, std=0.02)
            coords = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing="ij")).flatten(1)
            rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (w - 1)
            self.register_buffer("bias_index", rel[..., 0] * (2 * w - 1) + rel[..., 1], persistent=False)
            self.stats = None  # set to a list to record attention entropy (diagnostics)

        def forward(self, x, mask=None):
            bw, n, c = x.shape
            q, k, v = self.qkv(x).view(bw, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn + self.bias_table[self.bias_index].permute(2, 0, 1)[None]
            if mask is not None:
                nw = mask.shape[0]
                attn = attn.view(bw // nw, nw, self.heads, n, n) + mask[None, :, None]
                attn = attn.view(bw, self.heads, n, n)
            weights = attn.softmax(-1)
            if self.stats is not None:  # mean entropy, normalized by log(window tokens): 1 = uniform
                entropy = -(weights * torch.log(weights + 1e-12)).sum(-1).mean()
                self.stats.append(float(entropy) / math.log(n))
            out = (weights @ v).transpose(1, 2).reshape(bw, n, c)
            return self.proj(out)

    class SwinBlock(nn.Module):
        def __init__(self, dim, n_heads, res, w, shift, drop):
            super().__init__()
            self.res = res                                   # (H, W) tokens
            self.w = min(w, min(res))
            self.shift = shift if min(res) > self.w else 0
            self.padded = tuple(-(-r // self.w) * self.w for r in res)
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = WindowAttention(dim, n_heads, self.w)
            hidden = int(dim * config.mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.ada = nn.Linear(config.cond_embed, 6 * dim)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)  # adaLN-Zero: the block starts as the identity
            self.drop = drop
            self.stats = None  # set to a list to record residual update ratios (diagnostics)
            if self.shift:
                img = torch.zeros(1, *self.padded, 1)
                cuts = (slice(0, -self.w), slice(-self.w, -self.shift), slice(-self.shift, None))
                label = 0
                for hs in cuts:
                    for ws in cuts:
                        img[:, hs, ws, :] = label
                        label += 1
                windows = window_partition(img, self.w).squeeze(-1)
                diff = windows[:, None, :] - windows[:, :, None]
                self.register_buffer("mask", torch.where(diff != 0, -100.0, 0.0), persistent=False)
            else:
                self.mask = None

        def _drop_path(self, x):
            if not self.training or self.drop == 0.0:
                return x
            keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop).to(x.dtype)
            return x * keep / (1.0 - self.drop)

        def forward(self, x, c):
            b, n, ch = x.shape
            s1, g1, a1, s2, g2, a2 = self.ada(c)[:, None].chunk(6, dim=-1)
            h = self.norm1(x) * (1 + g1) + s1
            h = h.view(b, self.res[0], self.res[1], ch)
            ph, pw = self.padded
            if (ph, pw) != self.res:  # zero tokens up to whole windows (as in Swin for detection)
                h = F.pad(h, (0, 0, 0, pw - self.res[1], 0, ph - self.res[0]))
            if self.shift:
                h = torch.roll(h, (-self.shift, -self.shift), dims=(1, 2))
            h = self.attn(window_partition(h, self.w), self.mask)
            h = window_merge(h, self.w, ph, pw)
            if self.shift:
                h = torch.roll(h, (self.shift, self.shift), dims=(1, 2))
            h = h[:, :self.res[0], :self.res[1]]
            update = a1 * h.reshape(b, n, ch)
            if self.stats is not None:
                self.stats.append({"attn": float(update.norm() / (x.norm() + 1e-12))})
            x = x + self._drop_path(update)
            update = a2 * self.mlp(self.norm2(x) * (1 + g2) + s2)
            if self.stats is not None:
                self.stats[-1]["mlp"] = float(update.norm() / (x.norm() + 1e-12))
            return x + self._drop_path(update)

    class PatchMerging(nn.Module):
        def __init__(self, dim, res):
            super().__init__()
            self.res = res
            self.norm = nn.LayerNorm(4 * dim)
            self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

        def forward(self, x):
            b, _, c = x.shape
            x = x.view(b, self.res[0], self.res[1], c)
            x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
            return self.reduce(self.norm(x.view(b, -1, 4 * c)))

    def conv_block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    class FiLMResBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.norm1, self.norm2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
            self.film = nn.Linear(config.cond_embed, 2 * cout)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, c):
            h = F.gelu(self.norm1(self.conv1(x)))
            gamma, beta = self.film(c).chunk(2, -1)
            h = self.norm2(self.conv2(h)) * (1 + gamma[..., None, None]) + beta[..., None, None]
            return F.gelu(h + self.skip(x))

    class KeyboardNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            ce = config.cond_embed
            self.cond = nn.Sequential(nn.Linear(COND_DIM, ce), nn.GELU(), nn.Linear(ce, ce), nn.GELU())
            # high-resolution stem (skips for the decoder)
            in_planes = config.in_planes or IN_PLANES
            self.stem_full = nn.Sequential(conv_block(in_planes, 48), conv_block(48, 48))
            self.stem_half = nn.Sequential(nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                                           conv_block(96, 96))
            # Swin backbone
            self.patch = nn.Conv2d(in_planes, embed, config.patch, stride=config.patch)
            self.patch_norm = nn.LayerNorm(embed)
            res = (config.grid[0] // config.patch, config.grid[1] // config.patch)
            rates = np.linspace(0, config.drop_path, total_blocks).tolist()
            self.stages, self.merges, self.stage_norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            self.resolutions = []
            index = 0
            for i, (depth, n_heads) in enumerate(zip(depths, heads)):
                blocks = nn.ModuleList([SwinBlock(dims[i], n_heads, res, config.window,
                                                  0 if j % 2 == 0 else config.window // 2, rates[index + j])
                                        for j in range(depth)])
                index += depth
                self.stages.append(blocks)
                self.stage_norms.append(nn.LayerNorm(dims[i]))
                self.resolutions.append(res)
                if i < 3:
                    self.merges.append(PatchMerging(dims[i], res))
                    res = (res[0] // 2, res[1] // 2)
            # UPerNet head: PPM on the last stage + FPN
            d = config.head_dim
            self.lateral = nn.ModuleList([nn.Conv2d(dims[i], d, 1) for i in range(3)])
            self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(dims[3], d // 4, 1), nn.GELU())
                                      for s in (1, 2, 4)])
            self.ppm_fuse = conv_block(dims[3] + 3 * (d // 4), d)
            self.fpn = nn.ModuleList([conv_block(d, d) for _ in range(3)])
            self.fuse = conv_block(4 * d, d)
            self.fuse_film = FiLMResBlock(d, d)
            r_half, r_full = config.refine_dims
            self.refine_half = nn.ModuleList([FiLMResBlock(d + 96, r_half), FiLMResBlock(r_half, r_half)])
            self.refine_full = nn.ModuleList([FiLMResBlock(r_half + 48, r_full), FiLMResBlock(r_full, r_full)])
            self.out = nn.Conv2d(r_full, 1 + len(AUX_TARGETS), 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, planes, cond):
            c = self.cond(cond)
            full = self.stem_full(planes)
            half = self.stem_half(full)
            x = self.patch(planes)
            b = x.shape[0]
            x = self.patch_norm(x.flatten(2).transpose(1, 2))
            features = []
            for i, blocks in enumerate(self.stages):
                for block in blocks:
                    x = block(x, c)
                res = self.resolutions[i]
                features.append(self.stage_norms[i](x).transpose(1, 2).reshape(b, dims[i], *res))
                if i < 3:
                    x = self.merges[i](x)
            top = features[3]
            pooled = [F.interpolate(p(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for p in self.ppm]
            levels = [None, None, None, self.ppm_fuse(torch.cat([top] + pooled, 1))]
            for i in (2, 1, 0):
                up = F.interpolate(levels[i + 1], size=features[i].shape[-2:], mode="bilinear", align_corners=False)
                levels[i] = self.fpn[i](self.lateral[i](features[i]) + up)
            size = levels[0].shape[-2:]
            fused = self.fuse(torch.cat([levels[0]] + [F.interpolate(l, size=size, mode="bilinear", align_corners=False)
                                                       for l in levels[1:]], 1))
            h = self.fuse_film(fused, c)
            h = F.interpolate(h, size=half.shape[-2:], mode="bilinear", align_corners=False)
            h = self.refine_half[0](torch.cat([h, half], 1), c)
            h = self.refine_half[1](h, c)
            h = F.interpolate(h, size=full.shape[-2:], mode="bilinear", align_corners=False)
            h = self.refine_full[0](torch.cat([h, full], 1), c)
            h = self.refine_full[1](h, c)
            out = self.out(h)[:, :, :config.rows, :config.frames]
            limit = config.max_correction_db / DB_SCALE
            envelope = planes[:, BASE_PLANE, :config.rows, :config.frames] + limit * torch.tanh(out[:, 0] / limit)
            self.last_features = h  # full-resolution decoder features (used by KeyboardNet v3)
            return envelope, out[:, 1:]

    return KeyboardNetV2()


def save_keyboard_v2(path, model, extra=None):
    import torch
    torch.save({"config": asdict(model.config), "state": model.state_dict(), "extra": extra or {}}, Path(path))


def load_keyboard_v2(path):
    import torch
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    cfg["refine_dims"] = tuple(cfg["refine_dims"])
    if "grid" in cfg:
        cfg["grid"] = tuple(cfg["grid"])
    model = build_model_v2(KeyboardV2Config(**cfg))
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload.get("extra", {})


def predict_envelope_v2(model, examples, example_midis, target_midi):
    """Predicted target envelope (dB) and the baseline (dB) from 1-3 rich example feature stacks."""
    import torch
    planes, cond, base = network_inputs_v2(list(examples), list(example_midis), int(target_midi))
    with torch.no_grad():
        envelope, _ = model(torch.from_numpy(planes)[None], torch.from_numpy(cond)[None])
    return envelope[0].numpy() * DB_SCALE, base


def count_parameters(model) -> dict:
    groups = {"backbone": 0, "head": 0, "stem": 0, "conditioning": 0}
    for name, p in model.named_parameters():
        key = ("stem" if name.startswith("stem") else "conditioning" if name.startswith("cond") else
               "backbone" if name.startswith(("patch", "stages", "merges", "stage_norms")) else "head")
        groups[key] += p.numel()
    groups["total"] = sum(groups.values())
    return groups
