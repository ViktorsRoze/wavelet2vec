"""MorphNet-L: vox-synth hybrids with a pretrained Swin-L encoder (MorphNet v2).

Same task as :mod:`wavelet2vec.morph` (note_dataset, second purpose), built
from the lessons of the small run (docs/MORPH_MODEL.md):

- Encoder: KeyboardNet v3's Swin-L backbone (217 M parameters), warm-started
  from its checkpoint, shared by the vocal and the synth note. It reads the
  v3 input planes of each note (seven feature planes of the exact cells) and
  gives four scales: 4/8/16/32 semitones x 40/80/160/320 ms per token.
  v3's pretrained stems and its embedding of every exact sample give full and
  half resolution.
- Fusion (new): joint-attention transformers over vocal + synth tokens at
  the three coarse scales (depths 4 / 8 / 2), with adaLN-Zero conditioning
  on the vocal amount alpha; the finest scale and the stems are mixed by
  alpha-conditioned 1x1 convolutions. Every mixer starts as the average of
  the two notes.
- Decoder: v3's pretrained UPerNet head (pyramid pooling, FPN, full-
  resolution refinement), conditioned on the note and alpha.
- Output, stable pitch by construction: every hybrid has one **pitch
  source**, the synth or the vocal note (a control input like alpha; training
  uses both). Each exact sample is
      (1 - m) * A * U  +  m * N
  - Harmonic path: A is a learned, time-varying combination of both notes'
    sample-level **magnitudes** (gains per cell, interpolated inside cells,
    plus a phase-locked residual magnitude). U is the unit phasor of the
    pitch source: every partial keeps the pitch source's own phase, so it has
    no beating and no frequency modulation, and its pitch is as stable as the
    source's. Where the pitch source is more than 30 dB below the other note,
    the other note's phase takes over (smooth crossfade).
  - Noise path: N combines both notes' cells with their own phases, plus a
    learned residual. It is used where the source cells are noise-like
    (breath, consonants, noisy synths).
  - m is 0 for coherent (tonal) cells and 1 for noise-like cells, from the
    phase-increment coherence of the source cells (``noise_mask``). With the
    default ``noise_gate="cell"`` the one-sample cells below 440 Hz always
    read as coherent, so m = 0 there. ``noise_gate="scale"`` (opt-in, new
    runs) takes their coherence from wavelet2vec.semantic_bridge and keeps
    the note's harmonic rows tonal (m = 0), so pitch stays stable.

  The result is decoded exactly to 192 kHz stereo.
- Articulation (syllables, vowels): an optional per-frame target
  ``[B, 10, 120]`` over :data:`wavelet2vec.articulation.ART_CLASSES`. It is
  embedded and added at every scale: to the vocal note's encoder features
  before fusion, to the fused features, and to the full-resolution features
  of the output heads. All zero-initialized. Training asks the hybrid to
  keep the voice's own articulation, or to sing other vowels / glide between
  vowels, so at inference the vowels can be automated over time.
- ``critic_view``: the full-band critic's stereo view (mid and side
  levels, coherence, modulation; rows up to 20 kHz).
- ``vocality``: position of a judge embedding on the axis from the synth
  centre (0) to the vox centre (1): a graded measure that, unlike a
  classifier's logit, does not saturate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wavelet2vec.articulation import ART_CLASSES
from wavelet2vec.keyboard_v2 import IN_PLANES, KeyboardV2Config
from wavelet2vec.keyboard_v3 import CARRIERS, FRAMES, GRID, ROWS, KeyboardNetV3, cell_features, inputs_v3
from wavelet2vec.morph import (BANDS, CORE, GROUPS, SPEC, TOTAL, VIEW_STATS, _FusionBlock, _fourier,
                               _group_stats, _rows_index, _view_primitives, power)
from wavelet2vec.wavelet_image import WaveletImage

CRITIC_TOP_HZ = 20000.0
CRITIC_ROWS = sum(1 for b in BANDS if b.center_hz + 3 * b.sigma_hz <= CRITIC_TOP_HZ)
CRITIC_PLANES = 5
CRITIC_FLOOR_DB = -60.0
PITCH_SOURCES = ("synth", "vocal")
MATERIALS = ("synth", "vocal")
PATHS = ("harmonic", "noise")
FALLBACK_DB = 30.0           # the other note's phase takes over below this (pitch source quieter by > 30 dB)
TONAL_COHERENCE = 0.75       # phase-increment coherence above which a cell is fully tonal (m = 0) ...
NOISE_COHERENCE = 0.40       # ... and below which it is fully noise-like (m = 1)


def critic_view(image: WaveletImage, stats: str = "cell") -> torch.Tensor:
    """``[B, 5, CRITIC_ROWS, FRAMES]``: mid level and side level re the note's mid peak (floored at -60 dB,
    scaled to [-1, 1]), mid phase-increment coherence re/im and modulation depth (faded out near the floor).
    ``stats``: see ``wavelet2vec.morph.VIEW_STATS``; ``judge_view(image, stats)`` is an exact slice of it."""
    prim = _view_primitives(image, stats)
    first = next(iter(image.groups.values()))
    b, device, dtype = first.shape[0], first.device, first.real.dtype
    mid = torch.full((b, CRITIC_ROWS, TOTAL), -300.0, device=device, dtype=dtype)
    side, ure, uim, mod = mid.clone(), *(torch.zeros(b, CRITIC_ROWS, TOTAL, device=device, dtype=dtype) for _ in range(3))
    for c, values in image.groups.items():
        rows = _rows_index(image, c, device)
        keep = rows < CRITIC_ROWS
        if not bool(keep.any()):
            continue
        r = rows[keep]
        mean_p, side_p, re, im, md = _group_stats(values[:, :, keep], c, r, prim)
        mid = mid.index_copy(1, r, 10 * torch.log10(mean_p + 1e-30))
        side = side.index_copy(1, r, 10 * torch.log10(side_p + 1e-30))
        ure, uim, mod = ure.index_copy(1, r, re), uim.index_copy(1, r, im), mod.index_copy(1, r, md)
    mid, side, ure, uim, mod = (x[..., CORE] for x in (mid, side, ure, uim, mod))
    peak = mid.amax(dim=(1, 2), keepdim=True)
    rel = torch.clamp(mid - peak, min=CRITIC_FLOOR_DB)
    rel_side = torch.clamp(side - peak, min=CRITIC_FLOOR_DB)
    weight = torch.clamp((rel - CRITIC_FLOOR_DB) / 12.0, 0.0, 1.0)
    scale = -CRITIC_FLOOR_DB / 2
    return torch.stack([rel / scale + 1, rel_side / scale + 1, ure * weight, uim * weight, mod / 3 * weight], 1)


class FeatureCritic(nn.Module):
    """Spectrally normalized CNN critic that also returns its intermediate features (feature matching)."""

    def __init__(self, planes, width=32):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        chans = [width, 2 * width, 4 * width, 8 * width]
        self.blocks = nn.ModuleList()
        cin = planes + 1
        for ch in chans:
            self.blocks.append(nn.Sequential(sn(nn.Conv2d(cin, ch, 3, padding=1)), nn.LeakyReLU(0.2),
                                             sn(nn.Conv2d(ch, ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2)))
            cin = ch
        self.out = sn(nn.Linear(chans[-1], 1))

    def forward(self, view):
        rows = torch.linspace(-1, 1, view.shape[2], device=view.device, dtype=view.dtype)
        x = torch.cat([view, rows[None, None, :, None].expand(view.shape[0], 1, -1, view.shape[3])], 1)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return self.out(x.mean((2, 3)))[:, 0], feats


def vocality(embedding: torch.Tensor, centre_vox: torch.Tensor, centre_synth: torch.Tensor) -> torch.Tensor:
    """Projection of (normalized) judge embeddings onto the synth -> vox centre axis: 0 = synth centre,
    1 = vox centre; not bounded, does not saturate."""
    e = F.normalize(embedding, dim=-1)
    axis = centre_vox - centre_synth
    return ((e - centre_synth) * axis).sum(-1) / (axis * axis).sum()


def backbone_inputs(image: WaveletImage, midis):
    """v3 input planes ``[B, 28, 160, 128]`` and conditioning ``[B, 8]`` of single notes at their own key."""
    feats = cell_features(image).float().cpu()
    planes, conds = [], []
    for k, midi in enumerate(midis):
        p, c, _, _ = inputs_v3([feats[k]], [int(midi)], int(midi))
        planes.append(p)
        conds.append(c)
    return torch.from_numpy(np.stack(planes)), torch.from_numpy(np.stack(conds))


class _Mix2d(nn.Module):
    """alpha-conditioned 1x1 mixing of the vocal and synth maps; starts as their average."""

    def __init__(self, ch, cond):
        super().__init__()
        self.mix = nn.Conv2d(2 * ch, ch, 1)
        self.film = nn.Linear(cond, 2 * ch)
        with torch.no_grad():
            eye = torch.eye(ch)[:, :, None, None] * 0.5
            self.mix.weight.copy_(torch.cat([eye, eye], 1))
            self.mix.bias.zero_()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, v, s, cond):
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, 1)
        return self.mix(torch.cat([v, s], 1)) * (1 + scale) + shift


class _Fusion(nn.Module):
    """Joint attention over vocal + synth tokens of one scale; output = average + learned (zero-init) merge."""

    def __init__(self, dim, tokens, depth, heads, cond):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
        self.kind = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.kind, std=0.02)
        self.cond = nn.Linear(cond, dim)
        self.blocks = nn.ModuleList(_FusionBlock(dim, heads) for _ in range(depth))
        self.merge = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)

    def forward(self, v, s, cond):
        b, d, h, w = v.shape
        tv = v.flatten(2).transpose(1, 2)
        ts = s.flatten(2).transpose(1, 2)
        x = torch.cat([tv + self.pos + self.kind[0], ts + self.pos + self.kind[1]], 1)
        c = self.cond(cond)
        for block in self.blocks:
            x = block(x, c)
        n = h * w
        out = 0.5 * (tv + ts) + self.merge(torch.cat([x[:, :n], x[:, n:]], -1))
        return out.transpose(1, 2).reshape(b, d, h, w)


class MorphNetL(nn.Module):
    def __init__(self, v3_state=None, v2_config: dict | None = None, depths=(4, 8, 2), alpha_dim=256,
                 residual_hidden=128, max_gain_db=30.0, residual_scale=0.5, noise_gate: str = "cell"):
        super().__init__()
        if noise_gate not in VIEW_STATS:
            raise ValueError(f"noise_gate must be one of {VIEW_STATS}, not {noise_gate!r}")
        self.noise_gate = noise_gate
        cfg = KeyboardV2Config(**{k: v for k, v in (v2_config or {"size": "large"}).items()})
        self.v3 = KeyboardNetV3(None, cfg)
        if v3_state is not None:
            self.v3.load_state_dict(v3_state)
        net = self.v3.net
        ncfg = net.config
        dims = [net.stage_norms[i].normalized_shape[0] for i in range(4)]
        cond_dim = ncfg.cond_embed
        self.max_gain_db, self.residual_scale = max_gain_db, residual_scale
        self.alpha = nn.Sequential(nn.Linear(16, alpha_dim), nn.SiLU(), nn.Linear(alpha_dim, cond_dim))
        nn.init.zeros_(self.alpha[2].weight)
        nn.init.zeros_(self.alpha[2].bias)
        self.pitch = nn.Embedding(len(PITCH_SOURCES), cond_dim)
        nn.init.zeros_(self.pitch.weight)
        heads = [max(1, d // 64) for d in dims]
        self.fusion = nn.ModuleList([_Fusion(dims[i], net.resolutions[i][0] * net.resolutions[i][1], depths[i - 1],
                                             heads[i], cond_dim) for i in (1, 2, 3)])
        self.mix0 = _Mix2d(dims[0], cond_dim)
        with torch.no_grad():                       # stem widths, from the pretrained stems themselves
            full = net.stem_full(torch.zeros(1, IN_PLANES, *GRID))
            half = net.stem_half(full)
        self.mix_full = _Mix2d(full.shape[1], cond_dim)
        self.mix_half = _Mix2d(half.shape[1], cond_dim)
        self.mix_detail = _Mix2d(self.v3.detail_dim, cond_dim)
        r_full = ncfg.refine_dims[1]
        self.gain = nn.Conv2d(r_full, len(PATHS) * len(MATERIALS) * 2, 1)    # path x material x (mid, side): log gain
        nn.init.zeros_(self.gain.weight)
        with torch.no_grad():
            limit = max_gain_db / 20 * math.log(10)
            self.gain.bias.fill_(limit * math.atanh(math.log(0.5) / limit))   # start: both materials at -6 dB
        # per sample and channel: harmonic magnitude (phase-locked) and noise (re, im)
        self.residual = nn.ModuleDict({str(c): nn.Sequential(nn.Linear(r_full, residual_hidden), nn.GELU(),
                                                             nn.Linear(residual_hidden, 6 * c)) for c in GROUPS})
        k = len(ART_CLASSES)
        self.art_pre = nn.ModuleList(nn.Linear(k, d) for d in dims)        # vocal features before fusion
        self.art_post = nn.ModuleList(nn.Linear(k, d) for d in dims)       # fused features
        self.art_full = nn.Linear(k, r_full)                               # full-resolution output features
        for layer in [*self.art_pre, *self.art_post, self.art_full]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in self.residual.values():
            nn.init.zeros_(layer[2].weight)
            nn.init.zeros_(layer[2].bias)

    def pretrained_parameters(self):
        return list(self.v3.parameters())

    def new_parameters(self):
        ids = {id(p) for p in self.pretrained_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    # ----------------------------------------------------------------------------- encoder (shared)

    def encode(self, planes, cond, image: WaveletImage):
        """Backbone features of a batch of single notes: stages, stems, conditioning, exact-sample detail."""
        net = self.v3.net
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
        return {"stages": stages, "full": full, "half": half, "cond": c, "detail": self.v3.detail_map(image)}

    # ----------------------------------------------------------------------------- fusion + decoder

    @staticmethod
    def _art_map(articulation, layer, frames):
        """``[B, K, 120]`` articulation -> ``[B, d, 1, frames]`` (time-pooled to the scale, then embedded)."""
        a = F.pad(articulation, (0, GRID[1] - articulation.shape[-1]))
        a = F.adaptive_avg_pool1d(a, frames)
        return layer(a.transpose(1, 2)).transpose(1, 2)[:, :, None, :]

    def decode(self, enc_v, enc_s, alpha, vocal: WaveletImage, synth: WaveletImage, pitch_source=None,
               articulation=None, midis=None):
        """Hybrid of the encoded notes. ``pitch_source`` per item: 0 = synth (default), 1 = vocal.
        ``articulation``: ``[B, 10, 120]`` per-frame target (see wavelet2vec.articulation), or None.
        ``midis``: the notes' keys; with ``noise_gate="scale"`` their harmonic rows stay tonal."""
        net = self.v3.net
        if pitch_source is None:
            pitch_source = torch.zeros(alpha.shape[0], dtype=torch.long)
        pitch_source = torch.as_tensor(pitch_source, dtype=torch.long)
        c = enc_v["cond"] + self.alpha(_fourier(alpha)) + self.pitch(pitch_source)
        vocal_stages = list(enc_v["stages"])
        if articulation is not None:
            articulation = articulation.to(vocal_stages[0].dtype)
            vocal_stages = [x + self._art_map(articulation, self.art_pre[i], x.shape[-1]) for i, x in enumerate(vocal_stages)]
        stages = [self.mix0(vocal_stages[0], enc_s["stages"][0], c)]
        for k, i in enumerate((1, 2, 3)):
            stages.append(self.fusion[k](vocal_stages[i], enc_s["stages"][i], c))
        if articulation is not None:
            stages = [x + self._art_map(articulation, self.art_post[i], x.shape[-1]) for i, x in enumerate(stages)]
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
        half = self.mix_half(enc_v["half"], enc_s["half"], c)
        full = self.mix_full(enc_v["full"], enc_s["full"], c)
        detail = self.mix_detail(enc_v["detail"], enc_s["detail"], c)
        h = F.interpolate(h, size=half.shape[-2:], mode="bilinear", align_corners=False)
        h = net.refine_half[1](net.refine_half[0](torch.cat([h, half], 1), c), c)
        h = F.interpolate(h, size=full.shape[-2:], mode="bilinear", align_corners=False)
        h = net.refine_full[1](net.refine_full[0](torch.cat([h, full, detail], 1), c), c)[:, :, :ROWS, :FRAMES]
        if articulation is not None:
            h = h + self.art_full(articulation.transpose(1, 2)).transpose(1, 2)[:, :, None, :]
        return self._render(h, vocal, synth, pitch_source, midis)

    def _render(self, h, vocal: WaveletImage, synth: WaveletImage, pitch_source, midis=None):
        """Stable-pitch rendering (see the module docstring): exact cells out."""
        b = h.shape[0]
        raw = self.gain(h)                                                          # [B, 8, ROWS, FRAMES]
        limit = self.max_gain_db / 20 * math.log(10)
        level = limit * torch.tanh(raw / limit)
        pad = (CORE.start, TOTAL - CORE.stop, 0, 0)
        level_t = F.pad(level, pad, mode="replicate")                               # context frames hold the edges
        h_t = F.pad(h, pad, mode="replicate")
        vocal_pitch = (pitch_source.to(h.device) == 1).reshape(b, 1, 1, 1, 1)
        peak_v = max(float(power(v).mean(-1).amax()) for v in vocal.groups.values())
        peak_s = max(float(power(v).mean(-1).amax()) for v in synth.groups.values())
        peak = max(peak_v, peak_s)
        gate = noise_gate_inputs(vocal, synth, self.noise_gate, midis, h.device)
        groups, residual_energy = {}, 0.0
        for c in vocal.groups:
            rows = _rows_index(vocal, c, h.device)
            r = len(rows)
            v, s = vocal.groups[c], synth.groups[c]                                 # [B, 2, R, T, c]
            g = torch.exp(_within(level_t[:, :, rows], c)).reshape(b, len(PATHS), len(MATERIALS), 2, r, TOTAL, c)
            mag_s = torch.sqrt(power(s) + 1e-12 * peak)
            mag_v = torch.sqrt(power(v) + 1e-12 * peak)
            # phase carrier: the pitch source's unit phasor, the other note's where the source is > 30 dB quieter
            source = torch.where(vocal_pitch, v, s)
            other = torch.where(vocal_pitch, s, v)
            ps, po = power(source).mean(-1), power(other).mean(-1)                  # [B, 2, R, T] cell powers
            weight = torch.sigmoid((10 * torch.log10(ps + 1e-30) - 10 * torch.log10(po + 1e-30) + FALLBACK_DB) / 4)
            weight = _within(weight, c)
            carrier = weight * source / torch.sqrt(power(source) + 1e-10 * peak) \
                + (1 - weight) * other / torch.sqrt(power(other) + 1e-10 * peak)
            unit = carrier / torch.sqrt(power(carrier) + 1e-12)
            # tonal (m = 0) or noise-like (m = 1) cells, from the sources' phase-increment coherence (mid)
            m = _within(noise_mask(v, s, c, rows, gate)[:, None], c)              # [B, 1, R, T, c]
            # learned residual: phase-locked magnitude and noise, relative to the local level
            res = self.residual[str(c)](h_t[:, :, rows].permute(0, 2, 3, 1)).reshape(b, r, TOTAL, 2, 3, c)
            res = res.permute(0, 3, 1, 2, 4, 5)                                     # [B, 2, R, T, 3, c]
            ref = torch.sqrt(torch.maximum(power(v).sum(1).mean(-1), power(s).sum(1).mean(-1))
                             + 1e-4 * peak)[:, None, :, :, None]
            harm_extra = self.residual_scale * ref * F.softplus(res[..., 0, :] - 4.0)
            noise_extra = self.residual_scale * ref * torch.complex(torch.tanh(res[..., 1, :]), torch.tanh(res[..., 2, :]))
            harmonic = (g[:, 0, 0] * mag_s + g[:, 0, 1] * mag_v + harm_extra) * unit
            noise = g[:, 1, 0] * s + g[:, 1, 1] * v + noise_extra
            groups[c] = (1 - m) * harmonic + m * noise
            extra = (1 - m) * harm_extra * unit + m * noise_extra
            residual_energy = residual_energy + power(extra[:, :, :, CORE]).sum((1, 2, 3, 4)) / c
        extras = {"gain_level": level, "residual_energy": residual_energy}
        return WaveletImage(vocal.spec, groups, dict(vocal.rows)), extras


def harmonic_row_mask(midis, device=None, harmonics: int = 24, reach: float = 0.75) -> torch.Tensor:
    """``[B, rows]`` bool: rows within ``reach`` semitones of one of the first ``harmonics`` partials of each key
    (a partial between two semitone bands marks both)."""
    rows = torch.arange(SPEC.n_bands, device=device, dtype=torch.float64)
    out = torch.zeros(len(midis), SPEC.n_bands, dtype=torch.bool, device=device)
    for i, midi in enumerate(midis):
        for k in range(1, harmonics + 1):
            out[i] |= (rows - (float(midi) - SPEC.first_midi + 12 * math.log2(k))).abs() < reach
    return out


def noise_gate_inputs(vocal: WaveletImage, synth: WaveletImage, noise_gate: str = "cell", midis=None, device=None):
    """What ``noise_mask`` needs beyond the cells: semantic-bridge primitives of both notes and the protected
    harmonic rows (``noise_gate="scale"``), or None (``"cell"``)."""
    prim_v, prim_s = _view_primitives(vocal, noise_gate), _view_primitives(synth, noise_gate)
    if prim_v is None:
        return None
    protect = None if midis is None else harmonic_row_mask([int(k) for k in midis], device)
    return prim_v, prim_s, protect


def noise_mask(v: torch.Tensor, s: torch.Tensor, c: int, rows: torch.Tensor, gate=None) -> torch.Tensor:
    """``[B, R, T]`` noise share m of one sample-count group: 0 for tonal cells, 1 for noise-like cells, from the
    power-weighted phase-increment coherence of both notes' mid cells (``v``, ``s``: ``[B, 2, R, T, c]``).
    ``gate``: ``noise_gate_inputs(..., "scale")`` output; one-sample rows then use the semantic-bridge coherence
    and the notes' harmonic rows stay tonal."""
    prim_v, prim_s, protect = gate if gate is not None else (None, None, None)
    pv_cell, _, re_v, im_v, _ = _group_stats(v, c, rows, prim_v)
    ps_cell, _, re_s, im_s, _ = _group_stats(s, c, rows, prim_s)
    coherence = (pv_cell * torch.sqrt(re_v ** 2 + im_v ** 2 + 1e-12)
                 + ps_cell * torch.sqrt(re_s ** 2 + im_s ** 2 + 1e-12)) / (pv_cell + ps_cell + 1e-30)
    m = torch.clamp((TONAL_COHERENCE - coherence) / (TONAL_COHERENCE - NOISE_COHERENCE), 0.0, 1.0)
    if protect is not None and c == 1:
        m = m * (~protect[:, rows])[:, :, None].to(m.dtype)
    return m


def _within(cells: torch.Tensor, c: int) -> torch.Tensor:
    """``[B, K, R, TOTAL]`` per-cell values -> ``[B, K, R, TOTAL, c]`` per sample, linearly interpolated
    between cell centres (no 10 ms steps)."""
    b, k, r, t = cells.shape
    x = F.interpolate(cells.reshape(b, k * r, t), size=t * c, mode="linear", align_corners=False)
    return x.reshape(b, k, r, t, c)


def pitch_jitter(image: WaveletImage, midis, harmonics=(1, 2, 3), floor_db=25.0) -> list:
    """Per note: energy-weighted spread (std, cents) of the instantaneous frequency of harmonics 1-3 of
    the note's key over the core frames where each harmonic is within ``floor_db`` of its peak. A steady
    synth reads near 0; vibrato, beating or warble raise it."""
    position = {row: (c, k) for c, rows in image.rows.items() for k, row in enumerate(rows)}
    out = []
    for item, midi in enumerate(midis):
        f0 = 440.0 * 2 ** ((int(midi) - 69) / 12)
        spreads = []
        for harmonic in harmonics:
            row = int(midi) - SPEC.first_midi + round(12 * math.log2(harmonic))
            if row >= ROWS:
                continue
            c, k = position[row]
            x = image.groups[c][item, 0, k][CORE].detach().to(torch.complex128)    # [F, c]
            flat = x.reshape(-1)
            inc = flat[1:] * flat[:-1].conj()
            inc = torch.cat([inc[:1], inc]).reshape(x.shape).sum(-1)
            p = power(x).mean(-1)
            keep = p > p.max() * 10 ** (-floor_db / 10)
            if int(keep.sum()) < 5:
                continue
            freq = float(CARRIERS[row]) + torch.angle(inc) * (c * 1000.0 / SPEC.hop_ms) / (2 * math.pi)
            cents = 1200 * torch.log2(torch.clamp(freq, min=1e-3) / (harmonic * f0))
            w, cc = p[keep], cents[keep]
            mean = (w * cc).sum() / w.sum()
            spreads.append(float(torch.sqrt((w * (cc - mean) ** 2).sum() / w.sum())))
        out.append(float(np.mean(spreads)) if spreads else float("nan"))
    return out


def _take(image: WaveletImage, index) -> WaveletImage:
    return WaveletImage(image.spec, {c: v[index] for c, v in image.groups.items()}, dict(image.rows))


def load_v3_backbone(path):
    """Weights and config of a KeyboardNet v3 checkpoint (``{epoch}_epoch.pth`` or ``best.pt``)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["state"], state["v2_config"]


def load_morph_large(path, map_location="cpu") -> "MorphNetL":
    """A trained MorphNet-L from a training checkpoint (``{epoch}_epoch.pth``, ``best.pt`` or ``last.pt``)."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model = MorphNetL(None, state.get("v2_config"), noise_gate=state.get("noise_gate", "cell"))
    model.load_state_dict(state["G"], strict=False)
    return model.eval()


@torch.no_grad()
def generate_hybrid(model: "MorphNetL", vocal_audio, synth_audio, midi: int, alpha: float = 0.5,
                    pitch_source: str = "synth", vowels=None, vowel_amount=1.0, sample_rate: int = 192000):
    """One hybrid note, 192 kHz stereo ``[2, samples]``.

    ``vocal_audio`` / ``synth_audio``: ``[2, samples]`` (or mono) at ``sample_rate``, the same key ``midi``;
    loudness-matched here. ``alpha``: vocal amount. ``pitch_source``: "synth" or "vocal" (whose pitch the
    hybrid keeps). ``vowels``: None keeps the voice's own syllables and vowels; a vowel name ("a") sings it
    on the voice's vowel frames; a list of automation points ``[(time_s, vowel or {vowel: weight}), ...]``
    glides between them (``vowel_amount`` 0..1, or per frame, blends with the voice's own vowels)."""
    from wavelet2vec.articulation import apply_vowels, articulation_targets, vowel_automation
    from wavelet2vec.audio_io import resample
    from wavelet2vec.tools.vocal_detector import detect_vocal_parts
    from wavelet2vec.wavelet_image import decode, encode

    def prepare(x):
        x = np.asarray(x, dtype=np.float64)
        x = np.stack([x, x]) if x.ndim == 1 else x[:2]
        if sample_rate != SPEC.sample_rate:
            x = resample(x, sample_rate, SPEC.sample_rate)
        out = np.zeros((2, SPEC.audio_length))
        out[:, :min(x.shape[1], SPEC.audio_length)] = x[:, :SPEC.audio_length]
        rms = np.sqrt((out ** 2).mean())
        return out * (0.1 / max(rms, 1e-9))

    vocal, synth = prepare(vocal_audio), prepare(synth_audio)
    target = articulation_targets(detect_vocal_parts(vocal.mean(0), SPEC.sample_rate))
    if vowels is not None:
        curve = vowel_automation([(0.0, vowels)] if isinstance(vowels, (str, dict)) else vowels)
        target = apply_vowels(target, curve, vowel_amount)
    image = encode(torch.from_numpy(np.stack([vocal, synth])).float(), SPEC)
    planes, cond = backbone_inputs(image, [midi, midi])
    enc = model.encode(planes, cond, image)
    split = lambda sl: {k: (v[sl] if torch.is_tensor(v) else [y[sl] for y in v]) for k, v in enc.items()}
    hybrid, _ = model.decode(split(slice(0, 1)), split(slice(1, 2)), torch.tensor([float(alpha)]),
                             _take(image, slice(0, 1)), _take(image, slice(1, 2)),
                             torch.tensor([PITCH_SOURCES.index(pitch_source)]), torch.from_numpy(target)[None],
                             midis=[midi])
    return decode(hybrid)[0].double().numpy()
