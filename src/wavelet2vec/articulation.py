"""Articulation (syllables and vowels) for MorphNet: targets, a frame-wise judge, and automation curves.

The voice in a hybrid should keep recognizable syllables and vowels, and at
inference the vowels should be controllable over time, like an automation
lane in a DAW. Articulation is a distribution per 10 ms frame over
``ART_CLASSES``: silence, the six vowels of
:mod:`wavelet2vec.tools.vocal_detector` (i, e, eh, a, o, u), sibilant,
plosive and breath.

- ``articulation_targets``: soft targets from the algorithmic vocal
  detector. Vowel frames get a softmax over its template-fit errors, because
  neighbouring vowels are confused first.
- ``ArticulationJudge``: a frame-wise classifier on the judge view (mid,
  below 18 kHz, level re peak). It keeps full time resolution and pools over
  frequency. It is trained on vocal notes and reads the articulation of any
  sound, hybrids included.
- ``vowel_automation`` / ``morph_vowels``: build target curves for inference,
  e.g. "a" at 0 s gliding to "o" at 1 s. They are applied on the frames
  where the voice sings a vowel; consonants and silences of the source are
  kept.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ART_CLASSES = ("silence", "i", "e", "eh", "a", "o", "u", "sibilant", "plosive", "breath")
VOWELS = ("i", "e", "eh", "a", "o", "u")
VOWEL_INDEX = [ART_CLASSES.index(v) for v in VOWELS]
FRAMES = 120
HOP_S = 0.01


def articulation_targets(track, frames: int = FRAMES, hop_s: float = HOP_S, temperature: float = 0.35) -> np.ndarray:
    """``[len(ART_CLASSES), frames]`` soft articulation per 10 ms frame from a ``VocalTrack``."""
    k = len(ART_CLASSES)
    dist = np.zeros((len(track.times_s), k))
    vowel_cols = [list(track.vowels).index(v) if v in track.vowels else None for v in VOWELS]
    for i, part in enumerate(track.part):
        if part == "silence":
            dist[i, 0] = 1
        elif part == "sibilant":
            dist[i, ART_CLASSES.index("sibilant")] = 1
        elif part == "plosive":
            dist[i, ART_CLASSES.index("plosive")] = 1
        elif part == "breath":
            dist[i, ART_CLASSES.index("breath")] = 1
        else:                                   # vowel, voiced, transition, other
            err = np.array([track.vowel_errors[i, c] if c is not None else np.nan for c in vowel_cols])
            if np.isfinite(err).any():
                best = np.nanmin(err)
                logits = -(np.where(np.isfinite(err), err, np.inf) - best) / (temperature * best + 1e-6)
                p = np.exp(logits - logits.max())
                dist[i, VOWEL_INDEX] = p / p.sum()
            elif part in ("vowel", "voiced"):
                dist[i, VOWEL_INDEX] = 1 / len(VOWELS)
            else:
                dist[i, ART_CLASSES.index("breath")] = 1
    centres = (np.arange(frames) + 0.5) * hop_s
    out = np.zeros((frames, k))
    out[:, 0] = 1.0
    if len(track.times_s):
        idx = np.clip(np.searchsorted(track.times_s, centres), 0, len(track.times_s) - 1)
        prev = np.clip(idx - 1, 0, len(track.times_s) - 1)
        nearest = np.where(np.abs(track.times_s[prev] - centres) < np.abs(track.times_s[idx] - centres), prev, idx)
        inside = (centres >= track.times_s[0] - hop_s) & (centres <= track.times_s[-1] + hop_s)
        out[inside] = dist[nearest[inside]]
    return out.T.astype(np.float32)


def vowel_frames(target: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean ``[frames]``: frames whose target is mostly a vowel."""
    return target[VOWEL_INDEX].sum(0) > threshold


def vowel_automation(points, frames: int = FRAMES, hop_s: float = HOP_S, sharpness: float = 0.9) -> np.ndarray:
    """A vowel curve ``[len(VOWELS), frames]`` from automation points ``[(time_s, vowel or {vowel: weight}), ...]``,
    linearly interpolated between points and held before the first / after the last. ``sharpness`` is the
    weight of the named vowel (the rest is spread over the others: the judge's view of a sung vowel)."""
    def as_vector(v):
        if isinstance(v, str):
            vec = np.full(len(VOWELS), (1 - sharpness) / (len(VOWELS) - 1))
            vec[VOWELS.index(v)] = sharpness
            return vec
        vec = np.array([float(v.get(name, 0.0)) for name in VOWELS])
        return vec / max(vec.sum(), 1e-9)

    points = sorted(points, key=lambda p: p[0])
    times = np.array([p[0] for p in points], dtype=np.float64)
    vectors = np.stack([as_vector(p[1]) for p in points])
    centres = (np.arange(frames) + 0.5) * hop_s
    curve = np.stack([np.interp(centres, times, vectors[:, j]) for j in range(len(VOWELS))])
    return (curve / curve.sum(0, keepdims=True)).astype(np.float32)


def apply_vowels(target: np.ndarray, vowel_curve: np.ndarray, amount=1.0) -> np.ndarray:
    """Replace the vowel part of an articulation ``target`` by ``vowel_curve`` on the source's vowel frames
    (``amount`` 0 keeps the source's vowels, 1 uses the curve; may be an array per frame)."""
    out = target.copy()
    mask = vowel_frames(target)
    mass = target[VOWEL_INDEX].sum(0)
    amount = np.broadcast_to(np.asarray(amount, dtype=np.float32), (target.shape[1],))
    mixed = (1 - amount) * target[VOWEL_INDEX] + amount * vowel_curve * mass
    out[VOWEL_INDEX] = np.where(mask, mixed, target[VOWEL_INDEX])
    return out


def morph_vowels(target: np.ndarray, first: str, last: str, start_s=None, end_s=None, hop_s: float = HOP_S):
    """A glide from vowel ``first`` to ``last`` over the source's vowel frames (or between the given times)."""
    mask = vowel_frames(target)
    if mask.any():
        idx = np.nonzero(mask)[0]
        start_s = idx[0] * hop_s if start_s is None else start_s
        end_s = (idx[-1] + 1) * hop_s if end_s is None else end_s
    else:
        start_s, end_s = start_s or 0.0, end_s or target.shape[1] * hop_s
    return apply_vowels(target, vowel_automation([(start_s, first), (max(end_s, start_s + hop_s), last)],
                                                 target.shape[1], hop_s))


class ArticulationJudge(nn.Module):
    """Frame-wise articulation logits ``[B, len(ART_CLASSES), frames]`` from the judge view
    ``[B, 4, rows, frames]``: convolutions that halve frequency (not time), frequency pooling, then a
    dilated temporal stack."""

    def __init__(self, planes: int = 4, width: int = 48):
        super().__init__()
        layers = [nn.Conv2d(planes + 1, width, 3, padding=1), nn.GELU()]
        for _ in range(4):
            layers += [nn.Conv2d(width, width, 3, stride=(2, 1), padding=1), nn.GroupNorm(8, width), nn.GELU()]
        self.spectral = nn.Sequential(*layers)
        self.temporal = nn.Sequential(nn.Conv1d(width, 2 * width, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(2 * width, 2 * width, 5, padding=4, dilation=2), nn.GELU(),
                                      nn.Conv1d(2 * width, 2 * width, 5, padding=8, dilation=4), nn.GELU(),
                                      nn.Conv1d(2 * width, len(ART_CLASSES), 1))

    def forward(self, view):
        rows = torch.linspace(-1, 1, view.shape[2], device=view.device, dtype=view.dtype)
        x = torch.cat([view, rows[None, None, :, None].expand(view.shape[0], 1, -1, view.shape[3])], 1)
        return self.temporal(self.spectral(x).mean(2))


def articulation_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy per frame, weighted (``[B, frames]``) and normalized by the weight."""
    ce = -(target * F.log_softmax(logits, dim=1)).sum(1)
    return (ce * weight).sum() / weight.sum().clamp(min=1e-6)
