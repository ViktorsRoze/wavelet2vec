"""Deterministic decompositions in the coefficient domain (roadmap P-1, G-7).

Both decompositions are **exact by construction**: layers are produced by
complementary masks (summing to 1) applied to the same signal, so
``layer_a + layer_b == original`` to machine precision. They are the
no-model fallbacks for the tonality and transient imagination axes — and
useful tools in their own right.

- :func:`hpss` — harmonic/percussive split (median masks on the scalogram).
- :func:`transient_split` / :func:`shape_transients` — transient/sustain
  split via an onset-flux gate, and the classic transient-designer remix.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from wavelet2vec.frame import analyze, synthesize

_ENVELOPE_RATE = 200.0  # Hz — masks are smooth; computing them decimated is exact enough


def _envelopes(coeffs: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    hop = max(int(sample_rate / _ENVELOPE_RATE), 1)
    n_frames = coeffs.shape[1] // hop
    mag = np.abs(coeffs[:, : n_frames * hop]).reshape(coeffs.shape[0], n_frames, hop).mean(axis=2)
    return mag.astype(np.float64), hop


def _upsample_mask(mask_frames: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """Linear-interpolate a [bands, frames] mask to [bands, n_samples]."""
    n_frames = mask_frames.shape[1]
    frame_centers = (np.arange(n_frames) + 0.5) * hop
    positions = np.arange(n_samples)
    out = np.empty((mask_frames.shape[0], n_samples))
    for band in range(mask_frames.shape[0]):
        out[band] = np.interp(positions, frame_centers, mask_frames[band])
    return out


def hpss(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 96,
    harmonic_seconds: float = 0.15,
    percussive_bands: int = 13,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic/percussive separation; ``harmonic + percussive == audio`` exactly.

    Median-filters the (decimated) magnitude scalogram along time (enhances
    steady harmonics) and along frequency (enhances broadband hits), builds
    complementary soft masks, applies them to the complex coefficients, and
    inverts. Returns ``(harmonic, percussive)`` at the input length.
    """
    audio = np.asarray(audio, dtype=np.float64)
    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands)
    envelopes, hop = _envelopes(coeffs, sample_rate)

    time_kernel = max(3, int(harmonic_seconds * _ENVELOPE_RATE) | 1)
    band_kernel = max(3, percussive_bands | 1)
    harmonic_strength = median_filter(envelopes, size=(1, time_kernel))
    percussive_strength = median_filter(envelopes, size=(band_kernel, 1))

    hp = harmonic_strength**power
    pp = percussive_strength**power
    mask_frames = hp / (hp + pp + 1e-24)
    mask = _upsample_mask(mask_frames, hop, coeffs.shape[1])

    harmonic = synthesize(coeffs * mask, spec)
    percussive = synthesize(coeffs * (1.0 - mask), spec)
    return harmonic[: audio.shape[0]], percussive[: audio.shape[0]]


def _onset_gate(
    audio: np.ndarray,
    sample_rate: int,
    *,
    attack_ms: float,
    decay_ms: float,
    n_bands: int = 48,
) -> np.ndarray:
    """A 0..1 gate that opens fast at onsets and decays over ``decay_ms``."""
    coeffs, _spec = analyze(audio, sample_rate, n_bands=n_bands)
    envelopes, hop = _envelopes(coeffs, sample_rate)
    # Prepend zeros so a sound starting at sample 0 registers as an onset —
    # otherwise the first milliseconds of an attack escape the gate.
    flux = np.diff(np.log1p(envelopes), axis=1, prepend=np.zeros((envelopes.shape[0], 1)))
    novelty = np.clip(flux, 0.0, None).sum(axis=0)
    scale = float(novelty.max())
    if scale <= 1e-12:
        return np.zeros(audio.shape[0])
    novelty = np.clip(novelty / scale, 0.0, 1.0)

    frame_rate = sample_rate / hop
    attack_frames = max(1, int(attack_ms / 1000.0 * frame_rate))
    decay_coef = float(np.exp(-1.0 / max(decay_ms / 1000.0 * frame_rate, 1e-6)))
    gate = np.zeros_like(novelty)
    level = 0.0
    for index, value in enumerate(novelty):
        target = value
        if target > level:
            level = min(1.0, level + (target - level) / attack_frames)
        else:
            level *= decay_coef
        gate[index] = level

    positions = np.arange(audio.shape[0])
    frame_centers = (np.arange(gate.shape[0]) + 0.5) * hop
    return np.interp(positions, frame_centers, gate)


def transient_split(
    audio: np.ndarray,
    sample_rate: int,
    *,
    attack_ms: float = 8.0,
    decay_ms: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits into (transient, sustain); their sum is the input exactly.

    The transient layer is the audio inside an onset-triggered gate window
    (fast attack, ~``decay_ms`` release); the sustain layer is the
    complement. Deterministic; the no-model fallback of the transient axis.
    """
    audio = np.asarray(audio, dtype=np.float64)
    gate = _onset_gate(audio, sample_rate, attack_ms=attack_ms, decay_ms=decay_ms)
    transient = audio * gate
    return transient, audio - transient


def shape_transients(
    audio: np.ndarray,
    sample_rate: int,
    *,
    attack_gain: float = 1.0,
    sustain_gain: float = 1.0,
    attack_ms: float = 8.0,
    decay_ms: float = 70.0,
) -> np.ndarray:
    """The classic transient designer: remix the two layers.

    ``attack_gain > 1`` = more punch; ``attack_gain < 1`` = softer attacks
    (pad-ify); ``sustain_gain < 1`` = tighter, drier; gains of 1.0 return the
    input bit-exactly.
    """
    transient, sustain = transient_split(
        audio, sample_rate, attack_ms=attack_ms, decay_ms=decay_ms
    )
    return transient * float(attack_gain) + sustain * float(sustain_gain)
