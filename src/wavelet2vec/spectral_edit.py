"""Spectral editing in the invertible coefficient domain (roadmap Phase-3 core).

``attenuate_region`` scales a time × frequency rectangle of the complex
coefficients by a gain (down to erasure) with raised-cosine aprons in both
directions, then inverts exactly.

Exactness contract (stated honestly): coefficients outside the masked cells
are untouched, and synthesis is linear — so the output differs from the
input only within the edited rectangle **plus the wavelet support of the
edited bands** (Gaussian tails: ≲ milliseconds for high bands, a few tens of
ms for the lowest). Beyond that apron the difference is numerically zero
(< 1e-6, i.e. below −120 dBFS). This is physics, not implementation slack —
no time-frequency editor can do better.
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.frame import analyze, synthesize


def _time_weight(n: int, start: int, end: int, fade: int) -> np.ndarray:
    weight = np.zeros(n)
    start, end = max(0, start), min(n, end)
    if end <= start:
        return weight
    weight[start:end] = 1.0
    fade = min(fade, (end - start) // 2)
    if fade > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        lo = max(0, start - 0)  # ramps live inside the rectangle
        weight[start : start + fade] = ramp
        weight[end - fade : end] = ramp[::-1]
    return weight


def _band_weight(centers: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """1 inside [f_lo, f_hi], half-cosine rolloff over one band spacing outside."""
    ratio = float(centers[1] / centers[0]) if len(centers) > 1 else 2.0 ** (1 / 12)
    log_span = np.log(ratio)
    position = np.log(centers)
    lo, hi = np.log(f_lo), np.log(f_hi)
    weight = np.zeros_like(position)
    inside = (position >= lo) & (position <= hi)
    weight[inside] = 1.0
    below = (position < lo) & (position > lo - log_span)
    weight[below] = 0.5 + 0.5 * np.cos(np.pi * (lo - position[below]) / log_span)
    above = (position > hi) & (position < hi + log_span)
    weight[above] = 0.5 + 0.5 * np.cos(np.pi * (position[above] - hi) / log_span)
    return weight


def region_mask(
    centers: np.ndarray,
    n: int,
    sample_rate: int,
    regions: list[tuple[float, float, float, float]],
    time_fade_ms: float = 5.0,
) -> np.ndarray:
    """Union soft mask ``[bands, N]`` in 0..1 over several T-F rectangles.

    Each region is ``(start_s, end_s, f_lo_hz, f_hi_hz)``; raised-cosine
    aprons in time, half-cosine band rolloff in frequency — the same
    contract as :func:`attenuate_region`, generalized to region lists
    (rasterized polygons, brush strokes).
    """
    fade = int(time_fade_ms / 1000.0 * sample_rate)
    mask = np.zeros((len(centers), n))
    for start_s, end_s, f_lo_hz, f_hi_hz in regions:
        start = int(round(start_s * sample_rate))
        end = int(round(end_s * sample_rate))
        f_lo = max(min(f_lo_hz, f_hi_hz), float(centers[0]))
        f_hi = min(max(f_lo_hz, f_hi_hz), float(centers[-1]))
        if end <= start or f_hi <= f_lo:
            continue
        rect = np.outer(_band_weight(centers, f_lo, f_hi), _time_weight(n, start, end, fade))
        np.maximum(mask, rect, out=mask)
    return mask


def isolate_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float, float, float]],
    *,
    n_bands: int = 128,
    time_fade_ms: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits audio into ``(inside, outside)`` of the masked regions.

    Synthesis is linear, so ``inside + outside == audio`` to reconstruction
    precision — process the inside through any effect and add the outside
    back for region-limited processing.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    inside = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        mask = region_mask(spec.centers, n, sample_rate, regions, time_fade_ms)
        inside[index] = synthesize(coeffs * mask, spec)[:n]
    outside = channels - inside
    if mono_input:
        return inside[0], outside[0]
    return inside, outside


def process_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float, float, float]],
    effect,
    *,
    n_bands: int = 128,
    time_fade_ms: float = 5.0,
) -> np.ndarray:
    """Runs ``effect(inside_audio) -> same-length audio`` on the region
    content only; everything outside the regions passes through untouched
    (by linearity of the split)."""
    inside, outside = isolate_regions(
        audio, sample_rate, regions, n_bands=n_bands, time_fade_ms=time_fade_ms
    )
    processed = np.asarray(effect(inside), dtype=np.float64)
    if processed.shape != inside.shape:
        raise ValueError("The effect must preserve the audio shape.")
    return outside + processed


def phase_rotate_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float, float, float]],
    degrees: float,
    *,
    n_bands: int = 128,
    time_fade_ms: float = 5.0,
) -> np.ndarray:
    """Rotates the phase of the masked coefficients by ``degrees``.

    The frame is redundant, so synthesis projects the rotated coefficients
    back to the nearest consistent signal — moderate rotations behave as
    expected (180° flips the region content's sign), with the mask's soft
    edges preventing clicks.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    rotation = np.exp(1j * np.radians(degrees))
    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        mask = region_mask(spec.centers, n, sample_rate, regions, time_fade_ms)
        out[index] = synthesize(coeffs * (1.0 + (rotation - 1.0) * mask), spec)[:n]
    return out[0] if mono_input else out


def phase_scramble_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float, float, float]],
    *,
    block_ms: float = 15.0,
    seed: int = 0,
    n_bands: int = 128,
    time_fade_ms: float = 5.0,
) -> np.ndarray:
    """Randomizes phase inside the regions (per band, per ~``block_ms``
    block) — magnitudes stay, timing coherence dissolves: transients smear
    into texture while the spectral color is preserved."""
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    block = max(int(block_ms / 1000.0 * sample_rate), 1)
    n_blocks = n // block + 1
    rng = np.random.default_rng(seed)
    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        mask = region_mask(spec.centers, n, sample_rate, regions, time_fade_ms)
        angles = rng.uniform(-np.pi, np.pi, size=(coeffs.shape[0], n_blocks))
        rotation = np.exp(1j * np.repeat(angles, block, axis=1)[:, :n])
        out[index] = synthesize(coeffs * (1.0 + (rotation - 1.0) * mask), spec)[:n]
    return out[0] if mono_input else out


def attenuate_region(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    f_lo_hz: float,
    f_hi_hz: float,
    gain_db: float = -80.0,
    *,
    n_bands: int = 128,
    time_fade_ms: float = 5.0,
) -> np.ndarray:
    """Returns audio with the T-F rectangle attenuated by ``gain_db``.

    ``audio``: mono ``[N]`` or multichannel ``[C, N]`` (each channel gets the
    same mask). ``gain_db=-80`` is a practical erase; ``-12`` a repair duck.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio

    n = channels.shape[1]
    start = int(round(start_s * sample_rate))
    end = int(round(end_s * sample_rate))
    fade = int(time_fade_ms / 1000.0 * sample_rate)
    gain = 10.0 ** (gain_db / 20.0)

    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        f_lo = max(f_lo_hz, float(spec.centers[0]))
        f_hi = min(f_hi_hz, float(spec.centers[-1]))
        mask = np.outer(_band_weight(spec.centers, f_lo, f_hi), _time_weight(n, start, end, fade))
        coeffs *= 1.0 - (1.0 - gain) * mask
        out[index] = synthesize(coeffs, spec)[:n]
    return out[0] if mono_input else out
