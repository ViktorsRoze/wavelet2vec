from __future__ import annotations

import numpy as np

from wavelet2vec.filterbank import analytic_envelope


def _envelope_shape(envelope: np.ndarray, n_points: int) -> np.ndarray:
    frame = max(envelope.shape[0] // n_points, 1)
    n_frames = envelope.shape[0] // frame
    pooled = envelope[: n_frames * frame].reshape(n_frames, frame).mean(axis=1)
    indices = np.linspace(0, pooled.shape[0] - 1, n_points)
    shape = np.interp(indices, np.arange(pooled.shape[0]), pooled)
    return (shape / (shape.max() + 1e-12)).astype(np.float32)


def _log_attack_time(envelope: np.ndarray, sample_rate: int) -> float:
    peak = envelope.max() + 1e-12
    above_low = np.flatnonzero(envelope >= 0.1 * peak)
    above_high = np.flatnonzero(envelope >= 0.9 * peak)
    if above_low.size == 0 or above_high.size == 0:
        return 0.0
    attack_seconds = max(above_high[0] - above_low[0], 0) / sample_rate
    # Maps ~0.001s (click) to 0 and ~2s (slow swell) to ~1.
    return float(np.clip((np.log10(attack_seconds + 1e-3) + 3.0) / 3.5, 0.0, 1.0))


def _decay_slope(envelope: np.ndarray, sample_rate: int) -> float:
    peak_index = int(envelope.argmax())
    tail = envelope[peak_index:]
    if tail.shape[0] < 8:
        return 0.0
    times = np.arange(tail.shape[0]) / sample_rate
    slope = np.polyfit(times, np.log(tail + 1e-6), deg=1)[0]
    return float(np.clip(-slope, 0.0, 50.0) / 50.0)


def _onset_curve(scalogram: np.ndarray, sample_rate: int, frame_seconds: float = 0.01) -> np.ndarray:
    hop = max(int(sample_rate * frame_seconds), 1)
    n_frames = scalogram.shape[1] // hop
    if n_frames < 2:
        return np.zeros(1, dtype=np.float32)
    framed = scalogram[:, : n_frames * hop].reshape(scalogram.shape[0], n_frames, hop).mean(axis=2)
    flux = np.diff(np.log1p(framed), axis=1).clip(min=0.0).sum(axis=0)
    return flux.astype(np.float32)


def transient_features(
    mono: np.ndarray,
    scalogram: np.ndarray,
    sample_rate: int,
    envelope_points: int = 32,
) -> np.ndarray:
    """Temporal morphology of the snippet extracted from the waveform itself.

    This branch carries the information that phase encodes perceptually —
    attack sharpness, decay behavior, envelope shape, onset density — in a
    time-shift-robust form, instead of raw phase values which change with
    every sample of offset.

    Returns ``[envelope_points + 8]``.
    """
    smooth = max(int(sample_rate * 0.005), 1)
    envelope = analytic_envelope(mono, smooth_samples=smooth)

    shape = _envelope_shape(envelope, envelope_points)
    log_attack = _log_attack_time(envelope, sample_rate)
    decay = _decay_slope(envelope, sample_rate)

    total = envelope.sum() + 1e-12
    temporal_centroid = float((np.arange(envelope.shape[0]) * envelope).sum() / total / max(envelope.shape[0] - 1, 1))

    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    crest = float(np.clip(np.log1p(np.abs(mono).max() / rms) / 4.0, 0.0, 1.0))
    zero_crossing_rate = float((mono[:-1] * mono[1:] < 0).mean()) if mono.shape[0] > 1 else 0.0

    onsets = _onset_curve(scalogram, sample_rate)
    onset_scale = onsets.mean() + 1e-8
    onset_mean = float(np.clip(np.log1p(onsets.mean()), 0.0, 4.0) / 4.0)
    onset_peakiness = float(np.clip(onsets.max() / onset_scale, 0.0, 50.0) / 50.0)
    onset_variability = float(np.clip(onsets.std() / onset_scale, 0.0, 10.0) / 10.0)

    scalars = np.array(
        [
            log_attack,
            decay,
            temporal_centroid,
            crest,
            zero_crossing_rate,
            onset_mean,
            onset_peakiness,
            onset_variability,
        ],
        dtype=np.float32,
    )
    return np.concatenate([shape, scalars])
