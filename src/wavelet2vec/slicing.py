from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np

from wavelet2vec.audio_io import resample
from wavelet2vec.filterbank import log_spaced_frequencies, morlet_scalogram

_ANALYSIS_SR = 22050


def _normalize_starts(starts: np.ndarray, n_samples: int) -> np.ndarray:
    """Sorted, unique, in-range slice starts, always beginning at sample 0."""
    starts = np.unique(np.clip(np.asarray(starts, dtype=np.int64), 0, max(n_samples - 1, 0)))
    if starts.size == 0 or starts[0] != 0:
        starts = np.concatenate([[0], starts])
    return starts


def slice_grid(
    n_samples: int,
    sample_rate: int,
    *,
    bpm: float | None = None,
    division: int = 8,
    offset_beats: float = 0.0,
    spacing_seconds: float | None = None,
) -> np.ndarray:
    """Slice starts on a fixed grid — for quantized patterns.

    Either give ``bpm`` (+ ``division``: 8 = 8th notes, 4 = quarters, 16 =
    16ths) or a literal ``spacing_seconds`` between note starts.
    """
    if spacing_seconds is None:
        if bpm is None:
            raise ValueError("slice_grid needs either bpm or spacing_seconds.")
        spacing_seconds = 60.0 / bpm * 4.0 / division
    offset = (offset_beats * 60.0 / bpm) if bpm is not None else 0.0
    step = spacing_seconds * sample_rate
    if step <= 0:
        raise ValueError("Slice spacing must be positive.")
    starts = np.arange(offset * sample_rate, n_samples, step)
    return _normalize_starts(np.round(starts).astype(np.int64), n_samples)


def slice_markers(path: str | Path, sample_rate: int, n_samples: int) -> np.ndarray:
    """Slice starts from a marker file.

    Accepts a CSV with a ``time_seconds`` (or ``seconds``) column, or a plain
    text file with one time in seconds per line.
    """
    marker_path = Path(path)
    text = marker_path.read_text(encoding="utf-8")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} contains no markers.")
    times: list[float] = []
    try:
        times = [float(line.split(",")[0]) for line in lines]
    except ValueError:
        with marker_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("time_seconds", row.get("seconds"))
                if value is None:
                    raise ValueError("Marker CSV needs a time_seconds or seconds column.")
                times.append(float(value))
    return _normalize_starts(
        np.round(np.asarray(times) * sample_rate).astype(np.int64), n_samples
    )


def read_cue_points(path: str | Path) -> np.ndarray:
    """Sample offsets of cue markers embedded in a WAV file (RIFF ``cue `` chunk).

    Sound Forge, Edison, and FL Studio's slicing tools write standard cue
    points; this reads them with no extra dependency.
    """
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF WAVE file.")

    offsets: list[int] = []
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        (chunk_size,) = struct.unpack_from("<I", data, position + 4)
        body = position + 8
        if chunk_id == b"cue ":
            (count,) = struct.unpack_from("<I", data, body)
            for index in range(count):
                entry = body + 4 + index * 24
                # dwName, dwPosition, fccChunk, dwChunkStart, dwBlockStart, dwSampleOffset
                (sample_offset,) = struct.unpack_from("<I", data, entry + 20)
                offsets.append(int(sample_offset))
        position = body + chunk_size + (chunk_size & 1)
    return np.asarray(sorted(offsets), dtype=np.int64)


def slice_cues(path: str | Path, n_samples: int) -> np.ndarray:
    """Slice starts from a WAV file's embedded cue markers."""
    offsets = read_cue_points(path)
    if offsets.size == 0:
        raise ValueError(f"No cue markers found in {path}.")
    return _normalize_starts(offsets, n_samples)


def slice_onsets(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    sensitivity: float = 0.5,
    min_spacing_seconds: float = 0.05,
) -> np.ndarray:
    """Slice starts by onset detection — fallback for unquantized material.

    Uses the wavelet scalogram's spectral flux with simple peak picking.
    ``sensitivity`` in [0, 1]: higher finds more (weaker) onsets.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    n_samples = waveform.shape[1]
    mono = resample(waveform, int(sample_rate), _ANALYSIS_SR).mean(axis=0)
    peak = np.abs(mono).max()
    if peak > 1e-8:
        mono = mono / peak

    centers = log_spaced_frequencies(27.5, 0.45 * _ANALYSIS_SR, 48)
    scalogram = morlet_scalogram(mono, _ANALYSIS_SR, centers)
    hop = max(int(_ANALYSIS_SR * 0.01), 1)
    n_frames = scalogram.shape[1] // hop
    framed = scalogram[:, : n_frames * hop].reshape(scalogram.shape[0], n_frames, hop).mean(axis=2)
    flux = np.diff(np.log1p(framed), axis=1).clip(min=0.0).sum(axis=0)
    if flux.size == 0 or flux.max() <= 0:
        return np.zeros(1, dtype=np.int64)

    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    threshold = flux.mean() + (1.0 - sensitivity) * (flux.max() - flux.mean())
    min_gap = max(int(min_spacing_seconds / 0.01), 1)
    onsets: list[int] = []
    for frame in range(1, flux.shape[0] - 1):
        if flux[frame] >= threshold and flux[frame] >= flux[frame - 1] and flux[frame] > flux[frame + 1]:
            if not onsets or frame - onsets[-1] >= min_gap:
                onsets.append(frame)

    seconds = (np.asarray(onsets, dtype=np.float64) + 1) * hop / _ANALYSIS_SR
    return _normalize_starts(
        np.round(seconds * sample_rate).astype(np.int64), n_samples
    )
