"""Canonical ingestion, filename/folder metadata, and sliding windows.

Self-contained loader for the experiment suite: every input file — whatever
its bit depth, sample rate, channel count, or length — is normalized once to a
single float format at a common rate so mel and wavelet analyses see identical
input. See ``docs/EXPERIMENTS_PLAN.md`` §2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wavelet2vec.audio_io import AUDIO_EXTENSIONS, resample

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency behavior
    sf = None

# Pitch-class index by note name; flats map onto the same class as sharps.
_PITCH_CLASS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10,
    "BB": 10, "B": 11, "CB": 11,
}
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# A key/note token at the end of a stem: letter, optional accidental, optional
# octave digit(s), optional trailing 'm' for a minor key. Matches " A#",
# "_F#1", "_C#m", "_Cm", "_Bb".
_KEY_TOKEN = re.compile(r"[ _]([A-Ga-g])(#|b)?(\d+)?(m)?$")
_BPM_RE = re.compile(r"(\d+)\s*bpm", re.IGNORECASE)


@dataclass(frozen=True)
class Track:
    path: Path
    folder: str
    duration: float
    native_sr: int
    channels: int
    subtype: str
    pitch_class: int | None = None
    key_is_minor: bool | None = None
    octave: int | None = None
    bpm: float | None = None


@dataclass
class CanonicalAudio:
    """A file decoded to float and resampled to the canonical analysis rate."""

    stereo: np.ndarray  # [channels, samples], float64, peak <= 1 per window later
    mono: np.ndarray  # [samples], float64 mixdown
    sample_rate: int
    track: Track
    native_sr: int

    @property
    def n_samples(self) -> int:
        return self.mono.shape[0]


def parse_pitch(stem: str) -> tuple[int | None, bool | None, int | None]:
    """Pitch class, is-minor, octave from a filename stem; (None, …) if absent."""
    match = _KEY_TOKEN.search(stem)
    if not match:
        return None, None, None
    letter, accidental, octave, minor = match.groups()
    name = letter.upper() + (accidental.upper() if accidental else "")
    name = name.replace("B" * 2, "BB")  # guard, no-op for normal input
    pitch_class = _PITCH_CLASS.get(name)
    if pitch_class is None:
        return None, None, None
    return pitch_class, bool(minor), (int(octave) if octave else None)


def parse_bpm(folder: str) -> float | None:
    match = _BPM_RE.search(folder)
    return float(match.group(1)) if match else None


def list_tracks(audio_dir: str | Path) -> list[Track]:
    """Builds the manifest for every audio file under ``audio_dir``."""
    if sf is None:
        raise RuntimeError("The experiment suite requires soundfile.")
    root = Path(audio_dir)
    tracks: list[Track] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        try:
            info = sf.info(path)
        except Exception:
            continue
        folder = path.parent.name
        pitch_class, is_minor, octave = parse_pitch(path.stem)
        tracks.append(
            Track(
                path=path,
                folder=folder,
                duration=float(info.duration),
                native_sr=int(info.samplerate),
                channels=int(info.channels),
                subtype=str(info.subtype),
                pitch_class=pitch_class,
                key_is_minor=is_minor,
                octave=octave,
                bpm=parse_bpm(folder),
            )
        )
    return tracks


def canonical_rate(tracks: list[Track], requested: int | None = None) -> int:
    """Default canonical rate = the max native rate present (never down-sample)."""
    if requested is not None:
        return int(requested)
    return max((t.native_sr for t in tracks), default=44100)


def load_canonical(track: Track, sample_rate: int) -> CanonicalAudio:
    """Decodes to float and resamples to ``sample_rate`` (no down-sampling loss)."""
    if sf is None:
        raise RuntimeError("The experiment suite requires soundfile.")
    data, native_sr = sf.read(track.path, always_2d=True, dtype="float64")  # µ-law -> float
    channels = data.T  # [channels, samples]
    channels = resample(channels, int(native_sr), int(sample_rate))
    mono = channels.mean(axis=0)
    return CanonicalAudio(
        stereo=channels,
        mono=mono,
        sample_rate=int(sample_rate),
        track=track,
        native_sr=int(native_sr),
    )


def sliding_windows(
    n_samples: int,
    sample_rate: int,
    *,
    bpm: float | None = None,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    bars_per_window: int = 1,
    beats_per_hop: int = 1,
) -> list[tuple[int, int]]:
    """Window bounds (start, end) in samples.

    When ``bpm`` is given, windows are beat-aligned: width = ``bars_per_window``
    bars, hop = ``beats_per_hop`` beats. Otherwise a fixed ``window_seconds`` /
    ``hop_seconds`` grid is used. Files shorter than one window return a single
    whole-file frame.
    """
    if bpm:
        beat = 60.0 / bpm
        window = beat * 4.0 * bars_per_window
        hop = beat * beats_per_hop
    else:
        window, hop = window_seconds, hop_seconds
    window_n = max(int(round(window * sample_rate)), 1)
    hop_n = max(int(round(hop * sample_rate)), 1)

    if n_samples <= window_n:
        return [(0, n_samples)]
    bounds: list[tuple[int, int]] = []
    start = 0
    while start + window_n <= n_samples:
        bounds.append((start, start + window_n))
        start += hop_n
    if bounds[-1][1] < n_samples:  # tail remainder gets a final aligned window
        bounds.append((n_samples - window_n, n_samples))
    return bounds


@dataclass
class IngestReport:
    total: int = 0
    resampled: int = 0
    converted: list[str] = field(default_factory=list)

    def note(self, track: Track, canonical_sr: int) -> None:
        self.total += 1
        needs_resample = track.native_sr != canonical_sr
        if needs_resample:
            self.resampled += 1
        if needs_resample or track.subtype not in ("FLOAT", "PCM_24", "PCM_16"):
            self.converted.append(
                f"{track.path.name}: {track.native_sr} Hz {track.subtype} "
                f"{track.channels}ch -> {canonical_sr} Hz float"
            )

    def summary(self) -> str:
        return (
            f"{self.total} files decoded to float; {self.resampled} resampled to the "
            f"canonical rate; {len(self.converted)} required format conversion."
        )
