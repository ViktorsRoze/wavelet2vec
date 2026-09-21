"""Lossy coefficient-derived images + annotations for vision models (V-1).

These pooled images are recognition features, not lossless audio encodings.
Use wavelet2vec.hybrid for native stereo waveform and full complex carriers.

The bridge between the invertible wavelet frame and image models. One short
fragment becomes **one fixed-size 3-channel image** (128 bands × a fixed
frame count) — the model always sees the complete musical object, no tiling
needed. The three channels follow the roadmap encoding (raw phase is
unlearnable; instantaneous frequency is the GANSynth lesson):

    channel 0  log-magnitude          — "how loud, where"
    channel 1  IF deviation           — inst. frequency − band center, tanh-squashed
    channel 2  energy-weighted phase coherence — "how tonal / stable, where"

Events are boxes in (time, frequency) space with pitch/velocity/class
attributes, serialized COCO-style. Coordinate mapping is exact and
invertible: :func:`event_to_bbox` / :func:`bbox_to_event`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wavelet2vec.frame import analyze

DEFAULT_BANDS = 128
DEFAULT_FRAMES = 128
DEFAULT_F_MIN = 30.0


@dataclass
class ImageMeta:
    """Everything needed to map between seconds/Hz and image pixels."""

    centers_hz: np.ndarray  # [n_bands], low → high (row 0 = lowest)
    n_bands: int
    n_frames: int
    duration_s: float
    sample_rate: int

    def frame_of(self, time_s: float) -> float:
        return float(np.clip(time_s / max(self.duration_s, 1e-9), 0.0, 1.0)) * self.n_frames

    def time_of(self, frame: float) -> float:
        return float(frame) / self.n_frames * self.duration_s

    def row_of(self, hz: float) -> float:
        """Fractional band row of a frequency (log-interpolated)."""
        centers = self.centers_hz
        logc = np.log(centers)
        return float(np.interp(np.log(max(hz, centers[0])), logc, np.arange(len(centers))))

    def hz_of(self, row: float) -> float:
        centers = self.centers_hz
        logc = np.log(centers)
        return float(np.exp(np.interp(row, np.arange(len(centers)), logc)))

    def to_dict(self) -> dict:
        return {
            "centers_hz": self.centers_hz.tolist(), "n_bands": self.n_bands,
            "n_frames": self.n_frames, "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
        }


@dataclass
class Event:
    """A time × frequency box with musical attributes."""

    onset_s: float
    offset_s: float
    f_lo_hz: float
    f_hi_hz: float
    label: str = "note"
    pitch: int | None = None  # MIDI note
    velocity: int | None = None  # 0..127

    def to_dict(self) -> dict:
        return {
            "onset_s": self.onset_s, "offset_s": self.offset_s,
            "f_lo_hz": self.f_lo_hz, "f_hi_hz": self.f_hi_hz,
            "label": self.label, "pitch": self.pitch, "velocity": self.velocity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(**{k: data.get(k) for k in
                      ("onset_s", "offset_s", "f_lo_hz", "f_hi_hz", "label",
                       "pitch", "velocity")})


def _pool_to_frames(values: np.ndarray, n_frames: int, reduce: str = "mean") -> np.ndarray:
    """Adaptive-pool the time axis of ``[bands, N]`` into exactly ``n_frames``
    columns, so any-length input yields a fixed-width image."""
    n = values.shape[1]
    if n == 0:
        return np.zeros((values.shape[0], n_frames))
    edges = np.linspace(0, n, n_frames + 1).astype(int)
    out = np.empty((values.shape[0], n_frames))
    for i in range(n_frames):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        block = values[:, a:min(b, n)]
        out[:, i] = block.max(axis=1) if reduce == "max" else block.mean(axis=1)
    return out


def _channel_stack(coeffs: np.ndarray, centers: np.ndarray, sample_rate: int,
                   n_frames: int) -> np.ndarray:
    """The 3-channel encoding pooled to ``n_frames`` columns: [3, bands, T],
    row 0 = lowest band. Shared by :func:`coeff_image` (which flips for
    display) and :func:`sequence_tokens` (which keeps low→high)."""
    magnitude = np.abs(coeffs)

    # Channel 0: log-magnitude, globally normalized.
    log_mag = np.log1p(magnitude * 40.0)
    mag_img = _pool_to_frames(log_mag, n_frames, reduce="max")
    peak = mag_img.max()
    mag_img = mag_img / peak if peak > 0 else mag_img

    # Channel 1: instantaneous-frequency deviation from each band center.
    phase = np.angle(coeffs)
    dphase = np.diff(phase, axis=1, prepend=phase[:, :1])
    dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi  # wrap
    inst_freq = dphase * sample_rate / (2 * np.pi)
    deviation = (inst_freq - centers[:, None]) / centers[:, None]
    dev_img = _pool_to_frames(np.tanh(deviation), n_frames, reduce="mean")
    dev_img = (dev_img + 1.0) / 2.0  # [-1,1] -> [0,1]

    # Channel 2: energy-weighted local phase coherence.
    unit = coeffs / (magnitude + 1e-12)
    coh = np.abs(_pool_to_frames(unit.real, n_frames) + 1j * _pool_to_frames(unit.imag, n_frames))
    coh_img = coh * mag_img  # dark where there's no energy

    return np.stack([mag_img, dev_img, coh_img])


def sequence_tokens(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 64,
    hop_ms: float = 20.0,
    f_min: float = DEFAULT_F_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level tokens for sequence models: ``([T, 3·n_bands], times_s)``.

    Each token is one time frame of the same 3-channel encoding the vision
    tiles use — per-band log-magnitude, IF-deviation and coherence
    concatenated (bands ordered low → high within each block) — so sequence
    models and vision models learn from the same features. ``T = duration /
    hop_ms``; per-token values are in [0, 1]. Magnitude is normalized per
    clip (corpus-level standardization belongs to the training pipeline).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    n = audio.shape[0]
    hop = max(int(sample_rate * hop_ms / 1000.0), 1)
    n_frames = max(n // hop, 1)

    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)
    channels = _channel_stack(coeffs, spec.centers, sample_rate, n_frames)
    tokens = channels.transpose(2, 0, 1).reshape(n_frames, 3 * n_bands)
    times = (np.arange(n_frames) + 0.5) * (n / n_frames) / sample_rate
    return tokens.astype(np.float32), times


def coeff_image(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = DEFAULT_BANDS,
    n_frames: int = DEFAULT_FRAMES,
    f_min: float = DEFAULT_F_MIN,
) -> tuple[np.ndarray, ImageMeta]:
    """One fixed-size 3-channel image ``[3, n_bands, n_frames]`` in [0, 1].

    Channels: log-magnitude, IF deviation (tanh-squashed), energy-weighted
    phase coherence. Row 0 is the lowest band. Mono mixdown of stereo input.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    duration_s = audio.shape[0] / sample_rate

    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)
    channels = _channel_stack(coeffs, spec.centers, sample_rate, n_frames)

    # Display/COCO orientation: row 0 = highest band (top), matching
    # event_to_bbox and standard spectrogram display. The saved tensor and
    # the annotation boxes share this one coordinate system.
    image = channels[:, ::-1].astype(np.float32)
    meta = ImageMeta(
        centers_hz=spec.centers, n_bands=n_bands, n_frames=n_frames,
        duration_s=duration_s, sample_rate=int(sample_rate),
    )
    return image, meta


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """``[3, bands, frames]`` float → ``[bands, frames, 3]`` uint8 (for PNG or
    a vision backbone). The tensor is already in display orientation
    (row 0 = highest band), so this only transposes and scales."""
    hwc = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    return (hwc * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Event ↔ box coordinate mapping (COCO bbox = [x, y, w, h], origin top-left)
# ---------------------------------------------------------------------------
def event_to_bbox(event: Event, meta: ImageMeta) -> list[float]:
    x0 = meta.frame_of(event.onset_s)
    x1 = meta.frame_of(event.offset_s)
    # Rows increase downward in COCO; row 0 (low freq) sits at the bottom.
    row_lo = meta.row_of(event.f_lo_hz)
    row_hi = meta.row_of(event.f_hi_hz)
    y_top = meta.n_bands - 1 - max(row_lo, row_hi)
    y_bot = meta.n_bands - 1 - min(row_lo, row_hi)
    return [x0, y_top, max(x1 - x0, 1e-6), max(y_bot - y_top, 1e-6)]


def bbox_to_event(bbox: list[float], meta: ImageMeta, *, label: str = "note",
                  pitch: int | None = None, velocity: int | None = None) -> Event:
    x, y, w, h = bbox
    onset_s = meta.time_of(x)
    offset_s = meta.time_of(x + w)
    f_hi = meta.hz_of(meta.n_bands - 1 - y)
    f_lo = meta.hz_of(meta.n_bands - 1 - (y + h))
    return Event(onset_s, offset_s, min(f_lo, f_hi), max(f_lo, f_hi),
                 label=label, pitch=pitch, velocity=velocity)


# ---------------------------------------------------------------------------
# COCO-style dataset assembly + export
# ---------------------------------------------------------------------------
@dataclass
class CocoBuilder:
    categories: list[str]
    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    _cat_id: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cat_id = {name: i + 1 for i, name in enumerate(self.categories)}

    def add(self, file_name: str, meta: ImageMeta, events: list[Event]) -> int:
        image_id = len(self.images) + 1
        self.images.append({
            "id": image_id, "file_name": file_name,
            "width": meta.n_frames, "height": meta.n_bands,
            "meta": meta.to_dict(),
        })
        for event in events:
            category = event.label if event.label in self._cat_id else self.categories[0]
            self.annotations.append({
                "id": len(self.annotations) + 1, "image_id": image_id,
                "category_id": self._cat_id[category],
                "bbox": [round(v, 3) for v in event_to_bbox(event, meta)],
                "attributes": {"pitch": event.pitch, "velocity": event.velocity},
            })
        return image_id

    def to_dict(self) -> dict:
        return {
            "images": self.images, "annotations": self.annotations,
            "categories": [{"id": i, "name": n} for n, i in self._cat_id.items()],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def export_dataset(
    clips: list[tuple[np.ndarray, int, list[Event]]],
    out_dir: str | Path,
    *,
    categories: list[str] | None = None,
    n_bands: int = DEFAULT_BANDS,
    n_frames: int = DEFAULT_FRAMES,
) -> Path:
    """Writes ``clip_XXXX.npy`` images + ``annotations.json`` (COCO) for a list
    of ``(audio, sample_rate, events)``. Returns the annotations path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if categories is None:
        categories = sorted({event.label for _a, _sr, events in clips for event in events}) \
            or ["note"]
    builder = CocoBuilder(categories=categories)
    for index, (audio, sample_rate, events) in enumerate(clips):
        image, meta = coeff_image(audio, sample_rate, n_bands=n_bands, n_frames=n_frames)
        name = f"clip_{index:04d}.npy"
        np.save(out_dir / name, image)
        builder.add(name, meta, events)
    return builder.save(out_dir / "annotations.json")
