"""Lossless stereo wavelet image: audio <-> a bands x frames grid of exact complex samples.

The multirate compact Gaussian filter bank of :mod:`wavelet2vec.multirate`,
laid out so that it forms an image without losing anything:

- **Bands** sit on semitones (MIDI 12 = 16.35 Hz upward) up to 0.45 x the
  sample rate. The last band extends to Nyquist and the first to DC, so the
  whole spectrum is covered. At 192 kHz that is MIDI 12-160: 149 rows, up to
  86 kHz.
- **Frames** are a fixed hop (10 ms by default). Band b keeps
  ``c_b x n_frames`` complex samples: at least its frequency support (no
  aliasing) and a whole number per frame. Sample j of band b therefore lies
  in frame ``j // c_b``, and each image cell (band, frame) holds that band's
  exact complex samples for that frame. A cell holds one sample in the bass
  rows and dozens in the ultrasonic rows.
- **Stereo** is coded as mid ``(L+R)/2`` and side ``(L-R)/2``. This is an
  exact linear change of basis, so stereo width and phase are kept.

Inversion is the weighted inverse of the compact bank (window-weighted sum
over bands, divided by the squared-window coverage). It is exact in exact
arithmetic: float64 round trips sit at the ~1e-15 level, float32 storage at
roughly the float32 noise floor.

Cells are grouped by sample count: ``WaveletImage.groups[c]`` is a complex
tensor ``[batch, 2 (mid, side), rows, frames, c]``. A model embeds each group
with its own projection and writes coefficients back the same way, so both
its input and its output are the complete signal.

    spec = WaveletImageSpec(sample_rate=192000, n_frames=120)
    image = encode(audio, spec)          # audio [B, 2, n_frames*hop] (L, R)
    audio_back = decode(image)           # exact inverse
    view = magnitude_view(image)         # [B, 2, bands, frames] dB, for looking at
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


def midi_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


@dataclass(frozen=True)
class WaveletImageSpec:
    sample_rate: int = 192000
    n_frames: int = 120            # frames of the audio itself (1.2 s at 10 ms)
    hop_ms: float = 10.0
    pad_frames: int = 8            # zero context on each side (edge effects)
    first_midi: int = 12
    bandwidth_scale: float = 0.6   # Gaussian sigma = scale x one-semitone spacing
    truncate_sigma: float = 3.0    # finite support, rounded outward to FFT bins
    top_fraction: float = 0.45     # highest band centre, as a fraction of the sample rate
    pow2_cells: bool = True        # round samples-per-cell up to a power of two: ~one group per octave

    @property
    def hop(self) -> int:
        hop = self.sample_rate * self.hop_ms / 1000.0
        if abs(hop - round(hop)) > 1e-9:
            raise ValueError("hop_ms must be a whole number of samples")
        return int(round(hop))

    @property
    def total_frames(self) -> int:
        return self.n_frames + 2 * self.pad_frames

    @property
    def padded_length(self) -> int:
        return self.total_frames * self.hop

    @property
    def audio_length(self) -> int:
        return self.n_frames * self.hop

    @property
    def last_midi(self) -> int:
        return int(math.floor(69 + 12 * math.log2(self.top_fraction * self.sample_rate / 440.0)))

    @property
    def n_bands(self) -> int:
        return self.last_midi - self.first_midi + 1

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BandLayout:
    midi: int
    center_hz: float
    sigma_hz: float
    start_bin: int
    stop_bin: int
    carrier_bin: int
    per_frame: int                 # complex samples per frame (c_b)
    samples: int                   # per_frame * total_frames


def band_layout(spec: WaveletImageSpec) -> tuple[BandLayout, ...]:
    n, frames = spec.padded_length, spec.total_frames
    half, spacing = n // 2 + 1, spec.sample_rate / n
    step = 2.0 ** (1.0 / 12.0) - 1.0
    bands = []
    for i, midi in enumerate(range(spec.first_midi, spec.last_midi + 1)):
        center = float(midi_hz(midi))
        sigma = center * step * spec.bandwidth_scale
        radius = spec.truncate_sigma * sigma
        lo = max(0, min(half - 1, math.floor((center - radius) / spacing)))
        hi = max(lo + 1, min(half, math.ceil((center + radius) / spacing) + 1))
        if i == 0:
            lo = 0
        if midi == spec.last_midi:
            hi = half
        carrier = max(lo, min(hi - 1, round(center / spacing)))
        per_frame = max(1, math.ceil((hi - lo) / frames))
        if spec.pow2_cells:
            per_frame = 1 << (per_frame - 1).bit_length()
        bands.append(BandLayout(midi, center, sigma, lo, hi, carrier, per_frame, per_frame * frames))
    return tuple(bands)


class _Plan:
    """Packing tables for one spec, grouped by samples-per-frame; cached per device/dtype."""

    def __init__(self, spec: WaveletImageSpec, device, dtype):
        self.spec = spec
        self.bands = band_layout(spec)
        n = spec.padded_length
        freqs_all = np.arange(n // 2 + 1) * spec.sample_rate / n
        coverage = np.zeros(n // 2 + 1)
        windows = []
        for i, band in enumerate(self.bands):
            f = freqs_all[band.start_bin:band.stop_bin]
            w = np.exp(-0.5 * ((f - band.center_hz) / band.sigma_hz) ** 2)
            if i == 0:
                w[f <= band.center_hz] = 1.0
            if i == len(self.bands) - 1:
                w[f >= band.center_hz] = 1.0
            coverage[band.start_bin:band.stop_bin] += w ** 2
            windows.append(w)
        self.coverage_ratio = float(coverage.min() / coverage.max())
        if not np.isfinite(coverage).all() or self.coverage_ratio < 1e-6:
            raise ValueError("ill-conditioned wavelet image bank: raise bandwidth_scale or truncate_sigma")
        self.groups = {}
        for c in sorted({b.per_frame for b in self.bands}):
            rows = [i for i, b in enumerate(self.bands) if b.per_frame == c]
            m = c * spec.total_frames
            index, window = [], []
            for i in rows:
                b = self.bands[i]
                local = (np.arange(m) - (b.start_bin - b.carrier_bin)) % m
                valid = local < b.stop_bin - b.start_bin
                safe = np.minimum(local, b.stop_bin - b.start_bin - 1)
                index.append(safe + b.start_bin)
                window.append(np.where(valid, windows[i][safe], 0.0))
            self.groups[c] = (rows, torch.as_tensor(np.stack(index), device=device),
                              torch.as_tensor(np.stack(window), device=device, dtype=dtype))
        self.coverage = torch.as_tensor(coverage, device=device, dtype=dtype)
        analytic = torch.full((n // 2 + 1,), 2.0, device=device, dtype=dtype)
        analytic[0] = 1.0
        if n % 2 == 0:
            analytic[-1] = 1.0
        self.analytic = analytic
        weights = torch.full((n // 2 + 1,), 0.5, device=device, dtype=dtype)
        weights[0] = 1.0
        if n % 2 == 0:
            weights[-1] = 1.0
        self.weights = weights


_PLANS: dict = {}


def _plan(spec, device, dtype):
    key = (spec, str(device), dtype)
    if key not in _PLANS:
        _PLANS[key] = _Plan(spec, device, dtype)
    return _PLANS[key]


@dataclass
class WaveletImage:
    spec: WaveletImageSpec
    groups: dict  # samples-per-frame c -> complex tensor [B, 2, rows, total_frames, c]
    rows: dict = field(default_factory=dict)  # c -> band indices (row numbers) in that group

    def total_values(self) -> int:
        """Real numbers stored per batch item (both channels, real + imaginary)."""
        return int(sum(g[0].numel() * 2 for g in self.groups.values()))


def encode(audio: torch.Tensor, spec: WaveletImageSpec) -> WaveletImage:
    """``audio [B, 2, spec.audio_length]`` (left, right; shorter inputs are zero-padded) -> image."""
    if audio.ndim != 3 or audio.shape[1] != 2:
        raise ValueError("audio must be [batch, 2 (left, right), samples]")
    real = torch.float64 if audio.dtype == torch.float64 else torch.float32
    audio = audio.to(real)
    if audio.shape[-1] > spec.audio_length:
        raise ValueError(f"audio longer than {spec.audio_length} samples")
    plan = _plan(spec, audio.device, real)
    ms = torch.stack([(audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2], 1)
    left = spec.pad_frames * spec.hop
    padded = torch.nn.functional.pad(ms, (left, spec.padded_length - left - ms.shape[-1]))
    spectrum = torch.fft.rfft(padded, dim=-1) * plan.analytic
    n = spec.padded_length
    groups = {}
    for c, (rows, index, window) in plan.groups.items():
        packed = spectrum[..., index] * window          # [B, 2, rows, M]
        values = torch.fft.ifft(packed, dim=-1) * (index.shape[-1] / n)
        groups[c] = values.reshape(*values.shape[:-1], spec.total_frames, c)
    return WaveletImage(spec, groups, {c: v[0] for c, v in plan.groups.items()})


def decode(image: WaveletImage) -> torch.Tensor:
    """Exact inverse of :func:`encode`: image -> ``[B, 2 (left, right), audio_length]``."""
    spec = image.spec
    first = next(iter(image.groups.values()))
    real = first.real.dtype
    plan = _plan(spec, first.device, real)
    n = spec.padded_length
    summed = None
    for c, (rows, index, window) in plan.groups.items():
        values = image.groups[c].reshape(*image.groups[c].shape[:-2], -1)
        spectrum = torch.fft.fft(values, dim=-1) * (n / values.shape[-1]) * window
        if summed is None:
            summed = torch.zeros(*spectrum.shape[:-2], n // 2 + 1, dtype=spectrum.dtype, device=spectrum.device)
        summed.index_add_(-1, index.flatten(), spectrum.flatten(-2))
    ms = torch.fft.irfft(summed * plan.weights / plan.coverage, n=n, dim=-1)
    left = spec.pad_frames * spec.hop
    ms = ms[..., left:left + spec.audio_length]
    return torch.stack([ms[:, 0] + ms[:, 1], ms[:, 0] - ms[:, 1]], 1)


def magnitude_view(image: WaveletImage, floor_db: float = -140.0) -> torch.Tensor:
    """``[B, 2 (mid, side), bands, frames]`` RMS level in dB of each cell: for looking at, not for inversion."""
    spec = image.spec
    first = next(iter(image.groups.values()))
    out = torch.full((first.shape[0], 2, spec.n_bands, spec.n_frames), floor_db, dtype=first.real.dtype)
    keep = slice(spec.pad_frames, spec.pad_frames + spec.n_frames)
    for c, values in image.groups.items():
        level = 10 * torch.log10((values.abs() ** 2).mean(-1)[..., keep] + 1e-30)
        out[:, :, torch.as_tensor(image.rows[c])] = level.clamp(min=floor_db).cpu()
    return out


def describe(spec: WaveletImageSpec) -> dict:
    plan = _plan(spec, "cpu", torch.float64)
    per_channel = sum(b.samples for b in plan.bands) * 2
    return {"bands": spec.n_bands, "midi_range": [spec.first_midi, spec.last_midi],
            "top_center_hz": round(plan.bands[-1].center_hz, 1), "frames": spec.total_frames,
            "hop_samples": spec.hop, "groups": {c: len(v[0]) for c, v in plan.groups.items()},
            "real_values_per_channel": per_channel,
            "redundancy": round(per_channel / spec.padded_length, 2),
            "coverage_min_over_max": round(plan.coverage_ratio, 4)}


def save_image(image: WaveletImage, path) -> Path:
    """Float32 image files: one TIFF per sample-count group ([rows*2 (re,im) x frames*c] per channel,
    mid and side as 2 pages), plus a JSON sidecar with the spec. Inverse: :func:`load_image`."""
    import imageio.v3 as iio
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for c, values in image.groups.items():
        v = values[0].detach().cpu().numpy()  # [2, rows, frames, c]
        flat = v.reshape(2, v.shape[1], -1)
        pages = np.stack([np.vstack([flat[ch].real, flat[ch].imag]) for ch in range(2)]).astype(np.float32)
        iio.imwrite(path / f"group_c{c:03d}.tiff", pages)
    (path / "wavelet_image.json").write_text(json.dumps({"spec": image.spec.to_dict(),
                                                         "rows": {str(c): list(map(int, r)) for c, r in image.rows.items()}}),
                                             encoding="utf-8")
    return path


def load_image(path) -> WaveletImage:
    import imageio.v3 as iio
    path = Path(path)
    meta = json.loads((path / "wavelet_image.json").read_text(encoding="utf-8"))
    spec = WaveletImageSpec(**meta["spec"])
    groups, rows = {}, {}
    for key, row_list in meta["rows"].items():
        c = int(key)
        pages = np.asarray(iio.imread(path / f"group_c{c:03d}.tiff")).astype(np.float32)
        n_rows = len(row_list)
        complex_pages = pages[:, :n_rows] + 1j * pages[:, n_rows:]
        groups[c] = torch.from_numpy(complex_pages.reshape(1, 2, n_rows, spec.total_frames, c).astype(np.complex64))
        rows[c] = row_list
    return WaveletImage(spec, groups, rows)


# --------------------------------------------------------------------------- views
# Pictures of an image at screen resolution, for looking at only (decode never uses them). Unlike
# magnitude_view (one RMS value per cell), they use every complex sample at its own rate.

CHANNELS = {"mid": (1.0, 0.0), "side": (0.0, 1.0), "left": (1.0, 1.0), "right": (1.0, -1.0)}  # a*mid + b*side


def _tracks(image, item, midi_lo, midi_hi, t_lo, t_hi, dt):
    """Per sample-count group, the bands in [midi_lo, midi_hi] as three exact band signals: the value, its
    time derivative and the band filtered with the derivative of its window (for time reassignment). All
    three come from the cells, through the band's own spectrum, upsampled (band-limited, up to 4x) towards
    two samples per pixel. Yields (bands, carrier_hz [rows], times [n], signals [3, 2, rows, n], step)."""
    spec = image.spec
    fps = 1000.0 / spec.hop_ms
    pad = spec.pad_frames / fps
    spacing = spec.sample_rate / spec.padded_length
    layout = band_layout(spec)
    for c, values in image.groups.items():
        pick = [(k, int(i)) for k, i in enumerate(image.rows[c]) if midi_lo <= layout[int(i)].midi <= midi_hi]
        if not pick:
            continue
        bands = [layout[i] for _, i in pick]
        z = values[item][:, [k for k, _ in pick]].detach().cpu().to(torch.complex128).numpy()
        z = z.reshape(2, len(bands), -1)
        m, u = z.shape[-1], 1
        while u < 4 and c * fps * u * dt < 2:
            u *= 2
        spectrum = np.fft.fft(z, axis=-1)
        up = np.zeros((3, 2, len(bands), m * u), dtype=np.complex128)
        for r, b in enumerate(bands):
            pos, neg = b.stop_bin - b.carrier_bin, b.carrier_bin - b.start_bin
            src, dst = np.r_[0:pos, m - neg:m], np.r_[0:pos, m * u - neg:m * u]
            deviation = np.r_[0:pos, -neg:0] * spacing
            freq = b.carrier_bin * spacing + deviation
            slope = -(freq - b.center_hz) / b.sigma_hz ** 2      # window derivative / window (Gaussian)
            if b.midi == spec.first_midi:
                slope[freq <= b.center_hz] = 0.0                 # flat down to DC
            if b.midi == spec.last_midi:
                slope[freq >= b.center_hz] = 0.0                 # flat up to Nyquist
            values_r = spectrum[:, r][:, src]
            up[0, :, r][:, dst] = values_r
            up[1, :, r][:, dst] = values_r * (2j * np.pi * deviation)
            up[2, :, r][:, dst] = values_r * slope
        signals = np.fft.ifft(up, axis=-1) * u
        step = 1.0 / (c * fps * u)
        times = np.arange(m * u) * step - pad
        reach = 3.0 / (2 * np.pi * min(b.sigma_hz for b in bands))  # how far time reassignment can move energy
        keep = (times >= t_lo - reach) & (times <= t_hi + reach)
        carrier = np.array([b.carrier_bin * spacing for b in bands])
        yield bands, carrier, times[keep], signals[..., keep], step


def _locate(s, carrier, reassign_time):
    """Power, instantaneous frequency and time offset of each sample of one band-signal triple ``s [3, rows, n]``."""
    power = s[0].real ** 2 + s[0].imag ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = carrier[:, None] + np.imag(s[1] / s[0]) / (2 * np.pi)
        offset = np.imag(s[2] / s[0]) / (2 * np.pi) if reassign_time else np.zeros_like(power)
    bad = ~(np.isfinite(freq) & np.isfinite(offset))
    return np.where(bad, 0.0, power), np.where(bad, 0.0, freq), np.where(bad, 0.0, offset)


def _densify(times, step, layers, t0, t1, dt):
    """Inside [t0, t1], replace samples sparser than dt/2 by linear interpolation onto a dt/2 grid.
    ``layers`` are power-weighted quantities ([rows, n] each), so interpolation never invents positions."""
    if step <= dt / 2:
        return times, np.full(times.shape, step), layers
    inside = (times >= t0) & (times <= t1)
    grid = np.arange(t0 + dt / 4, t1, dt / 2)
    out = [np.concatenate([a[:, ~inside], np.stack([np.interp(grid, times, row) for row in a])], axis=1)
           for a in layers]
    duration = np.concatenate([np.full(int((~inside).sum()), step), np.full(grid.size, dt / 2)])
    return np.concatenate([times[~inside], grid]), duration, out


AGREE_SEMITONES = 0.3   # pitch disagreement between neighbouring bands that halves trust in a move (~e^-1)
AGREE_SPREAD = 0.25     # time disagreement, as a fraction of the band's time spread, with the same meaning


def _agreement(values, scale):
    """Soft trust in [0, 1] per sample: exp(-(d / scale)^2), d = distance to the closer neighbouring row's value."""
    if values.shape[0] < 2:
        return np.ones(values.shape)
    gap = np.abs(np.diff(values, axis=0)) / (np.minimum(scale[:-1], scale[1:]) if np.ndim(scale) else scale)
    nearest = np.full(values.shape, np.inf)
    nearest[:-1] = gap
    nearest[1:] = np.minimum(nearest[1:], gap)
    return np.exp(-nearest ** 2)


def _scatter(targets, layers, times, offset, freq, bands, t0, m0, dt, dm, width, height, reassign=True):
    """Add power-like layers ([rows, n], duration-weighted) of one group into flat ``[height*width]`` targets.

    Reassignment is trusted in proportion to agreement between neighbouring bands (soft, so there are no
    switching artefacts):
    - pitch: a resolved partial is placed at the same frequency by neighbouring bands, so that share of
      the energy moves to its reassigned frequency; the rest (noise, unresolved partials, broadband
      clicks) is spread over its own band with a triangle reaching the neighbouring band centres (the
      triangles of all bands sum to one, so neighbouring levels blend linearly);
    - time: attacks, clicks and steady partials are placed at the same time by neighbouring bands, so the
      sample moves that share of the way to its reassigned time (group delay).
    Noise therefore looks like the exact view, while partials become lines and attacks become edges."""
    midi = 69.0 + 12.0 * np.log2(np.maximum(freq, 1e-9) / 440.0)
    if reassign:
        spread_t = np.array([1.0 / (2 * np.pi * b.sigma_hz) for b in bands])[:, None]
        at = times + _agreement(times + offset, AGREE_SPREAD * spread_t) * offset
        trust = _agreement(midi, AGREE_SEMITONES)
    else:
        at, trust = np.broadcast_to(times, midi.shape), np.zeros(midi.shape)
    col = np.floor((at - t0) / dt)
    row = np.floor((midi - m0) / dm)
    in_time = (col >= 0) & (col < width)
    sharp = in_time & (freq > 0) & (row >= 0) & (row < height) & (trust > 1e-3)
    index = (row * width + col)[sharp].astype(np.int64)
    for target, layer in zip(targets, layers):
        target += np.bincount(index, (layer * trust)[sharp], height * width)
    spread = in_time & (trust < 1 - 1e-3)
    if not spread.any():
        return
    n_rows = midi.shape[0]
    flat = (np.arange(n_rows)[:, None] * width + col)[spread].astype(np.int64)
    edges = m0 + np.arange(height + 1) * dm
    overlaps = []
    for band in bands:
        x = np.clip(edges - band.midi, -1.0, 1.0)
        cumulative = np.where(x < 0, (x + 1) ** 2 / 2, 1 - (1 - x) ** 2 / 2)   # integral of the triangle
        overlap = np.diff(cumulative)
        overlaps.append((np.nonzero(overlap > 0)[0], overlap))
    for target, layer in zip(targets, layers):
        per_band = np.bincount(flat, (layer * (1 - trust))[spread], n_rows * width).reshape(n_rows, width)
        image = target.reshape(height, width)
        for r, (ys, overlap) in enumerate(overlaps):
            if ys.size:
                image[ys] += overlap[ys, None] * per_band[r]


def _antialias(power, pixels):
    """Gaussian blur of a power image (energy preserving), so one-pixel lines survive downscaling."""
    if not pixels:
        return power
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(power, pixels, mode="constant")


def _cell_power(image, item, a, b):
    """Mean power per cell of a*mid + b*side, ``[bands, frames]`` (core frames)."""
    spec = image.spec
    out = np.zeros((spec.n_bands, spec.n_frames))
    keep = slice(spec.pad_frames, spec.pad_frames + spec.n_frames)
    for c, values in image.groups.items():
        v = values[item].detach().cpu().to(torch.complex128).numpy()
        out[list(map(int, image.rows[c]))] = (np.abs(a * v[0] + b * v[1]) ** 2).mean(-1)[:, keep]
    return out


def _cells_to_pixels(cells, spec, t0, m0, dt, dm, width, height):
    row = np.clip(np.round(m0 + (np.arange(height) + 0.5) * dm).astype(int) - spec.first_midi, 0, spec.n_bands - 1)
    frame = np.clip(np.floor((t0 + (np.arange(width) + 0.5) * dt) * 1000.0 / spec.hop_ms).astype(int),
                    0, spec.n_frames - 1)
    return cells[row][:, frame]


def _view_frame(spec, time_range, midi_range, width, height):
    fps = 1000.0 / spec.hop_ms
    t0, t1 = time_range if time_range is not None else (0.0, spec.n_frames / fps)
    m0, m1 = midi_range if midi_range is not None else (spec.first_midi - 0.5, spec.last_midi + 0.5)
    if not (t1 > t0 and m1 > m0 and width > 0 and height > 0):
        raise ValueError("empty view")
    return float(t0), float(t1), float(m0), float(m1), (t1 - t0) / width, (m1 - m0) / height


def detail_view(image: WaveletImage, *, mode: str = "sharp", channel: str = "mid", item: int = 0,
                time_range=None, midi_range=None, width: int = 1200, height: int = 600,
                floor_db: float = -140.0, antialias: float = 0.7) -> np.ndarray:
    """``[height, width]`` level in dB of one channel at any resolution, row 0 = lowest pitch.

    - ``"sharp"``: reassigned. Each complex sample's power moves to its instantaneous frequency (the exact
      phase derivative) and to its own time (group delay, from the derivative of the band window). Partials
      become lines a fraction of a semitone wide and attacks become edges, in the bass as well. Each move is
      made in proportion to how well neighbouring bands agree on it, so noise keeps the smooth look of
      ``"exact"``.
    - ``"exact"``: each band's level at its true time resolution (band-limited interpolation where a band
      has fewer samples than pixels), interpolated between band centres. No reassignment.
    - ``"cells"``: the cell grid the backbone's feature planes are computed on (as magnitude_view).

    ``channel``: mid, side, left or right. ``time_range`` in seconds from the audio start (default: the
    whole note); ``midi_range`` in fractional MIDI numbers (default: all rows). Power is averaged over each
    pixel's duration and summed over bands, so a steady sine reads its own level (within ~1 dB) in every mode.
    ``antialias``: Gaussian blur in pixels applied to the sharp picture's power (0 = none)."""
    if mode not in ("sharp", "exact", "cells") or channel not in CHANNELS:
        raise ValueError("mode: sharp, exact or cells; channel: mid, side, left or right")
    spec = image.spec
    t0, t1, m0, m1, dt, dm = _view_frame(spec, time_range, midi_range, width, height)
    a, b = CHANNELS[channel]
    pixel_midi = m0 + (np.arange(height) + 0.5) * dm
    if mode == "cells":
        cells = 10 * np.log10(_cell_power(image, item, a, b) + 1e-30)
        return np.maximum(_cells_to_pixels(cells, spec, t0, m0, dt, dm, width, height), floor_db)
    sharp = mode == "sharp"
    margin = 3 * spec.bandwidth_scale + 1 if sharp else 1.5
    total = np.zeros(height * width)
    centres, rows_db = [], []
    for bands, carrier, times, sig, step in _tracks(image, item, m0 - margin, m1 + margin, t0, t1, dt):
        s = a * sig[:, 0] + b * sig[:, 1]
        power, freq, offset = _locate(s, carrier, sharp)
        times, duration, (power, pf, po) = _densify(times, step, [power, power * freq, power * offset], t0, t1, dt)
        weight = power * duration / dt
        if sharp:
            with np.errstate(divide="ignore", invalid="ignore"):
                freq = np.where(power > 0, pf / power, 0.0)
                offset = np.where(power > 0, po / power, 0.0)
            _scatter([total], [weight], times, offset, freq, bands, t0, m0, dt, dm, width, height)
        else:
            col = np.floor((times - t0) / dt).astype(np.int64)
            inside = (col >= 0) & (col < width)
            for r, band in enumerate(bands):
                centres.append(band.midi)
                rows_db.append(10 * np.log10(np.bincount(col[inside], weight[r, inside], width) + 1e-30))
    if sharp:
        power = _antialias(total.reshape(height, width), antialias)
        return np.maximum(10 * np.log10(power + 1e-30), floor_db)
    order = np.argsort(centres)
    centres, rows_db = np.asarray(centres, dtype=np.float64)[order], np.stack(rows_db)[order]
    if centres.size == 1:
        return np.maximum(np.repeat(rows_db, height, 0), floor_db)
    j = np.clip(np.searchsorted(centres, pixel_midi) - 1, 0, centres.size - 2)
    f = np.clip((pixel_midi - centres[j]) / (centres[j + 1] - centres[j]), 0.0, 1.0)[:, None]
    return np.maximum((1 - f) * rows_db[j] + f * rows_db[j + 1], floor_db)


def stereo_view(image: WaveletImage, *, mode: str = "sharp", item: int = 0, time_range=None, midi_range=None,
                width: int = 1200, height: int = 600, floor_db: float = -140.0, antialias: float = 0.7):
    """Stereo picture: three ``[height, width]`` maps (row 0 = lowest pitch), drawn ``"sharp"`` (reassigned,
    as detail_view), ``"exact"`` (each band at its own time resolution, blended between bands) or
    ``"cells"`` (per 10 ms x semitone cell).

    - level: left + right power in dB;
    - balance: (R − L) / (R + L) power: −1 = only left, 0 = centre, +1 = only right;
    - correlation: Re(L·conj R) / (|L||R|): +1 = in phase (mono), 0 = unrelated (wide), −1 = opposite phase.

    In the sharp picture each sample's energy goes to the power-weighted reassigned position of its left and
    right parts."""
    if mode not in ("sharp", "exact", "cells"):
        raise ValueError("mode: sharp, exact or cells")
    spec = image.spec
    t0, t1, m0, m1, dt, dm = _view_frame(spec, time_range, midi_range, width, height)
    if mode == "cells":
        pl, pr = _cell_power(image, item, 1.0, 1.0), _cell_power(image, item, 1.0, -1.0)
        cross = _cell_power(image, item, 1.0, 0.0) - _cell_power(image, item, 0.0, 1.0)  # Re(L conj R) = |M|^2 - |S|^2
        pl, pr, cross = (_cells_to_pixels(x, spec, t0, m0, dt, dm, width, height) for x in (pl, pr, cross))
        return _stereo_maps(pl, pr, cross, floor_db)
    acc = np.zeros((3, height * width))
    margin = 3 * spec.bandwidth_scale + 1 if mode == "sharp" else 1.5
    for bands, carrier, times, sig, step in _tracks(image, item, m0 - margin, m1 + margin, t0, t1, dt):
        left, right = sig[:, 0] + sig[:, 1], sig[:, 0] - sig[:, 1]
        pl, fl, ol = _locate(left, carrier, True)
        pr, fr, o_r = _locate(right, carrier, True)
        cross = np.real(left[0] * np.conj(right[0]))
        layers = [pl, pr, cross, pl * fl + pr * fr, pl * ol + pr * o_r]
        times, duration, (pl, pr, cross, pf, po) = _densify(times, step, layers, t0, t1, dt)
        both = pl + pr
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = np.where(both > 0, pf / both, 0.0)
            offset = np.where(both > 0, po / both, 0.0)
        scale = duration / dt
        _scatter(list(acc), [pl * scale, pr * scale, cross * scale], times, offset, freq, bands, t0, m0, dt, dm,
                 width, height, reassign=mode == "sharp")
    pl, pr, cross = (x.reshape(height, width) for x in acc)
    if mode == "sharp":
        pl, pr, cross = (_antialias(x, antialias) for x in (pl, pr, cross))
    return _stereo_maps(pl, pr, cross, floor_db)


def _stereo_maps(pl, pr, cross, floor_db):
    """Level (dB), balance and correlation from left power, right power and Re(L conj R) per pixel."""
    both = pl + pr
    with np.errstate(divide="ignore", invalid="ignore"):
        balance = np.where(both > 0, (pr - pl) / both, 0.0)
        correlation = np.where((pl > 0) & (pr > 0), np.clip(cross / np.sqrt(pl * pr), -1.0, 1.0), 0.0)
    return np.maximum(10 * np.log10(both + 1e-30), floor_db), balance, correlation
