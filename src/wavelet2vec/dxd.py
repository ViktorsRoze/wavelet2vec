"""DXD (Digital eXtreme Definition) audio: 352.8 kHz integer PCM at 24 or 32 bits.

DXD files are ordinary PCM containers (usually WAV/RF64, sometimes FLAC or
AIFF) at 8 x 44.1 kHz. This module keeps their integer samples exact:

* ``load_dxd`` reads the stored integers and scales them to float without
  touching libsndfile's float conversion. 24-bit PCM is exact in float32 and
  float64; 32-bit PCM is exact only in float64, so a float32 load of a PCM_32
  file must be requested as lossy explicitly.
* ``save_dxd`` quantizes explicitly (round-to-nearest, optional seeded TPDF
  dither) and refuses samples outside the integer range instead of letting
  the writer clip silently. Clipping is available, but only on request.
* ``dxd_hybrid_config`` gives a 352.8 kHz frontend whose hop/padding keep the
  physical durations of 128/512 samples at 44.1 kHz (an exact 8x scaling).

No resampling happens here. DXD is an integer storage format; the float32
contract in ``HIGH_RESOLUTION_AUDIO.md`` still governs model computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency behavior
    sf = None

DXD_SAMPLE_RATE = 352800
DXD_BITS = {"PCM_24": 24, "PCM_32": 32}
DXD_SUBTYPES = tuple(DXD_BITS)
_FORMATS = {".wav": "WAV", ".rf64": "RF64", ".w64": "W64", ".flac": "FLAC", ".aif": "AIFF", ".aiff": "AIFF",
            ".caf": "CAF"}
# Same conservative RIFF limit as bounded rendering: leave header room below 4 GiB.
_WAV_LIMIT = 2**32 - 1024**2


def _require_soundfile():
    if sf is None:
        raise RuntimeError("DXD audio requires the soundfile package.")


@dataclass(frozen=True)
class DXDInfo:
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    format: str

    @property
    def bits(self) -> int | None:
        return DXD_BITS.get(self.subtype)

    @property
    def is_dxd(self) -> bool:
        return self.sample_rate == DXD_SAMPLE_RATE and self.subtype in DXD_BITS

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def dxd_info(path: str | Path) -> DXDInfo:
    """Header facts for any audio file; ``is_dxd`` says whether it is DXD PCM."""
    _require_soundfile()
    info = sf.info(Path(path))
    return DXDInfo(int(info.samplerate), int(info.channels), int(info.frames), info.subtype, info.format)


def load_dxd(path: str | Path, *, dtype=np.float64, allow_precision_loss: bool = False,
             strict: bool = True) -> tuple[np.ndarray, DXDInfo]:
    """Load a DXD file as ``[channels, samples]`` floats in [-1, 1) plus its header.

    Samples are integer / 2**(bits-1), computed from the stored integers.
    ``dtype=np.float64`` (default) is exact for 24- and 32-bit files. float32
    is exact for PCM_24; for PCM_32 it rounds away up to 8 bits, so it raises
    unless ``allow_precision_loss=True``. ``strict=False`` also accepts other
    integer PCM rates/depths (still exact, never resampled).
    """
    _require_soundfile()
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")
    info = dxd_info(path)
    if strict and not info.is_dxd:
        raise ValueError(f"not DXD: {info.sample_rate} Hz {info.subtype}; DXD is {DXD_SAMPLE_RATE} Hz "
                         "PCM_24/PCM_32 (pass strict=False to load other integer PCM)")
    if not info.subtype.startswith("PCM_") or info.subtype == "PCM_U8":
        raise ValueError(f"{info.subtype} is not signed integer PCM; use load_audio for float files")
    if dtype == np.float32 and info.subtype == "PCM_32" and not allow_precision_loss:
        raise ValueError("float32 cannot hold 32-bit PCM exactly; load as float64 or pass allow_precision_loss=True")
    # libsndfile left-justifies every integer PCM depth into int32 exactly.
    data, _ = sf.read(Path(path), dtype="int32", always_2d=True)
    waveform = data.T.astype(np.float64) * 2.0**-31
    return waveform.astype(dtype, copy=False), info


def quantize_pcm(waveform: np.ndarray, bits: int = 24, *, dither: str | None = None, seed: int | None = None,
                 clip: bool = False) -> np.ndarray:
    """Quantize float audio to ``bits``-bit integers, left-justified in int32.

    Full scale is 2**(bits-1): +1.0 itself is not representable, the largest
    positive code is 1 - 2**-(bits-1). Rounding is to nearest (ties to even).
    ``dither="tpdf"`` adds triangular dither of +-1 LSB, reproducible via
    ``seed``. Out-of-range samples raise unless ``clip=True``; NaN/inf raise.
    """
    if bits not in (24, 32):
        raise ValueError("DXD bit depth must be 24 or 32")
    if dither not in (None, "tpdf"):
        raise ValueError("dither must be None or 'tpdf'")
    values = np.asarray(waveform, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("audio contains NaN or infinite samples")
    scale = 2.0 ** (bits - 1)
    scaled = values * scale
    if dither == "tpdf":
        rng = np.random.default_rng(seed)
        scaled = scaled + rng.random(scaled.shape) - rng.random(scaled.shape)
    codes = np.rint(scaled)
    low, high = -scale, scale - 1
    if not clip and codes.size and (codes.min() < low or codes.max() > high):
        peak = float(np.max(np.abs(values)))
        raise ValueError(f"peak {peak:.9g} exceeds {bits}-bit PCM full scale ({high / scale:.9g}); "
                         "reduce gain or pass clip=True")
    codes = np.clip(codes, low, high).astype(np.int64)
    return (codes << (32 - bits)).astype(np.int32)


def _container(path: Path, subtype: str, frames: int, channels: int, requested: str) -> str:
    if requested != "auto":
        return requested
    format_name = _FORMATS.get(path.suffix.lower())
    if format_name is None:
        raise ValueError(f"unknown DXD container for {path.suffix!r}; pass container= explicitly")
    if format_name == "WAV" and frames * channels * DXD_BITS[subtype] // 8 >= _WAV_LIMIT:
        return "RF64"  # keeps the .wav name; plain RIFF cannot address it
    return format_name


def save_dxd(path: str | Path, waveform: np.ndarray, *, bits: int = 24, dither: str | None = None,
             seed: int | None = None, clip: bool = False, container: str = "auto",
             sample_rate: int = DXD_SAMPLE_RATE) -> DXDInfo:
    """Write ``[channels, samples]`` float audio as DXD PCM and verify the header.

    The rate is fixed at 352.8 kHz; audio at another rate must be resampled
    by the caller deliberately (``audio_io.resample``). The container follows
    the extension (large .wav files switch to RF64); FLAC stores only 24-bit.
    """
    _require_soundfile()
    if sample_rate != DXD_SAMPLE_RATE:
        raise ValueError(f"DXD is {DXD_SAMPLE_RATE} Hz; resample explicitly before saving")
    values = np.asarray(waveform)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("waveform must be [samples] or [channels, samples]")
    subtype = f"PCM_{bits}"
    if subtype not in DXD_BITS:
        raise ValueError("DXD bit depth must be 24 or 32")
    output = Path(path)
    format_name = _container(output, subtype, values.shape[1], values.shape[0], container)
    if not sf.check_format(format_name, subtype):
        raise ValueError(f"{format_name} cannot store {subtype}" + ("; use WAV/RF64/AIFF for 32-bit DXD"
                                                                     if format_name == "FLAC" else ""))
    codes = quantize_pcm(values, bits, dither=dither, seed=seed, clip=clip)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, codes.T, DXD_SAMPLE_RATE, subtype=subtype, format=format_name)
    info = dxd_info(output)
    if (info.sample_rate, info.channels, info.frames, info.subtype) != (DXD_SAMPLE_RATE, values.shape[0],
                                                                        values.shape[1], subtype):
        raise ValueError("written DXD header does not match the request")
    return info


def dxd_hybrid_config(**overrides):
    """A ``HybridConfig`` at 352.8 kHz with durations scaled exactly from 44.1 kHz.

    hop_length=1024 and padding_samples=4096 are 128/512 samples at 44.1 kHz.
    Like the 192 kHz example, this is a resolution preset, not a trained one.
    """
    from wavelet2vec.hybrid import HybridConfig

    settings = dict(sample_rate=DXD_SAMPLE_RATE, backend="compact_wavelet", n_bands=48, q=2,
                    hop_length=1024, padding_samples=4096)
    settings.update(overrides)
    return HybridConfig(**settings)
