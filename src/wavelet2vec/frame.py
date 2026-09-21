"""The invertible constant-Q complex frame — wavelet2vec's signal core (C-1).

Public API for lossless audio <-> coefficient-image transforms:

    coeffs, spec = analyze(audio, sample_rate)      # mono [N] or multi [C, N]
    audio        = synthesize(coeffs, spec)         # exact to ~300 dB
    save_coefficients(coeffs, "c.tiff", extra=spec.to_dict())
    coeffs, extra = load_coefficients("c.tiff")

Magnitude is ``abs(coeffs)``, phase is ``angle(coeffs)``; both at full time
resolution. The frame is exactly invertible because the Gaussian analysis
windows cover the whole spectrum (coverage > 0 everywhere), so synthesis just
divides the summed band spectra by the window sum. Multichannel input yields
one coefficient plane per channel, ``[C, bands, N]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameSpec:
    """Everything needed to invert (and to rebuild the analysis windows)."""

    centers: np.ndarray
    windows: np.ndarray  # [n_bands, n_freqs] on the rfft grid
    coverage: np.ndarray  # sum of windows; > 0 everywhere
    length: int
    sample_rate: int
    n_even: bool
    bandwidth_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "centers": self.centers.tolist(),
            "length": self.length,
            "sample_rate": self.sample_rate,
            "bandwidth_scale": self.bandwidth_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FrameSpec":
        centers = np.asarray(data["centers"], dtype=np.float64)
        length = int(data["length"])
        freqs = np.fft.rfftfreq(length, d=1.0 / data["sample_rate"])
        windows = _frame_windows(freqs.shape[0], freqs, centers, float(data["bandwidth_scale"]))
        return cls(
            centers=centers, windows=windows, coverage=windows.sum(axis=0),
            length=length, sample_rate=int(data["sample_rate"]),
            n_even=(length % 2 == 0), bandwidth_scale=float(data["bandwidth_scale"]),
        )


def _frame_windows(n_freqs: int, freqs: np.ndarray, centers: np.ndarray, bandwidth_scale: float) -> np.ndarray:
    ratio = float(centers[1] / centers[0]) if len(centers) > 1 else 2.0 ** (1 / 12)
    windows = np.zeros((len(centers), n_freqs))
    for index, center in enumerate(centers):
        sigma = max(center * (ratio - 1.0) * bandwidth_scale, 1e-9)
        windows[index] = np.exp(-0.5 * ((freqs - center) / sigma) ** 2)
    # Plateau the edge bands to DC and Nyquist so coverage never drops to zero
    # — the exact-inversion condition.
    windows[0][freqs <= centers[0]] = 1.0
    windows[-1][freqs >= centers[-1]] = 1.0
    return windows


def analyze(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 128,
    f_min: float = 20.0,
    f_max: float | None = None,
    bandwidth_scale: float = 1.0,
) -> tuple[np.ndarray, FrameSpec]:
    """Complex constant-Q coefficients at full time resolution.

    ``audio``: mono ``[N]`` -> coeffs ``[n_bands, N]``; multichannel ``[C, N]``
    -> coeffs ``[C, n_bands, N]`` (all channels share one ``FrameSpec``).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        planes = [analyze(channel, sample_rate, n_bands=n_bands, f_min=f_min,
                          f_max=f_max, bandwidth_scale=bandwidth_scale) for channel in audio]
        return np.stack([plane for plane, _spec in planes]), planes[0][1]

    n = audio.shape[0]
    f_max = f_max if f_max is not None else 0.5 * sample_rate
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    centers = np.geomspace(max(f_min, freqs[1] if len(freqs) > 1 else f_min), f_max, n_bands)
    windows = _frame_windows(freqs.shape[0], freqs, centers, bandwidth_scale)

    n_even = (n % 2 == 0)
    coeffs = np.empty((n_bands, n), dtype=np.complex128)
    full = np.zeros(n, dtype=np.complex128)
    half = freqs.shape[0]
    for index in range(n_bands):
        full[:] = 0.0
        band = spectrum * windows[index]
        full[:half] = 2.0 * band
        full[0] = band[0]
        if n_even:
            full[half - 1] = band[half - 1]
        coeffs[index] = np.fft.ifft(full)
    spec = FrameSpec(
        centers=centers, windows=windows, coverage=windows.sum(axis=0),
        length=n, sample_rate=int(sample_rate), n_even=n_even,
        bandwidth_scale=bandwidth_scale,
    )
    return coeffs, spec


def synthesize(coeffs: np.ndarray, spec: FrameSpec) -> np.ndarray:
    """Exact inverse of :func:`analyze` (per channel for 3-D input)."""
    if coeffs.ndim == 3:
        return np.stack([synthesize(plane, spec) for plane in coeffs])

    n = spec.length
    half = spec.windows.shape[1]
    summed = np.zeros(half, dtype=np.complex128)
    for index in range(coeffs.shape[0]):
        summed += np.fft.fft(coeffs[index])[:half]
    coverage = spec.coverage.copy()
    coverage[coverage < 1e-12] = 1.0
    spectrum = summed / (2.0 * coverage)
    spectrum[0] = summed[0] / coverage[0]
    if spec.n_even:
        spectrum[half - 1] = summed[half - 1] / coverage[half - 1]
    return np.fft.irfft(spectrum, n=n)


# --------------------------------------------------------------------------
# Storage (audio-as-image) and fidelity
# --------------------------------------------------------------------------
STORAGE_FORMATS = {
    ".tiff": "float32 image — perfect-grade (~160 dB, beyond 24-bit audio)",
    ".tif": "float32 image — perfect-grade (~160 dB, beyond 24-bit audio)",
    ".npy": "float64 array — bit-exact (~300 dB, machine precision)",
    ".png": "16-bit image — near-transparent (~84 dB)",
}


def _affine_to_uint(values: np.ndarray, levels: int) -> tuple[np.ndarray, float, float]:
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else 1.0
    quantized = np.round((values - lo) / span * levels).astype(np.uint16)
    return quantized, lo, span


def save_coefficients(
    coeffs: np.ndarray,
    path: str | Path,
    extra: dict | None = None,
    *,
    dtype: str = "float32",
) -> Path:
    """Store complex coefficients as an image/array; format from the extension.

    ``.tiff``/``.tif`` float image (default float32; ``dtype="float64"`` for
    exact), ``.npy`` float array, ``.png`` 16-bit integer image. A JSON
    sidecar holds shape, scales (png), and ``extra`` (e.g. ``spec.to_dict()``)
    so the file inverts back to audio on its own.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    sidecar = {
        "n_bands": int(coeffs.shape[0]),
        "n_samples": int(coeffs.shape[1]),
        "format": ext,
        "extra": extra or {},
    }

    if ext in (".tiff", ".tif"):
        import imageio.v3 as iio

        stored = "float64" if dtype == "float64" else "float32"
        iio.imwrite(path, np.vstack([coeffs.real, coeffs.imag]).astype(stored))
        sidecar["dtype"] = stored
    elif ext == ".npy":
        np.save(path, np.stack([coeffs.real, coeffs.imag]).astype(dtype))
        sidecar["dtype"] = dtype
    elif ext == ".png":
        import imageio.v3 as iio

        levels = 2**16 - 1
        real_q, real_lo, real_span = _affine_to_uint(coeffs.real, levels)
        imag_q, imag_lo, imag_span = _affine_to_uint(coeffs.imag, levels)
        iio.imwrite(path, np.vstack([real_q, imag_q]))
        sidecar.update({
            "levels": levels,
            "real": {"lo": real_lo, "span": real_span},
            "imag": {"lo": imag_lo, "span": imag_span},
        })
    else:
        raise ValueError(f"Unsupported coefficient format {ext!r}; use .tiff, .npy, or .png.")

    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def load_coefficients(path: str | Path) -> tuple[np.ndarray, dict]:
    """Inverse of :func:`save_coefficients`: file -> (complex coeffs, extra)."""
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    ext = path.suffix.lower()
    n_bands = sidecar["n_bands"]

    if ext == ".npy":
        arr = np.load(path).astype(np.float64)
        coeffs = arr[0] + 1j * arr[1]
    elif ext in (".tiff", ".tif"):
        import imageio.v3 as iio

        image = np.asarray(iio.imread(path)).astype(np.float64)
        coeffs = image[:n_bands] + 1j * image[n_bands:]
    elif ext == ".png":
        import imageio.v3 as iio

        image = np.asarray(iio.imread(path)).astype(np.float64)
        levels = sidecar["levels"]
        real = image[:n_bands] / levels * sidecar["real"]["span"] + sidecar["real"]["lo"]
        imag = image[n_bands:] / levels * sidecar["imag"]["span"] + sidecar["imag"]["lo"]
        coeffs = real + 1j * imag
    else:
        raise ValueError(f"Unsupported coefficient format {ext!r}.")
    return coeffs, sidecar["extra"]


def quantize_complex(coeffs: np.ndarray, bits: int) -> np.ndarray:
    """Quantize real+imag to ``bits`` per channel (storage-fidelity studies)."""
    levels = 2**bits - 1
    real = None
    out = np.empty_like(coeffs)
    for part in ("real", "imag"):
        values = getattr(coeffs, part)
        lo, hi = values.min(), values.max()
        span = hi - lo if hi > lo else 1.0
        quantized = np.round((values - lo) / span * levels) / levels * span + lo
        if part == "real":
            real = quantized
        else:
            out = real + 1j * quantized
    return out


def reconstruction_stats(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Max abs error and SNR (dB) of a round trip."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)[..., : original.shape[-1]]
    error = original - reconstructed
    signal_power = float(np.mean(original**2))
    noise_power = float(np.mean(error**2))
    snr = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(noise_power)),
        "snr_db": snr,
    }


# Backwards-compatible aliases (experiments/ and earlier docs used these).
WaveletMeta = FrameSpec
complex_wavelet = analyze
inverse_complex_wavelet = synthesize
