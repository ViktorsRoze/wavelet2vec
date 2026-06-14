"""Lossless audio <-> image round trips via full complex coefficients.

The point: a *magnitude-only* spectrogram (mel, CQT magnitude, Griffin-Lim)
throws phase away and cannot reconstruct the audio. Keep the **full complex
coefficients** (magnitude *and* phase) at full resolution and the transform is
exactly invertible — the image literally is the audio. Two transforms are
provided:

- :func:`complex_stft` / :func:`inverse_complex_stft` — a COLA Hann STFT,
  exact to floating-point precision.
- :func:`complex_wavelet` / :func:`inverse_complex_wavelet` — an invertible
  constant-Q complex Morlet frame (the wavelet2vec-flavored transform), with
  log-frequency layout and per-sample time resolution.

Because the snippets are short, both keep every coefficient (no hop decimation
in the wavelet case), so magnitude and phase are stored at perfect resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Exact complex STFT (COLA Hann, 75% overlap)
# --------------------------------------------------------------------------
@dataclass
class STFTMeta:
    n_fft: int
    hop: int
    length: int
    pad: int


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)  # periodic (COLA)


def complex_stft(mono: np.ndarray, n_fft: int = 1024, hop: int | None = None) -> tuple[np.ndarray, STFTMeta]:
    """Full complex STFT, shape ``[n_fft//2+1, frames]``. Lossless with iSTFT."""
    mono = np.asarray(mono, dtype=np.float64)
    hop = hop or n_fft // 4
    pad = n_fft
    padded = np.concatenate([np.zeros(pad), mono, np.zeros(pad + n_fft)])
    window = _hann(n_fft)
    starts = range(0, len(padded) - n_fft + 1, hop)
    frames = np.stack([np.fft.rfft(padded[s : s + n_fft] * window) for s in starts], axis=1)
    return frames, STFTMeta(n_fft=n_fft, hop=hop, length=mono.shape[0], pad=pad)


def inverse_complex_stft(coeffs: np.ndarray, meta: STFTMeta) -> np.ndarray:
    """Exact inverse of :func:`complex_stft`."""
    window = _hann(meta.n_fft)
    total = meta.pad + meta.length + meta.pad + meta.n_fft
    out = np.zeros(total)
    norm = np.zeros(total)
    for index in range(coeffs.shape[1]):
        frame = np.fft.irfft(coeffs[:, index], n=meta.n_fft)
        start = index * meta.hop
        out[start : start + meta.n_fft] += frame * window
        norm[start : start + meta.n_fft] += window**2
    norm[norm < 1e-12] = 1.0
    recovered = out / norm
    return recovered[meta.pad : meta.pad + meta.length]


# --------------------------------------------------------------------------
# Invertible constant-Q complex Morlet frame (full time resolution)
# --------------------------------------------------------------------------
@dataclass
class WaveletMeta:
    centers: np.ndarray
    windows: np.ndarray  # [n_bands, n_freqs] real, on the rfft grid
    coverage: np.ndarray  # P(f) = sum_k windows; > 0 everywhere
    length: int
    sample_rate: int
    n_even: bool
    bandwidth_scale: float = 1.0

    def to_dict(self) -> dict:
        """Serializable parameters; windows/coverage are rebuilt from these."""
        return {
            "centers": self.centers.tolist(),
            "length": self.length,
            "sample_rate": self.sample_rate,
            "bandwidth_scale": self.bandwidth_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WaveletMeta":
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
    # Plateau the extreme bands to DC and Nyquist so coverage P(f) > 0 over the
    # whole band — the condition for exact inversion.
    windows[0][freqs <= centers[0]] = 1.0
    windows[-1][freqs >= centers[-1]] = 1.0
    return windows


def complex_wavelet(
    mono: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 128,
    f_min: float = 20.0,
    f_max: float | None = None,
    bandwidth_scale: float = 1.0,
) -> tuple[np.ndarray, WaveletMeta]:
    """Invertible constant-Q complex Morlet coefficients, shape ``[n_bands, n]``.

    Each row is an analytic band signal at full time resolution: ``|C_k|`` is
    the band envelope, ``angle(C_k)`` the band phase. Inverse is exact to
    floating-point precision (see :func:`inverse_complex_wavelet`).
    """
    mono = np.asarray(mono, dtype=np.float64)
    n = mono.shape[0]
    f_max = f_max if f_max is not None else 0.5 * sample_rate
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    centers = np.geomspace(max(f_min, freqs[1] if len(freqs) > 1 else f_min), f_max, n_bands)
    windows = _frame_windows(freqs.shape[0], freqs, centers, bandwidth_scale)
    coverage = windows.sum(axis=0)

    n_even = (n % 2 == 0)
    coeffs = np.empty((n_bands, n), dtype=np.complex128)
    full = np.zeros(n, dtype=np.complex128)
    half = freqs.shape[0]
    for index in range(n_bands):
        full[:] = 0.0
        band = spectrum * windows[index]
        full[:half] = 2.0 * band
        full[0] = band[0]  # DC not doubled
        if n_even:
            full[half - 1] = band[half - 1]  # Nyquist not doubled
        coeffs[index] = np.fft.ifft(full)
    meta = WaveletMeta(
        centers=centers, windows=windows, coverage=coverage,
        length=n, sample_rate=sample_rate, n_even=n_even, bandwidth_scale=bandwidth_scale,
    )
    return coeffs, meta


def inverse_complex_wavelet(coeffs: np.ndarray, meta: WaveletMeta) -> np.ndarray:
    """Exact inverse of :func:`complex_wavelet`."""
    n = meta.length
    half = meta.windows.shape[1]
    summed = np.zeros(half, dtype=np.complex128)
    for index in range(coeffs.shape[0]):
        band_spectrum = np.fft.fft(coeffs[index])[:half]
        summed += band_spectrum
    coverage = meta.coverage.copy()
    coverage[coverage < 1e-12] = 1.0
    spectrum = summed / (2.0 * coverage)
    spectrum[0] = summed[0] / coverage[0]  # undo the DC (×1) scaling
    if meta.n_even:
        spectrum[half - 1] = summed[half - 1] / coverage[half - 1]
    return np.fft.irfft(spectrum, n=n)


# --------------------------------------------------------------------------
# Reconstruction & storage-quantization analysis
# --------------------------------------------------------------------------
def reconstruction_stats(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Max abs error and signal-to-noise ratio (dB) of a round trip."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)[: original.shape[0]]
    error = original - reconstructed
    signal_power = float(np.mean(original**2))
    noise_power = float(np.mean(error**2))
    snr = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(noise_power)),
        "snr_db": snr,
    }


def _affine_to_uint(values: np.ndarray, levels: int) -> tuple[np.ndarray, float, float]:
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else 1.0
    quantized = np.round((values - lo) / span * levels).astype(np.uint16)
    return quantized, lo, span


# Reconstruction SNR by storage format (constant-Q wavelet coefficients of
# typical material): float keeps phase + magnitude essentially intact, integer
# image formats trade precision for a smaller, viewable picture.
STORAGE_FORMATS = {
    ".tiff": "float32 image — perfect-grade (~160 dB, beyond 24-bit audio)",
    ".tif": "float32 image — perfect-grade (~160 dB, beyond 24-bit audio)",
    ".npy": "float64 array — bit-exact (~300 dB, machine precision)",
    ".png": "16-bit image — near-transparent (~84 dB)",
}


def save_coefficients(
    coeffs: np.ndarray,
    path: str | Path,
    extra: dict | None = None,
    *,
    dtype: str = "float32",
) -> Path:
    """Store complex coefficients as an image/array; format from the extension.

    The base case is lossless float storage — real and imaginary parts stacked
    into one image:

    - ``.tiff`` / ``.tif`` — float image (default ``float32``: ~160 dB,
      perfect-grade and viewable; pass ``dtype="float64"`` for machine-exact).
    - ``.npy`` — numpy float array (``float64`` = bit-exact ~300 dB).
    - ``.png`` — 16-bit integer image, near-transparent (~84 dB), most compact.

    A small JSON sidecar holds the shape, any ``extra`` metadata (e.g. the
    wavelet parameters, so the file inverts back to audio on its own), and —
    for PNG only — the per-channel quantization scale. Returns the file path.
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
    """Quantize real+imag of coefficients to ``bits`` per channel.

    Models storing the coefficient 'image' in an integer image format: 8-bit
    (ordinary PNG) is visibly lossy, 16-bit is near-transparent, 32-bit float
    is exact. Used to show why a truly lossless picture needs float storage.
    """
    levels = 2**bits - 1
    out = np.empty_like(coeffs)
    for part in ("real", "imag"):
        values = getattr(coeffs, part)
        lo, hi = values.min(), values.max()
        span = hi - lo if hi > lo else 1.0
        q = np.round((values - lo) / span * levels) / levels * span + lo
        if part == "real":
            real = q
        else:
            out = real + 1j * q
    return out
