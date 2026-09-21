"""GPU/torch backend for the invertible constant-Q frame (roadmap C-3).

Mirrors :func:`wavelet2vec.frame.analyze` / :func:`synthesize` with torch
FFTs, so the same :class:`~wavelet2vec.frame.FrameSpec` inverts either way —
analyze on the GPU, synthesize on CPU numpy, or any mix. Runs on ``"cpu"`` or
``"cuda"`` (``device=...``) and batches over any leading dimensions:
``[..., N] -> [..., n_bands, N]``.

The payoff is throughput on **many short fragments at once** — the compute
unit of the whole library — not single-clip latency. The numpy path in
``frame.py`` is untouched and remains the default.

Numerics: with ``precision="double"`` (default) results match the numpy
frame to machine precision; ``precision="single"`` (complex64) is the fast
GPU path and round-trips to ~130 dB, still beyond 16-bit audio.
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.frame import FrameSpec, _frame_windows

try:  # torch is an optional backend
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None
    HAVE_TORCH = False

_DTYPES = {
    "double": ("float64", "complex128"),
    "single": ("float32", "complex64"),
}


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise RuntimeError(
            "The torch backend needs PyTorch. Install it or use the numpy "
            "wavelet2vec.frame.analyze/synthesize."
        )


def _resolve(precision: str):
    if precision not in _DTYPES:
        raise ValueError("precision must be 'double' or 'single'.")
    real_name, complex_name = _DTYPES[precision]
    return getattr(torch, real_name), getattr(torch, complex_name)


def _spec_for(n: int, sample_rate: int, n_bands: int, f_min: float,
              f_max: float | None, bandwidth_scale: float) -> FrameSpec:
    f_max = f_max if f_max is not None else 0.5 * sample_rate
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    centers = np.geomspace(
        max(f_min, freqs[1] if len(freqs) > 1 else f_min), f_max, n_bands
    )
    windows = _frame_windows(freqs.shape[0], freqs, centers, bandwidth_scale)
    return FrameSpec(
        centers=centers, windows=windows, coverage=windows.sum(axis=0),
        length=n, sample_rate=int(sample_rate), n_even=(n % 2 == 0),
        bandwidth_scale=bandwidth_scale,
    )


def analyze_torch(
    audio,
    sample_rate: int,
    *,
    n_bands: int = 128,
    f_min: float = 20.0,
    f_max: float | None = None,
    bandwidth_scale: float = 1.0,
    device=None,
    precision: str = "double",
):
    """Torch constant-Q analysis. ``audio`` is an array/tensor ``[..., N]``
    with any leading batch dimensions.

    Returns ``(coeffs, spec)`` where ``coeffs`` is a complex torch tensor
    ``[..., n_bands, N]`` on ``device`` and ``spec`` is a numpy
    :class:`FrameSpec` (interchangeable with the numpy backend).
    """
    _require_torch()
    real_dtype, complex_dtype = _resolve(precision)
    audio_t = torch.as_tensor(np.asarray(audio) if not torch.is_tensor(audio) else audio)
    audio_t = audio_t.to(device=device, dtype=real_dtype)
    n = audio_t.shape[-1]
    half = n // 2 + 1

    spec = _spec_for(n, sample_rate, n_bands, f_min, f_max, bandwidth_scale)
    windows_t = torch.as_tensor(spec.windows, device=audio_t.device, dtype=real_dtype)

    spectrum = torch.fft.rfft(audio_t, dim=-1)  # [..., half]
    band = spectrum.unsqueeze(-2) * windows_t  # [..., n_bands, half]

    full = torch.zeros(*band.shape[:-1], n, dtype=complex_dtype, device=audio_t.device)
    full[..., :half] = 2.0 * band
    full[..., 0] = band[..., 0]  # DC is not doubled
    if spec.n_even:
        full[..., half - 1] = band[..., half - 1]  # nor is Nyquist
    coeffs = torch.fft.ifft(full, dim=-1)
    return coeffs, spec


def synthesize_torch(coeffs, spec: FrameSpec, *, device=None, precision: str = "double"):
    """Exact torch inverse of :func:`analyze_torch`.

    ``coeffs`` is ``[..., n_bands, N]`` (torch tensor or array); returns a
    real torch tensor ``[..., N]``.
    """
    _require_torch()
    real_dtype, complex_dtype = _resolve(precision)
    coeffs_t = coeffs if torch.is_tensor(coeffs) else torch.as_tensor(np.asarray(coeffs))
    coeffs_t = coeffs_t.to(device=device, dtype=complex_dtype)
    n = spec.length
    half = spec.windows.shape[1]

    band_ffts = torch.fft.fft(coeffs_t, dim=-1)[..., :half]  # [..., n_bands, half]
    summed = band_ffts.sum(dim=-2)  # [..., half]

    coverage = torch.as_tensor(spec.coverage, device=coeffs_t.device, dtype=real_dtype)
    coverage = torch.where(coverage < 1e-12, torch.ones_like(coverage), coverage)
    spectrum = summed / (2.0 * coverage)
    spectrum[..., 0] = summed[..., 0] / coverage[0]
    if spec.n_even:
        spectrum[..., half - 1] = summed[..., half - 1] / coverage[half - 1]
    return torch.fft.irfft(spectrum, n=n, dim=-1)


def analyze_batch(
    audios,
    sample_rate: int,
    *,
    n_bands: int = 128,
    device=None,
    precision: str = "single",
    **kwargs,
):
    """Analyze a stack of equal-length fragments ``[B, N]`` in one pass.

    The batched short-fragment path is where the GPU (and even the torch CPU
    kernels) pull ahead of the per-band numpy loop. Defaults to single
    precision — the fast inference setting. Returns ``(coeffs [B, n_bands, N],
    spec)``.
    """
    stacked = np.asarray(audios)
    if stacked.ndim != 2:
        raise ValueError("analyze_batch expects [B, N] equal-length fragments.")
    return analyze_torch(
        stacked, sample_rate, n_bands=n_bands, device=device,
        precision=precision, **kwargs,
    )
