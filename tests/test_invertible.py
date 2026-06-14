"""Lossless round-trip tests for the complex coefficient transforms."""

from __future__ import annotations

import numpy as np

from experiments.invertible import (
    WaveletMeta,
    complex_stft,
    complex_wavelet,
    inverse_complex_stft,
    inverse_complex_wavelet,
    load_coefficients,
    quantize_complex,
    reconstruction_stats,
    save_coefficients,
)

SR = 22050


def _signals():
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    return {
        "noise": rng.standard_normal(SR),
        "sine": np.sin(2 * np.pi * 440 * t),
        "transient": np.concatenate([np.zeros(300), rng.standard_normal(200), 0.3 * np.sin(2 * np.pi * 200 * t[:5000])]),
    }


def test_complex_stft_is_lossless():
    for name, x in _signals().items():
        coeffs, meta = complex_stft(x, n_fft=1024)
        rec = inverse_complex_stft(coeffs, meta)
        stats = reconstruction_stats(x, rec)
        assert stats["snr_db"] > 200, (name, stats)
        assert stats["max_abs_error"] < 1e-9, (name, stats)


def test_complex_wavelet_is_lossless():
    for name, x in _signals().items():
        coeffs, meta = complex_wavelet(x, SR, n_bands=96)
        assert coeffs.shape == (96, x.shape[0])
        assert meta.coverage.min() > 0  # full spectral coverage -> invertible
        rec = inverse_complex_wavelet(coeffs, meta)
        stats = reconstruction_stats(x, rec)
        assert stats["snr_db"] > 200, (name, stats)


def test_complex_wavelet_handles_odd_length():
    x = np.sin(2 * np.pi * 330 * np.arange(4097) / SR)  # odd N, no Nyquist bin
    coeffs, meta = complex_wavelet(x, SR, n_bands=64)
    rec = inverse_complex_wavelet(coeffs, meta)
    assert reconstruction_stats(x, rec)["snr_db"] > 200


def test_magnitude_only_loses_information():
    # Discarding phase must wreck reconstruction — this is why we keep it.
    x = _signals()["transient"]
    coeffs, meta = complex_wavelet(x, SR, n_bands=96)
    full = inverse_complex_wavelet(coeffs, meta)
    mag_only = inverse_complex_wavelet(np.abs(coeffs).astype(np.complex128), meta)
    assert reconstruction_stats(x, full)["snr_db"] > 200
    assert reconstruction_stats(x, mag_only)["snr_db"] < 30


def test_quantization_snr_increases_with_bit_depth():
    x = _signals()["sine"]
    coeffs, meta = complex_wavelet(x, SR, n_bands=96)
    snrs = {}
    for bits in (8, 16):
        rec = inverse_complex_wavelet(quantize_complex(coeffs, bits), meta)
        snrs[bits] = reconstruction_stats(x, rec)["snr_db"]
    exact = reconstruction_stats(x, inverse_complex_wavelet(coeffs, meta))["snr_db"]
    assert snrs[8] < snrs[16] < exact  # more bits -> closer to lossless


def test_base_case_float_image_round_trip(tmp_path):
    # Base case: audio -> coefficients -> float32 image on disk -> audio.
    x = _signals()["sine"]
    coeffs, meta = complex_wavelet(x, SR, n_bands=96)
    path = save_coefficients(coeffs, tmp_path / "coeffs.tiff", extra=meta.to_dict())
    assert path.exists() and path.with_suffix(".json").exists()

    loaded, extra = load_coefficients(path)
    assert loaded.shape == coeffs.shape
    rec = inverse_complex_wavelet(loaded, WaveletMeta.from_dict(extra))
    # float32 image is perfect-grade — beyond 24-bit audio (~144 dB).
    assert reconstruction_stats(x, rec)["snr_db"] > 140


def test_float64_npy_is_bit_exact(tmp_path):
    x = _signals()["transient"]
    coeffs, meta = complex_wavelet(x, SR, n_bands=96)
    path = save_coefficients(coeffs, tmp_path / "c.npy", extra=meta.to_dict(), dtype="float64")
    loaded, extra = load_coefficients(path)
    rec = inverse_complex_wavelet(loaded, WaveletMeta.from_dict(extra))
    assert reconstruction_stats(x, rec)["snr_db"] > 250  # machine precision


def test_storage_formats_ordered_by_fidelity(tmp_path):
    x = _signals()["noise"]
    coeffs, meta = complex_wavelet(x, SR, n_bands=64)
    snr = {}
    for name, ext, kw in [("f64", ".npy", {"dtype": "float64"}), ("f32", ".tiff", {}), ("png", ".png", {})]:
        loaded, extra = load_coefficients(save_coefficients(coeffs, tmp_path / f"{name}{ext}", extra=meta.to_dict(), **kw))
        snr[name] = reconstruction_stats(x, inverse_complex_wavelet(loaded, WaveletMeta.from_dict(extra)))["snr_db"]
    assert snr["png"] < snr["f32"] < snr["f64"]


def test_float_image_is_actual_float32(tmp_path):
    import imageio.v3 as iio

    coeffs, _ = complex_wavelet(_signals()["noise"], SR, n_bands=64)
    image = iio.imread(save_coefficients(coeffs, tmp_path / "c.tiff"))
    assert image.dtype == np.float32
    assert image.shape == (2 * 64, coeffs.shape[1])  # real and imag stacked


def test_reconstruction_stats_identical_is_infinite():
    x = _signals()["sine"]
    stats = reconstruction_stats(x, x.copy())
    assert stats["snr_db"] == float("inf")
    assert stats["max_abs_error"] == 0.0
