"""DXD (352.8 kHz, 24/32-bit integer PCM) storage contract."""
import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import (
    DXD_SAMPLE_RATE, HybridSeparator, SourceRenderConfig, SynthesisConfig, Wavelet2Vec,
    dxd_hybrid_config, dxd_info, load_dxd, quantize_pcm, save_dxd, separate_file,
)


def on_grid(bits, shape, seed=0):
    """Random stereo audio that lies exactly on the ``bits``-bit PCM grid."""
    scale = 2 ** (bits - 1)
    codes = np.random.default_rng(seed).integers(-scale, scale, shape)
    codes[:, :2] = [[-scale, scale - 1]] * shape[0]  # both integer extremes
    return codes / scale


@pytest.mark.parametrize("suffix,bits", [(".wav", 24), (".wav", 32), (".flac", 24), (".aiff", 24), (".aiff", 32)])
def test_bit_exact_round_trip(tmp_path, suffix, bits):
    audio = on_grid(bits, (2, 4097))
    info = save_dxd(tmp_path / f"take{suffix}", audio, bits=bits)
    assert info.is_dxd and info.bits == bits and info.sample_rate == DXD_SAMPLE_RATE
    assert (info.channels, info.frames) == (2, 4097)
    restored, loaded = load_dxd(tmp_path / f"take{suffix}")
    assert restored.dtype == np.float64 and loaded == info
    np.testing.assert_array_equal(restored, audio)


def test_float32_is_exact_for_24_bit_and_explicitly_lossy_for_32_bit(tmp_path):
    save_dxd(tmp_path / "a24.wav", on_grid(24, (1, 300)), bits=24)
    restored, _ = load_dxd(tmp_path / "a24.wav", dtype=np.float32)
    np.testing.assert_array_equal(restored, on_grid(24, (1, 300)).astype(np.float32))
    save_dxd(tmp_path / "a32.wav", on_grid(32, (1, 300)), bits=32)
    with pytest.raises(ValueError, match="float32 cannot hold"):
        load_dxd(tmp_path / "a32.wav", dtype=np.float32)
    lossy, _ = load_dxd(tmp_path / "a32.wav", dtype=np.float32, allow_precision_loss=True)
    assert lossy.dtype == np.float32


def test_quantization_rounds_and_refuses_silent_clipping():
    lsb = 2.0 ** -23
    codes = quantize_pcm(np.array([[0.4 * lsb, 0.6 * lsb, -0.6 * lsb, 1 - lsb]]), 24) >> 8
    assert codes.tolist() == [[0, 1, -1, 2**23 - 1]]
    for peak in (1.0, -1.0 - lsb):
        with pytest.raises(ValueError, match="full scale"):
            quantize_pcm(np.array([peak]), 24)
    assert (quantize_pcm(np.array([1.5, -1.5]), 32, clip=True) == [2**31 - 1, -2**31]).all()
    with pytest.raises(ValueError, match="NaN"):
        quantize_pcm(np.array([np.nan]), 24)


def test_tpdf_dither_is_seeded_and_bounded():
    x = np.full((2, 5000), 0.25 + 0.3 * 2.0 ** -23)
    a = quantize_pcm(x, 24, dither="tpdf", seed=3)
    np.testing.assert_array_equal(a, quantize_pcm(x, 24, dither="tpdf", seed=3))
    error = (a >> 8) - x * 2**23
    assert np.abs(error).max() <= 1.5 and abs(error.mean()) < 0.05  # unbiased, within +-1 LSB of dither


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="FLAC cannot store PCM_32"):
        save_dxd(tmp_path / "x.flac", np.zeros((1, 8)), bits=32)
    with pytest.raises(ValueError, match="resample explicitly"):
        save_dxd(tmp_path / "x.wav", np.zeros((1, 8)), sample_rate=192000)
    with pytest.raises(ValueError, match="24 or 32"):
        save_dxd(tmp_path / "x.wav", np.zeros((1, 8)), bits=16)
    sf.write(tmp_path / "float.wav", np.zeros(8, np.float32), DXD_SAMPLE_RATE, subtype="FLOAT")
    sf.write(tmp_path / "cd.wav", np.zeros(8, np.float32), 44100, subtype="PCM_24")
    assert not dxd_info(tmp_path / "float.wav").is_dxd and not dxd_info(tmp_path / "cd.wav").is_dxd
    with pytest.raises(ValueError, match="not DXD"):
        load_dxd(tmp_path / "cd.wav")
    with pytest.raises(ValueError, match="not signed integer PCM"):
        load_dxd(tmp_path / "float.wav", strict=False)
    restored, info = load_dxd(tmp_path / "cd.wav", strict=False)
    assert info.sample_rate == 44100 and restored.shape == (1, 8)


def test_large_wav_selects_rf64():
    from wavelet2vec.dxd import _container
    from pathlib import Path
    assert _container(Path("a.wav"), "PCM_32", 2**29, 2, "auto") == "RF64"
    assert _container(Path("a.wav"), "PCM_24", 2**20, 2, "auto") == "WAV"


def test_embedding_from_dxd_file(tmp_path):
    t = np.arange(DXD_SAMPLE_RATE // 4) / DXD_SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-6 * t)
    save_dxd(tmp_path / "tone.wav", np.stack([tone, tone]), bits=24)
    embedder = Wavelet2Vec()
    from_file = embedder.embed_file(tmp_path / "tone.wav")
    reference = embedder.embed(np.stack([tone, tone]).astype(np.float32), DXD_SAMPLE_RATE)
    assert np.isfinite(from_file).all() and from_file.shape == reference.shape
    np.testing.assert_allclose(from_file, reference, atol=1e-4)


def test_dxd_file_renders_through_native_rate_separator(tmp_path):
    cfg = dxd_hybrid_config(n_bands=12, q=1, hop_length=16, padding_samples=32, f_min=40)
    assert cfg.sample_rate == DXD_SAMPLE_RATE and dxd_hybrid_config().hop_length == 1024
    torch.manual_seed(0)
    model = HybridSeparator(cfg, SynthesisConfig(dim=8))
    audio = on_grid(24, (2, 777), seed=4) * 0.1
    save_dxd(tmp_path / "mix.wav", audio, bits=24)
    result = separate_file(model, tmp_path / "mix.wav", tmp_path / "stems", SourceRenderConfig(256, 64, 64))
    assert result.sample_rate == DXD_SAMPLE_RATE and result.frames == 777 and result.subtype == "FLOAT"
    total = sum(sf.read(path, dtype="float64", always_2d=True)[0].T for path in result.files)
    np.testing.assert_allclose(total, audio, atol=1e-6)
