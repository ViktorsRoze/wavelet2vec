from __future__ import annotations

import math

import numpy as np

from wavelet2vec import (
    Wavelet2Vec,
    Wavelet2VecConfig,
    WaveletConvEncoder,
    blend,
    cosine_similarity,
    style_transfer,
)

SAMPLE_RATE = 22050


def _sine(frequency: float, seconds: float = 1.0, phase: float = 0.0) -> np.ndarray:
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return np.sin(2 * math.pi * frequency * times + phase)[None, :].astype(np.float32)


def _noise(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, int(SAMPLE_RATE * seconds))).astype(np.float32)


def _click(seconds: float = 1.0) -> np.ndarray:
    waveform = np.zeros((1, int(SAMPLE_RATE * seconds)), dtype=np.float32)
    waveform[0, 100:130] = 1.0
    return waveform


def _harmonic_tone(
    f0: float,
    seconds: float = 1.0,
    n_harmonics: int = 16,
    harmonic_phases: list[float] | None = None,
) -> np.ndarray:
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    waveform = np.zeros_like(times)
    for k in range(1, n_harmonics + 1):
        phase = harmonic_phases[k - 1] if harmonic_phases else 0.0
        waveform += np.sin(2 * math.pi * k * f0 * times + phase) / k
    return waveform[None, :].astype(np.float32)


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


def test_embedding_shape_and_determinism():
    embedder = Wavelet2Vec()
    sine = _sine(440.0)
    first = embedder.embed(sine, SAMPLE_RATE)
    second = embedder.embed(sine, SAMPLE_RATE)
    assert first.shape == (embedder.dim,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_phase_shift_invariance_beats_noise():
    embedder = Wavelet2Vec()
    sine = embedder.embed(_sine(440.0), SAMPLE_RATE)
    shifted = embedder.embed(_sine(440.0, phase=math.pi / 3), SAMPLE_RATE)
    noise = embedder.embed(_noise(), SAMPLE_RATE)

    same_pitch = cosine_similarity(sine, shifted)
    different_sound = cosine_similarity(sine, noise)
    assert same_pitch > 0.99
    assert same_pitch > different_sound + 0.1


def test_chroma_identifies_pitch_class():
    embedder = Wavelet2Vec()
    components = embedder.embed_components(_sine(440.0), SAMPLE_RATE)
    chroma = components["harmonic"][:12]
    assert int(np.argmax(chroma)) == 9  # A pitch class


def test_harmonicity_separates_tone_from_noise():
    embedder = Wavelet2Vec()
    tone = embedder.embed_components(_sine(220.0), SAMPLE_RATE)
    noise = embedder.embed_components(_noise(), SAMPLE_RATE)
    harmonicity_index = 12
    assert tone["harmonic"][harmonicity_index] > noise["harmonic"][harmonicity_index] + 0.3


def test_transient_attack_separates_click_from_pad():
    embedder = Wavelet2Vec()
    pad = _sine(220.0) * np.linspace(0, 1, int(SAMPLE_RATE * 1.0), dtype=np.float32)[None, :]
    click_components = embedder.embed_components(_click(), SAMPLE_RATE)
    pad_components = embedder.embed_components(pad, SAMPLE_RATE)
    attack_index = embedder.config.envelope_points  # first scalar after envelope shape
    assert click_components["transient"][attack_index] < pad_components["transient"][attack_index] - 0.2


def test_phase_signature_distinguishes_waveshape():
    # A saw-like tone and its phase-scrambled twin have identical magnitude
    # spectra; only the relative phase of the harmonics (the waveshape) differs.
    embedder = Wavelet2Vec()
    rng = np.random.default_rng(7)
    scrambled_phases = list(rng.uniform(0, 2 * math.pi, size=16))
    saw = embedder.embed_components(_harmonic_tone(110.0), SAMPLE_RATE)
    scrambled = embedder.embed_components(
        _harmonic_tone(110.0, harmonic_phases=scrambled_phases), SAMPLE_RATE
    )

    spectral_sim = float(_normalized(saw["spectral"]) @ _normalized(scrambled["spectral"]))
    phase_sim = float(_normalized(saw["phase"]) @ _normalized(scrambled["phase"]))
    assert spectral_sim > 0.99
    assert phase_sim < spectral_sim - 0.05


def test_phase_features_are_shift_invariant():
    embedder = Wavelet2Vec()
    saw = _harmonic_tone(110.0)
    shifted = np.roll(saw, shift=int(SAMPLE_RATE * 0.0037), axis=1)
    original = embedder.embed(saw, SAMPLE_RATE)
    rolled = embedder.embed(shifted, SAMPLE_RATE)
    assert cosine_similarity(original, rolled) > 0.98


def test_conv_encoder_deterministic_and_frozen():
    encoder = WaveletConvEncoder()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    mono = np.random.default_rng(3).standard_normal(SAMPLE_RATE).astype(np.float32)
    first = encoder.encode_mono(mono)
    second = encoder.encode_mono(mono)
    assert first.shape == (encoder.output_dim,)
    assert np.array_equal(first, second)

    trainable = WaveletConvEncoder(trainable=True)
    assert all(parameter.requires_grad for parameter in trainable.parameters())


def test_conv_section_can_be_disabled():
    embedder = Wavelet2Vec(Wavelet2VecConfig(include_conv=False))
    embedding = embedder.embed(_sine(440.0), SAMPLE_RATE)
    assert "conv" not in embedder.sections
    assert embedding.shape == (embedder.dim,)


def test_stereo_section_distinguishes_width():
    embedder = Wavelet2Vec()
    rng = np.random.default_rng(11)
    mono_noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    dual_mono = np.stack([mono_noise, mono_noise])
    wide = rng.standard_normal((2, SAMPLE_RATE)).astype(np.float32)

    centered = embedder.embed_components(dual_mono, SAMPLE_RATE)["stereo"]
    decorrelated = embedder.embed_components(wide, SAMPLE_RATE)["stereo"]
    n_groups = embedder.config.n_stereo_groups

    assert centered[0] < 0.01  # no side energy when channels are identical
    assert decorrelated[0] > centered[0] + 0.3
    assert decorrelated[1] > 0.3  # decorrelation
    coherence = slice(3 + n_groups, 3 + 2 * n_groups)
    assert centered[coherence].mean() > decorrelated[coherence].mean() + 0.3


def test_stereo_balance_detects_panning():
    embedder = Wavelet2Vec()
    signal = _sine(330.0)[0]
    left_only = np.stack([signal, np.zeros_like(signal)])
    features = embedder.embed_components(left_only, SAMPLE_RATE)["stereo"]
    assert features[2] < -0.9  # global balance fully left


def test_mono_input_gets_neutral_stereo_section():
    embedder = Wavelet2Vec()
    features = embedder.embed_components(_sine(440.0), SAMPLE_RATE)["stereo"]
    n_groups = embedder.config.n_stereo_groups
    assert np.allclose(features[:3], 0.0)
    assert np.allclose(features[3 + n_groups : 3 + 2 * n_groups], 1.0)


def test_style_transfer_moves_embedding_toward_target():
    embedder = Wavelet2Vec()
    rng = np.random.default_rng(5)
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    dark = (np.sin(2 * math.pi * 150.0 * times) * np.exp(-times * 3.0))[None, :].astype(np.float32)
    bright = (rng.standard_normal(SAMPLE_RATE) * np.exp(-times * 12.0))[None, :].astype(np.float32)

    target_vec = embedder.embed(bright, SAMPLE_RATE)
    similarities = []
    for amount in (0.0, 0.4, 0.8):
        morphed = style_transfer(dark, SAMPLE_RATE, bright, SAMPLE_RATE, amount)
        assert morphed.shape == dark.shape
        assert np.isfinite(morphed).all()
        similarities.append(cosine_similarity(embedder.embed(morphed, SAMPLE_RATE), target_vec))

    untouched = style_transfer(dark, SAMPLE_RATE, bright, SAMPLE_RATE, 0.0)
    source_vec = embedder.embed(dark, SAMPLE_RATE)
    assert cosine_similarity(embedder.embed(untouched, SAMPLE_RATE), source_vec) > 0.99
    assert similarities[0] < similarities[1] < similarities[2]


def test_blend_with_full_weight_returns_carrier():
    rng = np.random.default_rng(9)
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    pluck = (np.sin(2 * math.pi * 220.0 * times) * np.exp(-times * 6.0))[None, :].astype(np.float32)
    noise = rng.standard_normal((1, SAMPLE_RATE)).astype(np.float32)

    result, output_sr = blend([pluck, noise], [SAMPLE_RATE, SAMPLE_RATE], [1.0, 0.0])
    assert output_sr == SAMPLE_RATE
    embedder = Wavelet2Vec()
    assert cosine_similarity(
        embedder.embed(result, SAMPLE_RATE), embedder.embed(pluck, SAMPLE_RATE)
    ) > 0.99


def test_blend_moves_toward_heavier_reference():
    embedder = Wavelet2Vec()
    rng = np.random.default_rng(13)
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    pluck = (np.sin(2 * math.pi * 220.0 * times) * np.exp(-times * 6.0))[None, :].astype(np.float32)
    tone = _harmonic_tone(330.0)
    noise = (rng.standard_normal(SAMPLE_RATE) * np.exp(-times * 2.0))[None, :].astype(np.float32)
    references = [pluck, tone, noise]
    rates = [SAMPLE_RATE] * 3
    noise_vec = embedder.embed(noise, SAMPLE_RATE)

    light, _ = blend(references, rates, [0.7, 0.2, 0.1], carrier=0)
    heavy, _ = blend(references, rates, [0.3, 0.1, 0.6], carrier=0)
    light_sim = cosine_similarity(embedder.embed(light, SAMPLE_RATE), noise_vec)
    heavy_sim = cosine_similarity(embedder.embed(heavy, SAMPLE_RATE), noise_vec)
    assert heavy_sim > light_sim


def test_handles_short_and_stereo_input():
    embedder = Wavelet2Vec()
    short_stereo = np.random.default_rng(1).standard_normal((2, int(SAMPLE_RATE * 0.2)))
    embedding = embedder.embed(short_stereo.astype(np.float32), SAMPLE_RATE)
    assert embedding.shape == (embedder.dim,)
    assert np.isfinite(embedding).all()


def test_silence_yields_zero_embedding():
    embedder = Wavelet2Vec()
    silence = np.zeros((1, SAMPLE_RATE), dtype=np.float32)
    embedding = embedder.embed(silence, SAMPLE_RATE)
    assert float(np.linalg.norm(embedding)) == 0.0


def test_level_and_sample_rate_invariance():
    embedder = Wavelet2Vec()
    sine = _sine(440.0)
    quiet = embedder.embed(sine * 0.05, SAMPLE_RATE)
    loud = embedder.embed(sine, SAMPLE_RATE)
    assert cosine_similarity(quiet, loud) > 0.999

    times = np.arange(44100) / 44100
    high_rate = np.sin(2 * math.pi * 440.0 * times)[None, :].astype(np.float32)
    resampled = embedder.embed(high_rate, 44100)
    assert cosine_similarity(loud, resampled) > 0.98
