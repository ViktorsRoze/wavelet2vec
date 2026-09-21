"""Epic D: batch embedding (D-1), frontend (D-2), pitch shift (D-4), datasets (D-6)."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import Wavelet2Vec, coeff_image, cosine_similarity, pitch_shift
from wavelet2vec.scenes import render_note

torch = pytest.importorskip("torch")
from wavelet2vec import WaveletFrontend  # noqa: E402

SR = 22050


# --- D-1: embed_batch -----------------------------------------------------------
@pytest.fixture(scope="module")
def embedder():
    return Wavelet2Vec()


def _fragments(count=6):
    rng = np.random.default_rng(0)
    return np.stack([
        np.sin(2 * np.pi * (200 + 60 * i) * np.arange(SR) / SR)
        + 0.1 * rng.standard_normal(SR)
        for i in range(count)
    ]).astype(np.float32)


def test_embed_batch_matches_single(embedder):
    frags = _fragments()
    batch = embedder.embed_batch(frags, SR)
    assert batch.shape == (len(frags), embedder.dim)
    for i in range(3):
        single = embedder.embed(frags[i], SR)
        assert cosine_similarity(batch[i], single) > 0.999999
        assert np.abs(batch[i] - single).max() < 1e-4


def test_embed_batch_ragged_fallback(embedder):
    ragged = [_fragments(1)[0], _fragments(2)[1][: SR // 2]]
    out = embedder.embed_batch(ragged, SR)
    assert out.shape == (2, embedder.dim)
    assert np.all(np.isfinite(out))


def test_embed_batch_silent_fragment(embedder):
    frags = _fragments(2)
    frags[1] = 0.0
    out = embedder.embed_batch(frags, SR)
    assert np.allclose(out[1], 0.0)  # silence embeds to zeros, as embed() does


# --- D-2: WaveletFrontend ----------------------------------------------------------
def test_frontend_matches_coeff_image_where_energy_exists():
    audio = render_note(60, 1.0, SR, instrument="saw")
    frontend = WaveletFrontend(SR, SR)
    image_t = frontend(torch.from_numpy(audio.astype(np.float32))[None])
    image_t = image_t.detach().numpy()[0]
    image_np, _meta = coeff_image(audio, SR)
    assert image_t.shape == image_np.shape == (3, 128, 128)
    # Magnitude matches everywhere; IF/coherence match where there is signal
    # (near-silent cells hold precision-dependent noise that carries nothing).
    assert np.corrcoef(image_t[0].ravel(), image_np[0].ravel())[0, 1] > 0.999
    mask = image_np[0] > 0.3
    assert np.corrcoef(image_t[1][mask], image_np[1][mask])[0, 1] > 0.99
    assert np.corrcoef(image_t[2][mask], image_np[2][mask])[0, 1] > 0.95


def test_frontend_gradients_finite_through_silence():
    audio = render_note(64, 0.75, SR, instrument="pluck")
    padded = np.zeros(SR)
    padded[SR // 4: SR // 4 + audio.shape[0]] = audio[: SR - SR // 4]  # leading silence
    x = torch.from_numpy(padded.astype(np.float32))[None].requires_grad_(True)
    frontend = WaveletFrontend(SR, SR, n_bands=64, n_frames=64)
    frontend(x).mean().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())  # the NaN-at-zero trap is guarded
    assert float(x.grad.abs().sum()) > 0


def test_frontend_batched_and_range():
    frags = _fragments(3)
    frontend = WaveletFrontend(SR, SR, n_bands=64, n_frames=32)
    out = frontend(torch.from_numpy(frags))
    assert out.shape == (3, 3, 64, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_frontend_rejects_wrong_length():
    frontend = WaveletFrontend(SR, SR)
    with pytest.raises(ValueError):
        frontend(torch.zeros(1, SR // 2))


# --- D-3: sequence tokens -------------------------------------------------------------
def test_sequence_tokens_shape_times_range():
    from wavelet2vec import sequence_tokens

    audio = render_note(60, 1.2, SR, instrument="saw")
    tokens, times = sequence_tokens(audio, SR, n_bands=64, hop_ms=20.0)
    assert tokens.shape == (len(times), 192)  # 3 channels × 64 bands
    assert tokens.shape[0] == pytest.approx(1.2 / 0.02, abs=2)
    assert float(tokens.min()) >= 0.0 and float(tokens.max()) <= 1.0
    assert np.all(np.diff(times) > 0)


def test_sequence_tokens_localize_events():
    from wavelet2vec import sequence_tokens
    from wavelet2vec.scenes import Event, render_scene

    audio, _ = render_scene(
        [Event(0.2, 0.6, 0, 0, label="saw", pitch=60),
         Event(0.8, 0.85, 0, 0, label="hat")], SR, total_s=1.2)
    tokens, times = sequence_tokens(audio, SR, hop_ms=20.0)
    magnitude = tokens[:, :64].mean(axis=1)
    onset = times[int(np.argmax(np.diff(magnitude)))]
    assert onset == pytest.approx(0.2, abs=0.05)  # frame-accurate onset
    silent_tail = magnitude[times > 1.0].max()
    hat_region = magnitude[(times > 0.75) & (times < 0.95)].max()
    assert hat_region > silent_tail * 1.5


def test_sequence_tokens_hop_controls_resolution():
    from wavelet2vec import sequence_tokens

    audio = render_note(55, 1.0, SR, instrument="pluck")
    fine, _t1 = sequence_tokens(audio, SR, hop_ms=10.0)
    coarse, _t2 = sequence_tokens(audio, SR, hop_ms=100.0)
    assert fine.shape[0] > coarse.shape[0] * 5


def test_sequence_tokens_tonal_vs_noise_coherence():
    from wavelet2vec import sequence_tokens

    tone = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    noise = np.random.default_rng(0).standard_normal(SR) * 0.3
    tone_tokens = sequence_tokens(tone, SR)[0]
    noise_tokens = sequence_tokens(noise, SR)[0]
    # Coherence is energy-weighted, so compare it *where the energy is* —
    # a tone lights one band, noise lights all of them, and global means
    # would just count lit cells.
    tone_coh = tone_tokens[:, 128:][tone_tokens[:, :64] > 0.5].mean()
    noise_coh = noise_tokens[:, 128:][noise_tokens[:, :64] > 0.5].mean()
    assert tone_coh > noise_coh * 2  # tonal cells are far more phase-stable


def test_frontend_tokens_matches_numpy():
    from wavelet2vec import sequence_tokens

    audio = render_note(60, 1.0, SR, instrument="saw")
    tokens_np, _times = sequence_tokens(audio, SR, n_bands=64, hop_ms=20.0)
    frontend = WaveletFrontend(SR, SR, n_bands=64, n_frames=tokens_np.shape[0])
    tokens_t = frontend.tokens(
        torch.from_numpy(audio.astype(np.float32))[None]).detach().numpy()[0]
    assert tokens_t.shape == tokens_np.shape
    mask = tokens_np[:, :64] > 0.3  # energetic magnitude cells
    corr = np.corrcoef(tokens_t[:, :64][mask], tokens_np[:, :64][mask])[0, 1]
    assert corr > 0.999


# --- D-4: pitch_shift ----------------------------------------------------------------
def _peak_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.fft.rfftfreq(len(audio), 1.0 / SR)[int(np.argmax(spectrum))])


@pytest.mark.parametrize("semitones,expected", [(12, 880.0), (-12, 220.0), (7, 659.3)])
def test_pitch_shift_moves_fundamental(semitones, expected):
    tone = np.sin(2 * np.pi * 440.0 * np.arange(SR) / SR)
    shifted = pitch_shift(tone, SR, semitones)
    assert shifted.shape == tone.shape  # duration preserved
    assert _peak_hz(shifted.astype(np.float64)) == pytest.approx(expected, rel=0.02)


def test_pitch_shift_zero_is_identity_and_stereo():
    tone = np.sin(2 * np.pi * 330.0 * np.arange(SR) / SR)
    assert np.allclose(pitch_shift(tone, SR, 0).astype(np.float64), tone, atol=1e-6)
    stereo = np.stack([tone, tone * 0.7])
    up = pitch_shift(stereo, SR, 5)
    assert up.shape == stereo.shape


def test_pitch_shift_preserves_harmonic_structure():
    saw = render_note(48, 1.0, SR, instrument="saw")  # C3 ≈ 130.8 Hz
    up = pitch_shift(saw, SR, 12).astype(np.float64)
    assert _peak_hz(up) == pytest.approx(261.6, rel=0.05)  # fundamental doubled


# --- D-5: augmentation kit -------------------------------------------------------
def test_time_stretch_changes_duration_not_pitch():
    from wavelet2vec.augment import time_stretch

    tone = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    slow = time_stretch(tone, SR, 0.5).astype(np.float64)
    fast = time_stretch(tone, SR, 2.0).astype(np.float64)
    assert slow.shape[0] == pytest.approx(2 * SR, abs=4)
    assert fast.shape[0] == pytest.approx(SR // 2, abs=4)
    assert _peak_hz(slow) == pytest.approx(440.0, rel=0.01)
    assert _peak_hz(fast) == pytest.approx(440.0, rel=0.01)


def test_time_stretch_scales_event_spacing():
    from wavelet2vec.augment import time_stretch

    clicks = np.zeros(SR)
    clicks[SR // 4] = 1.0
    clicks[SR // 2] = 1.0
    slow = time_stretch(clicks, SR, 0.5).astype(np.float64)
    strongest = np.argsort(np.abs(slow))[-2:]
    spacing = abs(int(strongest[0]) - int(strongest[1]))
    assert spacing == pytest.approx(SR // 2, abs=64)  # spacing doubled


def test_effect_chain_and_shorthands():
    pytest.importorskip("pedalboard")
    from wavelet2vec.augment import effect_chain

    tone = (np.sin(2 * np.pi * 300 * np.arange(SR) / SR) * 0.4).astype(np.float32)
    out = effect_chain(tone, SR, ["reverb", ("gain", {"gain_db": -6.0})])
    assert out.shape == tone.shape
    assert not np.allclose(out, tone)
    stereo = np.stack([tone, tone])
    assert effect_chain(stereo, SR, ["chorus"]).shape == stereo.shape


def test_crop_bars_exact_length_and_padding():
    from wavelet2vec.augment import crop_bars

    audio = np.tile(np.sin(2 * np.pi * 220 * np.arange(SR) / SR), 4)
    bar = int(2.0 * SR)  # one 4/4 bar at 120 BPM
    crop = crop_bars(audio, SR, bpm=120.0, bars=1, start_bar=1)
    assert crop.shape[-1] == bar
    tail = crop_bars(audio, SR, bpm=120.0, bars=2, start_bar=1)  # runs past the end
    assert tail.shape[-1] == 2 * bar  # padded to exact length
    with pytest.raises(ValueError):
        crop_bars(audio, SR, bpm=120.0, start_bar=99)


def test_random_augment_deterministic_and_shape_stable():
    from wavelet2vec.augment import random_augment

    tone = (np.sin(2 * np.pi * 330 * np.arange(SR) / SR) * 0.5).astype(np.float32)
    a = random_augment(tone, SR, seed=7, index=3)
    b = random_augment(tone, SR, seed=7, index=3)
    c = random_augment(tone, SR, seed=7, index=4)
    assert np.array_equal(a, b)  # reproducible epochs
    assert not np.array_equal(a, c)  # but views vary by index
    assert a.shape == tone.shape  # batch shapes hold
    stereo = np.stack([tone, tone * 0.8])
    assert random_augment(stereo, SR, seed=1).shape == stereo.shape


# --- D-6: torch dataset wrappers ---------------------------------------------------
def test_scene_detection_dataset_items_and_determinism():
    from wavelet2vec.data import SCENE_CLASSES, SceneDetectionDataset

    ds = SceneDetectionDataset(4, seconds=1.0, seed=11, n_bands=64, n_frames=64)
    assert len(ds) == 4
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    boxes, labels = target["boxes"], target["labels"]
    assert boxes.shape == (labels.shape[0], 4) and labels.shape[0] > 0
    assert bool((boxes[:, 0] >= 0).all()) and bool((boxes[:, 1] >= 0).all())
    assert bool((boxes[:, 0] + boxes[:, 2] <= 64 + 1e-4).all())  # x+w in frame
    assert bool((boxes[:, 1] + boxes[:, 3] <= 64 + 1e-4).all())
    assert bool((labels >= 1).all()) and bool((labels <= len(SCENE_CLASSES)).all())
    assert target["pitches"].shape == labels.shape

    again_image, again_target = SceneDetectionDataset(
        4, seconds=1.0, seed=11, n_bands=64, n_frames=64)[0]
    assert torch.equal(image, again_image)  # deterministic per (seed, index)
    assert torch.equal(target["boxes"], again_target["boxes"])
    other_image, _ = ds[1]
    assert not torch.equal(image, other_image)  # indices differ


def test_detection_dataloader_with_collate():
    from torch.utils.data import DataLoader

    from wavelet2vec.data import SceneDetectionDataset, detection_collate

    ds = SceneDetectionDataset(4, seconds=1.0, seed=3, n_bands=64, n_frames=64)
    loader = DataLoader(ds, batch_size=2, collate_fn=detection_collate)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 64, 64)
    assert isinstance(targets, list) and len(targets) == 2
    assert targets[0]["boxes"].shape[1] == 4  # per-item variable-length targets


def test_ladder_axis_dataset_measured_labels():
    from wavelet2vec.data import LadderAxisDataset

    for axis, low, high in [("brightness", 0.0, 1.0),
                            ("harmonicity", 0.0, 1.0),
                            ("attack_ms", 0.0, 400.0)]:
        ds = LadderAxisDataset(axis, 6, seed=2, n_bands=64, n_frames=64)
        image, label = ds[0]
        assert image.shape == (3, 64, 64)
        assert low <= float(label) <= high  # measured, in the axis's range
        _again, label_again = LadderAxisDataset(
            axis, 6, seed=2, n_bands=64, n_frames=64)[0]
        assert float(label) == float(label_again)  # deterministic
    ds = LadderAxisDataset("brightness", 6, seed=2, n_bands=64, n_frames=64)
    labels = [float(ds[i][1]) for i in range(4)]
    assert max(labels) - min(labels) > 1e-3  # the ladder actually varies

    with pytest.raises(ValueError):
        LadderAxisDataset("loudness")


def test_contrastive_views_dataset():
    from torch.utils.data import DataLoader

    from wavelet2vec.data import ContrastiveViewsDataset

    rng = np.random.default_rng(0)
    clips = [
        (np.sin(2 * np.pi * (220 + 110 * i) * np.arange(SR) / SR)
         + 0.05 * rng.standard_normal(SR)).astype(np.float32)
        for i in range(3)
    ]
    ds = ContrastiveViewsDataset(clips, SR, seed=5, effect_probability=0.0)
    view_a, view_b = ds[0]
    assert view_a.shape == view_b.shape == (SR,)  # length-stable views
    assert not torch.equal(view_a, view_b)  # the two views differ
    again_a, _ = ContrastiveViewsDataset(clips, SR, seed=5,
                                         effect_probability=0.0)[0]
    assert torch.equal(view_a, again_a)  # deterministic per (seed, index)

    tokens = ContrastiveViewsDataset(
        clips, SR, seed=5, effect_probability=0.0,
        transform=lambda audio: torch.as_tensor(audio[::2].copy()))
    half_a, _half_b = tokens[1]
    assert half_a.shape == (SR // 2,)  # transform is applied per view

    loader = DataLoader(ds, batch_size=3)
    batch_a, batch_b = next(iter(loader))
    assert batch_a.shape == batch_b.shape == (3, SR)

    with pytest.raises(ValueError):
        ContrastiveViewsDataset([], SR)
