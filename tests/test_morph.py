"""MorphNet pieces: the judges' confound-controlled view, baselines, judges, generator shapes and gradients."""
import math

import numpy as np
import torch

from wavelet2vec.morph import (JUDGE_PLANES, JUDGE_ROWS, SPEC, TERMS, Critic, JudgeCNN, JudgeTCN, MorphConfig,
                               MorphNet, augment_view, judge_view, mix_baseline, power, vocoder_baseline)
from wavelet2vec.wavelet_image import encode

T = np.arange(SPEC.audio_length) / SPEC.sample_rate


def note(f0, partials=8, decay=0.6, width=0.0, vibrato=0.0):
    phase = 2 * np.pi * f0 * T + vibrato * np.sin(2 * np.pi * 5.5 * T)
    x = sum(np.sin(k * phase) / k for k in range(1, partials + 1)) * np.exp(-T / decay)
    return np.stack([x * (1 + width), x * (1 - width)]) * 0.1


def image_of(*notes):
    return encode(torch.from_numpy(np.stack(notes)).float(), SPEC)


def test_judge_view_is_band_limited_mid_only_and_level_invariant():
    assert JUDGE_ROWS < SPEC.n_bands
    top = SPEC.first_midi + JUDGE_ROWS - 1
    assert 440 * 2 ** ((top - 69) / 12) < 18000
    a, b = note(220.0), note(220.0, width=0.5)                 # same mid, different side
    view = judge_view(image_of(a, b, a * 0.01))
    assert view.shape == (3, JUDGE_PLANES, JUDGE_ROWS, SPEC.n_frames)
    assert torch.allclose(view[0], view[1], atol=1e-4)          # side is invisible to the judges
    assert torch.allclose(view[0], view[2], atol=1e-3)          # loudness is normalized away
    assert float(view[:, 0].min()) >= -1.0 - 1e-6 and float(view[:, 0].max()) <= 1.0 + 1e-6


def test_judge_view_has_finite_gradients_on_silence():
    image = image_of(np.zeros((2, SPEC.audio_length)), note(330.0))
    for v in image.groups.values():
        v.requires_grad_(True)
    judge_view(image).sum().backward()
    grads = [v.grad for v in image.groups.values() if v.grad is not None]    # groups above 18 kHz get none
    assert len(grads) < len(image.groups) and all(torch.isfinite(g).all() for g in grads)


def test_augment_view_shifts_and_cuts_with_the_floor():
    view = judge_view(image_of(note(440.0)))
    out = augment_view(view.repeat(8, 1, 1, 1), torch.Generator().manual_seed(0))
    assert out.shape == (8, *view.shape[1:])
    assert float(out[:, 0].min()) >= -1.0 - 1e-6


def test_baselines_mix_and_vocoder():
    vocal, synth = image_of(note(440.0, vibrato=0.5)), image_of(note(440.0, partials=20, decay=2.0))
    mix = mix_baseline(vocal, synth, torch.tensor([1.0]))
    assert all(torch.allclose(mix.groups[c], vocal.groups[c]) for c in vocal.groups)
    voc = vocoder_baseline(synth, vocal)
    for c in synth.groups:                                     # cell levels follow the envelope note
        pv = power(vocal.groups[c]).mean(-1).sum(1)
        po = power(voc.groups[c]).mean(-1).sum(1)
        loud = pv > pv.max() * 1e-3
        if loud.any():
            assert torch.allclose(10 * torch.log10(po[loud]), 10 * torch.log10(pv[loud]), atol=0.5)


def test_judges_and_critic_shapes():
    view = judge_view(image_of(note(220.0), note(330.0)))
    for judge in (JudgeCNN(), JudgeTCN()):
        logits, emb = judge(view)
        assert logits.shape == (2, 2) and emb.shape == (2, 128)
    assert Critic()(view).shape == (2,)


def test_morphnet_starts_as_an_even_blend_of_its_materials_and_trains():
    torch.manual_seed(0)
    model = MorphNet(MorphConfig(dims=(16, 24, 32, 48), detail=8, depth=1, heads=4))
    vocal, synth = image_of(note(261.6, vibrato=0.4)), image_of(note(261.6, partials=16, width=0.3))
    hybrid, extras = model(vocal, synth, torch.tensor([0.5]))
    c = 4
    v, s = vocal.groups[c], synth.groups[c]
    pv, ps = power(v), power(s)
    peak_v = max(float(power(x).mean(-1).amax()) for x in vocal.groups.values())
    peak_s = max(float(power(x).mean(-1).amax()) for x in synth.groups.values())
    mv, ms = torch.sqrt(pv + 1e-8 * peak_v), torch.sqrt(ps + 1e-8 * peak_s)
    expected = 0.25 * (s + v + s / ms * mv + v / mv * ms)    # zero-initialized residual, -12 dB gains
    assert torch.allclose(hybrid.groups[c], expected, atol=1e-5)
    assert extras["gain_level"].shape[1:3] == (len(TERMS), 2)
    loss = judge_view(hybrid).pow(2).mean() + extras["residual_energy"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert math.isclose(float(extras["residual_energy"][0]), 0.0, abs_tol=1e-12)
