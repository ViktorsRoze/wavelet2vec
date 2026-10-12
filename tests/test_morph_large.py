"""MorphNet-L pieces on a tiny Swin config: critic view, vocality axis, stable-pitch rendering, gradients."""
import math

import numpy as np
import torch

from wavelet2vec.morph import SPEC, judge_view, mix_baseline
from wavelet2vec.morph_large import (CRITIC_PLANES, CRITIC_ROWS, FeatureCritic, MorphNetL, backbone_inputs,
                                     critic_view, pitch_jitter, vocality)
from wavelet2vec.wavelet_image import WaveletImage, encode

T = np.arange(SPEC.audio_length) / SPEC.sample_rate


def note(f0, partials=8, width=0.0, vibrato=0.0):
    phase = 2 * np.pi * f0 * T + vibrato * np.sin(2 * np.pi * 5.5 * T)
    x = sum(np.sin(k * phase) / k for k in range(1, partials + 1)) * np.exp(-T / 0.8) * 0.1
    return np.stack([x * (1 + width), x * (1 - width)])


def image_of(*notes):
    return encode(torch.from_numpy(np.stack(notes)).float(), SPEC)


def take(image, index):
    return WaveletImage(image.spec, {c: v[index] for c, v in image.groups.items()}, dict(image.rows))


def tiny_model():
    torch.manual_seed(0)
    return MorphNetL(None, {"size": "tiny"}, depths=(1, 1, 1), alpha_dim=32, residual_hidden=16)


def encoded_pair(model, vocal_audio, synth_audio, midi=60):
    image = image_of(vocal_audio, synth_audio)
    planes, cond = backbone_inputs(image, [midi, midi])
    enc = model.encode(planes, cond, image)
    split = lambda sl: {k: (v[sl] if torch.is_tensor(v) else [x[sl] for x in v]) for k, v in enc.items()}
    return split(slice(0, 1)), split(slice(1, 2)), take(image, slice(0, 1)), take(image, slice(1, 2))


def test_critic_view_sees_stereo_up_to_20_khz():
    assert 440 * 2 ** ((SPEC.first_midi + CRITIC_ROWS - 1 - 69) / 12) < 20000
    view = critic_view(image_of(note(220.0), note(220.0, width=0.5)))
    assert view.shape == (2, CRITIC_PLANES, CRITIC_ROWS, SPEC.n_frames)
    assert torch.allclose(view[0, 0], view[1, 0], atol=1e-4)            # same mid
    assert float((view[1, 1] - view[0, 1]).max()) > 0.5                  # side differs
    score, feats = FeatureCritic(CRITIC_PLANES)(view)
    assert score.shape == (2,) and len(feats) == 4


def test_vocality_axis_maps_the_centres_to_0_and_1():
    cv, cs = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    v = vocality(torch.stack([cv, cs, (cv + cs) / 2]) * 3.0, cv, cs)       # scale-free (normalized)
    assert torch.allclose(v[:2], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_hybrid_pitch_is_exactly_as_stable_as_its_pitch_source():
    model = tiny_model()
    ev, es, vocal, synth = encoded_pair(model, note(261.63, vibrato=0.6), note(261.63, partials=16))
    jitter = {name: pitch_jitter(im, [60])[0] for name, im in
              (("synth", synth), ("vocal", vocal), ("mix", mix_baseline(vocal, synth, torch.tensor([0.5]))))}
    with torch.no_grad():
        for source, name in ((0, "synth"), (1, "vocal")):
            hybrid, _ = model.decode(ev, es, torch.tensor([0.5]), vocal, synth, torch.tensor([source]))
            assert math.isclose(pitch_jitter(hybrid, [60])[0], jitter[name], rel_tol=0.05, abs_tol=0.2)
    assert jitter["synth"] < 1.0 < jitter["mix"] < jitter["vocal"]       # a plain sum beats; the hybrid does not


def test_morphnet_l_trains_with_finite_gradients():
    model = tiny_model()
    ev, es, vocal, synth = encoded_pair(model, note(261.63), note(261.63, partials=16, width=0.3))
    hybrid, extras = model.decode(ev, es, torch.tensor([0.5]), vocal, synth, torch.tensor([1]))
    assert extras["gain_level"].shape[1] == 8                            # path x material x channel
    loss = judge_view(hybrid).pow(2).mean() + critic_view(hybrid).pow(2).mean() + extras["residual_energy"].mean()
    loss.backward()
    new = {id(p) for p in model.new_parameters()}
    assert any(p.grad is not None for p in model.parameters() if id(p) in new)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
