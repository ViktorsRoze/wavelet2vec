"""Articulation targets, vowel automation and the frame-wise judge; MorphNet-L accepts articulation."""
import numpy as np
import torch

from wavelet2vec.articulation import (ART_CLASSES, VOWEL_INDEX, VOWELS, ArticulationJudge, apply_vowels,
                                      articulation_loss, articulation_targets, morph_vowels, vowel_automation,
                                      vowel_frames)
from wavelet2vec.morph import SPEC, judge_view
from wavelet2vec.morph_large import MorphNetL, backbone_inputs
from wavelet2vec.tools.vocal_detector import VOWEL_FORMANTS, detect_vocal_parts, formant_envelope_db
from wavelet2vec.wavelet_image import WaveletImage, encode


def sung(vowel, f0=220.0, seconds=0.8, rate=48000, scale=1.0):
    """A harmonic tone shaped by a vowel's formant envelope, with silence after it."""
    t = np.arange(int(seconds * rate)) / rate
    harmonics = np.arange(1, int(5000 / f0))
    gains = 10 ** (formant_envelope_db(harmonics * f0, np.array(VOWEL_FORMANTS[vowel]) * scale) / 20)
    x = (gains[:, None] * np.sin(2 * np.pi * f0 * harmonics[:, None] * t)).sum(0)
    out = np.zeros(int(1.2 * rate))
    out[:x.size] = x / np.abs(x).max() * 0.3
    return out


def test_targets_follow_the_detector_and_cover_the_note():
    track = detect_vocal_parts(sung("a"), 48000, scale=1.0)
    target = articulation_targets(track)
    assert target.shape == (len(ART_CLASSES), 120)
    assert np.allclose(target.sum(0), 1.0, atol=1e-5)
    voiced = vowel_frames(target)
    assert voiced[10:70].mean() > 0.8 and not voiced[95:].any()             # sung, then silent
    dominant = VOWELS[int(target[VOWEL_INDEX][:, voiced].sum(1).argmax())]
    assert dominant == "a"


def test_vowel_automation_and_glide_only_touch_vowel_frames():
    curve = vowel_automation([(0.0, "a"), (1.0, "o")])
    assert curve.shape == (len(VOWELS), 120) and np.allclose(curve.sum(0), 1.0)
    assert curve[VOWELS.index("a"), 0] > curve[VOWELS.index("a"), 119]
    target = np.zeros((len(ART_CLASSES), 120), dtype=np.float32)
    target[0] = 1.0                                                        # silence ...
    target[ART_CLASSES.index("i"), 20:80] = 1.0                            # ... with an "i" in the middle
    target[0, 20:80] = 0.0
    glide = morph_vowels(target, "a", "o")
    assert np.allclose(glide[:, :20], target[:, :20]) and np.allclose(glide[:, 80:], target[:, 80:])
    a, o = ART_CLASSES.index("a"), ART_CLASSES.index("o")
    assert glide[a, 21] > glide[o, 21] and glide[o, 78] > glide[a, 78]
    kept = apply_vowels(target, curve, amount=0.0)
    assert np.allclose(kept, target)


def test_judge_shapes_and_loss_decrease():
    judge = ArticulationJudge()
    view = torch.randn(2, 4, 120, 120)
    logits = judge(view)
    assert logits.shape == (2, len(ART_CLASSES), 120)
    target = torch.zeros(2, len(ART_CLASSES), 120)
    target[:, ART_CLASSES.index("o")] = 1.0
    opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        loss = articulation_loss(judge(view), target, torch.ones(2, 120))
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_morphnet_l_ignores_zero_articulation_layers_at_start():
    torch.manual_seed(0)
    model = MorphNetL(None, {"size": "tiny"}, depths=(1, 1, 1), alpha_dim=32, residual_hidden=16)
    t = np.arange(SPEC.audio_length) / SPEC.sample_rate
    x = np.sin(2 * np.pi * 261.63 * t) * np.exp(-t / 0.8) * 0.1
    image = encode(torch.from_numpy(np.stack([np.stack([x, x]), np.stack([x, 0.5 * x])])).float(), SPEC)
    take = lambda sl: WaveletImage(image.spec, {c: v[sl] for c, v in image.groups.items()}, dict(image.rows))
    planes, cond = backbone_inputs(image, [60, 60])
    enc = model.encode(planes, cond, image)
    split = lambda sl: {k: (v[sl] if torch.is_tensor(v) else [y[sl] for y in v]) for k, v in enc.items()}
    target = torch.from_numpy(vowel_automation([(0.0, "a"), (1.0, "u")]))
    full = torch.zeros(1, len(ART_CLASSES), 120)
    full[0, VOWEL_INDEX] = target
    with torch.no_grad():
        plain, _ = model.decode(split(slice(0, 1)), split(slice(1, 2)), torch.tensor([0.5]), take(slice(0, 1)),
                                take(slice(1, 2)))
        steered, _ = model.decode(split(slice(0, 1)), split(slice(1, 2)), torch.tensor([0.5]), take(slice(0, 1)),
                                  take(slice(1, 2)), articulation=full)
    assert all(torch.allclose(plain.groups[c], steered.groups[c]) for c in plain.groups)   # zero-initialized
    assert judge_view(steered).shape[0] == 1


def test_generate_hybrid_with_vowel_automation():
    from wavelet2vec.morph_large import generate_hybrid
    torch.manual_seed(0)
    model = MorphNetL(None, {"size": "tiny"}, depths=(1, 1, 1), alpha_dim=32, residual_hidden=16).eval()
    vocal = sung("i", f0=220.0, rate=48000)
    t = np.arange(vocal.size) / 48000
    synth = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 12)) * np.exp(-t / 0.5) * 0.2
    for vowels in (None, "a", [(0.0, "a"), (0.8, "o")]):
        audio = generate_hybrid(model, vocal, synth, midi=57, vowels=vowels, sample_rate=48000)
        assert audio.shape == (2, SPEC.audio_length) and np.isfinite(audio).all() and np.abs(audio).max() > 0
