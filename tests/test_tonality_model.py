"""G-9 learned tonality axis: remix mechanics + imagine_tonality hook."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec.decompose import hpss
from wavelet2vec.scenes import render_note, render_drum
from wavelet2vec.tonality_model import (
    MODEL_BANDS,
    MODEL_FRAMES,
    layer_envelopes,
    remix_layers,
)

torch = pytest.importorskip("torch")
from wavelet2vec import TonalityNet, load_tonality_model, save_tonality_model  # noqa: E402
from wavelet2vec.imagine import imagine_tonality  # noqa: E402

SR = 22050


@pytest.fixture(scope="module")
def clip():
    note = render_note(55, 1.0, SR, instrument="saw")
    hit = render_drum("snare", SR)
    mix = note.copy()
    mix[: hit.shape[0]] += 0.6 * hit
    return mix / np.abs(mix).max()


def test_layer_envelopes_and_unit_gain_identity(clip):
    harmonic, percussive = hpss(clip, SR, n_bands=64)
    envelopes, centers, coeff_pair = layer_envelopes(harmonic, percussive, SR)
    assert envelopes.shape == (2, MODEL_BANDS, MODEL_FRAMES)
    assert centers.shape == (MODEL_BANDS,)

    # unit gains reproduce harmonic + percussive == the input (exact-sum HPSS)
    remix = remix_layers(coeff_pair, np.ones((2, MODEL_BANDS)), clip.shape[0])
    error = np.abs(remix - clip).max() / np.abs(clip).max()
    assert error < 1e-6

    # zeroing the percussive layer removes it: remix equals harmonic alone
    only_h = remix_layers(
        coeff_pair, np.stack([np.ones(MODEL_BANDS), np.zeros(MODEL_BANDS)]),
        clip.shape[0])
    assert np.abs(only_h - harmonic).max() / (np.abs(harmonic).max() + 1e-12) < 1e-6


def test_net_shapes_conditioning_roundtrip(tmp_path):
    net = TonalityNet(width=8)
    envelopes = torch.rand(3, 2, MODEL_BANDS, MODEL_FRAMES)
    tonal = net(envelopes, torch.full((3, 1), 0.9))
    noisy = net(envelopes, torch.full((3, 1), 0.1))
    assert tonal.shape == (3, 2, MODEL_BANDS)
    gains = tonal.detach().exp()
    assert float(gains.max()) <= 2.0 + 1e-5  # remix-family gains (1±a)
    assert float(gains.min()) >= 0.0
    assert torch.allclose(gains[:, 0] + gains[:, 1],
                          torch.full_like(gains[:, 0], 2.0), atol=1e-4)
    assert not torch.allclose(tonal, noisy)  # FiLM conditions the gains

    path = save_tonality_model(net, tmp_path / "t.pt")
    restored = load_tonality_model(path)
    again = restored(envelopes, torch.full((3, 1), 0.9))
    assert torch.allclose(tonal, again)


def test_imagine_tonality_model_hook(clip):
    net = TonalityNet(width=8)  # untrained: mechanics only
    out = imagine_tonality(clip, SR, 1.0, model=net, harmonicity=0.7)
    assert out.shape == clip.shape
    assert np.all(np.isfinite(out))

    # amount=0 in the model path is the input (log-gains scaled to zero;
    # peak-matched exact-sum remix)
    dry = imagine_tonality(clip, SR, 0.0, model=net, harmonicity=0.7)
    assert np.abs(dry - clip).max() < 1e-4

    # without a model the deterministic path is untouched
    legacy = imagine_tonality(clip, SR, 1.0)
    harmonic, _ = hpss(clip, SR)
    assert np.corrcoef(legacy, 2.0 * harmonic / np.abs(2.0 * harmonic).max()
                       * np.abs(clip).max())[0, 1] > 0.99


def test_anchored_harmonicity_calibration(clip):
    from wavelet2vec.harmonic_features import autocorrelation_pitch

    # anchored inversion (no model): the measured result lands near the
    # request when it is reachable, and requests order the outputs. (The
    # saw+snare clip's own harmonicity is ~0.98, so both targets pull down.)
    tonal = imagine_tonality(clip, SR, 1.0, harmonicity=0.95)
    noisy = imagine_tonality(clip, SR, 1.0, harmonicity=0.4)
    got_tonal = autocorrelation_pitch(tonal.astype(np.float64), SR)[0]
    got_noisy = autocorrelation_pitch(noisy.astype(np.float64), SR)[0]
    assert got_tonal > got_noisy
    assert abs(got_tonal - 0.95) < 0.15  # calibrated, not just directional
    # amount=0 stays the input in the calibrated path too
    dry = imagine_tonality(clip, SR, 0.0, harmonicity=0.95)
    assert np.abs(dry - clip).max() < 1e-4
