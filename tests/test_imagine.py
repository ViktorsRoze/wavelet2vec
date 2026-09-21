"""Deterministic imagination axes: brightness (G-2 fallback), tonality (G-9)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from wavelet2vec import imagine_highs, imagine_tonality
from wavelet2vec.datasets import axis_labels
from wavelet2vec.scenes import render_note

SR = 22050


def _dull_saw():
    """A saw with its highs removed — the thing "imagine highs" should fix."""
    saw = render_note(48, 1.0, SR, instrument="saw")
    b, a = butter(6, 3000 / (SR / 2), "low")
    return filtfilt(b, a, saw)


# --- imagine_highs (brightness) -------------------------------------------------------
def test_imagine_highs_amount_zero_is_bit_exact():
    dull = _dull_saw()
    out = imagine_highs(dull, SR, 0.0).astype(np.float64)
    assert np.max(np.abs(out - dull)) < 1e-5


def test_imagine_highs_brightness_monotone():
    dull = _dull_saw()
    brightness = [axis_labels(imagine_highs(dull, SR, a), SR)["brightness"]
                  for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert brightness[0] < 0.01  # the dull sound has ~no highs
    assert brightness[-1] > 0.4  # imagined highs are clearly audible
    assert all(brightness[i] <= brightness[i + 1] + 1e-9 for i in range(4))


def test_imagine_highs_preserves_low_end():
    from wavelet2vec.frame import analyze

    dull = _dull_saw()
    out = imagine_highs(dull, SR, 1.0).astype(np.float64)  # the natural setting
    c0, spec = analyze(dull, SR, n_bands=128)
    c1, _spec = analyze(out, SR, n_bands=128)
    low = spec.centers < 0.6 * 4000.0  # well below the split
    reference = np.abs(c0).max()
    change = np.abs(np.abs(c0[low]) - np.abs(c1[low])).max()
    assert change < 0.1 * reference  # low end kept to a few percent at amount 1


def test_imagine_highs_on_real_loop():
    from experiments.dataset import list_tracks, load_canonical

    tracks = [t for t in list_tracks("../soundgen/example_audio")
              if t.folder == "synth_loops_120BPM"]
    loop = load_canonical(tracks[0], SR).mono[: SR * 2]
    before = axis_labels(loop, SR)["brightness"]
    after = axis_labels(imagine_highs(loop, SR, 1.5), SR)["brightness"]
    assert after > before + 0.05  # measurably brighter


def test_imagine_highs_multichannel():
    dull = _dull_saw()
    stereo = np.stack([dull, dull * 0.8])
    out = imagine_highs(stereo, SR, 1.0)
    assert out.shape == stereo.shape


# --- imagine_tonality (harmonic <-> noisy) --------------------------------------------
def test_imagine_tonality_amount_zero_reconstructs_input():
    from experiments.dataset import list_tracks, load_canonical

    tracks = [t for t in list_tracks("../soundgen/example_audio")
              if t.folder == "synth_loops_120BPM"]
    loop = load_canonical(tracks[0], SR).mono[: SR * 2]
    out = imagine_tonality(loop, SR, 0.0).astype(np.float64)
    # HPSS is exact-sum, so amount 0 returns the input (peak-matched).
    assert np.corrcoef(out, loop)[0, 1] > 0.999


def test_imagine_tonality_harmonicity_monotone():
    from experiments.dataset import list_tracks, load_canonical

    tracks = [t for t in list_tracks("../soundgen/example_audio")
              if t.folder == "synth_loops_120BPM"]
    loop = load_canonical(tracks[0], SR).mono[: SR * 2]
    harmonicity = [axis_labels(imagine_tonality(loop, SR, a), SR)["harmonicity"]
                   for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    # More tonal as amount rises; noisier as it falls.
    assert harmonicity[-1] > harmonicity[0]
    assert all(harmonicity[i] <= harmonicity[i + 1] + 0.02 for i in range(4))


def test_imagine_tonality_clips_amount():
    dull = _dull_saw()
    # Out-of-range amounts are clipped, not errors.
    assert imagine_tonality(dull, SR, 5.0).shape == dull.shape
    assert imagine_tonality(dull, SR, -5.0).shape == dull.shape


def test_imagine_tonality_multichannel():
    saw = render_note(55, 1.0, SR, instrument="saw")
    stereo = np.stack([saw, saw * 0.9])
    out = imagine_tonality(stereo, SR, 0.7)
    assert out.shape == stereo.shape
