"""Expression automation over the imagination axes (roadmap G-11, det tier).

*The payoff*: "each next note gets a harder attack and more air" is one
automation clip per axis. Every deterministic axis becomes an automatable
lane over a note grid — the same slice contract as :func:`wavelet2vec.
perform.perform` (output has exactly the input's shape and rate; boundary
fades suppress clicks), with one value per lane sampled at each note.

Lane values are uniform **0 → 1** (FL Studio clip range): 0 is always
identity, 1 the full effect. Per-axis mapping (override via ``ranges``):

    brightness  imagine_highs amount            0 → 2
    tonality    imagine_tonality (harmonic)     0 → +1
    noisiness   imagine_tonality (percussive)   0 → −1
    attack      shape_transients attack gain    1 → 4
    sustain     shape_transients sustain gain   1 → 0.25   (1 = duck sustain)
    character   apply_character amount          0 → 1      (needs ``character``)

A lane is an :class:`~wavelet2vec.sequence.AutomationCurve` (e.g. parsed
from an FL clip), or a plain sequence: one value per note when the length
matches ``starts``, otherwise resampled across the timeline. When a learned
axis model ships (G-2/G-8/G-9) it slots behind the same lane names.
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.character import _as_channels, apply_character
from wavelet2vec.decompose import shape_transients
from wavelet2vec.imagine import imagine_highs, imagine_tonality
from wavelet2vec.sequence import AutomationCurve
from wavelet2vec.slicing import _normalize_starts

# Lane name -> (identity_value, full_effect_value); lane v maps linearly.
AXIS_RANGES: dict[str, tuple[float, float]] = {
    "brightness": (0.0, 2.0),
    "tonality": (0.0, 1.0),
    "noisiness": (0.0, 1.0),
    "attack": (1.0, 4.0),
    "sustain": (1.0, 0.25),
    "character": (0.0, 1.0),
}


def _lane_value(lane, fraction: float, index: int, count: int) -> float:
    """Sample a lane at a note: AutomationCurve by timeline fraction; a
    sequence per-note when its length matches, else linearly resampled."""
    if isinstance(lane, AutomationCurve):
        return float(np.clip(lane.value_at_fraction(fraction), 0.0, 1.0))
    values = np.asarray(lane, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    if values.size == count:
        return float(np.clip(values[index], 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, values.size)
    return float(np.clip(np.interp(fraction, grid, values), 0.0, 1.0))


def render_expression(
    waveform: np.ndarray,
    sample_rate: int,
    starts: np.ndarray,
    lanes: dict,
    *,
    character=None,
    ranges: dict | None = None,
    split_hz: float = 4000.0,
    boundary_fade_seconds: float = 0.005,
) -> np.ndarray:
    """Renders automation ``lanes`` over the note grid ``starts``.

    ``lanes`` maps axis names (see :data:`AXIS_RANGES`) to curves. Unknown
    names raise; a ``character`` lane requires the ``character`` argument.
    Effects apply per note in a fixed order — tonality/noisiness →
    attack/sustain → brightness → character — chosen so spectral extension
    sees the reshaped transients.
    """
    axis_ranges = {**AXIS_RANGES, **(ranges or {})}
    unknown = set(lanes) - set(axis_ranges)
    if unknown:
        raise ValueError(f"Unknown expression lanes: {sorted(unknown)}")
    if "character" in lanes and character is None:
        raise ValueError("A 'character' lane needs the character argument.")

    channels = _as_channels(waveform).astype(np.float64)
    n_samples = channels.shape[1]
    starts = _normalize_starts(np.asarray(starts), n_samples)
    bounds = np.concatenate([starts, [n_samples]])
    count = len(starts)

    def mapped(name: str, value: float) -> float:
        lo, hi = axis_ranges[name]
        return lo + value * (hi - lo)

    output = channels.copy()
    for index in range(count):
        begin, end = int(bounds[index]), int(bounds[index + 1])
        if end <= begin:
            continue
        fraction = begin / max(n_samples, 1)
        values = {name: _lane_value(lane, fraction, index, count)
                  for name, lane in lanes.items()}
        if all(v < 1e-3 for v in values.values()):
            continue  # identity note: leave the slice bit-untouched
        segment = channels[:, begin:end]

        net_tonality = values.get("tonality", 0.0) - values.get("noisiness", 0.0)
        if abs(net_tonality) >= 1e-3:
            segment = imagine_tonality(segment, sample_rate, net_tonality).astype(np.float64)
        attack_v = values.get("attack", 0.0)
        sustain_v = values.get("sustain", 0.0)
        if attack_v >= 1e-3 or sustain_v >= 1e-3:
            attack_gain = mapped("attack", attack_v)
            sustain_gain = mapped("sustain", sustain_v)
            segment = np.stack([
                shape_transients(channel, sample_rate,
                                 attack_gain=attack_gain, sustain_gain=sustain_gain)
                for channel in segment
            ]).astype(np.float64)
        brightness_v = values.get("brightness", 0.0)
        if brightness_v >= 1e-3:
            segment = imagine_highs(segment, sample_rate,
                                    mapped("brightness", brightness_v),
                                    split_hz=split_hz).astype(np.float64)
        character_v = values.get("character", 0.0)
        if character_v >= 1e-3:
            segment = apply_character(segment, sample_rate, character,
                                      amount=character_v).astype(np.float64)
        output[:, begin:end] = segment

    fade = int(boundary_fade_seconds * sample_rate)
    if fade > 0:
        ramp = np.linspace(1.0, 0.0, fade)
        for boundary in starts[1:]:
            lo = max(int(boundary) - fade, 0)
            output[:, lo:int(boundary)] *= ramp[fade - (int(boundary) - lo):]

    result = output if waveform_is_multichannel(waveform) else output[0]
    return result.astype(np.float32)


def waveform_is_multichannel(waveform) -> bool:
    return np.asarray(waveform).ndim == 2
