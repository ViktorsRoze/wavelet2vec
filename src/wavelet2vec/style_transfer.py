from __future__ import annotations

from typing import Sequence

import numpy as np

from wavelet2vec.character import (
    _as_channels,
    apply_character,
    average_characters,
    extract_character,
)


def blend(
    sounds: Sequence[np.ndarray],
    sample_rates: Sequence[int],
    weights: Sequence[float],
    *,
    carrier: int | None = None,
    transfer_spectrum: bool = True,
    transfer_envelope: bool = True,
    max_gain_db: float = 24.0,
) -> tuple[np.ndarray, int]:
    """Creates a new sound perceptually between several reference sounds.

    One reference (the ``carrier``, by default the one with the largest
    weight) provides the fine structure — pitch, waveshape, micro-detail —
    and is reshaped toward the weighted average character of all references
    (see :mod:`wavelet2vec.character`):

    - **spectral envelope**: the carrier's wavelet band energies are EQ-matched
      to the weighted geometric mean of all references' band energies;
    - **temporal envelope**: the carrier's amplitude envelope is morphed to the
      weighted average of all references' (time-normalized) envelope shapes.

    Because the blend includes the carrier at its own weight, weights of
    ``[1, 0, 0]`` return the carrier unchanged, and increasing a reference's
    weight moves the result monotonically toward it.

    Returns ``(waveform [channels, samples], sample_rate)`` — the carrier's
    shape and rate.
    """
    if not (len(sounds) == len(sample_rates) == len(weights)):
        raise ValueError("sounds, sample_rates, and weights must have equal length.")
    if len(sounds) < 1:
        raise ValueError("At least one sound is required.")
    weight_array = np.asarray(weights, dtype=np.float64)
    if (weight_array < 0).any() or weight_array.sum() <= 0:
        raise ValueError("Weights must be non-negative and not all zero.")
    weight_array = weight_array / weight_array.sum()

    carrier_index = int(np.argmax(weight_array)) if carrier is None else carrier
    channels = _as_channels(sounds[carrier_index])
    output_sr = int(sample_rates[carrier_index])
    if np.abs(channels).max() <= 1e-8:
        return channels.astype(np.float32), output_sr

    characters = [
        extract_character(sound, int(rate)) for sound, rate in zip(sounds, sample_rates)
    ]
    target = average_characters(characters, weight_array)
    result = apply_character(
        channels,
        output_sr,
        target,
        amount=1.0,
        transfer_spectrum=transfer_spectrum,
        transfer_envelope=transfer_envelope,
        max_gain_db=max_gain_db,
    )
    return result, output_sr


def style_transfer(
    source: np.ndarray,
    source_sr: int,
    target: np.ndarray,
    target_sr: int,
    amount: float,
    **kwargs,
) -> np.ndarray:
    """Makes ``source`` sound more like ``target`` by ``amount`` in [0, 1].

    ``amount=0`` returns the source unchanged; ``amount=1`` fully matches the
    target's spectral and temporal envelopes while keeping the source's fine
    structure. Equivalent to :func:`wavelet2vec.character.apply_character`
    with the target's extracted character.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    return apply_character(
        source,
        source_sr,
        extract_character(target, target_sr),
        amount=amount,
        **kwargs,
    )
