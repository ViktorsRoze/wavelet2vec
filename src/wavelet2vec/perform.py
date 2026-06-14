from __future__ import annotations

import numpy as np

from wavelet2vec.adsr import ADSR, apply_adsr_character, apply_adsr_volume
from wavelet2vec.character import SoundCharacter, _as_channels, apply_character
from wavelet2vec.sequence import AutomationCurve, _modulated_settings
from wavelet2vec.slicing import _normalize_starts


def perform(
    waveform: np.ndarray,
    sample_rate: int,
    character: SoundCharacter | None,
    starts: np.ndarray,
    *,
    automation: AutomationCurve | None = None,
    character_adsr: ADSR | None = None,
    volume_adsr: ADSR | None = None,
    modulate: str = "amount",
    max_amount: float = 1.0,
    boundary_fade_seconds: float = 0.005,
    transfer_spectrum: bool = True,
    transfer_envelope: bool = True,
    max_gain_db: float = 24.0,
) -> np.ndarray:
    """Transforms a continuous musical performance note by note.

    ``waveform`` is one long render (e.g. 16 bars of repeated 8th notes from
    the DAW) and ``starts`` are the note-start sample indices from one of the
    :mod:`wavelet2vec.slicing` strategies. Each note is morphed toward the
    character with its ADSR envelope (gated to the note length, so the
    release completes before the next note); the ``automation`` curve —
    sampled at each note's position on the shared timeline — modulates one
    morph parameter per note (see :func:`wavelet2vec.sequence.render_sequence`
    for the mapping).

    The output has **exactly** the input's shape, channel count, and sample
    rate: processing happens at the native rate (88.2/96 kHz passes through
    untouched), so the result re-imports into the DAW sample-aligned at
    bar 1, like an offline effect. A short fade at each note boundary
    (``boundary_fade_seconds``) suppresses clicks from per-note gain steps.
    """
    channels = _as_channels(waveform)
    n_samples = channels.shape[1]
    starts = _normalize_starts(np.asarray(starts), n_samples)
    bounds = np.concatenate([starts, [n_samples]])

    output = np.empty_like(channels)
    for index in range(len(starts)):
        begin, end = int(bounds[index]), int(bounds[index + 1])
        if end <= begin:
            continue
        segment = channels[:, begin:end]
        value = (
            automation.value_at_fraction(begin / max(n_samples, 1))
            if automation is not None
            else 1.0
        )
        adsr, amount = _modulated_settings(character_adsr or ADSR(), modulate, value, max_amount)
        if character is None:
            processed = segment.astype(np.float32)
        elif character_adsr is not None:
            processed = apply_adsr_character(
                segment,
                sample_rate,
                character,
                adsr,
                max_amount=amount,
                transfer_spectrum=transfer_spectrum,
                transfer_envelope=transfer_envelope,
                max_gain_db=max_gain_db,
            )
        else:
            processed = apply_character(
                segment,
                sample_rate,
                character,
                amount=amount,
                transfer_spectrum=transfer_spectrum,
                transfer_envelope=transfer_envelope,
                max_gain_db=max_gain_db,
            )
        if volume_adsr is not None:
            processed = apply_adsr_volume(processed, sample_rate, volume_adsr)
        output[:, begin:end] = processed

    fade = int(boundary_fade_seconds * sample_rate)
    if fade > 0:
        ramp = np.linspace(1.0, 0.0, fade)
        for boundary in starts[1:]:
            lo = max(int(boundary) - fade, 0)
            output[:, lo : int(boundary)] *= ramp[fade - (int(boundary) - lo) :]

    assert output.shape == channels.shape
    return output.astype(np.float32)
