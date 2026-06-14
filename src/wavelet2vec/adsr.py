from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from wavelet2vec.character import SoundCharacter, _as_channels, apply_character


@dataclass(frozen=True)
class ADSR:
    """Attack / Decay / Sustain / Release envelope.

    Times are in seconds; ``sustain`` is a level in [0, 1]. The same envelope
    can shape a snippet's **volume** (:func:`apply_adsr_volume`) or its
    **character** (:func:`apply_adsr_character`) — e.g. a slow attack on "how
    much this vocal sounds like a synth".
    """

    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.2

    def replace(self, **changes) -> "ADSR":
        return dataclasses.replace(self, **changes)

    def envelope(
        self, n_samples: int, sample_rate: int, gate: float | None = None
    ) -> np.ndarray:
        """Samples the envelope over ``n_samples``.

        ``gate`` is the note-off time in seconds; by default the release is
        placed so it finishes exactly at the end of the snippet.
        """
        duration = n_samples / sample_rate
        if gate is None:
            gate = duration - self.release
        gate = float(np.clip(gate, 1e-4, duration))

        times = np.arange(n_samples) / sample_rate
        envelope = np.full(n_samples, max(self.sustain, 0.0), dtype=np.float64)
        if self.attack > 0:
            mask = times < self.attack
            envelope[mask] = times[mask] / self.attack
        if self.decay > 0:
            mask = (times >= self.attack) & (times < self.attack + self.decay)
            progress = (times[mask] - self.attack) / self.decay
            envelope[mask] = 1.0 + (self.sustain - 1.0) * progress
        elif self.attack > 0:
            # No decay stage: hold the attack peak until the gate.
            envelope[times >= self.attack] = max(self.sustain, 0.0)

        if self.release > 0:
            release_curve = np.clip(1.0 - (times - gate) / self.release, 0.0, 1.0)
        else:
            release_curve = (times <= gate).astype(np.float64)
        return np.clip(envelope * release_curve, 0.0, 1.0)


def apply_adsr_volume(
    waveform: np.ndarray,
    sample_rate: int,
    adsr: ADSR,
    gate: float | None = None,
) -> np.ndarray:
    """Shapes a snippet's loudness with an ADSR envelope."""
    channels = _as_channels(waveform)
    envelope = adsr.envelope(channels.shape[1], sample_rate, gate=gate)
    return (channels * envelope[None, :]).astype(np.float32)


def apply_adsr_character(
    source: np.ndarray,
    source_sr: int,
    character: SoundCharacter,
    adsr: ADSR,
    *,
    max_amount: float = 1.0,
    gate: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Shapes how much a snippet takes on a character, over time, with an ADSR.

    The morph amount follows the ADSR envelope scaled by ``max_amount``: a
    slow attack means the snippet starts as itself and grows into the
    character; the release lets it fall back. Extra keyword arguments are
    forwarded to :func:`wavelet2vec.character.apply_character`
    (``transfer_spectrum``, ``transfer_envelope``, ``max_gain_db``).
    """
    channels = _as_channels(source)
    envelope = adsr.envelope(channels.shape[1], source_sr, gate=gate)
    amount = np.clip(envelope * float(max_amount), 0.0, 1.0)
    return apply_character(channels, source_sr, character, amount=amount, **kwargs)
