from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wavelet2vec.audio_io import list_audio_files, load_audio, resample
from wavelet2vec.filterbank import analytic_envelope, log_spaced_frequencies, morlet_scalogram

# The character analysis grid covers the full audible band (~27.5 Hz to
# ~19.8 kHz) so spectral character above 10 kHz — air, brightness, crunch —
# transfers too. Application happens at the source's native rate (gains above
# the grid hold the edge value), so 88.2/96 kHz material passes through
# without resampling or lowpassing.
CHARACTER_SAMPLE_RATE = 44100
CHARACTER_BANDS = 72
CHARACTER_F_MIN = 27.5
CHARACTER_F_MAX = 0.45 * CHARACTER_SAMPLE_RATE
CHARACTER_ENVELOPE_POINTS = 256

_CENTERS = log_spaced_frequencies(CHARACTER_F_MIN, CHARACTER_F_MAX, CHARACTER_BANDS)


def _as_channels(waveform: np.ndarray) -> np.ndarray:
    if hasattr(waveform, "detach"):  # accept torch tensors transparently
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    return waveform


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


@dataclass(frozen=True)
class SoundCharacter:
    """The transferable character of a sound (or a group of sounds).

    Holds the level-normalized wavelet band energy distribution (log domain)
    and the time-normalized amplitude envelope shape — the two dimensions
    :func:`apply_character` can impose on another sound. Characters live on a
    fixed analysis grid, so they can be extracted once (e.g. from a folder of
    electric guitar samples), saved, and applied to any other snippet later.
    """

    log_band_energies: np.ndarray  # [CHARACTER_BANDS]
    envelope_shape: np.ndarray  # [CHARACTER_ENVELOPE_POINTS], peak-normalized

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            log_band_energies=self.log_band_energies,
            envelope_shape=self.envelope_shape,
            band_centers_hz=_CENTERS,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "SoundCharacter":
        data = np.load(Path(path))
        energies = np.asarray(data["log_band_energies"], dtype=np.float64)
        if energies.shape[0] != CHARACTER_BANDS:
            raise ValueError(
                f"{path} was saved with a {energies.shape[0]}-band analysis grid; "
                f"this version uses {CHARACTER_BANDS} bands. Re-extract the character "
                "from the original sounds."
            )
        return cls(
            log_band_energies=energies,
            envelope_shape=np.asarray(data["envelope_shape"], dtype=np.float64),
        )


def extract_character(waveform: np.ndarray, sample_rate: int) -> SoundCharacter:
    """Extracts the spectral and temporal character of one snippet."""
    channels = _as_channels(waveform)
    mono = resample(channels, int(sample_rate), CHARACTER_SAMPLE_RATE).mean(axis=0)
    peak = np.abs(mono).max()
    if peak > 1e-8:
        mono = mono / peak

    scalogram = morlet_scalogram(mono, CHARACTER_SAMPLE_RATE, _CENTERS)
    energies = (scalogram.astype(np.float64) ** 2).mean(axis=1)
    energies = energies / (energies.sum() + 1e-12)
    log_band_energies = np.log(energies + 1e-10)

    smooth = max(int(CHARACTER_SAMPLE_RATE * 0.005), 1)
    envelope = analytic_envelope(mono, smooth_samples=smooth).astype(np.float64)
    positions = np.linspace(0.0, 1.0, CHARACTER_ENVELOPE_POINTS)
    source_positions = np.linspace(0.0, 1.0, envelope.shape[0])
    shape = np.interp(positions, source_positions, envelope)
    shape = shape / (shape.max() + 1e-12)
    return SoundCharacter(log_band_energies=log_band_energies, envelope_shape=shape)


def average_characters(
    characters: Sequence[SoundCharacter],
    weights: Sequence[float] | None = None,
) -> SoundCharacter:
    """Weighted average character: geometric mean of band energies, mean envelope."""
    if not characters:
        raise ValueError("At least one character is required.")
    if weights is None:
        weight_array = np.full(len(characters), 1.0 / len(characters))
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if len(weight_array) != len(characters):
            raise ValueError("weights must match the number of characters.")
        if (weight_array < 0).any() or weight_array.sum() <= 0:
            raise ValueError("Weights must be non-negative and not all zero.")
        weight_array = weight_array / weight_array.sum()

    log_energies = np.stack([c.log_band_energies for c in characters])
    shapes = np.stack([c.envelope_shape for c in characters])
    return SoundCharacter(
        log_band_energies=(weight_array[:, None] * log_energies).sum(axis=0),
        envelope_shape=(weight_array[:, None] * shapes).sum(axis=0),
    )


def character_from_files(paths: Sequence[str | Path]) -> SoundCharacter:
    """Average character of several audio files (e.g. a curated sound group)."""
    characters = []
    for path in paths:
        waveform, sample_rate = load_audio(path)
        characters.append(extract_character(waveform, sample_rate))
    return average_characters(characters)


def character_from_folder(folder: str | Path) -> SoundCharacter:
    """Average character of every audio file under a folder (recursive)."""
    paths = list_audio_files(folder)
    if not paths:
        raise FileNotFoundError(f"No audio files found under {folder}")
    return character_from_files(paths)


def apply_character(
    source: np.ndarray,
    source_sr: int,
    character: SoundCharacter,
    amount: float | np.ndarray = 1.0,
    *,
    transfer_spectrum: bool = True,
    transfer_envelope: bool = True,
    max_gain_db: float = 24.0,
) -> np.ndarray:
    """Pulls ``source`` toward ``character`` by ``amount``.

    ``amount`` may be a scalar in [0, 1] or a per-sample array (any length —
    it is resampled to the source length), enabling time-varying character
    control such as ADSR-shaped morphing. The time-varying path computes the
    full morph once and crossfades source and morph; because the morph keeps
    the source's phase, the crossfade interpolates the character smoothly
    without comb artifacts.

    The source's fine structure (pitch, waveshape, micro-detail) is preserved;
    only the spectral envelope and temporal envelope move. Returns a waveform
    with the source's shape and sample rate.
    """
    channels = _as_channels(source)
    original_peak = np.abs(channels).max()
    if original_peak <= 1e-8:
        return channels.astype(np.float32)

    if not np.isscalar(amount):
        envelope = np.clip(np.asarray(amount, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if envelope.shape[0] != channels.shape[1]:
            positions = np.linspace(0.0, 1.0, channels.shape[1])
            envelope = np.interp(positions, np.linspace(0.0, 1.0, envelope.shape[0]), envelope)
        morphed = apply_character(
            channels,
            source_sr,
            character,
            amount=1.0,
            transfer_spectrum=transfer_spectrum,
            transfer_envelope=transfer_envelope,
            max_gain_db=max_gain_db,
        ).astype(np.float64)
        mixed = (1.0 - envelope)[None, :] * channels + envelope[None, :] * morphed
        peak = np.abs(mixed).max()
        if peak > 1e-12:
            mixed *= min(original_peak, 0.99) / peak
        return mixed.astype(np.float32)

    amount = float(np.clip(amount, 0.0, 1.0))
    source_character = extract_character(channels, source_sr)
    result = channels.copy()

    if transfer_spectrum:
        log_gain = 0.5 * amount * (character.log_band_energies - source_character.log_band_energies)
        limit = max_gain_db / 20.0 * np.log(10.0)
        log_gain = np.clip(log_gain, -limit, limit)

        n_samples = result.shape[1]
        bin_freqs = np.fft.rfftfreq(n_samples, d=1.0 / source_sr)
        clipped = np.clip(bin_freqs, _CENTERS[0], _CENTERS[-1])
        gain_curve = np.exp(np.interp(np.log(clipped), np.log(_CENTERS), log_gain))
        for channel in range(result.shape[0]):
            spectrum = np.fft.rfft(result[channel])
            result[channel] = np.fft.irfft(spectrum * gain_curve, n=n_samples)

    if transfer_envelope:
        length = result.shape[1]
        positions = np.linspace(0.0, 1.0, length)
        grid = np.linspace(0.0, 1.0, CHARACTER_ENVELOPE_POINTS)
        target_shape = np.interp(positions, grid, character.envelope_shape)
        source_shape = np.interp(positions, grid, source_character.envelope_shape)
        limit = 10.0 ** (max_gain_db / 20.0)
        gain = ((target_shape + 1e-3) / (source_shape + 1e-3)) ** amount
        gain = np.clip(gain, 1.0 / limit, limit)
        gain = _smooth(gain, max(int(source_sr * 0.01), 1))
        result *= gain[None, :]

    peak = np.abs(result).max()
    if peak > 1e-12:
        result *= min(original_peak, 0.99) / peak
    return result.astype(np.float32)
