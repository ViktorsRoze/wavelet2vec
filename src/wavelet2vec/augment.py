"""Music-native augmentations for training pipelines (roadmap D-5).

Composes with :func:`wavelet2vec.imagine.pitch_shift` (D-4):

- :func:`time_stretch` — change duration without changing pitch, inside the
  invertible frame: per-band magnitude and per-sample phase *increments*
  are resampled to the new length and the phase re-integrated, so each
  band keeps its instantaneous frequency (a constant-Q phase vocoder).
- :func:`effect_chain` — run audio through pedalboard effects (reverb,
  distortion, chorus, …) named by string or given as plugin objects.
- :func:`crop_bars` — grid-aware crops: exact bar-length fragments at a
  known BPM, so training crops respect musical boundaries.
- :func:`swap_channels` / :func:`invert_polarity` / :func:`mono_mix` —
  cheap label-preserving channel ops.
- :func:`random_augment` — a seeded sampler combining the above for
  self-supervised views; deterministic per (seed, index).
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.frame import FrameSpec, analyze, synthesize


def time_stretch(
    audio: np.ndarray,
    sample_rate: int,
    rate: float,
    *,
    n_bands: int = 96,
) -> np.ndarray:
    """Stretches duration by ``1/rate`` with pitch preserved (rate 2 = twice
    as fast/half as long; 0.5 = half speed). Constant-Q phase vocoder:
    magnitudes are resampled in time, phase is rebuilt by integrating the
    resampled per-sample phase increments — so instantaneous frequency (and
    therefore pitch) is untouched."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    m = max(int(round(n / rate)), 2)
    positions = np.linspace(0.0, n - 1.0, m)

    out = np.empty((channels.shape[0], m))
    spec_out: FrameSpec | None = None
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        if spec_out is None:
            spec_out = FrameSpec.from_dict({
                "centers": spec.centers.tolist(), "length": m,
                "sample_rate": sample_rate, "bandwidth_scale": spec.bandwidth_scale,
            })
        magnitude = np.abs(coeffs)
        increment = np.angle(coeffs[:, 1:] * np.conj(coeffs[:, :-1]))
        increment = np.concatenate([np.zeros((n_bands, 1)), increment], axis=1)

        sample_grid = np.arange(n, dtype=np.float64)
        new_coeffs = np.empty((n_bands, m), dtype=np.complex128)
        start_phase = np.angle(coeffs[:, 0])
        for band in range(n_bands):
            mag_new = np.interp(positions, sample_grid, magnitude[band])
            inc_new = np.interp(positions, sample_grid, increment[band])
            phase_new = start_phase[band] + np.cumsum(inc_new)
            new_coeffs[band] = mag_new * np.exp(1j * phase_new)
        out[index] = synthesize(new_coeffs, spec_out)[:m]

    result = out[0] if mono_input else out
    return result.astype(np.float32)


def effect_chain(audio: np.ndarray, sample_rate: int, effects) -> np.ndarray:
    """Runs audio through a pedalboard chain. ``effects`` is a list of
    pedalboard plugin objects or ``(name, params)`` / ``"name"`` shorthands
    for built-ins (``reverb``, ``distortion``, ``chorus``, ``phaser``,
    ``compressor``, ``lowpass``, ``highpass``, ``gain``, ``delay``).
    Length and shape are preserved."""
    try:
        import pedalboard
    except Exception as error:  # pragma: no cover - optional dependency
        raise RuntimeError("effect_chain needs the 'pedalboard' package.") from error

    builtins = {
        "reverb": pedalboard.Reverb, "distortion": pedalboard.Distortion,
        "chorus": pedalboard.Chorus, "phaser": pedalboard.Phaser,
        "compressor": pedalboard.Compressor, "lowpass": pedalboard.LowpassFilter,
        "highpass": pedalboard.HighpassFilter, "gain": pedalboard.Gain,
        "delay": pedalboard.Delay,
    }
    plugins = []
    for effect in effects:
        if isinstance(effect, str):
            plugins.append(builtins[effect.lower()]())
        elif isinstance(effect, tuple):
            name, params = effect
            plugins.append(builtins[name.lower()](**params))
        else:
            plugins.append(effect)

    audio = np.asarray(audio, dtype=np.float32)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    board = pedalboard.Pedalboard(plugins)
    processed = np.asarray(board(channels, sample_rate), dtype=np.float32)
    if processed.shape[1] < channels.shape[1]:  # defensive length guard
        pad = np.zeros((processed.shape[0], channels.shape[1] - processed.shape[1]),
                       dtype=np.float32)
        processed = np.concatenate([processed, pad], axis=1)
    processed = processed[:, : channels.shape[1]]
    return processed[0] if mono_input else processed


def crop_bars(
    audio: np.ndarray,
    sample_rate: int,
    bpm: float,
    *,
    bars: int = 1,
    start_bar: int = 0,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """An exact ``bars``-long crop starting at ``start_bar`` — training crops
    on musical boundaries instead of arbitrary windows."""
    audio = np.asarray(audio)
    bar_samples = int(round(60.0 / bpm * beats_per_bar * sample_rate))
    start = start_bar * bar_samples
    end = start + bars * bar_samples
    n = audio.shape[-1]
    if start >= n:
        raise ValueError(f"start_bar {start_bar} is past the audio ({n} samples).")
    crop = audio[..., start:min(end, n)]
    if crop.shape[-1] < bars * bar_samples:  # pad a short tail to exact length
        pad_shape = list(crop.shape)
        pad_shape[-1] = bars * bar_samples - crop.shape[-1]
        crop = np.concatenate([crop, np.zeros(pad_shape, dtype=crop.dtype)], axis=-1)
    return crop


def swap_channels(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 2 or audio.shape[0] < 2:
        return audio.copy()
    return audio[::-1].copy()


def invert_polarity(audio: np.ndarray) -> np.ndarray:
    return -np.asarray(audio)


def mono_mix(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    return audio.mean(axis=0) if audio.ndim == 2 else audio.copy()


def random_augment(
    audio: np.ndarray,
    sample_rate: int,
    *,
    seed: int = 0,
    index: int = 0,
    max_semitones: int = 4,
    stretch_range: tuple[float, float] = (0.85, 1.18),
    effect_probability: float = 0.5,
) -> np.ndarray:
    """A random label-preserving view for self-supervised training —
    deterministic per ``(seed, index)`` so epochs are reproducible.

    Applies (each independently sampled): pitch shift within
    ``±max_semitones``, time stretch within ``stretch_range`` (output is
    trimmed/padded back to the input length so batch shapes hold), one
    random pedalboard effect with ``effect_probability``, and polarity/
    channel flips."""
    from wavelet2vec.imagine import pitch_shift

    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    out = np.asarray(audio, dtype=np.float32)
    n = out.shape[-1]

    semitones = int(rng.integers(-max_semitones, max_semitones + 1))
    if semitones:
        out = pitch_shift(out, sample_rate, semitones)
    rate = float(rng.uniform(*stretch_range))
    if abs(rate - 1.0) > 0.02:
        out = time_stretch(out, sample_rate, rate)
        if out.shape[-1] >= n:  # restore the input length for batching
            out = out[..., :n]
        else:
            pad_shape = list(out.shape)
            pad_shape[-1] = n - out.shape[-1]
            out = np.concatenate([out, np.zeros(pad_shape, dtype=out.dtype)], axis=-1)
    if rng.random() < effect_probability:
        try:
            effect = str(rng.choice(["reverb", "chorus", "distortion", "lowpass"]))
            out = effect_chain(out, sample_rate, [effect])
        except RuntimeError:
            pass  # pedalboard not installed: skip the effect stage
    if rng.random() < 0.5:
        out = invert_polarity(out)
    if out.ndim == 2 and rng.random() < 0.5:
        out = swap_channels(out)
    return out.astype(np.float32)
