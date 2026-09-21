"""Deterministic imagination axes: brightness and tonality (roadmap G-2/G-9
deterministic fallbacks).

Two `amount`-controlled creative knobs that work with no trained model — the
library never *requires* one (I-2). The learned versions (G-2 conditioned
bandwidth extension, G-9 conditioned harmonic/noise morph) replace or augment
these later; the API and the measured axis (`datasets.axis_labels`) stay the
same, so a model is a drop-in upgrade.

- :func:`imagine_highs` — "imagine more air/brightness": deterministic
  spectral band replication. Missing/quiet high bands are filled by
  heterodyning content up from the loudest mid band, rolled off gently, and
  high-passed onto the original. ``amount`` 0 = unchanged (bit-exact), 1 =
  natural extension, >1 exaggerated. The low end is essentially preserved
  (a few-percent blend around the split, like any spectral edit's apron).
- :func:`imagine_tonality` — "more tonal ↔ more noisy": HPSS-layer remix.
  ``amount`` in [-1, 1]: +1 fully harmonic (cut noise/transients), -1 fully
  percussive/noisy, 0 the original.
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.decompose import hpss
from wavelet2vec.frame import analyze, synthesize


def imagine_highs(
    audio: np.ndarray,
    sample_rate: int,
    amount: float = 1.0,
    *,
    n_bands: int = 128,
    split_hz: float = 4000.0,
    max_boost_db: float = 18.0,
    model=None,
    brightness: float | None = None,
) -> np.ndarray:
    """Extends the sound's brightness by imagining plausible high content.

    For each band above ``split_hz``, an analytic signal an octave below (or
    the loudest mid band, if that source is too quiet) is heterodyned up to
    that band — so the imagined highs inherit the body's dynamics and
    transients — scaled to a gentle −6 dB/octave rolloff from the loudest mid
    band, then high-passed and added with weight ``amount``. ``amount=0`` is
    bit-exact; the low end is essentially preserved (few-percent blend across
    the split, the apron every T-F edit has).

    ``model`` (G-2 learned v1): a :class:`~wavelet2vec.brightness_model.
    BrightnessNet` (or a checkpoint path) replaces only the fixed −6 dB/oct
    energy rule — it predicts per-band per-frame envelopes for the imagined
    highs, conditioned on ``brightness`` in [0, 1] (default: the input's
    measured brightness + 0.1). Carrier and phase stay deterministic, so the
    low end is preserved exactly as in the model-free path.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    boost_cap = 10.0 ** (max_boost_db / 20.0)

    if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
        from wavelet2vec.brightness_model import load_brightness_model

        model = load_brightness_model(model)

    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands)
        centers = spec.centers
        energies = (np.abs(coeffs) ** 2).mean(axis=1)

        # Reference the imagined highs to the loudest *real* mid band and roll
        # them off gently (~-6 dB/octave) — not to the sound's own HF rolloff,
        # which for a dull sound predicts (correctly) near-zero and would add
        # nothing. That reference is what makes "imagine" different from "EQ".
        mid_mask = centers < split_hz
        if not mid_mask.any():
            out[index] = channel
            continue
        peak_mid_energy = float(energies[mid_mask].max())
        peak_mid_band = int(np.argmax(np.where(mid_mask, energies, -1.0)))
        energy_floor = 0.01 * peak_mid_energy

        target_env = None  # model path: [model_bands, n] target envelopes
        model_centers = None
        if model is not None:
            from wavelet2vec.brightness_model import (
                envelope_brightness, log_envelopes, predict_full_envelopes,
            )

            env_log, model_centers = log_envelopes(channel, sample_rate)
            target = (brightness if brightness is not None
                      else min(envelope_brightness(env_log, model_centers) + 0.1, 1.0))
            predicted = np.expm1(
                predict_full_envelopes(model, env_log, float(target)))
            # Back to this channel's units (log_envelopes peak-normalizes),
            # then frame centers → per-sample envelopes.
            peak = np.abs(channel).max()
            predicted *= peak if peak > 1e-12 else 1.0
            hop = max(n // predicted.shape[1], 1)
            frame_times = (np.arange(predicted.shape[1]) + 0.5) * hop
            target_env = np.stack([
                np.interp(np.arange(n, dtype=np.float64), frame_times, row)
                for row in predicted
            ])

        t = np.arange(n) / sample_rate
        added = np.zeros_like(coeffs)  # only the imagined high content
        for b in np.flatnonzero(centers >= split_hz):
            source = int(np.argmin(np.abs(centers - centers[b] / 2.0)))  # octave below
            if energies[source] < energy_floor:
                source = peak_mid_band  # too quiet to transpose: use the loud mid band
            heterodyned = coeffs[source] * np.exp(2j * np.pi * (centers[b] - centers[source]) * t)
            if target_env is not None:
                nearest = int(np.argmin(np.abs(model_centers - centers[b])))
                carrier_env = np.abs(heterodyned)
                gain = np.minimum(
                    target_env[nearest] / (carrier_env + 1e-8 * carrier_env.max() + 1e-24),
                    boost_cap)
                added[b] = float(amount) * gain * heterodyned
                continue
            source_energy = (np.abs(heterodyned) ** 2).mean() + 1e-24
            target_energy = peak_mid_energy * (split_hz / centers[b]) ** 2  # -6 dB/oct
            gain = min(np.sqrt(target_energy / source_energy), boost_cap)
            added[b] = float(amount) * gain * heterodyned
        contribution = synthesize(added, spec)[:n]
        # High-pass the imagined content so its band tails can't recolor below
        # the split — the original low end is then preserved exactly.
        if np.any(added) and split_hz < 0.5 * sample_rate:
            from scipy.signal import butter, sosfiltfilt

            sos = butter(4, split_hz / (0.5 * sample_rate), btype="high", output="sos")
            contribution = sosfiltfilt(sos, contribution)
        out[index] = channel + contribution

    result = out[0] if mono_input else out
    return result.astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitones: int,
    *,
    bands_per_octave: int = 12,
    f_min: float = 27.5,
) -> np.ndarray:
    """Pitch shift by whole semitones via per-band heterodyning (roadmap D-4).

    The frame is semitone-spaced, so a shift is a *row remap* — but a naive
    row roll provably fails: coefficients are analytic band signals still
    oscillating at the source frequency, so synthesis puts the energy right
    back where it was. Each band must be **heterodyned** to its destination
    (multiplied by ``exp(2πj·Δf·t)``) before the remap. Duration is
    preserved (this shifts pitch, not speed); formants shift with the pitch
    (no formant correction — that is a learned-tier feature).
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    semitones = int(semitones)
    if semitones == 0:
        return channels[0].astype(np.float32) if mono_input else channels.astype(np.float32)

    octaves = np.log2(0.45 * sample_rate / f_min)
    n_bands = int(np.floor(octaves * bands_per_octave)) + 1
    f_max = f_min * 2.0 ** ((n_bands - 1) / bands_per_octave)
    t = np.arange(n) / sample_rate

    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=n_bands,
                               f_min=f_min, f_max=f_max)
        centers = spec.centers
        shifted = np.zeros_like(coeffs)
        for band in range(n_bands):
            target = band + semitones
            if not 0 <= target < n_bands:
                continue  # shifted off the analyzed range
            delta = centers[target] - centers[band]
            shifted[target] = coeffs[band] * np.exp(2j * np.pi * delta * t)
        out[index] = synthesize(shifted, spec)[:n]

    result = out[0] if mono_input else out
    return result.astype(np.float32)


def imagine_transient(
    audio: np.ndarray,
    sample_rate: int,
    amount: float = 1.0,
    *,
    model=None,
    sharpness: float = 0.6,
) -> np.ndarray:
    """Imagines an attack that isn't there — pad gets a pluck (G-8).

    Each band's attack is built from the band's *own body* (a segment of
    its analytic signal copied to the onset — the imagined attack inherits
    the sound's timbre), shaped by a 2 ms ramp × exponential decay. The
    per-band gains and decays come from :class:`~wavelet2vec.
    transient_model.TransientNet` (or a checkpoint path) conditioned on
    ``sharpness`` in [0, 1], or from the deterministic generic pluck when
    no model is given. ``amount`` weights the added attack; 0 is
    bit-exact. v1 targets one-shots (a single onset).
    """
    from wavelet2vec.transient_model import (
        MODEL_BANDS as _BANDS,
        MODEL_F_MIN as _F_MIN,
        MODEL_FRAMES as _FRAMES,
        build_attack,
        default_attack_params,
        onset_index,
    )

    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]

    if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
        from wavelet2vec.transient_model import load_transient_model

        model = load_transient_model(model)

    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        coeffs, spec = analyze(channel, sample_rate, n_bands=_BANDS,
                               f_min=_F_MIN)
        onset = onset_index(channel, sample_rate)
        if model is not None:
            from wavelet2vec.transient_model import predict_attack_params

            magnitude = np.abs(coeffs)
            hop = max(magnitude.shape[1] // _FRAMES, 1)
            pooled = magnitude[:, : hop * _FRAMES].reshape(
                _BANDS, _FRAMES, hop).mean(axis=2)
            peak = pooled.max()
            envelopes = np.log1p(pooled / peak if peak > 1e-12 else pooled)
            params = predict_attack_params(model, envelopes, float(sharpness))
        else:
            params = default_attack_params(spec.centers)
        added = build_attack(coeffs, sample_rate, onset, params)
        contribution = synthesize(added, spec)[:n]
        out[index] = channel + float(amount) * contribution

    result = out[0] if mono_input else out
    return result.astype(np.float32)


def imagine_tonality(
    audio: np.ndarray,
    sample_rate: int,
    amount: float = 0.0,
    *,
    n_bands: int = 96,
    model=None,
    harmonicity: float | None = None,
) -> np.ndarray:
    """Morphs a sound along the harmonic ↔ noisy axis via an HPSS remix.

    ``amount`` in [-1, 1]: the output is ``harmonic·(1+amount) +
    percussive·(1−amount)``, so +1 is fully tonal (transients/noise removed),
    −1 fully percussive/noisy, and 0 the original (``h + p == input``). The
    result is peak-matched to the input.

    ``harmonicity`` (G-9): a target in [0, 1] makes the knob *calibrated* —
    the layer remix is measured at five anchor amounts (the layers are
    already in hand, so this costs five autocorrelations) and the measured
    response curve is inverted to hit the request; the best any scalar
    remix can do, without a model. ``amount`` in [0, 1] then scales the
    chosen remix (0 = the input, 1 = fully calibrated remix).

    ``model`` (experimental): a :class:`~wavelet2vec.tonality_model.
    TonalityNet` (or checkpoint path) predicts the remix from envelopes in
    one forward pass instead of measuring — measured on held-out clips it
    calibrates *worse* than the anchored inversion (see FINDINGS); it
    exists as the substrate for the frequency-selective v2.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio

    if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
        from wavelet2vec.tonality_model import load_tonality_model

        model = load_tonality_model(model)

    out = np.empty_like(channels)
    for index, channel in enumerate(channels):
        harmonic, percussive = hpss(channel, sample_rate, n_bands=n_bands)
        if model is not None and harmonicity is not None:
            from wavelet2vec.tonality_model import (
                layer_envelopes, predict_layer_gains, remix_layers,
            )

            envelopes, _centers, coeff_pair = layer_envelopes(
                harmonic, percussive, sample_rate)
            gains = predict_layer_gains(model, envelopes, float(harmonicity))
            wet = float(np.clip(amount, 0.0, 1.0))
            gains = np.exp(wet * np.log(gains))
            mixed = remix_layers(coeff_pair, gains, channel.shape[0])
        elif harmonicity is not None:
            # Anchored inversion: measure the remix's harmonicity response
            # at five amounts and invert the (monotone-stabilized) curve.
            from wavelet2vec.harmonic_features import autocorrelation_pitch

            anchors = np.linspace(-1.0, 1.0, 5)
            curve = np.maximum.accumulate([
                autocorrelation_pitch(
                    harmonic * (1.0 + a) + percussive * (1.0 - a),
                    sample_rate)[0]
                for a in anchors
            ])
            calibrated = float(np.interp(float(harmonicity), curve, anchors))
            blend = float(np.clip(amount, 0.0, 1.0)) * calibrated
            mixed = harmonic * (1.0 + blend) + percussive * (1.0 - blend)
        else:
            blend = float(np.clip(amount, -1.0, 1.0))
            mixed = harmonic * (1.0 + blend) + percussive * (1.0 - blend)
        original_peak = np.abs(channel).max()
        peak = np.abs(mixed).max()
        if peak > 1e-12 and original_peak > 1e-12:
            mixed = mixed * (original_peak / peak)
        out[index] = mixed

    result = out[0] if mono_input else out
    return result.astype(np.float32)
