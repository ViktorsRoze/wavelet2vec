"""Reference-based percussive event diagnostics, not blind instrument recognition."""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks

from wavelet2vec.audio_io import resample
from wavelet2vec.filterbank import log_spaced_frequencies, morlet_scalogram


def _audio(audio, sample_rate):
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("audio must be finite nonempty [channels,samples]")
    return x


def detect_percussive_events(audio, sample_rate, *, sensitivity=.92, min_spacing_seconds=.07,
                             analysis_sample_rate=22050):
    """Wavelet-flux candidates with native-energy refinement and explicit onset uncertainty.

    Channels contribute power independently, avoiding anti-phase mono cancellation.
    Analysis-only resampling is declared in the result. No synthetic zero-time
    slice is inserted. Activity at a clip boundary is labeled separately.
    """
    x = _audio(audio, sample_rate)
    if not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
        raise ValueError("sensitivity must be in [0,1]")
    if not math.isfinite(min_spacing_seconds) or min_spacing_seconds <= 0:
        raise ValueError("min_spacing_seconds must be positive")
    if isinstance(analysis_sample_rate, bool) or not isinstance(analysis_sample_rate, int) or analysis_sample_rate < 1000:
        raise ValueError("analysis_sample_rate must be an integer >= 1000")
    rate = min(sample_rate, analysis_sample_rate)
    peak = float(np.max(np.abs(x)))
    result = {"analysis_sample_rate": rate, "native_sample_rate": sample_rate,
              "min_spacing_seconds": min_spacing_seconds, "sensitivity": sensitivity, "events": []}
    if peak <= 1e-12:
        return result
    normalized = resample(x, sample_rate, rate)/peak
    hop = max(1, round(rate*.005))
    centers = log_spaced_frequencies(min(27.5, rate*.01), rate*.45, 48)
    magnitude = np.sqrt(sum(morlet_scalogram(channel, rate, centers)**2 for channel in normalized)/len(normalized))
    count = magnitude.shape[-1]//hop
    candidates = []
    if count >= 3:
        framed = magnitude[:, :count*hop].reshape(48, count, hop).mean(-1)
        flux = np.maximum(np.diff(np.log1p(framed), axis=1), 0).sum(0)
        threshold = flux.mean()+(1-sensitivity)*(flux.max()-flux.mean())
        if flux.max() > 1e-12:
            peaks, _ = find_peaks(flux, height=max(threshold, 1e-12), distance=max(1, round(min_spacing_seconds*rate/hop)))
            candidates = [(int(round((i+1)*hop/ rate*sample_rate)), "onset", float(flux[i]/flux.max())) for i in peaks]
    first = x[:, :max(1, round(sample_rate*.02))]
    if np.sqrt(np.mean(first**2)) > np.sqrt(np.mean(x**2))*.15:
        candidates = [(0, "active_at_start", 1.)]+[row for row in candidates if row[0] >= sample_rate*min_spacing_seconds]
    energy = uniform_filter1d(np.mean(x*x, axis=0), max(1, round(sample_rate*.001)), mode="constant")
    previous = -int(sample_rate*min_spacing_seconds)
    for rough, kind, strength in candidates:
        if kind == "active_at_start":
            onset = 0
        else:
            left, right = max(0, rough-round(sample_rate*.025)), min(x.shape[-1], rough+round(sample_rate*.015))
            region = energy[left:right]
            if not len(region):
                continue
            floor = float(np.quantile(region, .05))
            threshold = floor + .05*(float(region.max())-floor)
            active = np.flatnonzero(region > max(threshold, peak*peak*1e-8))
            onset = left+int(active[0]) if len(active) else rough
        if onset-previous < round(sample_rate*min_spacing_seconds*.8):
            continue
        previous = onset
        result["events"].append({"sample": onset, "time_seconds": onset/sample_rate,
                                 "wavelet_peak_seconds": rough/sample_rate, "kind": kind,
                                 "relative_novelty": strength, "timing_resolution_seconds": hop/rate})
    return result


def compare_percussive_events(reference, target, sample_rate, *, preroll_samples=0,
                              search_pitch=True, max_semitones=12., max_lag_seconds=.02,
                              window_seconds=.06, confidence_threshold=.95):
    """Fit relative playback rate, sub-sample timing and independent channel gains.

    Pitch is reported only as playback-rate-equivalent semitones when the
    waveform fit is strong. This is not an F0 estimate for noisy/inharmonic
    drums and cannot determine MIDI velocity or synthesizer provenance.
    """
    a, b = _audio(reference, sample_rate), _audio(target, sample_rate)
    if a.shape[0] != b.shape[0]:
        raise ValueError("event channels must match")
    if isinstance(preroll_samples, bool) or not isinstance(preroll_samples, int) or preroll_samples < 0:
        raise ValueError("preroll_samples must be nonnegative integer")
    if (not isinstance(search_pitch, bool) or not math.isfinite(max_semitones) or not 0 < max_semitones <= 24
            or not math.isfinite(max_lag_seconds) or max_lag_seconds < 0
            or not math.isfinite(window_seconds) or window_seconds <= 0 or not 0 < confidence_threshold <= 1):
        raise ValueError("invalid comparison settings")
    lag = round(sample_rate*max_lag_seconds)
    maximum_rate = 2**(max_semitones/12) if search_pitch else 1.
    n = min(round(sample_rate*window_seconds), b.shape[-1]-preroll_samples,
            int((a.shape[-1]-preroll_samples)/maximum_rate)-lag)
    if n < max(16, round(sample_rate*.008)):
        return {"status": "insufficient_duration", "pitch_shift_semitones": None}
    y = b[:, preroll_samples:preroll_samples+n]
    target_energy = float(np.sum(y*y))
    if target_energy <= 1e-24 or np.sum(a*a) <= 1e-24:
        return {"status": "silent", "pitch_shift_semitones": None}
    grid = np.arange(a.shape[-1])
    time = np.arange(n)
    # Every coarse candidate uses the same target and FFT geometry. Batch
    # channels and reuse its spectrum instead of retransforming it per rate.
    fft_length = next_fast_len(2*n+2*lag-1, real=True)
    target_spectrum = rfft(y[:, ::-1], n=fft_length, axis=-1)
    def direct(semitones, offset):
        positions = preroll_samples+(time+offset)*2**(semitones/12)
        fitted = np.stack([np.interp(positions, grid, channel, left=0., right=0.) for channel in a])
        energy = np.sum(fitted*fitted, axis=1)
        gains = np.divide(np.sum(y*fitted, axis=1), energy, out=np.zeros(len(a)), where=energy > 1e-24)
        error = float(np.sum((y-gains[:, None]*fitted)**2)/target_energy)
        return error, gains
    def coarse(semitones):
        positions = preroll_samples+(np.arange(n+2*lag)-lag)*2**(semitones/12)
        warped = np.stack([np.interp(positions, grid, channel, left=0., right=0.) for channel in a])
        dots = irfft(rfft(warped, n=fft_length, axis=-1)*target_spectrum,
                     n=fft_length, axis=-1)[:, n-1:n+2*lag]
        prefix = np.pad(np.cumsum(warped*warped, axis=1), ((0, 0), (1, 0)))
        energies = prefix[:, n:]-prefix[:, :-n]
        explained = np.divide(dots*dots, energies, out=np.zeros_like(dots), where=energies > 1e-24).sum(0)
        best = int(np.argmax(explained))
        return direct(semitones, best-lag)[0], float(best-lag)
    zero_error, zero_lag = coarse(0.)
    fitted_zero = minimize_scalar(lambda value: direct(0., value)[0], bounds=(max(-lag, zero_lag-1), min(lag, zero_lag+1)), method="bounded") if lag else None
    if fitted_zero is not None and fitted_zero.fun < zero_error:
        zero_error, zero_lag = float(fitted_zero.fun), float(fitted_zero.x)
    best_semitones, best_lag, best_error = 0., zero_lag, zero_error
    if search_pitch:
        for semitones in np.arange(-max_semitones, max_semitones+.001, .5):
            error, offset = coarse(float(semitones))
            if error < best_error:
                best_semitones, best_lag, best_error = float(semitones), offset, error
        if lag:
            fit = minimize(lambda v: direct(v[0], v[1])[0], [best_semitones, best_lag], method="Powell",
                bounds=[(max(-max_semitones, best_semitones-.5), min(max_semitones, best_semitones+.5)),
                        (max(-lag, best_lag-2), min(lag, best_lag+2))], options={"maxiter": 40, "xtol": 1e-5, "ftol": 1e-8})
            if np.isfinite(fit.fun) and fit.fun < best_error:
                best_semitones, best_lag, best_error = float(fit.x[0]), float(fit.x[1]), float(fit.fun)
    _, gains = direct(best_semitones, best_lag)
    confidence = math.sqrt(max(0., 1-best_error))
    accepted = confidence >= confidence_threshold and abs(best_semitones) < max_semitones-.01
    integer_positions = preroll_samples+time+round(zero_lag)
    integer_reference = np.stack([np.interp(integer_positions, grid, channel, left=0., right=0.) for channel in a])
    best_positions = preroll_samples+(time+best_lag)*2**(best_semitones/12)
    best_reference = np.stack([np.interp(best_positions, grid, channel, left=0., right=0.) for channel in a])
    channel_energy = np.sum(best_reference*best_reference, axis=1)
    gain_ratio = np.sqrt(np.sum(channel_energy*gains*gains)/max(np.sum(channel_energy), 1e-30))
    return {"status": "matched" if accepted else "unresolved_waveform_change",
            "pitch_shift_semitones": best_semitones if accepted else None,
            "best_candidate_semitones": best_semitones, "waveform_fit": confidence,
            "zero_pitch_fit": math.sqrt(max(0., 1-zero_error)),
            "residual_db": 10*math.log10(max(best_error, 1e-30)),
            "lag_samples": best_lag, "channel_gains": gains.tolist(), "comparison_samples": n,
            "comparison_window_exact_pcm_match": bool(np.array_equal(integer_reference, y)),
            "fitted_level_change_db": 20*math.log10(max(float(gain_ratio), 1e-15)) if accepted else None,
            "max_lag_seconds": max_lag_seconds, "fit_acceptance_threshold": confidence_threshold,
            "pitch_model": "playback-rate equivalent, not fundamental frequency"}


def analyze_drum_track(audio, sample_rate, *, bpm=None, sensitivity=.92, min_spacing_seconds=.07):
    """Analyze isolated annotated drum audio; report every event and reference fit."""
    x = _audio(audio, sample_rate)
    if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
        raise ValueError("bpm must be positive")
    detection = detect_percussive_events(x, sample_rate, sensitivity=sensitivity, min_spacing_seconds=min_spacing_seconds)
    events = detection["events"]
    if not events:
        return {**detection, "reference_event": None, "repeat_groups": 0}
    pre = round(sample_rate*.02)
    snippets = []
    for index, event in enumerate(events):
        start = event["sample"]
        stop = min(x.shape[-1], start+round(sample_rate*.25),
                   events[index+1]["sample"] if index+1 < len(events) else x.shape[-1])
        end_measure = min(stop, start+round(sample_rate*.06))
        power = np.mean(x[:, start:end_measure]**2, axis=1)
        rms = float(np.sqrt(power.mean()))
        left = max(0, start-pre)
        snippets.append(np.pad(x[:, left:stop], ((0, 0), (pre-(start-left), 0))))
        event.update({"rms": rms, "level_dbfs": 20*math.log10(max(rms, 1e-15)),
                      "peak": float(np.abs(x[:, start:stop]).max()), "measurement_samples": end_measure-start})
        if len(power) == 2:
            event["stereo_energy_balance"] = float((power[1]-power[0])/max(power.sum(), 1e-30))
            event["right_over_left_db"] = float(10*math.log10(power[1]/power[0])) if np.all(power > 0) else None
        if bpm is not None:
            event["beat_position"] = start/sample_rate*bpm/60
            event["bar_number"] = int(event["beat_position"]//4)+1
    reference = int(np.argmax([event["rms"] for event in events]))
    templates = []
    for index, event in enumerate(events):
        event["relative_level_db"] = event["level_dbfs"]-events[reference]["level_dbfs"]
        event["reference_comparison"] = compare_percussive_events(snippets[reference], snippets[index], sample_rate, preroll_samples=pre)
        best_score, best_group = -1., None
        for group, representative in enumerate(templates):
            fit = compare_percussive_events(snippets[representative], snippets[index], sample_rate, preroll_samples=pre, search_pitch=False)
            score = fit.get("zero_pitch_fit", 0.)
            if score > best_score:
                best_score, best_group = score, group
        event["best_previous_template_fit"] = max(0., best_score) if templates else None
        if best_score < .995:
            best_group = len(templates)
            templates.append(index)
        event["candidate_repeat_group"] = best_group
    return {**detection, "reference_event": reference, "repeat_groups": len(templates),
            "repeat_fit_threshold": .995, "level_definition": "first up-to-60 ms RMS, relative to loudest detected event",
            "limits": "Reference-based waveform similarity; no MIDI velocity, pan-knob, sample-origin or blind instrument claim."}
