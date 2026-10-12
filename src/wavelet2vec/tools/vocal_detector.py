"""Vocal detector: label the parts of a voice recording, frame by frame.

Algorithmic (no training). Every 10 ms frame gets one part:

- ``silence``  quieter than ``floor_db`` below the loudest frame, or room
  background: no voice part and more than ``background_db`` below it;
- ``plosive``  the burst of p/t/k: a jump of >= 15 dB out of near-silence
  with a noise-like (flat) spectrum;
- ``sibilant`` "s"/"sh": most energy is noise above 2.5 kHz. ``sibilant_type``
  is "s" when that noise is centred above ~4.8 kHz, otherwise "sh";
- ``vowel``    a harmonic voice frame, with a ``vowel`` label (see below);
- ``voiced``   harmonic, but too few harmonics to judge a vowel (very high f0);
- ``breath``   other noise: h, f, aspiration, breathing;
- ``transition`` none of the above, within 50 ms of a vowel: note onsets
  and endings where the pitch is not yet (or no longer) steady;
- ``other``    none of the above.

``sibilance`` (0..1) is the share of a frame's energy above 2.5 kHz, useful
on its own, e.g. to find sibilant peaks for de-essing.

Vowels are judged from the harmonic spectral envelope on a 16 kHz analysis
copy. Sung notes reach f0 = 500-700 Hz, where harmonics are spaced wider than
a formant bandwidth and LPC locks onto harmonics instead of resonances. So
the envelope is sampled only at the harmonics k·f0 (f0 by subharmonic
summation, refined to 0.1 %). It is then fitted against vowel templates
(cascades of formant resonators at scale·F1..F5) with a free level and a
bounded spectral tilt. ``scale`` models vocal-tract length (1.0 = adult male
reference; about 1.15-1.2 adult female; 1.3 child). It is estimated from the
best-fitting frames unless given; for one speaker, estimate it once and pass
it to every call. Templates are adult male averages: the mean of Peterson &
Barney (1952) and Hillenbrand et al. (1995) where both list the vowel,
otherwise Hillenbrand.

Noise features (level, flatness, energy above 2.5 kHz, noise centroid) use
20 ms windows on a 32 kHz analysis copy, so "s" energy up to 16 kHz is seen.
Labels carry a confidence margin. They are not phonetic transcription:
neighbouring vowels ("e"/"i", "o"/"u", "a"/"aw") are confused first,
especially at high pitch and when singers darken vowels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet2vec.audio_io import resample

VOWEL_RATE = 16000
NOISE_RATE = 32000

# F1..F5 in Hz, adult male reference.
VOWEL_FORMANTS: dict[str, tuple[float, float, float, float, float]] = {
    "i": (306, 2306, 3005, 3700, 4500),   # "ee" as in "see"
    "e": (476, 2089, 2691, 3600, 4500),   # "ay" as in "say", without the glide
    "eh": (555, 1820, 2543, 3600, 4500),  # "eh" as in "bed"
    "a": (749, 1212, 2481, 3600, 4500),   # "ah" as in "father"
    "aw": (611, 919, 2474, 3600, 4500),   # "aw" as in "law" (not in the default set)
    "o": (497, 910, 2459, 3600, 4500),    # "oh" as in "go", without the glide
    "u": (339, 934, 2292, 3600, 4500),    # "oo" as in "boot"
}
DEFAULT_VOWELS = ("i", "e", "eh", "a", "o", "u")
VOWEL_DESCRIPTIONS = {
    "i": "ee (see)", "e": "ay (say)", "eh": "eh (bed)", "a": "ah (father)",
    "aw": "aw (law)", "o": "oh (go)", "u": "oo (boot)",
}
PARTS = ("silence", "plosive", "sibilant", "vowel", "voiced", "breath", "transition", "other")

_BANDWIDTHS = np.array([80.0, 100.0, 150.0, 200.0, 250.0])
_SCALES = np.round(np.arange(0.80, 1.4001, 0.02), 4)
_TILT_DB_PER_OCTAVE = (-15.0, 3.0)  # source + radiation slope beyond the template
_SIBILANT_SPLIT_HZ = 2500.0
_S_VS_SH_HZ = 4800.0


@dataclass(frozen=True)
class VocalTrack:
    times_s: np.ndarray          # frame centres
    part: np.ndarray             # one of PARTS per frame
    vowel: np.ndarray            # vowel label on "vowel" frames, else ""
    vowel_confidence: np.ndarray  # (second-best - best fit error) / best, 0 if no vowel
    vowel_errors: np.ndarray     # [frames, vowels] template fit error in dB², NaN if not judged
    vowels: tuple[str, ...]      # column order of ``vowel_errors``
    sibilant_type: np.ndarray    # "s" / "sh" on sibilant frames, else ""
    sibilance: np.ndarray        # share of frame energy above 2.5 kHz
    f0_hz: np.ndarray            # NaN where not harmonic
    level_db: np.ndarray         # frame RMS in dBFS
    scale: float                 # vocal-tract scale used for the vowel templates
    hop_s: float

    def segments(self) -> list[dict]:
        """Consecutive frames with the same part (and vowel / sibilant type) merged."""
        detail = np.where(self.part == "vowel", self.vowel, np.where(self.part == "sibilant", self.sibilant_type, ""))
        out = []
        start = 0
        for i in range(1, self.part.size + 1):
            if i < self.part.size and self.part[i] == self.part[start] and detail[i] == detail[start]:
                continue
            out.append({"start_s": round(float(self.times_s[start] - self.hop_s / 2), 3),
                        "end_s": round(float(self.times_s[i - 1] + self.hop_s / 2), 3),
                        "part": str(self.part[start]), "detail": str(detail[start])})
            start = i
        return out

    def summarize(self, start_s: float, end_s: float, *, min_confidence: float = 0.0) -> dict:
        """Majority vowel of a time span with its first-third and last-third vowels
        (showing glides such as "i" → "u"), plus the parts present in the span."""
        span = (self.times_s >= start_s) & (self.times_s < end_s)
        parts, counts = np.unique(self.part[span], return_counts=True)
        result = {"parts": {str(p): int(c) for p, c in zip(parts, counts)},
                  "sibilant_frames": int((self.part[span] == "sibilant").sum()),
                  "max_sibilance": round(float(self.sibilance[span].max()), 3) if span.any() else 0.0}
        inside = span & (self.part == "vowel") & (self.vowel_confidence >= min_confidence)
        labels = self.vowel[inside]
        if labels.size == 0:
            return {**result, "vowel": "", "share": 0.0, "start_vowel": "", "end_vowel": "", "vowel_frames": 0}

        def majority(values):
            names, n = np.unique(values, return_counts=True)
            return str(names[np.argmax(n)]), float(n.max() / n.sum())

        vowel, share = majority(labels)
        third = max(labels.size // 3, 1)
        return {**result, "vowel": vowel, "share": round(share, 3), "start_vowel": majority(labels[:third])[0],
                "end_vowel": majority(labels[-third:])[0], "vowel_frames": int(labels.size)}


def formant_envelope_db(freqs_hz, formants_hz, bandwidths_hz=_BANDWIDTHS) -> np.ndarray:
    """Magnitude (dB) of a cascade of 2nd-order resonators, 0 dB at DC."""
    f = np.asarray(freqs_hz, dtype=np.float64)[..., None]
    formant = np.asarray(formants_hz, dtype=np.float64)
    bandwidth = np.asarray(bandwidths_hz, dtype=np.float64)[: formant.shape[-1]]
    ratio = f / formant
    response = 1.0 / np.abs(1.0 - ratio**2 + 1j * ratio * bandwidth / formant)
    return 20.0 * np.log10(response).sum(axis=-1)


def _peaks(magnitude, positions_bins):
    index = np.clip(np.round(positions_bins).astype(int), 1, magnitude.size - 2)
    return np.maximum(np.maximum(magnitude[index - 1], magnitude[index]), magnitude[index + 1])


def _frame_f0(magnitude, bin_hz, f0_min, f0_max, decay=0.85):
    """Subharmonic summation on a compressed spectrum, then a 0.1 % refinement."""
    compressed = np.sqrt(magnitude)
    candidates = f0_min * 2.0 ** (np.arange(int(96 * np.log2(f0_max / f0_min)) + 1) / 96)
    harmonics = np.arange(1, 16)
    positions = candidates[:, None] * harmonics[None, :]
    scores = (_peaks(compressed, positions / bin_hz) * decay ** (harmonics - 1) * (positions < 4000.0)).sum(1)
    best = candidates[int(np.argmax(scores))]
    fine = best * (1.0 + np.arange(-30, 31) * 0.001)
    count = max(int(4000.0 // best), 1)
    fine_scores = _peaks(compressed, fine[:, None] * np.arange(1, count + 1)[None, :] / bin_hz).sum(1)
    return float(fine[int(np.argmax(fine_scores))])


def _harmonicity(power, bin_hz, f0):
    freqs = np.arange(power.size) * bin_hz
    band = (freqs >= 60.0) & (freqs <= 4000.0)
    distance = np.abs(freqs / f0 - np.round(freqs / f0)) * f0
    near = band & (distance <= max(0.12 * f0, 2.5 * bin_hz)) & (np.round(freqs / f0) >= 1)
    return float(power[near].sum() / max(power[band].sum(), 1e-30))


def _fit_errors(freqs, levels, templates):
    """templates: [..., harmonics] dB. Returns the bounded-tilt fit error per template."""
    x = np.log2(freqs / 1000.0)
    residual = levels - templates
    xc = x - x.mean()
    slope = np.clip((residual * xc).sum(-1) / max(float((xc**2).sum()), 1e-12), *_TILT_DB_PER_OCTAVE)
    offset = (residual - slope[..., None] * x).mean(-1)
    return ((residual - offset[..., None] - slope[..., None] * x) ** 2).mean(-1)


def _noise_features(signal, times_s, rate=NOISE_RATE, window_s=0.02):
    """Per frame: level (dBFS), spectral flatness over 200 Hz-12 kHz, flatness
    of the occupied high band (the 2.5-12 kHz span between the first and last
    bins within 20 dB of its peak: sibilants are band-limited noise, flat
    where they have energy), share of energy above 2.5 kHz, and the centroid
    of that energy."""
    length = int(rate * window_s)
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
    band = (freqs >= 200.0) & (freqs <= 12000.0)
    high = (freqs >= _SIBILANT_SPLIT_HZ) & (freqs <= 12000.0)
    window = np.hanning(length)
    level, flatness, high_flatness, share, centroid = (np.zeros(times_s.size) for _ in range(5))
    for i, t in enumerate(times_s):
        lo = max(int(t * rate) - length // 2, 0)
        frame = signal[lo:lo + length]
        if frame.size < length:
            frame = np.pad(frame, (0, length - frame.size))
        level[i] = 20.0 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-12)
        power = np.abs(np.fft.rfft(frame * window, n_fft)) ** 2 + 1e-20
        p = power[band]
        flatness[i] = float(np.exp(np.mean(np.log(p))) / np.mean(p))
        h = power[high]
        # Flatness over the contiguous span that holds energy, valleys included:
        # band-limited noise stays flat, a few bright harmonics do not.
        held = np.flatnonzero(h >= h.max() * 0.01)
        span = h[held[0]:held[-1] + 1]
        high_flatness[i] = float(np.exp(np.mean(np.log(span))) / np.mean(span))
        share[i] = float(h.sum() / p.sum())
        centroid[i] = float((freqs[high] * h).sum() / h.sum())
    return level, flatness, high_flatness, share, centroid


def detect_vocal_parts(
    audio: np.ndarray,
    sample_rate: int,
    *,
    scale: float | None = None,
    vowels: tuple[str, ...] = DEFAULT_VOWELS,
    floor_db: float = 50.0,
    background_db: float = 30.0,
    f_max_hz: float = 4500.0,
    min_harmonics: int = 4,
    f0_min: float = 60.0,
    f0_max: float = 1100.0,
    frame_ms: float = 10.0,
    min_harmonicity: float = 0.5,
    sibilance_threshold: float = 0.5,
    plosive_rise_db: float = 15.0,
) -> VocalTrack:
    """Frame-wise parts of a monophonic voice (``[samples]`` or ``[channels, samples]``)."""
    names = tuple(vowels)
    formants = np.array([VOWEL_FORMANTS[v] for v in names], dtype=np.float64)
    mono = np.asarray(audio, dtype=np.float64)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    voice = resample(mono[None], int(sample_rate), VOWEL_RATE)[0] if sample_rate != VOWEL_RATE else mono
    noise = resample(mono[None], int(sample_rate), NOISE_RATE)[0] if sample_rate != NOISE_RATE else mono

    hop = int(VOWEL_RATE * frame_ms / 1000.0)
    length = int(VOWEL_RATE * 0.064)  # >= 3 periods at 60 Hz
    n_fft = 8192
    bin_hz = VOWEL_RATE / n_fft
    starts = np.arange(0, max(voice.size - length, 0) + 1, hop)
    times = (starts + length / 2) / VOWEL_RATE
    level, flatness, high_flatness, sibilance, centroid = _noise_features(noise, times)
    audible = level > max(level.max() - floor_db, -100.0)

    window = np.hanning(length)
    f0 = np.full(times.size, np.nan)
    measured = []  # (frame, harmonic freqs, levels)
    for index in np.flatnonzero(audible):
        spectrum = np.abs(np.fft.rfft(voice[starts[index]:starts[index] + length] * window, n_fft))
        estimate = _frame_f0(spectrum, bin_hz, f0_min, f0_max)
        if _harmonicity(spectrum**2, bin_hz, estimate) < min_harmonicity:
            continue
        f0[index] = estimate
        count = int(min(f_max_hz, 0.45 * VOWEL_RATE) // estimate)
        if count >= min_harmonics:
            harmonics = estimate * np.arange(1, count + 1)
            measured.append((index, harmonics, 20.0 * np.log10(_peaks(spectrum, harmonics / bin_hz) + 1e-12)))

    if scale is None:
        best_scales, best_errors = [], []
        for _, freqs, levels in measured:
            grid = formant_envelope_db(freqs[None, None, :], formants[:, None, None, :] * _SCALES[None, :, None, None])
            err = _fit_errors(freqs, levels, grid)  # [vowels, scales]
            v, s = np.unravel_index(np.argmin(err), err.shape)
            best_scales.append(_SCALES[s])
            best_errors.append(err[v, s])
        keep = np.asarray(best_errors) <= np.median(best_errors) if best_errors else None
        scale = float(np.median(np.asarray(best_scales)[keep])) if best_scales else 1.0

    vowel = np.full(times.size, "", dtype=object)
    confidence = np.zeros(times.size)
    errors = np.full((times.size, len(names)), np.nan)
    for index, freqs, levels in measured:
        err = _fit_errors(freqs, levels, formant_envelope_db(freqs[None, :], formants[:, None, :] * scale))
        order = np.argsort(err)
        errors[index] = err
        vowel[index] = names[order[0]]
        confidence[index] = (err[order[1]] - err[order[0]]) / max(err[order[0]], 1e-6) if len(names) > 1 else 1.0

    part = np.full(times.size, "other", dtype=object)
    sibilant_type = np.full(times.size, "", dtype=object)
    noisy = flatness >= 0.2
    for i in range(times.size):
        quiet_before = level[max(i - 3, 0):i]
        if not audible[i]:
            part[i] = "silence"
        elif (i > 0 and level[i] - quiet_before.min() >= plosive_rise_db
              and quiet_before.min() < level.max() - 35.0 and noisy[i] and np.isnan(f0[i])):
            part[i] = "plosive"
        # 0.2: sibilants measured 0.26-0.30 median on Shevannai sa/shu/tus/tze,
        # bright sung vowels (ni/le/leh) 0.12-0.15.
        elif sibilance[i] >= sibilance_threshold and high_flatness[i] >= 0.2:
            part[i] = "sibilant"
            sibilant_type[i] = "s" if centroid[i] >= _S_VS_SH_HZ else "sh"
        elif vowel[i]:
            part[i] = "vowel"
        elif np.isfinite(f0[i]):
            part[i] = "voiced"
        elif noisy[i]:
            part[i] = "breath"
    near_vowel = np.convolve(part == "vowel", np.ones(11), "same") > 0  # within ±50 ms
    background = level < level.max() - background_db
    part[(part == "other") & near_vowel & ~background] = "transition"
    part[(part == "other") & background] = "silence"
    vowel = np.where(part == "vowel", vowel, "")
    confidence = np.where(part == "vowel", confidence, 0.0)
    return VocalTrack(times, part.astype(str), vowel.astype(str), confidence, errors, names,
                      sibilant_type.astype(str), sibilance, f0, level, float(scale), frame_ms / 1000.0)
