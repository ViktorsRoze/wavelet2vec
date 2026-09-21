"""Harmony & melody analysis (roadmap H-1, H-2).

- :func:`track_f0` — frame-wise fundamental-frequency track: YIN-style
  cumulative-mean-normalized difference per frame, smoothed by Viterbi over
  a semitone state grid with an explicit unvoiced state.
- :func:`segment_notes` — deterministic note events from an f0 track.
- :func:`chromagram` / :func:`estimate_key` / :func:`track_chords` —
  pitch-class analysis on semitone-aligned constant-Q bands, Krumhansl-
  Schmuckler key profiles, and template chords with Viterbi smoothing.

All deterministic, CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet2vec.audio_io import resample
from wavelet2vec.frame import analyze

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Kessler key profiles (probe-tone ratings).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_CHROMA_F_MIN = 55.0  # A1: band k has pitch class (9 + k) % 12


# ---------------------------------------------------------------------------
# H-1 — f0 tracking (YIN + Viterbi over semitone states)
# ---------------------------------------------------------------------------
@dataclass
class F0Track:
    times_s: np.ndarray
    f0_hz: np.ndarray  # NaN where unvoiced
    voicing: np.ndarray  # 0..1 periodicity strength per frame
    frame_rate: float


@dataclass(frozen=True)
class NoteEvent:
    start_s: float
    end_s: float
    f0_hz: float  # median over the event
    midi: int
    confidence: float  # mean voicing over the event


def _viterbi(costs: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Minimum-cost state path. ``costs``: [states, frames]; ``transition``:
    [from, to] additive costs."""
    n_states, n_frames = costs.shape
    total = costs[:, 0].copy()
    back = np.zeros((n_states, n_frames), dtype=np.int32)
    for frame in range(1, n_frames):
        stepped = total[:, None] + transition
        best_from = np.argmin(stepped, axis=0)
        total = stepped[best_from, np.arange(n_states)] + costs[:, frame]
        back[:, frame] = best_from
    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmin(total))
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = back[path[frame], frame]
    return path


def track_f0(
    audio: np.ndarray,
    sample_rate: int,
    *,
    f_min: float = 55.0,
    f_max: float = 1000.0,
    frame_ms: float = 10.0,
    voicing_threshold: float = 0.35,
) -> F0Track:
    """Frame-wise f0 with Viterbi smoothing and an explicit unvoiced state.

    Per frame the YIN cumulative-mean-normalized difference (CMND) is
    evaluated at the lag of every semitone between ``f_min`` and ``f_max``;
    the Viterbi path over these states (plus "unvoiced", whose constant cost
    is ``voicing_threshold``) trades frame evidence against pitch jumps.
    Voiced frames are refined by parabolic interpolation of the CMND dip.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio / peak

    hop = max(int(sample_rate * frame_ms / 1000.0), 1)
    lag_max = int(sample_rate / f_min)
    window = lag_max
    segment = window + lag_max
    if audio.shape[0] < segment:
        empty = np.array([])
        return F0Track(empty, empty, empty, sample_rate / hop)

    n_frames = (audio.shape[0] - segment) // hop + 1
    starts = np.arange(n_frames) * hop
    frames = audio[starts[:, None] + np.arange(segment)]

    # d(tau) = r(0) + E(tau) - 2*corr(tau), batched over frames via FFT.
    n_fft = 1 << int(np.ceil(np.log2(segment + lag_max)))
    head = frames[:, :window]
    spectrum_full = np.fft.rfft(frames, n_fft)
    spectrum_head = np.fft.rfft(head, n_fft)
    correlation = np.fft.irfft(np.conj(spectrum_head) * spectrum_full, n_fft)[:, : lag_max + 1]
    squares = np.concatenate(
        [np.zeros((n_frames, 1)), np.cumsum(frames**2, axis=1)], axis=1
    )
    energy = squares[:, window : window + lag_max + 1] - squares[:, : lag_max + 1]
    difference = np.maximum(energy[:, [0]] + energy - 2.0 * correlation, 0.0)
    running_mean = np.cumsum(difference[:, 1:], axis=1) / np.arange(1, lag_max + 1)
    cmnd = np.ones_like(difference)
    cmnd[:, 1:] = difference[:, 1:] / np.maximum(running_mean, 1e-12)

    n_semitones = int(np.floor(12.0 * np.log2(f_max / f_min))) + 1
    state_freqs = f_min * 2.0 ** (np.arange(n_semitones) / 12.0)
    state_lags = sample_rate / state_freqs
    low = np.floor(state_lags).astype(int)
    frac = state_lags - low
    emissions = np.empty((n_semitones + 1, n_frames))
    emissions[:-1] = (
        cmnd[:, low] * (1.0 - frac) + cmnd[:, np.minimum(low + 1, lag_max)] * frac
    ).T
    emissions[-1] = voicing_threshold  # unvoiced state

    transition = np.zeros((n_semitones + 1, n_semitones + 1))
    indices = np.arange(n_semitones)
    transition[:-1, :-1] = 0.02 * np.abs(indices[:, None] - indices[None, :])
    transition[-1, :-1] = transition[:-1, -1] = 0.1  # voiced <-> unvoiced switch

    path = _viterbi(emissions, transition)

    f0 = np.full(n_frames, np.nan)
    voicing = np.zeros(n_frames)
    voiced = path < n_semitones
    for frame in np.flatnonzero(voiced):
        lag = int(round(state_lags[path[frame]]))
        lag = min(max(lag, 1), lag_max - 1)
        left, center, right = cmnd[frame, lag - 1 : lag + 2]
        denominator = left - 2.0 * center + right
        offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        refined = lag + float(np.clip(offset, -1.0, 1.0))
        f0[frame] = sample_rate / refined
        voicing[frame] = float(np.clip(1.0 - center, 0.0, 1.0))

    times = (starts + window / 2.0) / sample_rate
    return F0Track(times, f0, voicing, sample_rate / hop)


def segment_notes(
    track: F0Track,
    *,
    min_duration_s: float = 0.05,
) -> list[NoteEvent]:
    """Groups consecutive voiced frames of one MIDI pitch into note events."""
    if track.times_s.size == 0:
        return []
    midi = np.full(track.f0_hz.shape, -1, dtype=int)
    voiced = np.isfinite(track.f0_hz)
    midi[voiced] = np.round(69.0 + 12.0 * np.log2(track.f0_hz[voiced] / 440.0)).astype(int)

    events: list[NoteEvent] = []
    frame_s = 1.0 / track.frame_rate
    run_start = 0
    for index in range(1, midi.size + 1):
        if index < midi.size and midi[index] == midi[run_start]:
            continue
        if midi[run_start] >= 0:
            start = float(track.times_s[run_start])
            end = float(track.times_s[index - 1] + frame_s)
            if end - start >= min_duration_s:
                span = slice(run_start, index)
                events.append(
                    NoteEvent(
                        start_s=start,
                        end_s=end,
                        f0_hz=float(np.nanmedian(track.f0_hz[span])),
                        midi=int(midi[run_start]),
                        confidence=float(track.voicing[span].mean()),
                    )
                )
        run_start = index
    return events


# ---------------------------------------------------------------------------
# H-2 — chroma, key, chords
# ---------------------------------------------------------------------------
_CHROMA_SR = 16000


def chromagram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_s: float = 0.25,
    octaves: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """``([12, frames] pitch-class energies, frame times)``.

    Bands of the constant-Q frame are placed exactly on semitones (A1 up,
    ``octaves`` octaves), so folding band k onto pitch class (9 + k) mod 12
    is exact — no bin-to-note rounding.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sample_rate != _CHROMA_SR:
        audio = resample(audio[None, :], sample_rate, _CHROMA_SR)[0]
        sample_rate = _CHROMA_SR

    n_bands = 12 * octaves + 1
    f_max = _CHROMA_F_MIN * 2.0**octaves
    coeffs, _spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=_CHROMA_F_MIN, f_max=f_max)
    hop = max(int(sample_rate * frame_s), 1)
    n_frames = max(coeffs.shape[1] // hop, 1)
    clipped = np.abs(coeffs[:, : n_frames * hop]) ** 2
    band_energy = clipped.reshape(n_bands, n_frames, -1).mean(axis=2)

    chroma = np.zeros((12, n_frames))
    for band in range(n_bands):
        chroma[(9 + band) % 12] += band_energy[band]
    times = (np.arange(n_frames) + 0.5) * hop / sample_rate
    return chroma, times


@dataclass(frozen=True)
class KeyEstimate:
    pitch_class: int
    is_minor: bool
    confidence: float  # profile correlation of the winner, 0..1
    margin: float  # winner minus runner-up correlation

    @property
    def name(self) -> str:
        suffix = " minor" if self.is_minor else " major"
        return PITCH_CLASS_NAMES[self.pitch_class] + suffix


def estimate_key(audio: np.ndarray, sample_rate: int) -> KeyEstimate:
    """Global key via Krumhansl-Schmuckler profile correlation on the mean
    chroma (log-compressed so pads and leads weigh comparably)."""
    chroma, _times = chromagram(audio, sample_rate)
    profile = np.log1p(chroma / (chroma.mean() + 1e-24)).mean(axis=1)

    scores = np.empty(24)
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        scores[tonic] = np.corrcoef(rotated, _KS_MAJOR)[0, 1]
        scores[12 + tonic] = np.corrcoef(rotated, _KS_MINOR)[0, 1]
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    return KeyEstimate(
        pitch_class=best % 12,
        is_minor=best >= 12,
        confidence=float(np.clip(scores[best], 0.0, 1.0)),
        margin=float(scores[order[0]] - scores[order[1]]),
    )


def track_chords(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_s: float = 0.5,
    switch_penalty: float = 0.15,
) -> list[tuple[float, float, str]]:
    """Major/minor triad segments ``(start_s, end_s, label)`` via template
    matching + Viterbi smoothing (``switch_penalty`` per chord change)."""
    chroma, times = chromagram(audio, sample_rate, frame_s=frame_s)
    norms = np.linalg.norm(chroma, axis=0) + 1e-24
    unit = chroma / norms

    templates = np.zeros((24, 12))
    for tonic in range(12):
        templates[tonic, [tonic, (tonic + 4) % 12, (tonic + 7) % 12]] = 1.0  # major
        templates[12 + tonic, [tonic, (tonic + 3) % 12, (tonic + 7) % 12]] = 1.0  # minor
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    similarity = templates @ unit  # [24, frames]
    transition = np.full((24, 24), switch_penalty)
    np.fill_diagonal(transition, 0.0)
    path = _viterbi(1.0 - similarity, transition)

    frame_step = float(times[1] - times[0]) if times.size > 1 else frame_s
    segments: list[tuple[float, float, str]] = []
    run_start = 0
    for index in range(1, path.size + 1):
        if index < path.size and path[index] == path[run_start]:
            continue
        state = int(path[run_start])
        label = PITCH_CLASS_NAMES[state % 12] + ("m" if state >= 12 else "")
        segments.append(
            (float(times[run_start] - frame_step / 2.0),
             float(times[index - 1] + frame_step / 2.0),
             label)
        )
        run_start = index
    return segments
