"""Rhythm & groove analysis with known BPM (roadmap R-1..R-5).

Deterministic: onset curves per register (low/mid/high ≈ kick/snare/hats),
grid deviations, swing percentage, per-slot microtiming, a registers×16
pattern matrix, and a syncopation index. BPM is an input (the DAW knows it);
tempo estimation without BPM is roadmap R-6.

Timing honesty: absolute onset times carry a register-dependent constant
advance (zero-phase constant-Q wavelets spread energy symmetrically in time —
a 60 Hz band leads by up to ~10 ms). ``deviations_ms`` is therefore reported
raw, while ``microtiming_*`` and ``swing`` are relative measures where the
constant cancels (per-register median-centering / intra-beat fractions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelet2vec.frame import analyze

REGISTERS = ("low", "mid", "high")
REGISTER_EDGES_HZ = {"low": (20.0, 150.0), "mid": (150.0, 1500.0), "high": (1500.0, 20000.0)}
SLOTS_PER_BAR = 16  # 16th-note grid, 4/4
# Metrical weight per 16th slot (higher = stronger position).
METRICAL_WEIGHTS = np.array([4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0], dtype=float)

_FRAME_MS = 5.0
_MIN_ONSET_GAP_MS = 50.0


@dataclass
class RhythmProfile:
    bpm: float
    sample_rate: int
    frame_rate: float
    onset_curves: dict[str, np.ndarray] = field(default_factory=dict)
    onsets: dict[str, list[tuple[float, float]]] = field(default_factory=dict)  # (time_s, strength)
    deviations_ms: dict[str, np.ndarray] = field(default_factory=dict)
    microtiming_mean_ms: np.ndarray = field(default_factory=lambda: np.zeros(SLOTS_PER_BAR))
    microtiming_std_ms: np.ndarray = field(default_factory=lambda: np.zeros(SLOTS_PER_BAR))
    swing_percent: dict[str, float | None] = field(default_factory=dict)
    pattern: np.ndarray = field(default_factory=lambda: np.zeros((len(REGISTERS), SLOTS_PER_BAR)))
    syncopation: dict[str, float] = field(default_factory=dict)

    @property
    def swing_overall(self) -> float | None:
        values = [value for value in self.swing_percent.values() if value is not None]
        return float(np.mean(values)) if values else None


def _register_slices(centers: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero((centers >= lo) & (centers < hi))
        for name, (lo, hi) in REGISTER_EDGES_HZ.items()
    }


def _pick_onsets(curve: np.ndarray, frame_rate: float) -> list[tuple[int, float]]:
    """Local maxima above an adaptive threshold, ≥ 50 ms apart (index 0 allowed)."""
    if curve.size < 3 or curve.max() <= 0:
        return []
    threshold = curve.mean() + 1.0 * curve.std()
    min_gap = max(1, int(_MIN_ONSET_GAP_MS / 1000.0 * frame_rate))
    onsets: list[tuple[int, float]] = []
    for index in range(curve.shape[0] - 1):
        rising_ok = index == 0 or curve[index] >= curve[index - 1]
        if curve[index] >= threshold and rising_ok and curve[index] > curve[index + 1]:
            if not onsets or index - onsets[-1][0] >= min_gap:
                onsets.append((index, float(curve[index])))
    return onsets


def analyze_rhythm(audio: np.ndarray, sample_rate: int, bpm: float) -> RhythmProfile:
    """Full rhythm profile of a fragment whose tempo is ``bpm`` (4/4 grid)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio / peak

    coeffs, spec = analyze(audio, sample_rate, n_bands=48)
    magnitude = np.abs(coeffs)
    hop = max(int(sample_rate * _FRAME_MS / 1000.0), 1)
    n_frames = magnitude.shape[1] // hop
    envelopes = magnitude[:, : n_frames * hop].reshape(48, n_frames, hop).mean(axis=2)
    frame_rate = sample_rate / hop

    beat_s = 60.0 / bpm
    slot_s = beat_s / 4.0
    profile = RhythmProfile(bpm=float(bpm), sample_rate=int(sample_rate), frame_rate=frame_rate)

    slot_hits: dict[int, list[float]] = {slot: [] for slot in range(SLOTS_PER_BAR)}
    register_bands = _register_slices(spec.centers)

    for name in REGISTERS:
        bands = register_bands[name]
        if bands.size == 0:
            profile.onset_curves[name] = np.zeros(n_frames)
            profile.onsets[name] = []
            profile.deviations_ms[name] = np.array([])
            profile.swing_percent[name] = None
            profile.syncopation[name] = 0.0
            continue
        register_env = np.log1p(envelopes[bands])
        flux = np.clip(
            np.diff(register_env, axis=1, prepend=np.zeros((bands.size, 1))), 0.0, None
        ).sum(axis=0)
        profile.onset_curves[name] = flux

        picked = _pick_onsets(flux, frame_rate)
        times = [(index / frame_rate, strength) for index, strength in picked]
        profile.onsets[name] = times

        deviations = []
        swing_fractions = []
        strengths_per_slot = np.zeros(SLOTS_PER_BAR)
        slots_of_onsets = []
        for time_s, strength in times:
            nearest_slot = round(time_s / slot_s)
            deviations.append((time_s - nearest_slot * slot_s) * 1000.0)
            slot_index = int(nearest_slot) % SLOTS_PER_BAR
            slots_of_onsets.append(slot_index)
            strengths_per_slot[slot_index] += strength
            # Swing: onsets in the off-8th zone of a beat (fraction 0.35..0.85).
            fraction = (time_s % beat_s) / beat_s
            if 0.35 <= fraction <= 0.85:
                swing_fractions.append(fraction)
        deviations = np.asarray(deviations)
        profile.deviations_ms[name] = deviations
        # Microtiming is *relative* groove: remove the register's constant
        # analysis latency (zero-phase wavelets advance low-band onsets by up
        # to ~half the band's time spread) by median-centering per register.
        if deviations.size:
            centered = deviations - float(np.median(deviations))
            for slot_index, dev in zip(slots_of_onsets, centered):
                slot_hits[slot_index].append(float(dev))
        profile.swing_percent[name] = (
            float(np.mean(swing_fractions) * 100.0) if len(swing_fractions) >= 2 else None
        )

        total = strengths_per_slot.sum()
        if total > 0:
            weights_gap = METRICAL_WEIGHTS.max() - METRICAL_WEIGHTS
            profile.syncopation[name] = float(
                (strengths_per_slot * weights_gap).sum() / (total * METRICAL_WEIGHTS.max())
            )
            register_index = REGISTERS.index(name)
            profile.pattern[register_index] = strengths_per_slot / strengths_per_slot.max()
        else:
            profile.syncopation[name] = 0.0

    for slot in range(SLOTS_PER_BAR):
        hits = slot_hits[slot]
        if hits:
            profile.microtiming_mean_ms[slot] = float(np.mean(hits))
            profile.microtiming_std_ms[slot] = float(np.std(hits))
    return profile


# ---------------------------------------------------------------------------
# R-6 — tempo estimation when BPM is not given
# ---------------------------------------------------------------------------
# Autocorrelation comb: (multiple of the beat lag, weight). The bar term
# (4 beats) dominates because loops repeat at bar scale; the 3-beat term
# discriminates 4:3 metrical aliases (a 16th-note grid makes 4/3·BPM score
# well on 1- and 2-beat terms alone).
_TEMPO_COMB = ((1, 0.5), (2, 0.3), (3, 0.3), (4, 1.0), (8, 0.5), (16, 0.3))


def estimate_tempo(
    audio: np.ndarray,
    sample_rate: int,
    *,
    bpm_range: tuple[float, float] = (60.0, 180.0),
) -> tuple[float, float]:
    """Estimates (bpm, confidence 0..1) from the onset novelty curve.

    Autocorrelation of broadband onset flux, scored against a bar-aware comb
    (1, 2, 3, 4, 8 and 16 beats — the 4-beat *bar* term carries the largest
    weight because loops repeat at bar scale, and the 3-beat term vetoes 4:3
    metrical aliases such as reading 100 BPM material as 133), with a mild
    log-normal prior centered on 120 BPM (documented bias: between a 60 and
    a 120 reading of the same loop, the faster common interpretation wins).
    The autocorrelation uses capped-unbiased normalization and lags up to 3/4
    of the analysis window, so long-lag terms stay statistically stable.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio / peak

    coeffs, _spec = analyze(audio, sample_rate, n_bands=48)
    magnitude = np.abs(coeffs)
    hop = max(int(sample_rate * _FRAME_MS / 1000.0), 1)
    n_frames = magnitude.shape[1] // hop
    if n_frames < 8:
        return 120.0, 0.0
    envelopes = magnitude[:, : n_frames * hop].reshape(48, n_frames, hop).mean(axis=2)
    flux = np.clip(
        np.diff(np.log1p(envelopes), axis=1, prepend=np.zeros((48, 1))), 0.0, None
    ).sum(axis=0)
    flux = flux - flux.mean()

    autocorr = np.correlate(flux, flux, mode="full")[flux.size - 1 :]
    if autocorr[0] <= 0:
        return 120.0, 0.0
    # Unbiased normalization compensates the shrinking overlap at long lags;
    # capping the divisor at N/4 keeps near-window lags from blowing up.
    counts = np.maximum(flux.size - np.arange(flux.size), flux.size // 4)
    autocorr = autocorr / counts
    autocorr = autocorr / autocorr[0]
    max_lag = int(0.75 * flux.size)

    frame_rate = sample_rate / hop
    best_bpm, best_score, best_weight = 120.0, -np.inf, 1.0
    candidates = np.arange(bpm_range[0], bpm_range[1] + 0.25, 0.25)
    for bpm in candidates:
        beat_lag = 60.0 / bpm * frame_rate
        if beat_lag < 2:
            continue
        score, weight_total = 0.0, 0.0
        for multiple, weight in _TEMPO_COMB:
            index = int(round(multiple * beat_lag))
            if index <= max_lag:
                score += weight * autocorr[index]
                weight_total += weight
        if weight_total < 1.0:  # not even beat + bar fit the window
            continue
        prior = float(np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.6) ** 2))
        score *= prior
        if score > best_score:
            best_bpm, best_score, best_weight = float(bpm), score, weight_total
    confidence = float(np.clip(best_score / best_weight, 0.0, 1.0))
    return best_bpm, confidence


# ---------------------------------------------------------------------------
# R-8 — groove transfer: re-time a loop's 16th slices by per-slot offsets
# ---------------------------------------------------------------------------
def swing_offsets(swing_fraction: float, bpm: float) -> np.ndarray:
    """Per-slot offsets (ms) that move off-8ths to ``swing_fraction`` of the
    beat (0.5 = straight, 2/3 = full triplet swing)."""
    offsets = np.zeros(SLOTS_PER_BAR)
    beat_ms = 60_000.0 / bpm
    offsets[[2, 6, 10, 14]] = (swing_fraction - 0.5) * beat_ms
    return offsets


def groove_offsets(profile: RhythmProfile) -> np.ndarray:
    """Transferable per-slot offsets of an analyzed loop.

    Groove = coarse swing + fine microtiming. Microtiming on a 16th grid can
    only represent shifts within ±half a slot; heavy swing (e.g. full
    triplet, +1/6 beat) aliases into the neighboring slot. So the off-8th
    slots are driven by the measured swing fraction, and — when the swing is
    strong — the aliased neighbor slots are cleared; everything else keeps
    its measured microtiming.
    """
    offsets = profile.microtiming_mean_ms.copy()
    swing = profile.swing_overall
    if swing is not None:
        beat_ms = 60_000.0 / profile.bpm
        offsets[[2, 6, 10, 14]] = (swing / 100.0 - 0.5) * beat_ms
        if abs(swing - 50.0) > 8.0:  # heavy swing: neighbor slots hold aliases
            offsets[[3, 7, 11, 15]] = 0.0
    return offsets


def apply_groove(
    audio: np.ndarray,
    sample_rate: int,
    bpm: float,
    offsets_ms: np.ndarray,
    *,
    strength: float = 1.0,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Re-times a loop by shifting each 16th-note slice by its slot's offset.

    Slices are placed at their shifted positions with short crossfade ramps
    at the seams; total length is preserved. ``strength`` scales the
    offsets; 0 returns the input bit-exactly.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]

    scaled = np.asarray(offsets_ms, dtype=np.float64) * float(strength)
    if np.max(np.abs(scaled)) < 1e-9:
        out = channels.copy()
        return out[0] if mono_input else out

    from wavelet2vec.slicing import slice_grid

    starts = slice_grid(n, sample_rate, bpm=bpm, division=16)
    bounds = np.concatenate([starts, [n]])
    fade = max(1, int(fade_ms / 1000.0 * sample_rate))
    out = np.zeros_like(channels)

    for index in range(len(starts)):
        begin, end = int(bounds[index]), int(bounds[index + 1])
        if end <= begin:
            continue
        slot = index % SLOTS_PER_BAR
        shift = int(round(scaled[slot] / 1000.0 * sample_rate))
        segment = channels[:, begin:end].copy()
        length = segment.shape[1]
        ramp = min(fade, length // 2)
        if ramp > 1:
            window = np.ones(length)
            window[:ramp] = np.linspace(0.0, 1.0, ramp)
            window[-ramp:] = np.linspace(1.0, 0.0, ramp)
            segment = segment * window
        target = begin + shift
        lo = max(0, target)
        hi = min(n, target + length)
        if hi > lo:
            out[:, lo:hi] += segment[:, lo - target : hi - target]
    return out[0] if mono_input else out


# ---------------------------------------------------------------------------
# R-7 — tempo-normalized rhythm embedding for groove search
# ---------------------------------------------------------------------------
# Section layout of the rhythm embedding. Every quantity is expressed
# relative to the grid (slot shares, beat fractions), so the same groove at
# a different tempo lands on (nearly) the same vector.
RHYTHM_SECTIONS: dict[str, int] = {
    "pattern": len(REGISTERS) * SLOTS_PER_BAR,  # 48: registers × 16 slots
    "swing": len(REGISTERS),  #  3: (swing% − 50) / 50 per register
    "syncopation": len(REGISTERS),  #  3: syncopation index per register
    "microtiming": 2 * SLOTS_PER_BAR,  # 32: per-slot mean, std in slot units
    "balance": len(REGISTERS),  #  3: onset-strength share per register
}
RHYTHM_DIM = sum(RHYTHM_SECTIONS.values())

_SECTION_NORM_FLOOR = 0.1  # same convention as Wavelet2Vec.embed()


def rhythm_embedding(
    audio: np.ndarray,
    sample_rate: int,
    *,
    bpm: float | None = None,
) -> np.ndarray:
    """Tempo-normalized groove vector (``RHYTHM_DIM`` dims) for similarity
    search. Compare with :func:`wavelet2vec.cosine_similarity`.

    All sections are grid-relative — pattern slots, swing as beat fraction,
    microtiming in slot units — so the same groove rendered at 100 and at
    140 BPM embeds nearly identically. When ``bpm`` is omitted it is taken
    from :func:`estimate_tempo` (pass the DAW's BPM whenever you have it).
    Each section is L2-normalized (floor 0.1) so patterns don't drown the
    scalar sections.
    """
    if bpm is None:
        bpm, _confidence = estimate_tempo(audio, sample_rate)
    profile = analyze_rhythm(audio, sample_rate, bpm=bpm)
    return profile_embedding(profile)


def profile_embedding(profile: RhythmProfile) -> np.ndarray:
    """The rhythm embedding of an already-computed :class:`RhythmProfile`."""
    slot_ms = 60_000.0 / profile.bpm / 4.0

    sections: dict[str, np.ndarray] = {}
    sections["pattern"] = profile.pattern.ravel().astype(np.float64)
    swing_values = []
    for name in REGISTERS:
        swing = profile.swing_percent.get(name)
        swing_values.append(0.0 if swing is None else (swing - 50.0) / 50.0)
    sections["swing"] = np.array(swing_values)
    sections["syncopation"] = np.array(
        [profile.syncopation.get(name, 0.0) for name in REGISTERS]
    )
    sections["microtiming"] = np.concatenate(
        [profile.microtiming_mean_ms / slot_ms, profile.microtiming_std_ms / slot_ms]
    )
    strengths = np.array(
        [sum(s for _t, s in profile.onsets.get(name, [])) for name in REGISTERS]
    )
    total = strengths.sum()
    sections["balance"] = strengths / total if total > 0 else strengths

    parts = []
    for name, dim in RHYTHM_SECTIONS.items():
        vector = sections[name]
        assert vector.size == dim, f"{name}: {vector.size} != {dim}"
        norm = float(np.linalg.norm(vector))
        parts.append(vector / max(norm, _SECTION_NORM_FLOOR))
    return np.concatenate(parts)
