"""Structure segmentation and loop-point detection (roadmap H-3, H-4).

- :func:`segment_structure` — sections/drops of a longer clip: cosine
  self-similarity of windowed Wavelet2Vec embeddings, Foote checkerboard
  novelty, peak picking.
- :func:`find_loop_points` — "find the perfect loop": periodicity of the
  band-envelope sequence picks the loop length, raw-audio correlation
  refines the seam to the sample, zero-crossing snap makes it click-safe.

All deterministic, CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet2vec.frame import analyze


# ---------------------------------------------------------------------------
# H-3 — structure segmentation on windowed embeddings
# ---------------------------------------------------------------------------
@dataclass
class StructureSegments:
    boundaries_s: list[float]
    sections: list[tuple[float, float]]  # (start_s, end_s)
    novelty: np.ndarray  # per analysis window, 0..1
    times_s: np.ndarray


def segment_structure(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    kernel_s: float = 4.0,
    threshold: float | None = None,
    embedder=None,
) -> StructureSegments:
    """Splits a clip into sections where its embedding sequence changes.

    Each ``window_s`` window is embedded with :class:`~wavelet2vec.embedder.
    Wavelet2Vec` (pass a shared ``embedder`` to reuse one instance); the
    cosine self-similarity matrix is convolved along its diagonal with a
    Gaussian-tapered checkerboard kernel (Foote novelty), and local maxima
    above the threshold (with ≥ half a kernel of separation) become section
    boundaries.

    ``threshold=None`` (default) adapts to the clip: median + 10·MAD of its
    own novelty curve, floored at 0.02. Musically similar sections (e.g.
    two loops of the same genre) produce small absolute novelty — what
    matters is standing out from the clip's own baseline. Pass an explicit
    float for an absolute cut instead.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    duration = audio.shape[0] / sample_rate

    window = int(window_s * sample_rate)
    hop = max(int(hop_s * sample_rate), 1)
    n_windows = max((audio.shape[0] - window) // hop + 1, 0)
    if n_windows < 4:
        return StructureSegments([], [(0.0, duration)], np.array([]), np.array([]))

    if embedder is None:
        from wavelet2vec.embedder import Wavelet2Vec

        embedder = Wavelet2Vec()
    embeddings = np.stack(
        [
            embedder.embed(audio[index * hop : index * hop + window], sample_rate)
            for index in range(n_windows)
        ]
    ).astype(np.float64)
    unit = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-24)
    similarity = unit @ unit.T

    half = max(int(round(kernel_s / hop_s / 2.0)), 2)
    taper = np.exp(-0.5 * (np.arange(-half, half) + 0.5) ** 2 / (half / 2.0) ** 2)
    checker = np.sign(np.arange(-half, half) + 0.5)
    kernel = np.outer(taper * checker, taper * checker)
    kernel /= np.abs(kernel).sum()

    # Kernel is positive on the two within-section quadrants and negative on
    # the cross quadrants; |kernel| sums to 1, so a boundary scores
    # (within − cross) / 2 and a homogeneous region scores 0. Embedding
    # cosines rarely drop below ~0.5 (norm-floored sections share mass), so
    # real boundaries land around 0.1–0.25 — hence the 0.1 default threshold.
    novelty = np.zeros(n_windows)
    for center in range(half, n_windows - half):
        block = similarity[center - half : center + half, center - half : center + half]
        novelty[center] = max((kernel * block).sum(), 0.0)

    if threshold is None:
        median = float(np.median(novelty))
        mad = float(np.median(np.abs(novelty - median)))
        threshold = max(0.02, median + 8.0 * mad)

    times = (np.arange(n_windows) * hop + window / 2.0) / sample_rate
    boundaries: list[float] = []
    min_gap = half
    for index in range(1, n_windows - 1):
        if novelty[index] < threshold:
            continue
        if novelty[index] < novelty[index - 1] or novelty[index] < novelty[index + 1]:
            continue
        time = float(times[index])
        if boundaries and time - boundaries[-1] < min_gap * hop_s:
            continue
        boundaries.append(time)

    edges = [0.0] + boundaries + [duration]
    sections = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return StructureSegments(boundaries, sections, novelty, times)


# ---------------------------------------------------------------------------
# H-4 — loop-point detection
# ---------------------------------------------------------------------------
@dataclass
class LoopPoints:
    start_sample: int
    end_sample: int
    loop_s: float
    periodicity: float  # centered envelope-sequence match at the chosen lag
    seam_error: float  # spectral continuation mismatch at the seam, 0 = seamless


def find_loop_points(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_loop_s: float = 0.5,
    max_loop_s: float | None = None,
    frame_ms: float = 10.0,
) -> LoopPoints:
    """Finds loop points ``[start_sample, end_sample)``.

    1. **Loop length**: the *centered* 48-band log-envelope sequence is
       compared with itself at every lag; the smallest lag scoring within
       2 % of the best wins, so the fundamental period beats its own
       multiples (centering matters: a steady synth spectrum makes raw
       frame cosines high at every lag).
    2. **Start**: the position whose following spectral context best
       matches the context one loop later.
    3. **Seam**: raw-audio correlation around the end refines the length to
       the sample; both points then snap to rising zero crossings.

    ``seam_error`` is spectral: 1 − the envelope-context match at the seam
    (0 = the music continues seamlessly). A *sample-exact* seam often does
    not exist even for musically perfect loops — free-running synth
    oscillators never repeat their waveform — so for playback use
    :func:`render_loop`, which blends the seam with a short equal-power
    crossfade.

    When the BPM is known, pass ``min_loop_s`` just under one bar
    (``0.95 * 240 / bpm`` for 4/4) — unconstrained, the detector honestly
    reports the material's fundamental spectral period, which for evolving
    synth loops is usually a couple of *beats*, not a full bar.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    n = audio.shape[0]

    coeffs, _spec = analyze(audio, sample_rate, n_bands=48)
    hop = max(int(sample_rate * frame_ms / 1000.0), 1)
    n_frames = coeffs.shape[1] // hop
    envelopes = np.log1p(
        np.abs(coeffs[:, : n_frames * hop]).reshape(48, n_frames, hop).mean(axis=2)
    )
    # Center each band over time: a steady synth spectrum makes raw frame
    # cosines ~0.9 at *every* lag, hiding the pattern; centering leaves only
    # the pattern, so wrong lags score near (or below) zero.
    envelopes = envelopes - envelopes.mean(axis=1, keepdims=True)
    unit = envelopes / (np.linalg.norm(envelopes, axis=0, keepdims=True) + 1e-24)

    frame_rate = sample_rate / hop
    min_lag = max(int(min_loop_s * frame_rate), 1)
    max_lag = n_frames // 2 if max_loop_s is None else min(int(max_loop_s * frame_rate), n_frames - 2)
    if max_lag <= min_lag:
        raise ValueError("Audio is too short for the requested loop length range.")

    scores = np.full(max_lag + 1, -1.0)
    for lag in range(min_lag, max_lag + 1):
        scores[lag] = float((unit[:, : n_frames - lag] * unit[:, lag:]).sum(axis=0).mean())
    best_score = scores.max()
    candidates = np.flatnonzero(scores >= best_score - 0.02 * abs(best_score))
    lag = int(candidates[0])  # smallest lag near the max: the fundamental period
    periodicity = float(np.clip(scores[lag], 0.0, 1.0))

    context = min(max(int(0.25 * frame_rate), 2), n_frames - lag - 1)
    start_scores = [
        float((unit[:, s : s + context] * unit[:, s + lag : s + lag + context]).sum(axis=0).mean())
        for s in range(0, n_frames - lag - context)
    ]
    start_frame = int(np.argmax(start_scores)) if start_scores else 0
    seam_error = float(np.clip(1.0 - start_scores[start_frame], 0.0, 1.0)) if start_scores else 1.0

    start = start_frame * hop
    end = start + lag * hop
    # Sample-exact refinement: slide the end so the raw audio after it best
    # matches the raw audio after the start.
    match = min(int(0.02 * sample_rate), n - end - hop - 1)
    if match > 8:
        reference = audio[start : start + match]
        best_offset, best_err = 0, np.inf
        for offset in range(-hop, hop + 1):
            seam = end + offset
            if seam + match > n or seam <= start:
                continue
            err = float(np.mean((audio[seam : seam + match] - reference) ** 2))
            if err < best_err:
                best_offset, best_err = offset, err
        end += best_offset

    def _snap_rising_zero(position: int) -> int:
        radius = int(0.005 * sample_rate)
        lo, hi = max(position - radius, 1), min(position + radius, n - 1)
        crossings = np.flatnonzero((audio[lo:hi] <= 0) & (audio[lo + 1 : hi + 1] > 0)) + lo + 1
        if crossings.size == 0:
            return position
        return int(crossings[np.argmin(np.abs(crossings - position))])

    start = _snap_rising_zero(start)
    end = _snap_rising_zero(end)

    return LoopPoints(
        start_sample=int(start),
        end_sample=int(end),
        loop_s=float((end - start) / sample_rate),
        periodicity=periodicity,
        seam_error=seam_error,
    )


def render_loop(
    audio: np.ndarray,
    sample_rate: int,
    loop: LoopPoints,
    *,
    repeats: int = 2,
    crossfade_ms: float = 30.0,
) -> np.ndarray:
    """Renders ``repeats`` seamless passes of a detected loop.

    At every seam the natural continuation (the audio right after the loop
    end) is equal-power crossfaded into the loop restart, so free-running
    oscillator phase — which no splice point can align — is blended instead
    of cut. Output length is exactly ``repeats * (end - start)`` samples.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mono_input = audio.ndim == 1
    channels = audio[np.newaxis, :] if mono_input else audio
    n = channels.shape[1]
    start, end = loop.start_sample, loop.end_sample
    if not (0 <= start < end <= n):
        raise ValueError("Loop points are outside the audio.")

    fade = min(int(crossfade_ms / 1000.0 * sample_rate), n - end, end - start)
    body = channels[:, start:end]
    passes = [body]
    if fade > 0 and repeats > 1:
        # Equal-power gains: the two sides carry independent oscillator
        # phase, so powers (not amplitudes) should sum to one.
        theta = np.linspace(0.0, np.pi / 2.0, fade)
        continuation = channels[:, end : end + fade]
        blended_head = continuation * np.cos(theta) + channels[:, start : start + fade] * np.sin(theta)
        repeat_body = np.concatenate([blended_head, channels[:, start + fade : end]], axis=1)
        passes.extend([repeat_body] * (repeats - 1))
    else:
        passes.extend([body] * (repeats - 1))
    out = np.concatenate(passes, axis=1)
    return out[0] if mono_input else out
