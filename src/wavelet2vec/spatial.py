"""Multichannel spatial analysis and design (roadmap M-2, M-3, M-4, M-5).

- :func:`stereo_phase_spectrum` — the editor view's data source: per-band
  inter-channel phase difference, coherence, and balance **over time**.
- :func:`spatial_matrix` — generalized N-channel analysis: the same three
  quantities per band for every channel pair, pooled over time.
- :func:`extract_spatial_character` / :func:`apply_spatial_character` —
  width-vs-frequency profile matching ("make it wide like this reference,
  per band").
- :func:`design_width` — phase-safe per-band width designer with a
  mono-compatibility guard.

All analysis quantities compare channels against each other, so common
gain/delay cancels: level- and shift-invariant by construction. All design
operations touch only the Side signal — the mono downmix (L+R)/2 is
preserved to reconstruction precision by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet2vec.frame import analyze, synthesize


@dataclass
class StereoPhaseSpectrum:
    """Time-resolved stereo image: arrays are ``[bands, frames]``."""

    centers_hz: np.ndarray
    times_s: np.ndarray
    ipd_rad: np.ndarray      # circular-mean inter-channel phase difference
    coherence: np.ndarray    # 0..1 — how phase-stable L/R are in this cell
    balance: np.ndarray      # -1 (left) .. +1 (right) energy balance
    energy: np.ndarray       # summed L+R band energy per cell (display weighting)


def _framed_cross(
    left_coeffs: np.ndarray, right_coeffs: np.ndarray, sample_rate: int, frame_ms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    hop = max(int(sample_rate * frame_ms / 1000.0), 1)
    n_frames = left_coeffs.shape[1] // hop
    shape = (left_coeffs.shape[0], n_frames, hop)
    left = left_coeffs[:, : n_frames * hop].reshape(shape)
    right = right_coeffs[:, : n_frames * hop].reshape(shape)
    cross = (left * np.conj(right)).sum(axis=2)
    energy_left = (np.abs(left) ** 2).sum(axis=2)
    energy_right = (np.abs(right) ** 2).sum(axis=2)
    return cross, energy_left, energy_right, hop


def stereo_phase_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 96,
    frame_ms: float = 10.0,
    f_min: float = 20.0,
    coherence_times: float = 1.5,
) -> StereoPhaseSpectrum:
    """Per-band, per-frame stereo phase image of a ``[2, N]`` signal.

    Coherence is integrated over ~``coherence_times`` band coherence times
    (∝ 1/bandwidth): a narrow low band needs a long window before "coherent"
    means anything — one frame of a 1 Hz-wide band is a single phasor and
    would always read 1.0. Narrow-band cells therefore trade time resolution
    for estimator validity; wide (high) bands stay fast.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] < 2:
        raise ValueError("stereo_phase_spectrum expects [2, N] audio.")
    coeffs, spec = analyze(audio[:2], sample_rate, n_bands=n_bands, f_min=f_min)
    cross, energy_left, energy_right, hop = _framed_cross(
        coeffs[0], coeffs[1], sample_rate, frame_ms
    )
    n_frames = cross.shape[1]
    frame_seconds = hop / sample_rate

    ratio = float(spec.centers[1] / spec.centers[0]) if n_bands > 1 else 2.0 ** (1 / 12)
    sigma_f = spec.centers * (ratio - 1.0)
    window_frames = np.clip(
        np.ceil(coherence_times / np.maximum(sigma_f, 1e-9) / frame_seconds).astype(int),
        1, n_frames,
    )

    def _rolling(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(values, axis=1)
        out = np.empty_like(values)
        for band in range(values.shape[0]):
            length = int(window_frames[band])
            padded = np.concatenate([np.zeros((1,), dtype=values.dtype), cumulative[band]])
            ends = np.arange(1, n_frames + 1)
            starts = np.maximum(ends - length, 0)
            out[band] = padded[ends] - padded[starts]
        return out

    cross_w = _rolling(cross)
    energy_left_w = _rolling(energy_left)
    energy_right_w = _rolling(energy_right)

    denominator = np.sqrt(energy_left_w * energy_right_w) + 1e-24
    coherence = np.abs(cross_w) / denominator
    ipd = np.angle(cross_w)
    balance = (energy_right_w - energy_left_w) / (energy_left_w + energy_right_w + 1e-24)
    times = (np.arange(n_frames) + 0.5) * frame_seconds
    return StereoPhaseSpectrum(
        centers_hz=spec.centers, times_s=times,
        ipd_rad=ipd, coherence=np.clip(coherence, 0.0, 1.0), balance=balance,
        energy=energy_left + energy_right,  # per-frame (not windowed): display weighting
    )


def spatial_matrix(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 48,
    f_min: float = 20.0,
) -> dict:
    """N-channel spatial analysis pooled over time.

    Returns ``{"centers_hz", "pairs": [(i, j), …], "coherence" [pairs, bands],
    "ipd_rad" [pairs, bands], "balance" [pairs, bands]}`` — the generalized
    spatial description (stereo is the 1-pair special case).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("spatial_matrix expects [channels, N] audio.")
    n_channels = audio.shape[0]
    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)

    energies = (np.abs(coeffs) ** 2).sum(axis=2)  # [channels, bands]
    pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    coherence = np.zeros((len(pairs), n_bands))
    ipd = np.zeros((len(pairs), n_bands))
    balance = np.zeros((len(pairs), n_bands))
    for index, (i, j) in enumerate(pairs):
        cross = (coeffs[i] * np.conj(coeffs[j])).sum(axis=1)
        denominator = np.sqrt(energies[i] * energies[j]) + 1e-24
        coherence[index] = np.clip(np.abs(cross) / denominator, 0.0, 1.0)
        ipd[index] = np.angle(cross)
        balance[index] = (energies[j] - energies[i]) / (energies[i] + energies[j] + 1e-24)
    return {
        "centers_hz": spec.centers,
        "pairs": pairs,
        "coherence": coherence,
        "ipd_rad": ipd,
        "balance": balance,
    }


# ---------------------------------------------------------------------------
# M-4 — spatial character: width-vs-frequency profile matching
# ---------------------------------------------------------------------------
@dataclass
class SpatialCharacter:
    """Width-vs-frequency fingerprint of a stereo signal.

    ``width`` is the per-band Side-energy fraction ``Es / (Em + Es)``:
    0 = mono, 0.5 = fully decorrelated L/R, 1 = pure antiphase.
    ``balance`` is the per-band L/R energy balance in −1..+1.
    """

    centers_hz: np.ndarray
    width: np.ndarray
    balance: np.ndarray


def extract_spatial_character(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_bands: int = 48,
    f_min: float = 20.0,
) -> SpatialCharacter:
    """Per-band width and balance profile of a ``[2, N]`` signal."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("extract_spatial_character expects [2, N] audio.")
    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)
    mid = (coeffs[0] + coeffs[1]) / 2.0
    side = (coeffs[0] - coeffs[1]) / 2.0
    energy_mid = (np.abs(mid) ** 2).sum(axis=1)
    energy_side = (np.abs(side) ** 2).sum(axis=1)
    energy_left = (np.abs(coeffs[0]) ** 2).sum(axis=1)
    energy_right = (np.abs(coeffs[1]) ** 2).sum(axis=1)
    width = energy_side / (energy_mid + energy_side + 1e-24)
    balance = (energy_right - energy_left) / (energy_left + energy_right + 1e-24)
    return SpatialCharacter(centers_hz=spec.centers, width=width, balance=balance)


def _interp_to_grid(values: np.ndarray, from_hz: np.ndarray, to_hz: np.ndarray) -> np.ndarray:
    if from_hz.shape == to_hz.shape and np.allclose(from_hz, to_hz):
        return values
    return np.interp(np.log(to_hz), np.log(from_hz), values)


def apply_spatial_character(
    source: np.ndarray,
    sample_rate: int,
    character: SpatialCharacter,
    amount: float = 1.0,
    *,
    match_balance: bool = False,
    max_gain_db: float = 20.0,
    n_bands: int = 48,
    f_min: float = 20.0,
) -> np.ndarray:
    """Pulls the source's width profile toward ``character`` by ``amount``.

    Matching works by scaling the Side signal per band, so the mono downmix
    (L+R)/2 is untouched (to reconstruction precision) and no phase is
    manipulated. Bands whose Side energy is (near) zero cannot be widened by
    scaling — creating width from nothing is :func:`design_width`'s job.
    ``match_balance=True`` additionally applies per-band L/R gain curves to
    match the reference balance.
    """
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != 2:
        raise ValueError("apply_spatial_character expects [2, N] audio.")
    coeffs, spec = analyze(source, sample_rate, n_bands=n_bands, f_min=f_min)
    mid = (coeffs[0] + coeffs[1]) / 2.0
    side = (coeffs[0] - coeffs[1]) / 2.0
    energy_mid = (np.abs(mid) ** 2).sum(axis=1)
    energy_side = (np.abs(side) ** 2).sum(axis=1)

    width_now = energy_side / (energy_mid + energy_side + 1e-24)
    width_ref = np.clip(
        _interp_to_grid(character.width, character.centers_hz, spec.centers), 0.0, 0.995
    )
    width_target = width_now + float(amount) * (width_ref - width_now)

    gain_cap = 10.0 ** (max_gain_db / 20.0)
    has_side = energy_side > energy_mid * 1e-12
    gain_squared = np.where(
        has_side,
        (width_target / np.maximum(1.0 - width_target, 1e-6))
        * (energy_mid / np.maximum(energy_side, 1e-24)),
        1.0,
    )
    side_gain = np.clip(np.sqrt(np.maximum(gain_squared, 0.0)), 1.0 / gain_cap, gain_cap)

    scaled_side = side * side_gain[:, np.newaxis]
    new_coeffs = np.stack([mid + scaled_side, mid - scaled_side])

    if match_balance:
        balance_ref = np.clip(
            _interp_to_grid(character.balance, character.centers_hz, spec.centers),
            -0.995, 0.995,
        )
        energy_left = (np.abs(new_coeffs[0]) ** 2).sum(axis=1)
        energy_right = (np.abs(new_coeffs[1]) ** 2).sum(axis=1)
        balance_now = (energy_right - energy_left) / (energy_left + energy_right + 1e-24)
        balance_target = np.clip(
            balance_now + float(amount) * (balance_ref - balance_now), -0.995, 0.995
        )
        gain_left = np.sqrt((1.0 - balance_target) / np.maximum(1.0 - balance_now, 1e-6))
        gain_right = np.sqrt((1.0 + balance_target) / np.maximum(1.0 + balance_now, 1e-6))
        gain_left = np.clip(gain_left, 1.0 / gain_cap, gain_cap)
        gain_right = np.clip(gain_right, 1.0 / gain_cap, gain_cap)
        new_coeffs[0] *= gain_left[:, np.newaxis]
        new_coeffs[1] *= gain_right[:, np.newaxis]

    return synthesize(new_coeffs, spec)


# ---------------------------------------------------------------------------
# M-5 — phase-safe width designer with mono-compatibility guard
# ---------------------------------------------------------------------------
def design_width(
    audio: np.ndarray,
    sample_rate: int,
    width: float | np.ndarray,
    *,
    coherence_floor: float = 0.2,
    delay_ms: float = 15.0,
    n_bands: int = 48,
    f_min: float = 20.0,
) -> np.ndarray:
    """Sets the per-band stereo width of mono or stereo audio.

    ``width`` is the target Side-energy fraction (scalar or per-band array):
    0 = mono, 0.5 = fully decorrelated. Narrowing scales the existing Side
    down (always phase-safe). Widening beyond the available Side adds a
    *delayed* copy of the Mid as antiphase Side — a real decorrelator, not a
    phase trick: a constant per-band phase rotation provably keeps L/R
    coherence at 1, while a delay ≫ the band's coherence time makes the
    added Side orthogonal, dropping coherence to ``1 − 2·width``. Physics
    follows bandwidth: high bands (short coherence time) decorrelate fully,
    narrow low bands stay coherent — bass stays tight, and pure tones can
    only take a phase image, not true width.

    ``coherence_floor`` is the mono-compatibility guard: the target width is
    capped at ``(1 − floor) / 2`` per band so L/R never sound phasier than
    the floor allows. The mono downmix itself is preserved exactly (to
    reconstruction precision) by antiphase construction. Output peaks may
    exceed the input's — widening adds energy to L and R individually.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("design_width expects mono [N] or stereo [2, N] audio.")

    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)
    mid = (coeffs[0] + coeffs[1]) / 2.0
    side = (coeffs[0] - coeffs[1]) / 2.0
    energy_mid = (np.abs(mid) ** 2).sum(axis=1)
    energy_side = (np.abs(side) ** 2).sum(axis=1)
    width_now = energy_side / (energy_mid + energy_side + 1e-24)

    width_max = (1.0 - float(np.clip(coherence_floor, 0.0, 1.0))) / 2.0
    width_target = np.clip(np.broadcast_to(np.asarray(width, dtype=np.float64), (n_bands,)),
                           0.0, width_max)

    # Narrowing: scale existing Side down toward the target.
    ratio_target = width_target / np.maximum(1.0 - width_target, 1e-6)
    narrow_gain = np.sqrt(
        np.minimum(ratio_target * energy_mid / np.maximum(energy_side, 1e-24), 1.0)
    )
    new_side = side * np.where(width_target < width_now, narrow_gain, 1.0)[:, np.newaxis]

    # Widening: add delayed Mid as decorrelated Side. One sign for every
    # band — adjacent constant-Q bands overlap heavily, so per-band sign
    # flips would cancel each other at synthesis; the delay itself spreads
    # IPD across frequency (2π·f·τ), so no static lateral lean builds up.
    deficit = np.maximum(ratio_target * energy_mid - energy_side, 0.0)
    alpha = np.where(
        width_target > width_now,
        np.sqrt(deficit / np.maximum(energy_mid, 1e-24)),
        0.0,
    )
    if np.any(alpha > 0):
        delay = min(max(int(round(delay_ms / 1000.0 * sample_rate)), 1), mid.shape[1] - 1)
        shifted = np.zeros_like(mid)
        shifted[:, delay:] = mid[:, : mid.shape[1] - delay]
        new_side = new_side + shifted * alpha[:, np.newaxis]

    new_coeffs = np.stack([mid + new_side, mid - new_side])
    return synthesize(new_coeffs, spec)
