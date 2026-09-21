"""Learned transient axis (roadmap G-8): imagine an attack that isn't there.

Pad gets a pluck. The family lesson (G-9) says learn only what cannot be
measured at inference — a *missing* attack qualifies: there is nothing to
measure, only to imagine.

The G-2 pattern, applied to time instead of frequency:

- **Deterministic carrier**: each band's attack content is the band's own
  *body* (a segment of its analytic signal from the sustained region,
  copied to the onset) — so the imagined attack inherits the sound's
  timbre, and phase stays coherent with what follows.
- **Learned energy**: :class:`TransientNet` predicts per-band attack
  parameters — gain over the body level and exponential decay time —
  conditioned on a sharpness target in [0, 1]. The deterministic fallback
  (:func:`default_attack_params`) is a generic pluck: HF-tilted gains,
  decays shrinking with frequency; the library never requires weights.
- **Free supervision**: suppress the attack of sounds that *have* one and
  fit (gain, decay) per band on the original's attack window — measured
  labels, no annotation (``experiments/train_transient.py``).

``imagine_transient(audio, sr, amount)`` in :mod:`wavelet2vec.imagine`
is the user surface; ``amount=0`` is bit-exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    HAVE_TORCH = False

from wavelet2vec.filterbank import analytic_envelope

MODEL_BANDS = 64
MODEL_FRAMES = 128
MODEL_F_MIN = 30.0
ATTACK_S = 0.12  # imagined-attack window
BODY_START_S = 0.06  # body carrier begins here after the onset
BODY_S = 0.2
LOG_GAIN_LO, LOG_GAIN_HI = float(np.log(0.2)), float(np.log(30.0))
LOG_TAU_LO, LOG_TAU_HI = float(np.log(0.005)), float(np.log(0.2))


def onset_index(mono: np.ndarray, sample_rate: int, threshold: float = 0.2) -> int:
    """First index where the smoothed envelope crosses ``threshold``·peak —
    a one-shot's attack instant (v1 targets one-shots)."""
    envelope = analytic_envelope(
        np.asarray(mono, dtype=np.float64),
        smooth_samples=max(int(sample_rate * 0.005), 1))
    peak = envelope.max()
    if peak <= 1e-12:
        return 0
    crossings = np.flatnonzero(envelope >= threshold * peak)
    return int(crossings[0]) if crossings.size else 0


def band_envelopes(coeffs: np.ndarray, sample_rate: int) -> np.ndarray:
    """|coeffs| smoothed by a 5 ms box — per-band amplitude envelopes."""
    magnitude = np.abs(coeffs)
    width = max(int(sample_rate * 0.005), 1)
    kernel = np.ones(width) / width
    return np.stack([np.convolve(row, kernel, mode="same") for row in magnitude])


def sharpness_label(mono: np.ndarray, sample_rate: int) -> float:
    """Attack-to-body energy ratio, squashed to [0, 1] — the conditioning
    axis. ~0 = swell/pad, ~0.5 = balanced hit, →1 = click-dominated."""
    onset = onset_index(mono, sample_rate)
    attack = mono[onset: onset + int(ATTACK_S * sample_rate)]
    body_begin = onset + int(BODY_START_S * sample_rate)
    body = mono[body_begin: body_begin + int(BODY_S * sample_rate)]
    if attack.size == 0 or body.size == 0:
        return 0.0
    ratio = float(np.mean(np.square(attack)) / (np.mean(np.square(body)) + 1e-12))
    return float(ratio / (1.0 + ratio))


def fit_attack_params(
    coeffs: np.ndarray, sample_rate: int, onset: int,
) -> np.ndarray:
    """Measured labels: ``[2, bands]`` (log gain-over-body, log decay τ)
    fitted on the attack window of a sound that has an attack."""
    envelopes = band_envelopes(coeffs, sample_rate)
    n = envelopes.shape[1]
    attack_n = int(ATTACK_S * sample_rate)
    body_begin = min(onset + int(BODY_START_S * sample_rate), n - 1)
    body_end = min(body_begin + int(BODY_S * sample_rate), n)
    log_gain = np.zeros(MODEL_BANDS)
    log_tau = np.full(MODEL_BANDS, np.log(0.05))
    for b in range(envelopes.shape[0]):
        window = envelopes[b, onset: min(onset + attack_n, n)]
        body = float(envelopes[b, body_begin:body_end].mean()) + 1e-12
        if window.size < 8:
            continue
        peak_at = int(np.argmax(window))
        log_gain[b] = np.log(np.clip(window[peak_at] / body,
                                     np.exp(LOG_GAIN_LO), np.exp(LOG_GAIN_HI)))
        decay = window[peak_at:]
        keep = decay > 0.05 * window[peak_at]
        if keep.sum() >= 8:
            t = np.arange(keep.sum()) / sample_rate
            slope = np.polyfit(t, np.log(decay[keep] + 1e-12), 1)[0]
            if slope < -1e-3:
                log_tau[b] = np.log(np.clip(-1.0 / slope,
                                            np.exp(LOG_TAU_LO),
                                            np.exp(LOG_TAU_HI)))
    return np.stack([log_gain, log_tau]).astype(np.float32)


def default_attack_params(centers: np.ndarray) -> np.ndarray:
    """The deterministic generic pluck: HF-tilted gains over body, decays
    shrinking with frequency (τ ∝ f^−0.4)."""
    centers = np.asarray(centers, dtype=np.float64)
    gain = np.clip(3.0 * (centers / 300.0) ** 0.2, 1.5, 8.0)
    tau = np.clip(0.09 * (150.0 / centers) ** 0.4, 0.006, 0.12)
    return np.stack([np.log(gain), np.log(tau)]).astype(np.float32)


def build_attack(
    coeffs: np.ndarray, sample_rate: int, onset: int, params: np.ndarray,
) -> np.ndarray:
    """The imagined attack as added coefficients (same shape as ``coeffs``).

    Per band: carrier = the band's own body segment copied to the onset
    (timbre-matched, phase-coherent), shaped by ``ramp(2 ms) ·
    exp(−t/τ_b)`` and scaled so its peak sits ``gain_b`` × the body level.
    """
    n = coeffs.shape[1]
    attack_n = min(int(ATTACK_S * sample_rate), n - onset)
    if attack_n <= 8:
        return np.zeros_like(coeffs)
    envelopes = band_envelopes(coeffs, sample_rate)
    body_begin = min(onset + int(BODY_START_S * sample_rate), n - 1)
    body_end = min(body_begin + int(BODY_S * sample_rate), n)
    gain = np.exp(np.asarray(params[0], dtype=np.float64))
    tau = np.exp(np.asarray(params[1], dtype=np.float64))

    t = np.arange(attack_n) / sample_rate
    ramp = np.minimum(t / 0.002, 1.0)
    added = np.zeros_like(coeffs)
    for b in range(coeffs.shape[0]):
        body_level = float(envelopes[b, body_begin:body_end].mean())
        if body_level <= 1e-10:
            continue
        segment = coeffs[b, body_begin:body_end]
        if segment.size == 0:
            continue
        reps = int(np.ceil(attack_n / segment.size))
        carrier = np.tile(segment, reps)[:attack_n]
        carrier_level = np.abs(carrier).mean() + 1e-12
        shape = ramp * np.exp(-t / tau[b])
        added[b, onset: onset + attack_n] = (
            carrier / carrier_level * body_level * gain[b] * shape)
    return added


if HAVE_TORCH:

    class TransientNet(nn.Module):
        """(envelopes [B, 1, bands, frames], sharpness [B, 1]) →
        per-band attack parameters [B, 2, bands] (log gain-over-body,
        log τ), bounded to the fit ranges. FiLM on the sharpness target."""

        def __init__(self, width: int = 32) -> None:
            super().__init__()
            self.width = width
            act = nn.SiLU()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, width, 3, stride=(1, 2), padding=1), act,
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1), act,
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), act,
                nn.AdaptiveAvgPool2d((8, 4)),
            )
            hidden = width * 2 * 8 * 4
            self.film = nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 512))
            self.head = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU())
            self.out = nn.Linear(256, 2 * MODEL_BANDS)

        def forward(self, envelopes, sharpness):
            encoded = self.encoder(envelopes).flatten(1)
            hidden = self.head(encoded)
            scale, shift = self.film(sharpness).chunk(2, dim=1)
            hidden = hidden * (1.0 + scale) + shift
            raw = torch.tanh(self.out(hidden)).view(-1, 2, MODEL_BANDS)
            gain_mid = 0.5 * (LOG_GAIN_LO + LOG_GAIN_HI)
            gain_half = 0.5 * (LOG_GAIN_HI - LOG_GAIN_LO)
            tau_mid = 0.5 * (LOG_TAU_LO + LOG_TAU_HI)
            tau_half = 0.5 * (LOG_TAU_HI - LOG_TAU_LO)
            return torch.stack([
                gain_mid + gain_half * raw[:, 0],
                tau_mid + tau_half * raw[:, 1],
            ], dim=1)

    def save_transient_model(model: "TransientNet", path: str | Path) -> Path:
        path = Path(path)
        torch.save({"state_dict": model.state_dict(), "width": model.width,
                    "bands": MODEL_BANDS, "frames": MODEL_FRAMES}, path)
        return path

    def load_transient_model(path: str | Path) -> "TransientNet":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = TransientNet(width=int(payload["width"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def predict_attack_params(
        model: "TransientNet", envelopes_log: np.ndarray, sharpness: float,
    ) -> np.ndarray:
        """[2, bands] (log gain, log τ) for one envelope grid at the
        requested sharpness."""
        with torch.no_grad():
            batch = torch.from_numpy(
                np.asarray(envelopes_log, dtype=np.float32))[None, None]
            condition = torch.tensor([[float(sharpness)]], dtype=torch.float32)
            device = next(model.parameters()).device
            output = model(batch.to(device), condition.to(device))
        return output[0].cpu().numpy()

else:  # pragma: no cover - exercised only without torch

    class TransientNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("TransientNet needs PyTorch (pip install torch).")

    def save_transient_model(model, path):  # type: ignore[no-redef]
        raise RuntimeError("save_transient_model needs PyTorch.")

    def load_transient_model(path):  # type: ignore[no-redef]
        raise RuntimeError("load_transient_model needs PyTorch.")

    def predict_attack_params(model, envelopes_log, sharpness):  # type: ignore[no-redef]
        raise RuntimeError("predict_attack_params needs PyTorch.")
