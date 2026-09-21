"""Learned tonality axis (roadmap G-9): conditioned harmonic/percussive remix.

The deterministic :func:`~wavelet2vec.imagine.imagine_tonality` remixes the
exact-sum HPSS layers with one scalar. The learned version keeps every part
of that machinery — the layers still come from :func:`~wavelet2vec.
decompose.hpss`, gains are real numbers on complex coefficients (phase
untouched), and unit gains reproduce the input exactly — and replaces only
the *choice of gains*: :class:`TonalityNet` maps (layer envelopes, target
harmonicity) → per-band gains for both layers, so one forward pass lands
the measured harmonicity near the request, with frequency-dependent
character (air is percussive highs, body is harmonic mids) instead of a
blanket scalar.

Supervision follows the G-2 recipe with one twist: harmonicity is a
*waveform* feature, so labels cannot be read off envelopes — the training
set renders scalar+tilt remixes and measures each one
(``experiments/train_tonality.py``, cached and resumable).
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

from wavelet2vec.frame import analyze, synthesize

MODEL_BANDS = 64
MODEL_FRAMES = 128
MODEL_F_MIN = 30.0
LOG_GAIN_MAX = float(np.log(8.0))  # ±18 dB per band, same cap family as G-2


def layer_envelopes(
    harmonic: np.ndarray, percussive: np.ndarray, sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    """``(envelopes [2, bands, frames], centers, coeff_pair)`` for the two
    HPSS layers on the model grid. ``coeff_pair`` holds each layer's complex
    coefficients + spec so a remix can reuse the analysis."""
    envelopes, coeff_pair = [], []
    centers = None
    for layer in (harmonic, percussive):
        coeffs, spec = analyze(np.asarray(layer, dtype=np.float64), sample_rate,
                               n_bands=MODEL_BANDS, f_min=MODEL_F_MIN)
        magnitude = np.abs(coeffs)
        hop = max(magnitude.shape[1] // MODEL_FRAMES, 1)
        pooled = magnitude[:, : hop * MODEL_FRAMES].reshape(
            MODEL_BANDS, MODEL_FRAMES, hop).mean(axis=2)
        envelopes.append(np.log1p(pooled).astype(np.float32))
        coeff_pair.append((coeffs, spec))
        centers = spec.centers.astype(np.float32)
    return np.stack(envelopes), centers, coeff_pair


def remix_layers(coeff_pair: list, gains: np.ndarray, length: int) -> np.ndarray:
    """Synthesizes ``Σ layer_coeffs · gain[layer][band]`` — real per-band
    gains, phase untouched; unit gains reproduce ``harmonic + percussive``
    (the exact-sum input) to synthesis precision."""
    gains = np.asarray(gains, dtype=np.float64)
    output = np.zeros(length)
    for (coeffs, spec), layer_gain in zip(coeff_pair, gains):
        output += synthesize(coeffs * layer_gain[:, None], spec)[:length]
    return output


if HAVE_TORCH:

    class TonalityNet(nn.Module):
        """(layer envelopes [B, 2, bands, frames], harmonicity [B, 1]) →
        per-band log-gains [B, 2, bands].

        The head predicts a single remix amount ``a ∈ (−1, 1)`` and the
        gains are ``(1 + a, 1 − a)`` — *exactly* the deterministic knob's
        family, so the reachable set matches the scalar oracle and a
        perfect predictor equals it. Two designs were measured and
        rejected first: a free [2, 64] gain field (regresses to the
        conditional mean of every profile sharing a label — calibration
        0.23 vs oracle 0.167) and a 4-param base+slope family (0.29 —
        smoother but the ±31 dB cap cannot *remove* a layer, which the
        oracle's ``a=±1`` can, so its reach is shorter). FiLM on the
        target harmonicity conditions the bottleneck; frequency-selective
        gains are the v2 idea once the scalar is calibrated."""

        def __init__(self, width: int = 32) -> None:
            super().__init__()
            self.width = width
            act = nn.SiLU()
            self.encoder = nn.Sequential(
                nn.Conv2d(2, width, 3, stride=(1, 2), padding=1), act,
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1), act,
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), act,
                nn.AdaptiveAvgPool2d((8, 4)),
            )
            hidden = width * 2 * 8 * 4
            self.film = nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 512))
            self.head = nn.Sequential(
                nn.Linear(hidden, 256), nn.SiLU())
            self.out = nn.Linear(256, 1)

        def amount_from(self, envelopes, harmonicity):
            """Remix amount ``a`` [B, 1] in (−1, 1)."""
            encoded = self.encoder(envelopes).flatten(1)
            hidden = self.head(encoded)
            scale, shift = self.film(harmonicity).chunk(2, dim=1)
            hidden = hidden * (1.0 + scale) + shift
            return torch.tanh(self.out(hidden))

        def forward(self, envelopes, harmonicity):
            amount = self.amount_from(envelopes, harmonicity)
            gains = torch.cat([1.0 + amount, 1.0 - amount], dim=1)
            log_gains = torch.log(torch.clamp(gains, min=1e-6))
            return log_gains[:, :, None].expand(-1, -1, MODEL_BANDS)

    def save_tonality_model(model: "TonalityNet", path: str | Path) -> Path:
        path = Path(path)
        torch.save({"state_dict": model.state_dict(), "width": model.width,
                    "bands": MODEL_BANDS, "frames": MODEL_FRAMES,
                    "f_min": MODEL_F_MIN, "log_gain_max": LOG_GAIN_MAX}, path)
        return path

    def load_tonality_model(path: str | Path) -> "TonalityNet":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = TonalityNet(width=int(payload["width"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def predict_layer_gains(
        model: "TonalityNet", envelopes: np.ndarray, harmonicity: float,
    ) -> np.ndarray:
        """Linear per-band gains ``[2, bands]`` for one envelope stack at the
        requested target harmonicity."""
        with torch.no_grad():
            batch = torch.from_numpy(
                np.asarray(envelopes, dtype=np.float32))[None]
            condition = torch.tensor([[float(harmonicity)]], dtype=torch.float32)
            device = next(model.parameters()).device
            log_gains = model(batch.to(device), condition.to(device))
        return np.exp(log_gains[0].cpu().numpy().astype(np.float64))

else:  # pragma: no cover - exercised only without torch

    class TonalityNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("TonalityNet needs PyTorch (pip install torch).")

    def save_tonality_model(model, path):  # type: ignore[no-redef]
        raise RuntimeError("save_tonality_model needs PyTorch.")

    def load_tonality_model(path):  # type: ignore[no-redef]
        raise RuntimeError("load_tonality_model needs PyTorch.")

    def predict_layer_gains(model, envelopes, harmonicity):  # type: ignore[no-redef]
        raise RuntimeError("predict_layer_gains needs PyTorch.")
