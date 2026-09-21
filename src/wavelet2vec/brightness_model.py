"""Learned brightness axis v1 (roadmap G-2): conditioned high-band envelopes.

The deterministic :func:`~wavelet2vec.imagine.imagine_highs` fills missing
highs by heterodyning mid-band content up and scaling it to a fixed
−6 dB/octave target. The learned v1 keeps the physics — carrier and phase
still come from the deterministic heterodyne, the low bands stay exactly
preserved — and replaces only the *energy rule*: :class:`BrightnessNet`
predicts per-band, per-frame log-envelopes for the missing bands,
conditioned on a target brightness in [0, 1] (the same measured axis as
``datasets.axis_labels``). A model is therefore a drop-in upgrade via
``imagine_highs(..., model=...)`` and the library never requires weights.

Training pairs are free (G-1): a coefficient-domain lowpass scales each
band's envelope by the same mask, so ``(masked envelopes → full
envelopes)`` supervision needs no audio synthesis at all — see
``experiments/train_brightness.py``.
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

from wavelet2vec.frame import analyze

MODEL_BANDS = 64
MODEL_FRAMES = 128
MODEL_F_MIN = 30.0
SPLIT_HZ = 4000.0  # brightness = HF energy fraction above this


def log_envelopes(
    audio: np.ndarray, sample_rate: int, *,
    n_bands: int = MODEL_BANDS, n_frames: int = MODEL_FRAMES,
    f_min: float = MODEL_F_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """``(log1p band envelopes [n_bands, n_frames], centers)`` on the model's
    fixed grid. Audio is peak-normalized so levels are absolute; envelopes
    are mean ``|coeff|`` per frame (hop = length // n_frames)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio / peak
    coeffs, spec = analyze(audio, sample_rate, n_bands=n_bands, f_min=f_min)
    magnitude = np.abs(coeffs)
    hop = max(magnitude.shape[1] // n_frames, 1)
    pooled = magnitude[:, : hop * n_frames].reshape(n_bands, n_frames, hop).mean(axis=2)
    return np.log1p(pooled).astype(np.float32), spec.centers.astype(np.float32)


def cutoff_mask(centers: np.ndarray, cutoff_hz: float,
                transition_octaves: float = 0.4) -> np.ndarray:
    """The smooth lowpass mask of ``datasets.lowpass_pair`` — cosine rolloff
    over ``transition_octaves`` around ``cutoff_hz`` — as per-band gains."""
    octaves = np.log2(np.asarray(centers, dtype=np.float64) / cutoff_hz)
    ramp = np.clip(octaves / transition_octaves, -0.5, 0.5)
    return np.clip(0.5 - 0.5 * np.sin(np.pi * ramp), 0.0, 1.0).astype(np.float32)


def envelope_brightness(envelopes_log: np.ndarray, centers: np.ndarray,
                        split_hz: float = SPLIT_HZ) -> float:
    """HF energy fraction of a log1p envelope grid — the conditioning label
    (envelope-domain twin of ``axis_labels``' brightness)."""
    energy = (np.expm1(np.asarray(envelopes_log, dtype=np.float64)) ** 2).mean(axis=1)
    high = energy[np.asarray(centers) >= split_hz].sum()
    return float(high / (energy.sum() + 1e-24))


if HAVE_TORCH:

    class BrightnessNet(nn.Module):
        """FiLM-conditioned encoder–decoder over the envelope grid.

        ``forward(envelopes [B, 1, bands, frames], brightness [B, 1])``
        → predicted full log-envelopes ``[B, 1, bands, frames]``. The
        brightness scalar modulates the bottleneck (FiLM scale + shift),
        which is what makes the knob a knob.
        """

        def __init__(self, width: int = 32) -> None:
            super().__init__()
            self.width = width
            act = nn.SiLU()
            self.enc1 = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), act)
            self.enc2 = nn.Sequential(
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1), act)
            self.enc3 = nn.Sequential(
                nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), act)
            self.film = nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width * 4))
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(width * 2, width * 2, 4, stride=2, padding=1), act)
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(width * 4, width, 4, stride=2, padding=1), act)
            self.out = nn.Conv2d(width * 2, 1, 3, padding=1)

        def forward(self, envelopes, brightness):
            e1 = self.enc1(envelopes)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            scale, shift = self.film(brightness).chunk(2, dim=1)
            e3 = e3 * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            d3 = self.dec3(e3)
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            prediction = self.out(torch.cat([d2, e1], dim=1))
            return nn.functional.softplus(prediction)  # log1p envelopes are ≥ 0

    def save_brightness_model(model: "BrightnessNet", path: str | Path) -> Path:
        path = Path(path)
        torch.save({"state_dict": model.state_dict(), "width": model.width,
                    "bands": MODEL_BANDS, "frames": MODEL_FRAMES,
                    "f_min": MODEL_F_MIN, "split_hz": SPLIT_HZ}, path)
        return path

    def load_brightness_model(path: str | Path) -> "BrightnessNet":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = BrightnessNet(width=int(payload["width"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def predict_full_envelopes(
        model: "BrightnessNet", envelopes_log: np.ndarray, brightness: float,
    ) -> np.ndarray:
        """Predicted full log1p envelopes ``[bands, frames]`` for one input
        grid at the requested target brightness."""
        with torch.no_grad():
            batch = torch.from_numpy(
                np.asarray(envelopes_log, dtype=np.float32))[None, None]
            condition = torch.tensor([[float(brightness)]], dtype=torch.float32)
            device = next(model.parameters()).device
            output = model(batch.to(device), condition.to(device))
        return output[0, 0].cpu().numpy()

else:  # pragma: no cover - exercised only without torch

    class BrightnessNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("BrightnessNet needs PyTorch (pip install torch).")

    def save_brightness_model(model, path):  # type: ignore[no-redef]
        raise RuntimeError("save_brightness_model needs PyTorch.")

    def load_brightness_model(path):  # type: ignore[no-redef]
        raise RuntimeError("load_brightness_model needs PyTorch.")

    def predict_full_envelopes(model, envelopes_log, brightness):  # type: ignore[no-redef]
        raise RuntimeError("predict_full_envelopes needs PyTorch.")
