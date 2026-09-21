"""Lossy differentiable wavelet features for recognition models (roadmap D-2).

This pooled image is not invertible and does not retain all phase or stereo
information. For a waveform + full-complex carrier with a defined inverse,
use HybridAudioFrontend / HybridTokenEncoder from wavelet2vec.hybrid.

:class:`WaveletFrontend` is a ``torch.nn.Module`` that turns raw audio into
the library's 3-channel coefficient image **inside the autograd graph** —
the same encoding as :func:`wavelet2vec.vision.coeff_image` (log-magnitude,
IF-deviation, energy-weighted coherence), batched, on any device, with
gradients flowing back to the samples. Drop it in front of any vision or
audio model; use it for trainable front ends, feature-inversion losses, or
DDSP-style decoders.

The filter windows are built once for a fixed ``(n_samples, sample_rate)``
and registered as buffers (they move with ``.to(device)``).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    HAVE_TORCH = False

from wavelet2vec.frame import _frame_windows

if HAVE_TORCH:

    class WaveletFrontend(nn.Module):
        """``[B, N] float`` → ``[B, 3, n_bands, n_frames]`` in [0, 1].

        Rows follow display/COCO orientation (row 0 = highest band), matching
        :func:`wavelet2vec.vision.coeff_image` and its annotations.
        """

        def __init__(self, n_samples: int, sample_rate: int, *,
                     n_bands: int = 128, n_frames: int = 128,
                     f_min: float = 30.0) -> None:
            super().__init__()
            self.n_samples = int(n_samples)
            self.sample_rate = int(sample_rate)
            self.n_bands = int(n_bands)
            self.n_frames = int(n_frames)
            self.n_even = self.n_samples % 2 == 0
            # Mirror frame.analyze() exactly (same windows incl. edge
            # plateaus, same center grid to 0.5·sr) so this module's channel
            # 0 matches vision.coeff_image and the COCO annotations.
            freqs = np.fft.rfftfreq(self.n_samples, d=1.0 / sample_rate)
            centers = np.geomspace(
                max(f_min, freqs[1] if len(freqs) > 1 else f_min),
                0.5 * sample_rate, n_bands,
            )
            windows = _frame_windows(freqs.shape[0], freqs, centers, 1.0)
            self.register_buffer("windows", torch.from_numpy(windows.astype(np.float32)))
            self.register_buffer("centers", torch.from_numpy(centers.astype(np.float32)))

        def _analytic_bands(self, audio: "torch.Tensor") -> "torch.Tensor":
            """[B, N] → complex analytic band signals [B, bands, N], mirroring
            analyze(): positive bins doubled, DC (and Nyquist) single."""
            spectrum = torch.fft.rfft(audio, dim=-1)
            half = spectrum.shape[-1]
            band = spectrum[:, None, :] * self.windows
            full = torch.zeros(audio.shape[0], self.n_bands, self.n_samples,
                               dtype=torch.complex64, device=audio.device)
            full[..., :half] = 2.0 * band
            full[..., 0] = band[..., 0]
            if self.n_even:
                full[..., half - 1] = band[..., half - 1]
            return torch.fft.ifft(full, dim=-1)

        def _pool(self, values: "torch.Tensor", reduce: str) -> "torch.Tensor":
            """Adaptive time pooling [B, bands, N] → [B, bands, n_frames]."""
            if reduce == "max":
                return torch.nn.functional.adaptive_max_pool1d(values, self.n_frames)
            return torch.nn.functional.adaptive_avg_pool1d(values, self.n_frames)

        def forward(self, audio: "torch.Tensor") -> "torch.Tensor":
            if audio.dim() == 1:
                audio = audio[None, :]
            if audio.shape[-1] != self.n_samples:
                raise ValueError(
                    f"WaveletFrontend was built for {self.n_samples} samples, "
                    f"got {audio.shape[-1]}. Build one per fragment length."
                )
            audio = audio.to(self.windows.dtype)
            coeffs = self._analytic_bands(audio)
            # Smooth magnitude: |c| has a NaN gradient at exactly zero
            # coefficients (silent regions would poison the whole backward).
            magnitude = torch.sqrt(coeffs.real**2 + coeffs.imag**2 + 1e-12)

            # Channel 0: log-magnitude, per-example max-normalized.
            log_mag = torch.log1p(magnitude * 40.0)
            mag_img = self._pool(log_mag, "max")
            peak = mag_img.amax(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            mag_img = mag_img / peak

            # Channel 1: instantaneous-frequency deviation from band centers.
            # angle() also has NaN gradients at 0: route near-zero increments
            # through a safe constant so no NaN is ever *produced*.
            increment = coeffs[..., 1:] * coeffs[..., :-1].conj()
            inc_mag = torch.sqrt(increment.real**2 + increment.imag**2 + 1e-24)
            active = (inc_mag > 1e-9).to(audio.dtype)
            safe = torch.where(inc_mag > 1e-9, increment,
                               torch.ones_like(increment))
            angles = torch.atan2(safe.imag, safe.real) * active
            inst_freq = angles * (self.sample_rate / (2 * np.pi))
            zero_first = torch.zeros_like(inst_freq[..., :1])  # coeff_image convention
            inst_freq = torch.cat([zero_first, inst_freq], dim=-1)
            deviation = torch.tanh(
                (inst_freq - self.centers[None, :, None]) / self.centers[None, :, None]
            )
            dev_img = (self._pool(deviation, "mean") + 1.0) / 2.0

            # Channel 2: energy-weighted local phase coherence.
            unit_real = coeffs.real / magnitude
            unit_imag = coeffs.imag / magnitude
            coherence = torch.sqrt(
                self._pool(unit_real, "mean") ** 2
                + self._pool(unit_imag, "mean") ** 2 + 1e-24
            )
            coh_img = coherence * mag_img

            image = torch.stack([mag_img, dev_img, coh_img], dim=1)
            return torch.flip(image, dims=(2,))  # row 0 = highest band

        def tokens(self, audio: "torch.Tensor") -> "torch.Tensor":
            """Sequence tokens ``[B, n_frames, 3·n_bands]`` — the batched,
            differentiable twin of :func:`wavelet2vec.vision.sequence_tokens`
            (bands low → high within each channel block). Build the module
            with ``n_frames = duration/hop`` for a chosen hop."""
            image = torch.flip(self.forward(audio), dims=(2,))  # back to low→high
            return image.permute(0, 3, 1, 2).reshape(
                image.shape[0], self.n_frames, 3 * self.n_bands
            )

else:  # pragma: no cover - import-time fallback without torch

    class WaveletFrontend:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "WaveletFrontend needs PyTorch. Install torch or use "
                "wavelet2vec.vision.coeff_image for the numpy path."
            )
