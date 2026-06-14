from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wavelet2vec.audio_io import resample
from wavelet2vec.filterbank import log_spaced_frequencies

N_TEMPORAL_FILTERS = 4


def _morlet_kernels(center_frequencies: np.ndarray, sample_rate: int, kernel_size: int) -> torch.Tensor:
    """Gaussian-windowed cosine/sine kernel pairs, one quadrature pair per band."""
    times = (np.arange(kernel_size) - kernel_size // 2) / sample_rate
    kernels = np.zeros((2 * len(center_frequencies), 1, kernel_size), dtype=np.float32)
    for index, center in enumerate(center_frequencies):
        # ~6 cycles of support, capped so low bands still fit the kernel.
        sigma = min(kernel_size / (6.0 * sample_rate), 1.0 / center)
        window = np.exp(-0.5 * (times / sigma) ** 2)
        cos_kernel = window * np.cos(2.0 * np.pi * center * times)
        sin_kernel = window * np.sin(2.0 * np.pi * center * times)
        cos_kernel -= cos_kernel.mean()
        kernels[index, 0] = cos_kernel / (np.linalg.norm(cos_kernel) + 1e-12)
        kernels[len(center_frequencies) + index, 0] = sin_kernel / (np.linalg.norm(sin_kernel) + 1e-12)
    return torch.from_numpy(kernels)


def _temporal_kernels(kernel_size: int, frame_rate: float) -> torch.Tensor:
    """Smoothing, derivative, and two Gabor modulation kernels (shared across bands)."""
    times = (np.arange(kernel_size) - kernel_size // 2) / frame_rate
    window = np.exp(-0.5 * (times / (times.max() / 2.5 + 1e-12)) ** 2)
    kernels = np.stack(
        [
            window,
            -times * window * frame_rate,
            window * np.cos(2.0 * np.pi * 16.0 * times),
            window * np.cos(2.0 * np.pi * 64.0 * times),
        ]
    ).astype(np.float32)
    kernels /= np.linalg.norm(kernels, axis=1, keepdims=True) + 1e-12
    return torch.from_numpy(kernels)


class WaveletConvEncoder(nn.Module):
    """1D convolutional snippet encoder initialized as a Morlet wavelet filterbank.

    Layer one is a strided quadrature (cosine/sine) convolution whose modulus
    gives translation-stable band envelopes — a learnable version of the FFT
    filterbank. Layer two applies depthwise temporal filters (smoothing,
    derivative, 16 Hz and 64 Hz Gabors) to those envelopes, capturing local
    dynamics, and the result is pooled over time.

    With ``trainable=False`` (default) the encoder is deterministic and frozen;
    set ``trainable=True`` to fine-tune it, e.g. with a contrastive objective.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_bands: int = 32,
        f_min: float = 27.5,
        f_max: float | None = None,
        kernel_size: int = 1023,
        hop: int = 64,
        temporal_kernel_size: int = 31,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        centers = log_spaced_frequencies(f_min, f_max if f_max is not None else 0.45 * sample_rate, n_bands)

        self.analysis = nn.Conv1d(
            1, 2 * n_bands, kernel_size, stride=hop, padding=kernel_size // 2, bias=False
        )
        self.analysis.weight.data.copy_(_morlet_kernels(centers, sample_rate, kernel_size))

        self.temporal = nn.Conv1d(
            n_bands,
            n_bands * N_TEMPORAL_FILTERS,
            temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            groups=n_bands,
            bias=False,
        )
        temporal = _temporal_kernels(temporal_kernel_size, frame_rate=sample_rate / hop)
        self.temporal.weight.data.copy_(temporal.unsqueeze(1).repeat(n_bands, 1, 1))

        self.requires_grad_(trainable)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_bands * N_TEMPORAL_FILTERS

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encodes ``[batch, 1, samples]`` at ``self.sample_rate`` into ``[batch, output_dim]``."""
        analyzed = self.analysis(waveform)
        real, imag = analyzed[:, : self.n_bands], analyzed[:, self.n_bands :]
        envelopes = torch.log1p(torch.sqrt(real**2 + imag**2 + 1e-12))
        responses = self.temporal(envelopes).abs()
        return torch.cat([responses.mean(dim=-1), responses.std(dim=-1)], dim=1)

    @torch.no_grad()
    def encode_mono(self, mono: np.ndarray) -> np.ndarray:
        """Encodes a prepared mono signal already at ``self.sample_rate``."""
        batch = torch.from_numpy(mono.astype(np.float32)).reshape(1, 1, -1)
        return self.forward(batch).squeeze(0).numpy()

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encodes ``[channels, samples]`` audio at any sample rate and level."""
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        waveform = resample(waveform.astype(np.float32), sample_rate, self.sample_rate)
        mono = waveform.mean(axis=0)
        peak = np.abs(mono).max()
        if peak > 1e-8:
            mono = mono / peak
        return self.encode_mono(mono)
