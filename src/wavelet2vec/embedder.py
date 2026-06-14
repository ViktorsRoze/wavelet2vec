from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wavelet2vec.audio_io import list_audio_files, load_audio, resample
from wavelet2vec.conv_encoder import N_TEMPORAL_FILTERS, WaveletConvEncoder
from wavelet2vec.filterbank import log_spaced_frequencies, morlet_scalogram
from wavelet2vec.harmonic_features import harmonic_features
from wavelet2vec.phase_features import phase_features
from wavelet2vec.stereo_features import neutral_stereo_features, stereo_features
from wavelet2vec.transient_features import transient_features
from wavelet2vec.wavelet_features import modulation_features, spectral_features

SECTION_NAMES = ("spectral", "modulation", "transient", "harmonic", "phase", "stereo", "conv")


@dataclass
class Wavelet2VecConfig:
    """Settings for the wavelet2vec embedding.

    All inputs are resampled to ``analysis_sample_rate`` and peak-normalized,
    so embeddings are comparable across files with different rates and levels.
    """

    analysis_sample_rate: int = 22050
    n_bands: int = 64
    f_min: float = 27.5
    f_max: float | None = None
    bandwidth_scale: float = 1.0
    envelope_points: int = 32
    envelope_rate: int = 256
    n_band_groups: int = 8
    n_modulation_bands: int = 8
    mod_f_min: float = 0.5
    mod_f_max: float = 64.0
    n_phase_groups: int = 8
    n_phase_harmonics: int = 8
    n_stereo_groups: int = 8
    include_conv: bool = True
    conv_bands: int = 32
    conv_kernel_size: int = 1023
    conv_hop: int = 64
    section_weights: dict[str, float] = field(
        default_factory=lambda: {
            "spectral": 1.0,
            "modulation": 0.75,
            "transient": 1.0,
            "harmonic": 0.75,
            "phase": 1.0,
            "stereo": 0.5,
            "conv": 0.5,
        }
    )

    def resolved_f_max(self) -> float:
        return self.f_max if self.f_max is not None else 0.45 * self.analysis_sample_rate

    def section_dims(self) -> dict[str, int]:
        dims = {
            "spectral": 2 * self.n_bands,
            "modulation": self.n_band_groups * self.n_modulation_bands,
            "transient": self.envelope_points + 8,
            "harmonic": 12 + 3,
            "phase": 2 * self.n_phase_groups + 3 + 3 * (self.n_phase_harmonics - 1),
            "stereo": 3 + 3 * self.n_stereo_groups,
        }
        if self.include_conv:
            dims["conv"] = 2 * self.conv_bands * N_TEMPORAL_FILTERS
        return dims


class Wavelet2Vec:
    """Turns a short audio snippet into a fixed-size perceptual embedding.

    The embedding combines complementary views of the sound:

    - ``spectral``: constant-Q Morlet wavelet band statistics (timbre, color)
    - ``modulation``: second-order modulation spectrum (texture, roughness)
    - ``transient``: waveform envelope morphology (attack, decay, punch)
    - ``harmonic``: chroma, harmonicity, pitch height (musical pitch content)
    - ``phase``: shift-invariant phase information — instantaneous-frequency
      statistics, cross-band onset phase coherence, and the pitch-synchronous
      harmonic phase signature that encodes waveshape
    - ``stereo``: spatial image — width, inter-channel coherence, and balance,
      globally and per band group (mono input gets the canonical centered
      vector, so dimensions are fixed)
    - ``conv``: wavelet-initialized 1D convolutional encoder output (local
      band-envelope dynamics; deterministic by default, trainable later)

    Each section is L2-normalized and weighted before concatenation, so cosine
    similarity between embeddings behaves like a weighted blend of per-aspect
    similarities. The embedding is deterministic — no training required — and
    invariant to input sample rate, playback level, and small time shifts.
    """

    def __init__(self, config: Wavelet2VecConfig | None = None) -> None:
        self.config = config or Wavelet2VecConfig()
        self._centers = log_spaced_frequencies(
            self.config.f_min, self.config.resolved_f_max(), self.config.n_bands
        )
        self.conv_encoder: WaveletConvEncoder | None = None
        if self.config.include_conv:
            self.conv_encoder = WaveletConvEncoder(
                sample_rate=self.config.analysis_sample_rate,
                n_bands=self.config.conv_bands,
                f_min=self.config.f_min,
                f_max=self.config.resolved_f_max(),
                kernel_size=self.config.conv_kernel_size,
                hop=self.config.conv_hop,
            )
            self.conv_encoder.eval()

        dims = self.config.section_dims()
        self.sections: dict[str, slice] = {}
        offset = 0
        for name in SECTION_NAMES:
            if name not in dims:
                continue
            self.sections[name] = slice(offset, offset + dims[name])
            offset += dims[name]
        self.dim = offset

    def _prepare(self, waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Returns (peak-normalized mono, first two channels or None if mono input)."""
        if hasattr(waveform, "detach"):  # accept torch tensors transparently
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float64)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        waveform = resample(waveform, sample_rate, self.config.analysis_sample_rate)
        peak = np.abs(waveform).max()
        if peak > 1e-8:
            waveform = waveform / peak
        stereo = waveform[:2] if waveform.shape[0] >= 2 else None
        return waveform.mean(axis=0), stereo

    def embed_components(self, waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """Raw (unnormalized) feature vectors per section, useful for inspection."""
        config = self.config
        mono, stereo = self._prepare(waveform, sample_rate)
        if np.abs(mono).max() <= 1e-8:
            return {name: np.zeros(dim, dtype=np.float32) for name, dim in config.section_dims().items()}

        complex_scalogram = morlet_scalogram(
            mono,
            config.analysis_sample_rate,
            self._centers,
            bandwidth_scale=config.bandwidth_scale,
            return_complex=True,
        )
        scalogram = np.abs(complex_scalogram).astype(np.float32)
        components = {
            "spectral": spectral_features(scalogram),
            "modulation": modulation_features(
                scalogram,
                config.analysis_sample_rate,
                envelope_rate=config.envelope_rate,
                n_band_groups=config.n_band_groups,
                n_modulation_bands=config.n_modulation_bands,
                mod_f_min=config.mod_f_min,
                mod_f_max=config.mod_f_max,
            ),
            "transient": transient_features(
                mono,
                scalogram,
                config.analysis_sample_rate,
                envelope_points=config.envelope_points,
            ),
            "harmonic": harmonic_features(mono, scalogram, config.analysis_sample_rate),
            "phase": phase_features(
                mono,
                complex_scalogram,
                self._centers,
                config.analysis_sample_rate,
                n_groups=config.n_phase_groups,
                n_harmonics=config.n_phase_harmonics,
            ),
        }
        if stereo is None:
            components["stereo"] = neutral_stereo_features(config.n_stereo_groups)
        else:
            components["stereo"] = stereo_features(
                stereo[0],
                stereo[1],
                config.analysis_sample_rate,
                self._centers,
                bandwidth_scale=config.bandwidth_scale,
                n_groups=config.n_stereo_groups,
            )
        if self.conv_encoder is not None:
            components["conv"] = self.conv_encoder.encode_mono(mono)
        return components

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embeds ``[channels, samples]`` (or 1D mono) audio at any sample rate."""
        components = self.embed_components(waveform, sample_rate)
        parts = []
        for name in self.sections:
            vector = components[name]
            # The norm floor keeps near-empty sections (e.g. modulation of a
            # steady tone, where only numerical/resampling noise remains) from
            # being amplified to unit length with a random direction.
            vector = vector / max(np.linalg.norm(vector), 0.1)
            parts.append(vector * float(self.config.section_weights.get(name, 1.0)))
        return np.concatenate(parts).astype(np.float32)

    def embed_file(self, path: str | Path) -> np.ndarray:
        waveform, sample_rate = load_audio(path)
        return self.embed(waveform, sample_rate)

    def embed_folder(self, folder: str | Path) -> dict[str, np.ndarray]:
        root = Path(folder)
        embeddings: dict[str, np.ndarray] = {}
        for path in list_audio_files(root):
            embeddings[path.relative_to(root).as_posix()] = self.embed_file(path)
        return embeddings


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
    if denominator <= 1e-12:
        return 0.0
    return float(embedding_a @ embedding_b / denominator)


def pairwise_similarity(embeddings: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Cosine similarity matrix over a named embedding collection."""
    names = list(embeddings)
    matrix = np.eye(len(names), dtype=np.float32)
    for row in range(len(names)):
        for column in range(row + 1, len(names)):
            value = cosine_similarity(embeddings[names[row]], embeddings[names[column]])
            matrix[row, column] = matrix[column, row] = value
    return names, matrix
