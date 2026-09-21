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


def _batched_scalogram_stats(coeffs, centers: np.ndarray, sample_rate: int,
                             n_groups: int):
    """Batched torch twins of ``spectral_features``,
    ``instantaneous_frequency_features`` and ``onset_phase_coherence`` (D-1b).

    ``coeffs`` is a ``[B, bands, N]`` complex64 tensor (the batch scalogram
    already computed in :meth:`Wavelet2Vec.embed_batch`). Operation order and
    dtypes mirror the numpy paths exactly — float32 temporaries, float64
    accumulation — so per-clip results match to float precision. Returns
    ``(spectral [B, 2·bands], if_features [B, 2·n_groups], onset [B, 3])``
    as numpy float32.
    """
    import torch

    from wavelet2vec.phase_features import group_if_stats

    magnitude = coeffs.abs()  # float32 [B, bands, N]

    log_mag = torch.log1p(magnitude)
    spectral = torch.cat(
        [log_mag.mean(dim=2), log_mag.std(dim=2, correction=0)], dim=1
    ).cpu().numpy().astype(np.float32)

    increment = coeffs[:, :, 1:] * coeffs[:, :, :-1].conj()
    inst_freq = torch.angle(increment) * np.float32(sample_rate / (2.0 * np.pi))
    weights = magnitude[:, :, :-1] ** 2
    centers_t = torch.from_numpy(
        centers.astype(np.float32)).to(coeffs.device)[None, :, None]
    deviation = torch.clamp((inst_freq - centers_t) / centers_t, -1.0, 1.0)
    band_weight = weights.sum(dim=2, dtype=torch.float64) + 1e-12
    weighted_dev = weights * deviation
    # w ≥ 0 makes w·|d| = |w·d| (mad reuses the mean's product), and the
    # variance uses the one-pass identity E[d²] − E[d]² — two fewer full
    # passes over the [B, bands, N] tensors; float64 accumulation keeps the
    # identity numerically safe (matches the numpy two-pass path to ~1e-7).
    band_mad = weighted_dev.abs().sum(dim=2, dtype=torch.float64) / band_weight
    band_mean = weighted_dev.sum(dim=2, dtype=torch.float64) / band_weight
    band_sq = (weighted_dev * deviation).sum(dim=2, dtype=torch.float64) / band_weight
    band_std = torch.sqrt(torch.clamp(band_sq - band_mean ** 2, min=0.0))
    band_weight_np = band_weight.cpu().numpy()
    band_mad_np = band_mad.cpu().numpy()
    band_std_np = band_std.cpu().numpy()
    if_features = np.stack([
        group_if_stats(band_weight_np[i], band_mad_np[i], band_std_np[i],
                       n_groups)
        for i in range(coeffs.shape[0])
    ])

    phasors = coeffs / (magnitude + 1e-12)
    active = magnitude > 0.05 * magnitude.amax(dim=(1, 2), keepdim=True)
    n_active = active.sum(dim=1)  # [B, N]
    coherence = (phasors * active.to(phasors.dtype)).sum(dim=1).abs() \
        / n_active.clamp(min=1)
    coherence = coherence.masked_fill(n_active < 4, 0.0)
    onset = torch.stack([
        coherence.amax(dim=1),
        coherence.mean(dim=1),
        (coherence > 0.5).float().mean(dim=1),
    ], dim=1).cpu().numpy().astype(np.float32)

    return spectral, if_features.astype(np.float32), onset

#: Version tag of the embedding layout (roadmap D-7). Bump when the section
#: layout or the features feeding it change; indexes and standardizers record
#: it so stale artifacts are detected instead of silently mixed.
EMBEDDING_VERSION = "w2v570-v1"


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
        self.version = EMBEDDING_VERSION

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
        conv = (self.conv_encoder.encode_mono(mono)
                if self.conv_encoder is not None else None)
        return self._components_from(mono, stereo, complex_scalogram, conv)

    def _components_from(self, mono, stereo, complex_scalogram,
                         conv: np.ndarray | None, *,
                         spectral: np.ndarray | None = None,
                         if_features: np.ndarray | None = None,
                         onset_coherence: np.ndarray | None = None,
                         ) -> dict[str, np.ndarray]:
        """Feature sections from a prepared mono + precomputed scalogram —
        the shared core of :meth:`embed_components` and :meth:`embed_batch`.
        The keyword blocks accept batched-torch precomputations (D-1b)."""
        config = self.config
        scalogram = np.abs(complex_scalogram).astype(np.float32)
        components = {
            "spectral": (spectral if spectral is not None
                         else spectral_features(scalogram)),
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
                if_features=if_features,
                onset_coherence=onset_coherence,
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
        if conv is not None:
            components["conv"] = conv
        return components

    def _finalize(self, components: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for name in self.sections:
            vector = components[name]
            vector = vector / max(np.linalg.norm(vector), 0.1)
            parts.append(vector * float(self.config.section_weights.get(name, 1.0)))
        return np.concatenate(parts).astype(np.float32)

    def embed_batch(self, waveforms, sample_rate: int, *, chunk: int = 8,
                    device: str | None = None) -> np.ndarray:
        """Embeds a batch of equal-length fragments in one pass → ``[B, dim]``.

        The scalograms are computed batched on the torch frame (identical
        Gaussian windows to the numpy path) and the conv section in one
        forward — the two hot spots of :meth:`embed`. Fragments are
        ``[B, N]`` mono or a list of equal-shaped arrays; results match
        :meth:`embed` per clip. ``chunk`` bounds the batched-FFT memory.

        ``device`` (e.g. ``"cuda"``) runs the scalograms and the batched
        statistics there; ``None`` uses the CPU. The per-clip feature tail
        and the conv section stay on CPU either way.
        """
        config = self.config
        prepared = [self._prepare(np.asarray(w), sample_rate) for w in waveforms]
        monos = [mono for mono, _stereo in prepared]
        lengths = {mono.shape[0] for mono in monos}
        if len(lengths) != 1:
            return np.stack([  # ragged: honest per-clip fallback
                self.embed(np.asarray(w), sample_rate) for w in waveforms
            ])
        rate = config.analysis_sample_rate

        try:
            import torch

            from wavelet2vec.filterbank import constant_q_sigmas

            n = monos[0].shape[0]
            freqs = np.fft.rfftfreq(n, d=1.0 / rate)
            sigmas = constant_q_sigmas(self._centers, config.bandwidth_scale)
            windows = np.exp(
                -0.5 * ((freqs[None, :] - self._centers[:, None])
                        / np.maximum(sigmas[:, None], 1e-6)) ** 2
            )
            torch_device = torch.device(device) if device is not None else None
            windows_t = torch.from_numpy(windows.astype(np.float32))
            if torch_device is not None:
                windows_t = windows_t.to(torch_device)
            half = freqs.shape[0]
            scalograms: list[np.ndarray] = []
            spectral_pre: list[np.ndarray] | None = []
            if_pre: list[np.ndarray] | None = []
            onset_pre: list[np.ndarray] | None = []
            with torch.no_grad():
                for start in range(0, len(monos), max(chunk, 1)):
                    block = torch.from_numpy(
                        np.stack(monos[start: start + chunk]).astype(np.float32))
                    if torch_device is not None:
                        block = block.to(torch_device)
                    spectrum = torch.fft.rfft(block, dim=-1)
                    full = torch.zeros(block.shape[0], len(self._centers), n,
                                       dtype=torch.complex64, device=block.device)
                    full[..., :half] = spectrum[:, None, :] * windows_t * 2.0
                    coeffs = torch.fft.ifft(full, dim=-1)
                    scalograms.extend(coeffs.cpu().numpy())
                    # D-1b: the scalogram-statistics sections, batched while
                    # the coefficients are hot — the per-clip numpy tail.
                    spec_block, if_block, onset_block = _batched_scalogram_stats(
                        coeffs, self._centers, rate, config.n_phase_groups)
                    spectral_pre.extend(spec_block)
                    if_pre.extend(if_block)
                    onset_pre.extend(onset_block)
        except Exception:  # torch missing: numpy per-clip scalograms
            if device is not None:
                raise  # an explicit device request must not silently degrade
            spectral_pre = if_pre = onset_pre = None
            scalograms = [
                morlet_scalogram(mono, rate, self._centers,
                                 bandwidth_scale=config.bandwidth_scale,
                                 return_complex=True)
                for mono in monos
            ]

        convs: list[np.ndarray | None] = [None] * len(monos)
        if self.conv_encoder is not None:
            import torch

            with torch.no_grad():
                batch = torch.from_numpy(
                    np.stack(monos).astype(np.float32)).unsqueeze(1)
                convs = list(self.conv_encoder.forward(batch).numpy())

        vectors = []
        for i, ((mono, stereo), scalogram, conv) in enumerate(
                zip(prepared, scalograms, convs)):
            if np.abs(mono).max() <= 1e-8:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
                continue
            components = self._components_from(
                mono, stereo, scalogram, conv,
                spectral=None if spectral_pre is None else spectral_pre[i],
                if_features=None if if_pre is None else if_pre[i],
                onset_coherence=None if onset_pre is None else onset_pre[i])
            vectors.append(self._finalize(components))
        return np.stack(vectors)

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
