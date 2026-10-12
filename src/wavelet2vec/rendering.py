"""Bounded offline source rendering and staged floating-point file exports."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import soundfile as sf
import torch

from wavelet2vec.feature_cache import model_fingerprint, source_fingerprint
from wavelet2vec.synthesis import HybridSeparator, SourceSynthesis


class SeparationCancelled(RuntimeError):
    """Rendering stopped at a window/write boundary before publication."""


@dataclass(frozen=True)
class SourceRenderConfig:
    window_samples: int
    overlap_samples: int
    context_samples: int = 0
    max_buffer_bytes: int = 256 * 1024**2

    def __post_init__(self):
        for key in ("window_samples", "overlap_samples", "context_samples", "max_buffer_bytes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.window_samples == 0 or self.max_buffer_bytes == 0 or self.overlap_samples >= self.window_samples:
            raise ValueError("require positive window/budget and 0 <= overlap < window")

    @classmethod
    def from_seconds(cls, window, overlap, context=0., *, sample_rate, hop_length, **kwargs):
        for value in (sample_rate, hop_length):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("sample_rate and hop_length must be positive integers")
        if any(not math.isfinite(v) or v < 0 for v in (window, overlap, context)) or window == 0:
            raise ValueError("durations must be finite, with positive window and nonnegative overlap/context")
        return cls(*(math.ceil(v*sample_rate/hop_length)*hop_length for v in (window, overlap, context)), **kwargs)

    def to_dict(self):
        return {"version": "source-render-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("version", None) != "source-render-v1":
            raise ValueError("unsupported source render configuration")
        return cls(**data)

    def buffer_bytes(self, *, batch_size, sources, channels, dtype):
        """Conservative explicit rolling-buffer/reader payload, not total RAM.

        Includes output/shift temporaries, reader audio, core weights and model
        waveform output. Excludes parameters, carriers and neural/FFT workspaces.
        """
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("rendering supports float32 or float64")
        for value in (batch_size, sources, channels):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("batch, source and channel counts must be positive integers")
        width, read = self.window_samples, self.window_samples+2*self.context_samples
        element = 4 if dtype == torch.float32 else 8
        return element*(batch_size*channels*(4*sources*width + (sources+2)*read) + 4*width) + 24*width


def _cancel(callback):
    if callback is not None and callback():
        raise SeparationCancelled("source rendering cancelled")


def _weights(length, reference):
    # Form coordinates in float64: long float32 windows can round the final
    # midpoint to 1 and lose a strictly positive edge weight.
    phase = (torch.arange(length, dtype=torch.float64, device=reference.device)+.5)/length
    return torch.sin(torch.pi*phase).square().to(reference.dtype)


def iter_separate_reader(model: HybridSeparator, reader, n_samples: int, config: SourceRenderConfig, *,
                         sample_origin: int = 0, branches: str = "both", cancel=None):
    """Yield completed [B,S,C,Nchunk] intervals using bounded rolling buffers.

    reader(lo, hi) returns [B,C,hi-lo] on CPU or the model device, in model
    precision. Reads include real context. Chunks own their memory; retaining
    them increases caller memory. Model modes and grad state are restored
    before every yield, so abandoning the iterator leaves no changed modes.
    This is offline windowed processing, not causal/live separation.
    """
    if not callable(reader) or (cancel is not None and not callable(cancel)):
        raise ValueError("reader and optional cancel must be callable")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(sample_origin, bool) or not isinstance(sample_origin, int):
        raise ValueError("sample_origin must be an integer")
    hop = model.encoder.frontend.config.hop_length
    if any(value % hop for value in (config.window_samples, config.overlap_samples, config.context_samples)):
        raise ValueError("render windows, overlap and context must be aligned to hop_length")
    if branches not in ("both", "waveform", "spectral"):
        raise ValueError("unknown feature branches")
    parameter = next(model.parameters())
    sources, channels = len(model.config.source_names), model.config.channels
    def guard(batch):
        size = config.buffer_bytes(batch_size=batch, sources=sources, channels=channels, dtype=parameter.dtype)
        if size > config.max_buffer_bytes:
            raise ValueError(f"render buffer estimate {size} exceeds max_buffer_bytes")
    guard(1)  # reject impossible requests before invoking a reader
    width, step = config.window_samples, config.window_samples-config.overlap_samples
    accumulation = denominator = None
    batch_size = None
    for start in range(0, n_samples, step):
        _cancel(cancel)
        end = min(n_samples, start+width)
        left, right = max(0, start-config.context_samples), min(n_samples, end+config.context_samples)
        with torch.no_grad():
            audio = reader(left, right)
            if (not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] <= 0
                    or audio.shape[1:] != (channels, right-left) or audio.dtype != parameter.dtype):
                raise ValueError("reader must return [B,model channels,hi-lo] in model precision")
            if batch_size is not None and audio.shape[0] != batch_size:
                raise ValueError("reader batch size changed")
            batch_size = audio.shape[0]
            guard(batch_size)
            audio = audio.to(parameter.device)
            if not torch.isfinite(audio).all():
                raise ValueError("reader returned nonfinite audio")
            _cancel(cancel)
            modes = [(module, module.training) for module in model.modules()]
            try:
                model.eval()
                predicted = model(audio, sample_origin=sample_origin+left, branches=branches).waveform
            finally:
                for module, training in modes:
                    module.training = training
            if (predicted.shape != (batch_size, sources, channels, right-left)
                    or predicted.dtype != parameter.dtype or predicted.device != parameter.device):
                raise ValueError("model output shape, precision or device differs from render contract")
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("nonfinite source prediction")
            _cancel(cancel)
            core = predicted[..., start-left:end-left]
            if accumulation is None:
                accumulation = audio.new_zeros(batch_size, sources, channels, width)
                denominator = audio.new_zeros(width)
            length = end-start
            weights = _weights(length, audio)
            accumulation[..., :length].add_(core*weights)
            denominator[:length].add_(weights)
            emit = length if end == n_samples else step
            if not (denominator[:emit] > 0).all():
                raise FloatingPointError("invalid overlap-add normalization")
            waveform = accumulation[..., :emit] / denominator[:emit]
            if not torch.isfinite(waveform).all():
                raise FloatingPointError("nonfinite overlap-add output")
            remaining = width-emit
            if remaining:
                accumulation[..., :remaining].copy_(accumulation[..., emit:].clone())
                denominator[:remaining].copy_(denominator[emit:].clone())
            accumulation[..., remaining:].zero_()
            denominator[remaining:].zero_()
            output = SourceSynthesis(waveform, torch.full((batch_size,), emit, device=parameter.device, dtype=torch.long),
                                     model.config.source_names, model.encoder.frontend.config.sample_rate, sample_origin+start)
            # Do not hold input/carrier output views across suspension.
            del audio, predicted, core, weights, waveform
        yield output
        if end == n_samples:
            break


@dataclass(frozen=True)
class FileSeparationResult:
    directory: Path
    files: tuple[Path, ...]
    manifest: Path
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    container: str


def _container(frames, channels, subtype, requested):
    if requested not in ("auto", "WAV", "RF64"):
        raise ValueError("container must be auto, WAV or RF64")
    # Reserve conservative header space rather than approach RIFF's 32-bit
    # length limit. RF64 retains .wav filenames and floating-point subtypes.
    large = frames*channels*(4 if subtype == "FLOAT" else 8) >= 2**32-1024**2
    if requested == "WAV" and large:
        raise ValueError("output exceeds conservative WAV capacity; choose RF64 or auto")
    result = "RF64" if requested == "auto" and large else "WAV" if requested == "auto" else requested
    if not sf.check_format(result, subtype):
        raise ValueError("requested floating-point container is unavailable")
    return result


def separate_file(model: HybridSeparator, source_path, output_directory, config: SourceRenderConfig, *,
                  subtype: str = "FLOAT", container: str = "auto", branches: str = "both",
                  cancel=None, progress=None, provenance=None) -> FileSeparationResult:
    """Render a native-rate file into a new, atomically published stem folder.

    FLOAT requires a float32 model; DOUBLE requires float64. No resampling,
    normalization, clipping or integer fallback. Cancellation/failure cleans
    only this operation's staging directory. Existing destinations are refused.
    Source names are metadata, never filesystem paths. Reads/writes and model
    activations are window-bounded; retaining no full-track waveforms is needed.
    """
    if subtype not in ("FLOAT", "DOUBLE"):
        raise ValueError("source export requires FLOAT or DOUBLE subtype")
    provenance = json.loads(json.dumps({} if provenance is None else provenance, allow_nan=False))
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    parameter = next(model.parameters())
    dtype = torch.float32 if subtype == "FLOAT" else torch.float64
    if parameter.dtype != dtype:
        raise ValueError("file subtype must match model precision explicitly")
    if (cancel is not None and not callable(cancel)) or (progress is not None and not callable(progress)):
        raise ValueError("cancel and progress must be callable")
    source_path, destination = Path(source_path).resolve(), Path(output_directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("output directory must be new; existing destinations are not overwritten")
    _cancel(cancel)
    def snapshot():
        stat = source_path.stat()
        return stat.st_size, stat.st_mtime_ns
    original = snapshot()
    digest = source_fingerprint(source_path)
    if snapshot() != original:
        raise ValueError("source changed while hashing")
    with sf.SoundFile(source_path) as source:
        rate, channels, frames = source.samplerate, source.channels, len(source)
        if rate != model.encoder.frontend.config.sample_rate or channels != model.config.channels:
            raise ValueError("file rate/channels differ from model; no implicit conversion")
        if frames <= 0:
            raise ValueError("source file must contain samples")
        format_name = _container(frames, channels, subtype, container)
        def reader(lo, hi):
            if snapshot() != original:
                raise ValueError("source changed during rendering")
            source.seek(lo)
            value = source.read(hi-lo, dtype="float32" if dtype == torch.float32 else "float64", always_2d=True)
            if snapshot() != original:
                raise ValueError("source changed during reading")
            return torch.from_numpy(value.T.copy())[None]
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            fingerprint = model_fingerprint(model)
        finally:
            for module, training in modes:
                module.training = training
        destination.parent.mkdir(parents=True, exist_ok=True)
        names = tuple(f"source_{i:03d}.wav" for i in range(len(model.config.source_names)))
        with tempfile.TemporaryDirectory(prefix=".wavelet2vec-render-", dir=destination.parent) as temporary:
            staging = Path(temporary)
            emitted = 0
            with ExitStack() as stack:
                writers = [stack.enter_context(sf.SoundFile(staging/name, mode="w", samplerate=rate,
                            channels=channels, format=format_name, subtype=subtype)) for name in names]
                for chunk in iter_separate_reader(model, reader, frames, config, branches=branches, cancel=cancel):
                    if chunk.sample_origin != emitted:
                        raise ValueError("noncontiguous source render output")
                    values = chunk.waveform[0].cpu().numpy()
                    for writer, value in zip(writers, values):
                        writer.write(value.T)
                    emitted += values.shape[-1]
                    if progress is not None:
                        progress(emitted, frames)
            _cancel(cancel)
            if emitted != frames or snapshot() != original:
                raise ValueError("source length changed or render did not complete")
            for name in names:
                info = sf.info(staging/name)
                if (info.frames, info.samplerate, info.channels, info.subtype, info.format) != (frames, rate, channels, subtype, format_name):
                    raise ValueError("export header does not match source render")
            receipt = {"version": "source-file-render-v1", "source_sha256": digest,
                       "source_name": source_path.name, "model": fingerprint, "render": config.to_dict(),
                       "branches": branches, "sample_rate": rate, "channels": channels, "frames": frames,
                       "subtype": subtype, "container": format_name, "provenance": provenance,
                       "sources": [{"name": label, "file": name} for label, name in zip(model.config.source_names, names)]}
            (staging/"render.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
            _cancel(cancel)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("output destination appeared during rendering")
            os.rename(staging, destination)
    return FileSeparationResult(destination, tuple(destination/name for name in names), destination/"render.json",
                                rate, channels, frames, subtype, format_name)
