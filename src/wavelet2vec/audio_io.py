from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency behavior
    sf = None

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - optional dependency behavior
    resample_poly = None


AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg"}


def list_audio_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Loads an audio file as ``[channels, samples]`` float32 plus sample rate."""
    if sf is None:
        raise RuntimeError("Audio loading requires the soundfile package.")
    data, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    return data.T, int(sample_rate)


def save_audio(
    path: str | Path,
    waveform: np.ndarray,
    sample_rate: int,
    subtype: str | None = None,
) -> Path:
    """Saves ``[channels, samples]`` audio to a file.

    The whole processing chain is floating point, so by default WAV/AIFF
    output is written as 32-bit float (no quantization on the way out);
    formats without float support (e.g. FLAC) fall back to 24-bit. Pass
    ``subtype`` (e.g. ``"PCM_24"``, ``"PCM_16"``) to override.
    """
    if sf is None:
        raise RuntimeError("Audio saving requires the soundfile package.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if subtype is None:
        format_name = output_path.suffix.lstrip(".").upper()
        format_name = {"AIF": "AIFF"}.get(format_name, format_name)
        for candidate in ("FLOAT", "PCM_24"):
            if sf.check_format(format_name, candidate):
                subtype = candidate
                break

    sf.write(output_path, np.asarray(waveform, dtype=np.float32).T, sample_rate, subtype=subtype)
    return output_path


def resample(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling of ``[channels, samples]`` audio."""
    if source_sr == target_sr:
        return waveform
    if resample_poly is None:
        raise RuntimeError("Resampling requires scipy.")
    ratio_gcd = math.gcd(source_sr, target_sr)
    up = target_sr // ratio_gcd
    down = source_sr // ratio_gcd
    return np.stack([resample_poly(channel, up, down) for channel in waveform]).astype(
        waveform.dtype
    )
