"""Training-data tooling for the imagination axes (roadmap G-1).

Produces graded, self-supervised pairs and ladders whose control labels are
computed by the *same* features the models will be conditioned on:

- ``axis_labels`` — brightness / harmonicity / attack, measured, not assumed;
- ``lowpass_pair`` — (band-limited input, full target) for brightness training;
- ``filter_ladder`` — the same synth pluck at rising filter cutoffs;
- ``sine_distortion_ladder`` — a pure tone at rising drive;
- ``noise_mix_ladder`` — tonal→noisy gradations for the tonality axis;
- ``attack_ladder`` — the same tone with rising amplitude-attack times;
- ``find_ladder_files`` — ingest user-rendered ladders named
  ``name__cut30.wav`` / ``name__drive0.7.wav`` (double-underscore + key +
  number);
- ``render_ladder_dataset`` — write a synthetic corpus + ``manifest.json``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from wavelet2vec.audio_io import save_audio
from wavelet2vec.filterbank import analytic_envelope
from wavelet2vec.frame import analyze
from wavelet2vec.harmonic_features import autocorrelation_pitch

_LADDER_RE = re.compile(r"__([a-z]+)([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Axis labels — measured with the features the models condition on
# --------------------------------------------------------------------------
def axis_labels(audio: np.ndarray, sample_rate: int, *, hf_split_hz: float = 4000.0) -> dict:
    """Measured control labels: brightness, harmonicity, attack_ms."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio / peak

    coeffs, spec = analyze(audio, sample_rate, n_bands=64)
    energies = (np.abs(coeffs) ** 2).mean(axis=1)
    high = energies[spec.centers >= hf_split_hz].sum()
    brightness = float(high / (energies.sum() + 1e-24))

    harmonicity, _f0 = autocorrelation_pitch(audio, sample_rate)

    envelope = analytic_envelope(audio, smooth_samples=max(int(sample_rate * 0.005), 1))
    peak_env = envelope.max() + 1e-12
    above_low = np.flatnonzero(envelope >= 0.1 * peak_env)
    above_high = np.flatnonzero(envelope >= 0.9 * peak_env)
    attack_ms = 0.0
    if above_low.size and above_high.size:
        attack_ms = max(0.0, (above_high[0] - above_low[0]) / sample_rate * 1000.0)

    return {"brightness": brightness, "harmonicity": float(harmonicity), "attack_ms": attack_ms}


# --------------------------------------------------------------------------
# Pair/ladder generators
# --------------------------------------------------------------------------
def lowpass_pair(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float, *, transition_octaves: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """(band-limited input, full-band target) — phase-coherent, in-domain.

    The lowpass is a smooth per-band mask in the coefficient domain (cosine
    rolloff over ``transition_octaves``), so the kept region matches the
    target exactly — precisely the supervision BWE training wants.
    """
    audio = np.asarray(audio, dtype=np.float64)
    coeffs, spec = analyze(audio, sample_rate, n_bands=128)
    octaves = np.log2(spec.centers / cutoff_hz)
    mask = np.clip(0.5 - 0.5 * np.sin(np.pi * np.clip(octaves / transition_octaves, -0.5, 0.5)), 0.0, 1.0)
    from wavelet2vec.frame import synthesize

    limited = synthesize(coeffs * mask[:, None], spec)[: audio.shape[0]]
    return limited, audio


def synth_pluck(
    f0: float,
    sample_rate: int,
    cutoff_hz: float,
    *,
    seconds: float = 1.0,
    resonance: float = 2.0,
    decay_per_second: float = 5.0,
) -> np.ndarray:
    """A simple subtractive pluck: band-limited saw → resonant lowpass → decay.

    Keep ``cutoff_hz`` below ~0.3 · ``sample_rate``: the RBJ biquad's response
    warps near Nyquist (render at 44.1 kHz for cutoffs up to ~10 kHz).
    """
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    saw = np.zeros(n)
    for k in range(1, int(0.45 * sample_rate / f0) + 1):
        saw += np.sin(2 * math.pi * k * f0 * t) / k
    # RBJ resonant lowpass biquad.
    w0 = 2 * math.pi * cutoff_hz / sample_rate
    alpha = math.sin(w0) / (2 * resonance)
    b = [(1 - math.cos(w0)) / 2, 1 - math.cos(w0), (1 - math.cos(w0)) / 2]
    a = [1 + alpha, -2 * math.cos(w0), 1 - alpha]
    filtered = lfilter(np.asarray(b) / a[0], np.asarray(a) / a[0], saw)
    pluck = filtered * np.exp(-t * decay_per_second)
    return pluck / (np.abs(pluck).max() + 1e-12)


def filter_ladder(
    f0: float, sample_rate: int, cutoffs_hz: list[float], **kwargs
) -> list[tuple[np.ndarray, float]]:
    """Same patch, rising filter cutoff — the brightness axis, musically."""
    return [(synth_pluck(f0, sample_rate, cutoff, **kwargs), cutoff) for cutoff in cutoffs_hz]


def sine_distortion_ladder(
    f0: float, sample_rate: int, drives: list[float], *, seconds: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """A sine at rising tanh drive — harmonic generation from a pure tone."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    sine = np.sin(2 * math.pi * f0 * t)
    out = []
    for drive in drives:
        shaped = np.tanh(drive * sine) / math.tanh(drive)
        out.append((shaped * np.exp(-t * 3.0), drive))
    return out


def noise_mix_ladder(
    f0: float, sample_rate: int, noise_mixes: list[float], *, seconds: float = 1.0, seed: int = 0
) -> list[tuple[np.ndarray, float]]:
    """Tonal→noisy gradations: the tonality axis. ``noise_mix`` in [0, 1]."""
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    tone = sum(np.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 7))
    tone = tone / np.abs(tone).max()
    envelope = np.exp(-t * 3.0)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    noise = noise / np.abs(noise).max()
    out = []
    for mix in noise_mixes:
        blended = ((1.0 - mix) * tone + mix * noise) * envelope
        out.append((blended / (np.abs(blended).max() + 1e-12), mix))
    return out


def attack_ladder(
    f0: float, sample_rate: int, attacks_ms: list[float], *, seconds: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """Same tone, rising amplitude-attack time — the transient axis."""
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    tone = sum(np.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 9))
    tone = tone / np.abs(tone).max()
    out = []
    for attack_ms in attacks_ms:
        attack_n = max(1, int(attack_ms / 1000.0 * sample_rate))
        envelope = np.minimum(np.arange(n) / attack_n, 1.0) * np.exp(-t * 2.5)
        shaped = tone * envelope
        out.append((shaped / (np.abs(shaped).max() + 1e-12), attack_ms))
    return out


# --------------------------------------------------------------------------
# User-rendered ladders and corpus rendering
# --------------------------------------------------------------------------
def find_ladder_files(folder: str | Path) -> dict[str, dict[str, list[tuple[float, Path]]]]:
    """Groups ``base__key<value>.wav`` files: {base: {key: [(value, path), …]}}.

    Example: ``pluck__cut30.wav`` + ``pluck__cut70.wav`` → one 'cut' ladder.
    """
    ladders: dict[str, dict[str, list[tuple[float, Path]]]] = {}
    for path in sorted(Path(folder).glob("*.wav")):
        match = _LADDER_RE.search(path.stem)
        if not match:
            continue
        base = path.stem[: match.start()]
        key, value = match.group(1).lower(), float(match.group(2))
        ladders.setdefault(base, {}).setdefault(key, []).append((value, path))
    for base in ladders.values():
        for key in base:
            base[key].sort()
    return ladders


def render_ladder_dataset(
    out_dir: str | Path,
    sample_rate: int = 44100,
    *,
    f0s: tuple[float, ...] = (110.0, 220.0),
    cutoffs_hz: tuple[float, ...] = (800.0, 2000.0, 5000.0, 10000.0),
    drives: tuple[float, ...] = (0.5, 2.0, 8.0),
    noise_mixes: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9),
    attacks_ms: tuple[float, ...] = (1.0, 20.0, 80.0, 200.0),
    seconds: float = 1.0,
) -> Path:
    """Renders a synthetic ladder corpus + ``manifest.json`` with measured labels."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(kind: str, base: str, key: str, value: float, audio: np.ndarray) -> None:
        name = f"{base}__{key}{value:g}.wav"
        save_audio(out / name, audio[np.newaxis, :], sample_rate)
        manifest.append({
            "file": name, "kind": kind, "param_key": key, "param_value": value,
            "labels": axis_labels(audio, sample_rate),
        })

    for f0 in f0s:
        for audio, cutoff in filter_ladder(f0, sample_rate, list(cutoffs_hz), seconds=seconds):
            emit("filter", f"pluck{f0:g}", "cut", cutoff, audio)
        for audio, drive in sine_distortion_ladder(f0, sample_rate, list(drives), seconds=seconds):
            emit("distortion", f"sine{f0:g}", "drive", drive, audio)
        for audio, mix in noise_mix_ladder(f0, sample_rate, list(noise_mixes), seconds=seconds):
            emit("noise", f"tone{f0:g}", "mix", mix, audio)
        for audio, attack in attack_ladder(f0, sample_rate, list(attacks_ms), seconds=seconds):
            emit("attack", f"tone{f0:g}", "atk", attack, audio)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest_path
