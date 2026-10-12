"""Keyboard generation: from one or two notes of a sound, generate its other notes.

KeyboardNet v1 follows the library's "deterministic carrier + learned
envelopes" pattern (as in the brightness and transient axes):

- Envelopes: a note becomes log band energies on a semitone grid, one band
  per MIDI note from MIDI 12 (16.35 Hz) to MIDI 135 (19.9 kHz). Frames are
  10 ms over the 1.2 s note, analysed at 48 kHz with the invertible frame.
- Carrier: an example note is pitch-shifted to the target by heterodyning in
  the same frame. With semitone-spaced bands, a shift by d semitones moves
  the envelope by exactly d bands. So envelopes are shifted, not audio, and
  training needs no audio processing.
- Learned part: a small FiLM-conditioned 2-D CNN sees the shifted example
  envelope(s), their validity masks and a baseline. It predicts a dB
  correction to that baseline: how this sound's brightness, decay, level and
  spectral shape change at the target key. With two examples it also sees
  the trend between them.
- Audio: the shifted carrier (nearest example) is reshaped band by band and
  frame by frame toward the predicted envelope, then resynthesized.

v1 models 16 Hz-20 kHz at 48 kHz. Content above 20 kHz and native 192 kHz
output are a later tier.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wavelet2vec.frame import analyze, synthesize

ENV_RATE = 48000
ENV_FIRST_MIDI = 12
ENV_BANDS = 124            # MIDI 12..135
ENV_HOP = 480              # 10 ms at 48 kHz
NOTE_SAMPLES = 57600       # 1.2 s at 48 kHz
ENV_FRAMES = NOTE_SAMPLES // ENV_HOP
FLOOR_DB = -120.0
DB_SCALE = 40.0            # network inputs/outputs are dB / DB_SCALE


def midi_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def note_frame(audio, sample_rate=ENV_RATE, n_bands=None):
    """Semitone frame (bands at MIDI 12, 13, ...) up to 0.45·rate unless n_bands is given."""
    if n_bands is None:
        top = 69 + 12 * np.log2(0.45 * sample_rate / 440.0)
        n_bands = int(np.floor(top)) - ENV_FIRST_MIDI + 1
    return analyze(audio, sample_rate, n_bands=n_bands, f_min=float(midi_hz(ENV_FIRST_MIDI)),
                   f_max=float(midi_hz(ENV_FIRST_MIDI + n_bands - 1)))


def note_envelope(note_48k: np.ndarray) -> np.ndarray:
    """``[ENV_BANDS, ENV_FRAMES]`` log band energy (dB) of a mono 1.2 s note at 48 kHz."""
    note = np.zeros(NOTE_SAMPLES)
    note[:min(NOTE_SAMPLES, note_48k.shape[-1])] = note_48k[:NOTE_SAMPLES]
    coeffs, _ = note_frame(note, ENV_RATE, ENV_BANDS)
    power = (np.abs(coeffs) ** 2).reshape(ENV_BANDS, ENV_FRAMES, ENV_HOP).mean(-1)
    return np.maximum(10.0 * np.log10(power + 1e-30), FLOOR_DB).astype(np.float32)


def shift_envelope(envelope, semitones):
    """Move an envelope by whole semitones; returns (shifted, valid mask)."""
    shifted = np.full_like(envelope, FLOOR_DB)
    valid = np.zeros(envelope.shape, dtype=np.float32)
    d = int(semitones)
    if d >= 0:
        shifted[d:] = envelope[:envelope.shape[0] - d]
        valid[d:] = 1.0
    else:
        shifted[:d] = envelope[-d:]
        valid[:d] = 1.0
    return shifted, valid


def baseline_envelope(shifted, valid, distances):
    """Distance-weighted mean of the shifted examples (the nearest dominates)."""
    weights = np.array([1.0 / (1.0 + abs(d)) for d in distances], dtype=np.float32)[:, None, None] * valid
    total = weights.sum(0)
    blended = (weights * shifted).sum(0) / np.maximum(total, 1e-9)
    nearest = shifted[int(np.argmin([abs(d) for d in distances]))]
    return np.where(total > 0, blended, nearest).astype(np.float32)


def network_inputs(examples, example_midis, target_midi):
    """Inputs for 1 or 2 examples: (planes [7, bands, frames], conditioning [6])."""
    distances = [target_midi - m for m in example_midis]
    shifted, valid = zip(*(shift_envelope(e, d) for e, d in zip(examples, distances)))
    shifted, valid = np.stack(shifted), np.stack(valid)
    base = baseline_envelope(shifted, valid, distances)
    if len(examples) == 1:
        shifted = np.concatenate([shifted, np.full_like(shifted, FLOOR_DB)])
        valid = np.concatenate([valid, np.zeros_like(valid)])
        distances = distances + [0]
    band = np.broadcast_to(np.linspace(0.0, 1.0, ENV_BANDS, dtype=np.float32)[:, None], base.shape)
    planes = np.stack([shifted[0] / DB_SCALE, valid[0], shifted[1] / DB_SCALE, valid[1],
                       base / DB_SCALE, band, (shifted[0] - shifted[1]) / DB_SCALE * valid[0] * valid[1]])
    cond = np.array([target_midi / 127.0, distances[0] / 48.0, distances[1] / 48.0, float(len(examples) == 2),
                     abs(distances[0]) / 48.0, abs(distances[1]) / 48.0], dtype=np.float32)
    return planes.astype(np.float32), cond, base


@dataclass
class KeyboardConfig:
    channels: tuple[int, ...] = (16, 48, 96)  # narrow at full resolution: CPU training speed
    cond_dim: int = 6
    in_planes: int = 7
    max_correction_db: float = 40.0
    version: str = "keyboard-v1"


def _torch():
    import torch
    from torch import nn
    return torch, nn


def build_model(config: KeyboardConfig | None = None):
    torch, nn = _torch()
    config = config or KeyboardConfig()

    class FiLMBlock(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.norm = nn.GroupNorm(8, cout)
            self.film = nn.Linear(64, 2 * cout)
            self.skip = nn.Conv2d(cin, cout, 1, stride=stride)

        def forward(self, x, c):
            h = torch.nn.functional.gelu(self.conv1(x))
            h = self.norm(self.conv2(h))
            gamma, beta = self.film(c).chunk(2, dim=-1)
            h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
            return torch.nn.functional.gelu(h + self.skip(x))

    class KeyboardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            ch = config.channels
            self.cond = nn.Sequential(nn.Linear(config.cond_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            self.stem = FiLMBlock(config.in_planes, ch[0], 1)
            self.down1 = FiLMBlock(ch[0], ch[1], 2)
            self.down2 = FiLMBlock(ch[1], ch[2], 2)
            self.mid = FiLMBlock(ch[2], ch[2], 1)
            self.up2 = FiLMBlock(ch[2] + ch[1], ch[1], 1)
            self.up1 = FiLMBlock(ch[1] + ch[0], ch[0], 1)
            self.head = nn.Conv2d(ch[0], 1, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)  # starts as the baseline

        def forward(self, planes, cond):
            c = self.cond(cond)
            s0 = self.stem(planes, c)
            s1 = self.down1(s0, c)
            h = self.mid(self.down2(s1, c), c)
            h = torch.nn.functional.interpolate(h, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up2(torch.cat([h, s1], 1), c)
            h = torch.nn.functional.interpolate(h, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up1(torch.cat([h, s0], 1), c)
            limit = config.max_correction_db / DB_SCALE
            correction = limit * torch.tanh(self.head(h)[:, 0] / limit)
            return planes[:, 4] + correction  # baseline + correction, in dB / DB_SCALE

    return KeyboardNet()


def save_keyboard_model(path, model, extra=None):
    torch, _ = _torch()
    payload = {"config": asdict(model.config), "state": model.state_dict(), "extra": extra or {}}
    torch.save(payload, Path(path))


def load_keyboard_model(path):
    torch, _ = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = KeyboardConfig(**{k: tuple(v) if k == "channels" else v for k, v in payload["config"].items()})
    model = build_model(config)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload.get("extra", {})


def predict_envelope(model, examples, example_midis, target_midi):
    """Predicted target envelope (dB) from 1-2 example envelopes; also returns the baseline."""
    torch, _ = _torch()
    planes, cond, base = network_inputs(list(examples), list(example_midis), int(target_midi))
    with torch.no_grad():
        out = model(torch.from_numpy(planes)[None], torch.from_numpy(cond)[None])[0].numpy()
    return out * DB_SCALE, base


def render_note(example_audio, example_midi, target_midi, target_envelope, sample_rate=ENV_RATE,
                max_gain_db=30.0):
    """Shift ``example_audio`` (``[channels, samples]``, 1.2 s) to ``target_midi`` and reshape it
    toward ``target_envelope``. Returns ``[channels, samples]`` at ``sample_rate``."""
    audio = np.atleast_2d(np.asarray(example_audio, dtype=np.float64))
    d = int(target_midi) - int(example_midi)
    out = []
    t = np.arange(audio.shape[1]) / sample_rate
    for channel in audio:
        coeffs, spec = note_frame(channel, sample_rate)
        shifted = coeffs.copy()  # bands left empty by the shift keep the example's own content
        n = coeffs.shape[0]
        for band in range(n):
            target = band + d
            if 0 <= target < n:
                delta = spec.centers[target] - spec.centers[band]
                shifted[target] = coeffs[band] * np.exp(2j * np.pi * delta * t)
        out.append((shifted, spec))
    # Gains from the mid-channel envelope of the shifted carrier, on the model grid
    # (bands are MIDI-aligned at any rate; a 1.2 s note always has ENV_FRAMES frames).
    hop = audio.shape[1] // ENV_FRAMES
    mid = sum(s[0] for s in out) / len(out)
    carrier_db = 10 * np.log10((np.abs(mid[:ENV_BANDS, :hop * ENV_FRAMES]) ** 2).reshape(ENV_BANDS, ENV_FRAMES, hop).mean(-1) + 1e-30)
    gain_db = np.clip(np.asarray(target_envelope) - np.maximum(carrier_db, FLOOR_DB), -max_gain_db, max_gain_db)
    centres = (np.arange(ENV_FRAMES) + 0.5) * hop
    sample_index = np.arange(audio.shape[1])
    gains = np.stack([np.interp(sample_index, centres, g) for g in gain_db])
    result = []
    for shifted, spec in out:
        shaped = shifted.copy()
        shaped[:ENV_BANDS] *= 10.0 ** (gains / 20.0)
        result.append(synthesize(shaped, spec)[:audio.shape[1]])
    return np.stack(result)


def describe(model) -> dict:
    return {"config": asdict(model.config), "parameters": int(sum(p.numel() for p in model.parameters()))}


def to_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
