"""Synthetic ground-truth scenes + weak labels (roadmap V-2).

The unfair advantage: render note/drum scenes where every onset, pitch,
duration and instrument is known exactly, so we get **images + perfect
labels at unlimited scale** — plus :func:`weak_labels` for turning real
audio into approximate labels from the deterministic DSP (onsets, f0).

An :class:`~wavelet2vec.vision.Event`'s ``label`` names the instrument/class
(``"sine" | "saw" | "pluck" | "kick" | "snare" | "hat"``) and ``pitch`` its
MIDI note; :func:`render_scene` reads them back to synthesize.
"""

from __future__ import annotations

import numpy as np

from wavelet2vec.vision import Event

DRUM_KINDS = ("kick", "snare", "hat")
PITCHED_KINDS = ("sine", "saw", "pluck")

# Frequency extent (Hz) of each drum's box.
_DRUM_BANDS = {"kick": (30.0, 130.0), "snare": (150.0, 1800.0), "hat": (4000.0, 13000.0)}


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _adsr(n: int, sample_rate: int, attack=0.005, decay=0.05, sustain=0.7,
          release=0.05) -> np.ndarray:
    env = np.ones(n)
    a = min(int(attack * sample_rate), n)
    d = min(int(decay * sample_rate), n - a)
    r = min(int(release * sample_rate), n)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if d:
        env[a:a + d] = np.linspace(1.0, sustain, d)
    env[a + d:] = sustain
    if r:
        env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


def render_note(pitch: int, duration_s: float, sample_rate: int, *,
                instrument: str = "saw", velocity: int = 100) -> np.ndarray:
    """A single pitched note. ``saw`` is bright (many harmonics), ``sine``
    pure, ``pluck`` a decaying harmonic stack."""
    n = max(int(duration_s * sample_rate), 1)
    t = np.arange(n) / sample_rate
    f0 = midi_to_hz(pitch)
    amp = velocity / 127.0

    if instrument == "sine":
        wave = np.sin(2 * np.pi * f0 * t)
    else:
        nyquist = 0.5 * sample_rate
        n_harm = max(int(nyquist / f0), 1)
        wave = np.zeros(n)
        for k in range(1, min(n_harm, 24) + 1):
            weight = 1.0 / k
            if instrument == "pluck":
                weight *= np.exp(-k * 0.15)  # rolls off high harmonics
            wave += weight * np.sin(2 * np.pi * k * f0 * t)
        wave /= np.abs(wave).max() + 1e-12

    if instrument == "pluck":
        env = np.exp(-t / max(duration_s * 0.4, 1e-3)) * _adsr(n, sample_rate, attack=0.002)
    else:
        env = _adsr(n, sample_rate)
    return (wave * env * amp).astype(np.float64)


def render_drum(kind: str, sample_rate: int, *, velocity: int = 100) -> np.ndarray:
    """A short percussion hit; kind in :data:`DRUM_KINDS`."""
    amp = velocity / 127.0
    rng = np.random.default_rng(abs(hash(kind)) % (2**32))
    if kind == "kick":
        n = int(0.18 * sample_rate)
        t = np.arange(n) / sample_rate
        sweep = 120.0 * np.exp(-t * 30.0) + 45.0
        wave = np.sin(2 * np.pi * np.cumsum(sweep) / sample_rate) * np.exp(-t * 18.0)
    elif kind == "snare":
        n = int(0.15 * sample_rate)
        t = np.arange(n) / sample_rate
        tone = np.sin(2 * np.pi * 180.0 * t)
        noise = rng.standard_normal(n)
        wave = (0.5 * tone + noise) * np.exp(-t * 22.0)
    else:  # hat
        n = int(0.05 * sample_rate)
        t = np.arange(n) / sample_rate
        noise = rng.standard_normal(n)
        wave = np.diff(noise, prepend=0.0) * np.exp(-t * 90.0)  # HF emphasis
    return (wave / (np.abs(wave).max() + 1e-12) * amp).astype(np.float64)


def frequency_extent(event: Event) -> tuple[float, float]:
    """Box (f_lo, f_hi) for an event from its class/pitch."""
    if event.label in _DRUM_BANDS:
        return _DRUM_BANDS[event.label]
    f0 = midi_to_hz(event.pitch if event.pitch is not None else 60)
    return f0 * 2 ** (-1.5 / 12), f0 * 2 ** (1.5 / 12)


def render_scene(events: list[Event], sample_rate: int,
                 total_s: float | None = None) -> tuple[np.ndarray, list[Event]]:
    """Renders events to mono audio and returns ``(audio, events)`` with each
    event's frequency extent filled in. Events are placed at ``onset_s`` and
    summed; the returned audio is peak-normalized to 0.9."""
    if total_s is None:
        total_s = max((e.offset_s for e in events), default=1.0) + 0.2
    n = int(total_s * sample_rate)
    buffer = np.zeros(n)
    filled: list[Event] = []
    for event in events:
        if event.label in DRUM_KINDS:
            hit = render_drum(event.label, sample_rate, velocity=event.velocity or 100)
        else:
            hit = render_note(event.pitch if event.pitch is not None else 60,
                              max(event.offset_s - event.onset_s, 0.05), sample_rate,
                              instrument=event.label, velocity=event.velocity or 100)
        start = int(event.onset_s * sample_rate)
        end = min(start + hit.shape[0], n)
        if end > start:
            buffer[start:end] += hit[: end - start]
        f_lo, f_hi = frequency_extent(event)
        filled.append(Event(event.onset_s, event.offset_s, f_lo, f_hi,
                            label=event.label, pitch=event.pitch, velocity=event.velocity))
    peak = np.abs(buffer).max()
    if peak > 0:
        buffer *= 0.9 / peak
    return buffer, filled


def random_scene(sample_rate: int, *, seconds: float = 2.0, bpm: float = 120.0,
                 seed: int = 0, pitched=("saw", "pluck"), drums=True,
                 scale=(0, 2, 4, 5, 7, 9, 11)) -> tuple[np.ndarray, list[Event]]:
    """A procedural note+drum scene with exact labels. Notes land on the 8th
    grid with pitches from ``scale`` (C major by default); drums lay down a
    simple kick/snare/hat pattern."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    step = beat / 2.0  # 8th notes
    n_steps = int(seconds / step)
    events: list[Event] = []
    for i in range(n_steps):
        t = i * step
        if rng.random() < 0.6:  # a note this step
            root = 48 + rng.integers(0, 3) * 12
            pitch = int(root + rng.choice(scale))
            instrument = str(rng.choice(pitched))
            events.append(Event(t, t + step * rng.uniform(0.6, 1.4), 0.0, 0.0,
                               label=instrument, pitch=pitch,
                               velocity=int(rng.integers(70, 120))))
        if drums:
            if i % 4 == 0:
                events.append(Event(t, t + 0.18, 0.0, 0.0, label="kick", velocity=110))
            if i % 4 == 2:
                events.append(Event(t, t + 0.15, 0.0, 0.0, label="snare", velocity=100))
            events.append(Event(t, t + 0.05, 0.0, 0.0, label="hat", velocity=70))
    return render_scene(events, sample_rate, total_s=seconds)


def weak_labels(audio: np.ndarray, sample_rate: int, *, bpm: float | None = None) -> list[Event]:
    """Approximate event boxes for *real* audio from the deterministic DSP:
    pitched notes from f0 tracking, drum hits from the rhythm registers.
    Labels are weak (``"note"``/``"kick"``/``"snare"``/``"hat"``) — training
    signal, not ground truth."""
    from wavelet2vec.harmony import segment_notes, track_f0
    from wavelet2vec.rhythm import REGISTER_EDGES_HZ, analyze_rhythm

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    events: list[Event] = []

    for note in segment_notes(track_f0(audio, sample_rate)):
        f0 = midi_to_hz(note.midi)
        events.append(Event(note.start_s, note.end_s, f0 * 2 ** (-1.5 / 12),
                            f0 * 2 ** (1.5 / 12), label="note", pitch=note.midi,
                            velocity=int(np.clip(note.confidence * 127, 1, 127))))

    if bpm is not None:
        profile = analyze_rhythm(audio, sample_rate, bpm=bpm)
        register_label = {"low": "kick", "mid": "snare", "high": "hat"}
        for register, onsets in profile.onsets.items():
            f_lo, f_hi = REGISTER_EDGES_HZ[register]
            for time_s, strength in onsets:
                events.append(Event(time_s, time_s + 0.12, f_lo, min(f_hi, 13000.0),
                                   label=register_label[register],
                                   velocity=int(np.clip(strength * 20, 1, 127))))
    return events
