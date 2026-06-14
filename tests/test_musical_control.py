from __future__ import annotations

import math

import numpy as np

from wavelet2vec import (
    ADSR,
    AutomationCurve,
    Wavelet2Vec,
    apply_adsr_character,
    apply_adsr_volume,
    apply_character,
    average_characters,
    cosine_similarity,
    extract_character,
    render_sequence,
)
from wavelet2vec.character import SoundCharacter

SAMPLE_RATE = 22050


def _vocal_like(seconds: float = 1.0, f0: float = 180.0) -> np.ndarray:
    """Vibrato tone with formant-weighted harmonics."""
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    vibrato = f0 * (1 + 0.01 * np.sin(2 * math.pi * 5.5 * times))
    phase = 2 * math.pi * np.cumsum(vibrato) / SAMPLE_RATE
    formant = lambda k: math.exp(-(((f0 * k - 700) / 400) ** 2)) + 0.7 * math.exp(
        -(((f0 * k - 1800) / 600) ** 2)
    )
    waveform = sum(np.sin(k * phase) * formant(k) for k in range(1, 14))
    waveform *= np.minimum(times * 8, 1) * np.exp(-times * 1.2)
    return (waveform / np.abs(waveform).max())[None, :].astype(np.float32)


def _guitar_like(f0: float, seconds: float = 1.0) -> np.ndarray:
    """Clipped saw with fast decay — crunchy, harmonically dense."""
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    saw = sum(np.sin(2 * math.pi * f0 * k * times) / k for k in range(1, 24))
    crunchy = np.clip(saw * 4.0, -1.0, 1.0) * np.exp(-times * 4.0)
    return (crunchy / np.abs(crunchy).max())[None, :].astype(np.float32)


def _guitar_group_character() -> SoundCharacter:
    return average_characters(
        [extract_character(_guitar_like(f0), SAMPLE_RATE) for f0 in (82.4, 110.0, 146.8, 196.0)]
    )


def test_adsr_envelope_stages():
    adsr = ADSR(attack=0.1, decay=0.1, sustain=0.5, release=0.2)
    envelope = adsr.envelope(SAMPLE_RATE, SAMPLE_RATE)  # 1 second
    attack_end = int(0.1 * SAMPLE_RATE)
    decay_end = int(0.2 * SAMPLE_RATE)
    assert envelope[0] < 0.01
    assert abs(envelope[attack_end] - 1.0) < 0.02
    assert abs(envelope[decay_end] - 0.5) < 0.02
    assert abs(envelope[int(0.5 * SAMPLE_RATE)] - 0.5) < 0.02  # sustain plateau
    assert envelope[-1] < 0.02  # released by the end
    assert np.all(envelope >= 0.0) and np.all(envelope <= 1.0)


def test_adsr_volume_shapes_loudness():
    adsr = ADSR(attack=0.3, decay=0.1, sustain=0.8, release=0.2)
    tone = np.ones((1, SAMPLE_RATE), dtype=np.float32)
    shaped = apply_adsr_volume(tone, SAMPLE_RATE, adsr)
    assert shaped.shape == tone.shape
    early = np.abs(shaped[0, : SAMPLE_RATE // 20]).mean()
    middle = np.abs(shaped[0, SAMPLE_RATE // 2 : SAMPLE_RATE // 2 + SAMPLE_RATE // 20]).mean()
    assert early < 0.2
    assert middle > 0.7


def test_group_character_makes_vocal_crunchier():
    embedder = Wavelet2Vec()
    vocal = _vocal_like()
    character = _guitar_group_character()

    morphed = apply_character(vocal, SAMPLE_RATE, character, amount=0.9)
    guitar_vec = embedder.embed(_guitar_like(110.0), SAMPLE_RATE)
    before = cosine_similarity(embedder.embed(vocal, SAMPLE_RATE), guitar_vec)
    after = cosine_similarity(embedder.embed(morphed, SAMPLE_RATE), guitar_vec)
    assert after > before + 0.05

    untouched = apply_character(vocal, SAMPLE_RATE, character, amount=0.0)
    assert cosine_similarity(
        embedder.embed(untouched, SAMPLE_RATE), embedder.embed(vocal, SAMPLE_RATE)
    ) > 0.99


def test_character_save_and_load_roundtrip(tmp_path):
    character = _guitar_group_character()
    path = character.save(tmp_path / "guitar.npz")
    loaded = SoundCharacter.load(path)
    assert np.allclose(loaded.log_band_energies, character.log_band_energies)
    assert np.allclose(loaded.envelope_shape, character.envelope_shape)


def test_adsr_character_grows_into_the_morph():
    vocal = _vocal_like()
    character = _guitar_group_character()
    adsr = ADSR(attack=0.6, decay=0.1, sustain=1.0, release=0.1)
    morphed = apply_adsr_character(vocal, SAMPLE_RATE, character, adsr)
    assert morphed.shape == vocal.shape

    difference = np.abs(morphed.astype(np.float64) - vocal.astype(np.float64))
    n = vocal.shape[1]
    early = difference[:, : n // 10].mean()
    late = difference[:, n // 2 : (4 * n) // 5].mean()
    assert early < late  # slow attack: starts as itself, grows into the character


def test_automation_curve_interpolation(tmp_path):
    curve = AutomationCurve.from_points([(0.0, 0.0), (2.0, 1.0)])
    assert curve.value_at(0.0) == 0.0
    assert abs(curve.value_at(1.0) - 0.5) < 1e-9
    assert curve.value_at(5.0) == 1.0
    assert abs(curve.value_at_fraction(0.25) - 0.25) < 1e-9

    csv_path = tmp_path / "clip.csv"
    csv_path.write_text("time_seconds,value\n0.0,0.2\n1.0,0.8\n", encoding="utf-8")
    loaded = AutomationCurve.from_file(csv_path)
    assert abs(loaded.value_at(0.5) - 0.5) < 1e-9


def test_fl_studio_automation_parsing(tmp_path, monkeypatch):
    pyflp = __import__("pytest").importorskip("pyflp")

    class FakePoint:
        def __init__(self, position, value):
            self.position = position
            self.value = value

    class FakeAutomation:  # name contains "automation", like pyflp.channel.Automation
        def __iter__(self):
            # PPQ ticks at 96 ppq / 120 bpm: 0 -> 0.0 s, 96 -> 0.5 s, 192 -> 1.0 s
            return iter([FakePoint(0, 0.0), FakePoint(96, 0.5), FakePoint(192, 1.0)])

    class FakeProject:
        ppq = 96
        tempo = 120.0
        channels = [object(), FakeAutomation()]  # non-automation channels are skipped

    monkeypatch.setattr(pyflp, "parse", lambda path: FakeProject())
    clip = tmp_path / "clip.flp"
    clip.write_bytes(b"")
    curve = AutomationCurve.from_file(clip)

    assert abs(curve.duration - 1.0) < 1e-9
    assert abs(curve.value_at(0.5) - 0.5) < 1e-9
    assert abs(curve.value_at_fraction(0.25) - 0.25) < 1e-9


def test_render_sequence_develops_texture():
    vocal = _vocal_like(seconds=0.5)
    snippets = [vocal, vocal, vocal]
    rates = [SAMPLE_RATE] * 3
    character = _guitar_group_character()
    automation = AutomationCurve.from_points([(0.0, 0.0), (1.0, 1.0)])

    result, output_sr = render_sequence(
        snippets,
        rates,
        character=character,
        automation=automation,
        modulate="amount",
        gap_seconds=0.1,
    )
    assert output_sr == SAMPLE_RATE
    gap = int(0.1 * SAMPLE_RATE)
    n = vocal.shape[1]
    assert result.shape == (1, 3 * n + 2 * gap)
    assert np.isfinite(result).all()

    # Rising automation on the morph amount: the first snippet stays close to
    # the original, the last one is the most transformed.
    first = np.abs(result[:, :n].astype(np.float64) - vocal.astype(np.float64)).mean()
    last = np.abs(result[:, -n:].astype(np.float64) - vocal.astype(np.float64)).mean()
    assert first < last


def test_render_sequence_attack_modulation_runs():
    vocal = _vocal_like(seconds=0.5)
    character = _guitar_group_character()
    automation = AutomationCurve.from_points([(0.0, 0.0), (1.0, 1.0)])
    adsr = ADSR(attack=0.4, decay=0.05, sustain=1.0, release=0.05)

    result, _ = render_sequence(
        [vocal, vocal, vocal],
        [SAMPLE_RATE] * 3,
        character=character,
        character_adsr=adsr,
        automation=automation,
        modulate="attack",
        volume_adsr=ADSR(attack=0.01, decay=0.05, sustain=0.9, release=0.1),
    )
    assert np.isfinite(result).all()

    # Faster transformation attack on later snippets: early in each snippet,
    # the last snippet should already be further from the original.
    n = vocal.shape[1]
    head = slice(0, n // 4)
    first_head = np.abs(result[:, :n][:, head].astype(np.float64))
    last_head = np.abs(result[:, -n:][:, head].astype(np.float64))
    raw_head = np.abs(vocal[:, head].astype(np.float64))
    first_dev = np.abs(first_head - raw_head).mean()
    last_dev = np.abs(last_head - raw_head).mean()
    assert last_dev > first_dev
