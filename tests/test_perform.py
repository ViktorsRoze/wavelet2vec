from __future__ import annotations

import math
import struct

import numpy as np
import soundfile as sf

from wavelet2vec import (
    ADSR,
    AutomationCurve,
    Wavelet2Vec,
    average_characters,
    cosine_similarity,
    extract_character,
    perform,
    read_cue_points,
    slice_grid,
    slice_markers,
    slice_onsets,
)
from wavelet2vec.audio_io import save_audio

HIGH_SR = 88200


def _note(sample_rate: int, seconds: float, f0: float = 180.0) -> np.ndarray:
    times = np.arange(int(sample_rate * seconds)) / sample_rate
    formant = lambda k: math.exp(-(((f0 * k - 700) / 400) ** 2)) + 0.7 * math.exp(
        -(((f0 * k - 1800) / 600) ** 2)
    )
    waveform = sum(np.sin(2 * math.pi * f0 * k * times) * formant(k) for k in range(1, 14))
    waveform *= np.minimum(times * 30, 1) * np.exp(-times * 6)
    return waveform / np.abs(waveform).max()


def _performance(sample_rate: int, n_notes: int = 8, spacing: float = 0.25) -> np.ndarray:
    """Repeated vocal-like notes at a fixed spacing, like an FL render."""
    total = int(sample_rate * spacing * n_notes)
    render = np.zeros(total)
    note = _note(sample_rate, spacing * 0.8)
    for index in range(n_notes):
        start = int(index * spacing * sample_rate)
        render[start : start + note.shape[0]] += note
    return (render / np.abs(render).max())[None, :].astype(np.float32)


def _crunch_reference(f0: float = 110.0) -> np.ndarray:
    sample_rate = 44100
    times = np.arange(sample_rate) / sample_rate
    saw = sum(np.sin(2 * math.pi * f0 * k * times) / k for k in range(1, 24))
    crunchy = np.clip(saw * 4.0, -1.0, 1.0) * np.exp(-times * 4.0)
    return (crunchy / np.abs(crunchy).max())[None, :].astype(np.float32)


def _crunch_character():
    return average_characters(
        [extract_character(_crunch_reference(f0), 44100) for f0 in (82.4, 110.0)]
    )


def test_slice_grid_bpm_and_spacing():
    n = HIGH_SR * 4  # 4 seconds
    starts = slice_grid(n, HIGH_SR, bpm=120, division=8)  # 8ths at 120 bpm = 0.25 s
    assert starts[0] == 0
    assert len(starts) == 16
    assert starts[1] == int(0.25 * HIGH_SR)

    by_spacing = slice_grid(n, HIGH_SR, spacing_seconds=0.25)
    assert np.array_equal(starts, by_spacing)


def test_slice_markers_plain_and_csv(tmp_path):
    plain = tmp_path / "markers.txt"
    plain.write_text("0.0\n0.5\n1.0\n", encoding="utf-8")
    starts = slice_markers(plain, 44100, 2 * 44100)
    assert np.array_equal(starts, [0, 22050, 44100])

    csv_file = tmp_path / "markers.csv"
    csv_file.write_text("time_seconds,label\n0.5,a\n1.0,b\n", encoding="utf-8")
    starts = slice_markers(csv_file, 44100, 2 * 44100)
    assert np.array_equal(starts, [0, 22050, 44100])  # 0 is always prepended


def test_read_cue_points_from_wav(tmp_path):
    # Minimal 16-bit mono WAV with a standard RIFF 'cue ' chunk, as written
    # by Sound Forge / Edison.
    sr, n = 44100, 1000
    pcm = (np.zeros(n, dtype=np.int16)).tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    cues = [(1, 100), (2, 500)]
    cue_body = struct.pack("<I", len(cues))
    for cue_id, offset in cues:
        cue_body += struct.pack("<II4sIII", cue_id, offset, b"data", 0, 0, offset)
    chunks = b""
    for cid, body in ((b"fmt ", fmt), (b"data", pcm), (b"cue ", cue_body)):
        chunks += cid + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")
    riff = b"WAVE" + chunks
    path = tmp_path / "cued.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)

    assert np.array_equal(read_cue_points(path), [100, 500])


def test_slice_onsets_finds_fixed_grid_notes():
    render = _performance(44100, n_notes=4, spacing=0.5)
    starts = slice_onsets(render, 44100, sensitivity=0.5)
    expected = np.array([0.0, 0.5, 1.0, 1.5]) * 44100
    assert len(starts) == 4
    assert np.all(np.abs(starts - expected) < 0.03 * 44100)  # within 30 ms


def test_perform_high_rate_round_trip_and_development():
    render = _performance(HIGH_SR, n_notes=8, spacing=0.25)
    character = _crunch_character()
    starts = slice_grid(render.shape[1], HIGH_SR, spacing_seconds=0.25)
    automation = AutomationCurve.from_points([(0.0, 0.0), (1.0, 1.0)])

    result = perform(
        render,
        HIGH_SR,
        character,
        starts,
        automation=automation,
        character_adsr=ADSR(attack=0.05, decay=0.05, sustain=1.0, release=0.05),
        modulate="amount",
    )
    # Sample-exact round trip at the native high rate: drop back in at bar 1.
    assert result.shape == render.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()

    # Rising automation: the first note is untouched, the last is the most
    # transformed.
    n = int(0.25 * HIGH_SR)
    first = np.abs(result[:, :n].astype(np.float64) - render[:, :n].astype(np.float64)).mean()
    last = np.abs(result[:, -n:].astype(np.float64) - render[:, -n:].astype(np.float64)).mean()
    assert first < last

    # The development is also visible in embedding space: the last note is
    # closer to the crunch character's source material than the first.
    embedder = Wavelet2Vec()
    crunch = embedder.embed(_crunch_reference(), 44100)
    first_sim = cosine_similarity(embedder.embed(result[:, :n], HIGH_SR), crunch)
    last_sim = cosine_similarity(embedder.embed(result[:, -n:], HIGH_SR), crunch)
    assert last_sim > first_sim


def test_perform_without_character_is_identity_except_fades():
    render = _performance(44100, n_notes=4, spacing=0.25)
    starts = slice_grid(render.shape[1], 44100, spacing_seconds=0.25)
    result = perform(render, 44100, None, starts, boundary_fade_seconds=0.0)
    assert np.allclose(result, render, atol=1e-6)


def test_save_audio_writes_float_at_high_rate(tmp_path):
    render = _performance(HIGH_SR, n_notes=2, spacing=0.25)
    path = save_audio(tmp_path / "high.wav", render, HIGH_SR)
    info = sf.info(path)
    assert info.samplerate == HIGH_SR
    assert info.subtype == "FLOAT"

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == HIGH_SR
    assert np.array_equal(audio.T, render)  # 32-bit float round trip is lossless
