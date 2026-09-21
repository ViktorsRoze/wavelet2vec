"""V-1 coefficient images + annotations, V-2 synthetic scenes + weak labels."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wavelet2vec import (
    CocoBuilder,
    Event,
    bbox_to_event,
    coeff_image,
    event_to_bbox,
    export_dataset,
    image_to_uint8,
    midi_to_hz,
    random_scene,
    render_note,
    render_scene,
    weak_labels,
)

SR = 22050


# --- V-1 image ----------------------------------------------------------------------
def test_coeff_image_fixed_size_and_range():
    for seconds in (0.5, 2.0, 5.0):
        audio = render_note(60, seconds, SR, instrument="saw")
        image, meta = coeff_image(audio, SR, n_bands=128, n_frames=128)
        assert image.shape == (3, 128, 128)  # any length -> same image
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert meta.n_bands == 128 and meta.n_frames == 128
        assert meta.duration_s == pytest.approx(seconds, abs=0.02)


def test_image_channels_are_distinct():
    audio, _events = random_scene(SR, seconds=2.0, seed=2)
    image, _meta = coeff_image(audio, SR)
    mag, dev, coh = image
    assert not np.allclose(mag, dev)
    assert not np.allclose(mag, coh)
    # A tonal note has high coherence where it's loud; noise-only hats less so.
    assert coh[mag > 0.5].mean() > coh[mag < 0.05].mean()


def test_image_to_uint8():
    audio = render_note(64, 1.0, SR)
    image, _meta = coeff_image(audio, SR)
    rgb = image_to_uint8(image)
    assert rgb.shape == (128, 128, 3) and rgb.dtype == np.uint8


# --- V-1 coordinate mapping ----------------------------------------------------------
def test_event_box_round_trip():
    audio = render_note(60, 2.0, SR)
    _image, meta = coeff_image(audio, SR)
    event = Event(0.4, 1.2, 400.0, 900.0, label="saw", pitch=60, velocity=100)
    bbox = event_to_bbox(event, meta)
    back = bbox_to_event(bbox, meta, label="saw", pitch=60)
    assert back.onset_s == pytest.approx(event.onset_s, abs=0.03)
    assert back.offset_s == pytest.approx(event.offset_s, abs=0.03)
    assert back.f_lo_hz == pytest.approx(event.f_lo_hz, rel=0.05)
    assert back.f_hi_hz == pytest.approx(event.f_hi_hz, rel=0.05)


def test_boxes_land_on_energy():
    audio, events = random_scene(SR, seconds=2.0, bpm=120, seed=3)
    image, meta = coeff_image(audio, SR)
    magnitude = image[0]
    hits = 0
    for event in events:
        x, y, w, h = event_to_bbox(event, meta)
        patch = magnitude[int(y): int(y + h) + 1, int(x): int(x + w) + 1]
        if patch.size and patch.mean() > magnitude.mean():
            hits += 1
    assert hits >= 0.8 * len(events)  # ground-truth boxes sit on real energy


# --- V-1 COCO export -----------------------------------------------------------------
def test_export_dataset_coco(tmp_path):
    clips = []
    for seed in range(3):
        audio, events = random_scene(SR, seconds=1.5, seed=seed)
        clips.append((audio, SR, events))
    path = export_dataset(clips, tmp_path, n_frames=96)
    coco = json.loads(path.read_text(encoding="utf-8"))

    assert len(coco["images"]) == 3
    assert {img["width"] for img in coco["images"]} == {96}
    assert {img["height"] for img in coco["images"]} == {128}
    assert coco["annotations"] and coco["categories"]
    # Every annotation references a real image and category; bbox in-bounds.
    image_ids = {img["id"] for img in coco["images"]}
    category_ids = {c["id"] for c in coco["categories"]}
    for ann in coco["annotations"]:
        assert ann["image_id"] in image_ids
        assert ann["category_id"] in category_ids
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= 96 and 0 <= y <= 128 and w > 0 and h > 0
    # Tiles saved.
    assert (tmp_path / "clip_0000.npy").exists()
    assert np.load(tmp_path / "clip_0000.npy").shape == (3, 128, 96)


def test_coco_builder_categories():
    builder = CocoBuilder(categories=["kick", "snare", "note"])
    audio = render_note(60, 1.0, SR)
    _image, meta = coeff_image(audio, SR)
    builder.add("a.npy", meta, [Event(0.1, 0.5, 40, 120, label="kick"),
                                Event(0.2, 0.3, 200, 400, label="note", pitch=60)])
    result = builder.to_dict()
    assert len(result["categories"]) == 3
    assert len(result["annotations"]) == 2


# --- V-2 synthetic scenes ------------------------------------------------------------
def test_midi_to_hz():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(60) == pytest.approx(261.63, abs=0.1)


def test_render_note_has_energy_at_fundamental():
    from wavelet2vec.frame import analyze

    audio = render_note(69, 1.0, SR, instrument="saw")  # A4 = 440
    coeffs, spec = analyze(audio, SR, n_bands=96)
    energy = (np.abs(coeffs) ** 2).mean(axis=1)
    peak_band = int(np.argmax(energy))
    assert spec.centers[peak_band] == pytest.approx(440.0, rel=0.1)


def test_render_scene_fills_extent_and_places_hits():
    events = [Event(0.0, 0.5, 0, 0, label="saw", pitch=60),
              Event(0.5, 0.7, 0, 0, label="kick")]
    audio, filled = render_scene(events, SR, total_s=1.0)
    assert audio.shape[0] == SR
    assert all(e.f_hi_hz > e.f_lo_hz > 0 for e in filled)  # extents filled
    # Energy appears around each onset.
    assert np.abs(audio[: SR // 2]).max() > 0.1  # the note
    assert np.abs(audio[SR // 2:]).max() > 0.1  # the kick


def test_random_scene_labels_and_determinism():
    a1, e1 = random_scene(SR, seconds=2.0, seed=7)
    a2, e2 = random_scene(SR, seconds=2.0, seed=7)
    assert np.array_equal(a1, a2)  # deterministic
    assert [e.to_dict() for e in e1] == [e.to_dict() for e in e2]
    labels = {e.label for e in e1}
    assert labels & {"saw", "pluck"} and labels & {"kick", "snare", "hat"}


# --- V-2 weak labels -----------------------------------------------------------------
def test_weak_labels_find_notes_and_drums():
    audio, _events = random_scene(SR, seconds=2.0, bpm=120, seed=4)
    labels = weak_labels(audio, SR, bpm=120.0)
    assert labels
    kinds = {e.label for e in labels}
    assert "note" in kinds  # f0-derived pitched notes
    assert kinds & {"kick", "snare", "hat"}  # rhythm-register drums
    for event in labels:
        assert event.offset_s > event.onset_s
        assert event.f_hi_hz > event.f_lo_hz > 0


def test_weak_labels_without_bpm_still_gives_notes():
    audio = render_note(60, 0.5, SR)
    audio = np.concatenate([audio, render_note(67, 0.5, SR)])
    labels = weak_labels(audio, SR)  # no bpm -> pitched notes only
    assert all(e.label == "note" for e in labels)
    assert len(labels) >= 1
