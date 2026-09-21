"""Aligned targets, provenance, sample-weighted training and exact resume."""
from dataclasses import replace
import copy
import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import (
    AudioRecord, AudioCropConfig, MultitrackRecord, MultitrackManifest, MultitrackDataset,
    collate_multitracks, HybridConfig, SynthesisConfig, HybridSeparator, MusicContextConfig,
    SeparationRunConfig, SeparationTrainingRun, evaluate_separation,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def manifest(tmp_path, rate=8192, lengths=(128, 128, 128, 128)):
    records = []
    for i, (split, n) in enumerate(zip(("train", "train", "validation", "validation"), lengths)):
        rng = np.random.default_rng(i)
        low = rng.normal(0, .05, (n, 2)).astype(np.float32)
        high = rng.normal(0, .03, (n, 2)).astype(np.float32)
        paths = {}
        for name, audio in (("low", low), ("high", high), ("mix", low+high)):
            path = tmp_path/f"{i}-{name}.wav"
            sf.write(path, audio, rate, subtype="FLOAT")
            paths[name] = path
        records.append(MultitrackRecord.inspect(str(i), paths["mix"],
            {"low": paths["low"], "high": paths["high"], "silent": None},
            group_id=f"group-{i}", split=split, additive=True))
    return MultitrackManifest(records, source_order=("low", "high", "silent"))


def datasets(m):
    cfg = AudioCropConfig(m.records[0].mixture.sample_rate, 2, 96, hop_length=16, crops_per_record=2, seed=13)
    return MultitrackDataset(m, "train", cfg), MultitrackDataset(m, "validation", replace(cfg, mode="fixed"))


def model(m, dropout=.1):
    cfg = HybridConfig(sample_rate=m.records[0].mixture.sample_rate, backend="compact_wavelet",
                       n_bands=8, q=1, f_min=40, hop_length=16, padding_samples=16)
    return HybridSeparator(cfg, SynthesisConfig(source_names=m.source_order, dim=8),
        MusicContextConfig(dim=8, heads=2, local_layers=1, global_layers=1, local_tokens=4, dropout=dropout))


def run(m, **kwargs):
    train, val = datasets(m)
    cfg = SeparationRunConfig(total_steps=4, warmup_steps=0, batch_size=1, accumulation_steps=2,
                              fft_sizes=(32, 64), mixture_weight=.1, shuffle_seed=8, **kwargs)
    return SeparationTrainingRun(model(m), train, val, cfg)


def test_manifest_relative_roundtrip_and_immutable_sources(tmp_path):
    m = manifest(tmp_path)
    path = tmp_path/"metadata"/"corpus.json"
    m.save(path)
    loaded = MultitrackManifest.load(path)
    assert loaded.identity == m.identity and loaded.records == m.records
    assert not os.path.isabs(json.loads(path.read_text())["records"][0]["mixture"]["path"])
    with pytest.raises(TypeError):
        m.records[0].sources["new"] = None


def test_source_reuse_groups_split_and_alignment_guards(tmp_path):
    m = manifest(tmp_path)
    source = replace(m.records[0].sources["low"], path=str(tmp_path/"copy.wav"), group_id="group-2", split="validation")
    bad = replace(m.records[2], sources={**m.records[2].sources, "low": source})
    with pytest.raises(ValueError, match="reused"):
        MultitrackManifest((m.records[0], bad), source_order=m.source_order)
    with pytest.raises(ValueError, match="rate, channels and length"):
        replace(m.records[0], sources={"low": replace(m.records[0].sources["low"], frames=99)})
    with pytest.raises(ValueError, match="all source slots"):
        MultitrackManifest([replace(m.records[0], sources={"low": m.records[0].sources["low"]})], source_order=m.source_order)
    with pytest.raises(ValueError, match="sample_aligned"):
        replace(m.records[0], alignment="infer")


@pytest.mark.parametrize("rate", [8192, 192000])
def test_native_aligned_crops_silence_and_epoch_determinism(tmp_path, rate):
    train, val = datasets(manifest(tmp_path, rate))
    first = train[0]
    assert first.example.mixture.dtype == torch.float32 and first.example.sample_rate == rate
    assert first.sample_origin % 16 == 0
    torch.testing.assert_close(sum(first.example.sources.values()), first.example.mixture)
    assert first.example.sources["silent"].count_nonzero() == 0
    torch.manual_seed(900)
    assert train[0].window_id == first.window_id
    original = [train.plan(i) for i in range(len(train))]
    train.set_epoch(1)
    assert original != [train.plan(i) for i in range(len(train))]
    fixed = [val.plan(i) for i in range(len(val))]
    val.set_epoch(9)
    assert fixed == [val.plan(i) for i in range(len(val))]
    batch = collate_multitracks([first], source_order=train.manifest.source_order)
    assert batch.source_valid.tolist() == [[True, True, True]]


def test_missing_annotations_are_not_silent(tmp_path):
    m = manifest(tmp_path)
    record = replace(m.records[0], sources={"low": m.records[0].sources["low"], "silent": None}, additive=False)
    partial = MultitrackManifest([record], source_order=m.source_order)
    dataset = MultitrackDataset(partial, "train", AudioCropConfig(8192, 2, 64, hop_length=16))
    batch = collate_multitracks([dataset[0]], source_order=m.source_order)
    assert batch.source_valid.tolist() == [[True, False, True]]
    with pytest.raises(ValueError, match="additive"):
        SeparationTrainingRun(model(partial), dataset, config=SeparationRunConfig(mixture_weight=.1))


def test_full_track_additivity_and_source_mutation(tmp_path):
    m = manifest(tmp_path)
    train, _ = datasets(m)
    path = m.records[0].sources["low"].path
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    data[-1] += 1  # invalid even if a selected crop excludes this final sample
    sf.write(path, data, rate, subtype="FLOAT")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+10000000))
    with pytest.raises(ValueError, match="changed"):
        train[0]
    with pytest.raises(ValueError, match="matches"):
        datasets(m)
    changed = replace(m.records[0], sources={**m.records[0].sources,
        "low": AudioRecord.inspect(path, group_id="group-0", split="train")})
    fresh = MultitrackManifest([changed], source_order=m.source_order)
    with pytest.raises(ValueError, match="sum to"):
        MultitrackDataset(fresh, "train", AudioCropConfig(8192, 2, 32, hop_length=16))


@pytest.mark.parametrize("rate", [8192, 192000])
@pytest.mark.parametrize("completed", [1, 2])
def test_resume_restores_next_crops_dropout_loss_and_weights(tmp_path, rate, completed):
    torch.manual_seed(5)
    m = manifest(tmp_path, rate)
    original = run(m)
    for _ in range(completed):
        original.step()
    path = tmp_path/"checkpoint.pt"
    original.save_checkpoint(path)
    expected = original.step()
    expected_state = copy.deepcopy(original.model.state_dict())
    resumed = run(m)
    resumed.load_checkpoint(path)
    actual = resumed.step()
    assert actual == expected
    for name, value in expected_state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
        else:
            assert resumed.model.state_dict()[name] == value


def test_provenance_and_runtime_rejection_before_model_mutation(tmp_path):
    m = manifest(tmp_path)
    original = run(m)
    original.step()
    path = tmp_path/"checkpoint.pt"
    original.save_checkpoint(path)
    train, val = datasets(m)
    different = MultitrackDataset(m, "train", replace(train.config, seed=900))
    changed = SeparationTrainingRun(model(m), different, val, original.config)
    before = {k: v.clone() for k, v in changed.model.state_dict().items() if isinstance(v, torch.Tensor)}
    with pytest.raises(ValueError, match="provenance"):
        changed.load_checkpoint(path)
    for key, value in before.items():
        assert torch.equal(value, changed.model.state_dict()[key])
    payload = torch.load(path, weights_only=True)
    payload["runtime"]["threads"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="runtime"):
        original.load_checkpoint(path)


def test_failed_accumulation_rewinds_rng_and_data_position(tmp_path, monkeypatch):
    torch.manual_seed(3)
    active = run(manifest(tmp_path))
    path = tmp_path/"before.pt"
    active.save_checkpoint(path)
    expected = active.step()
    active.load_checkpoint(path)
    forward = active.model.forward
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return forward(*args, **kwargs)
    monkeypatch.setattr(active.model, "forward", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        active.step()
    assert active.update_step == active.epoch == active.cursor == 0
    assert all(p.grad is None for p in active.model.parameters())
    monkeypatch.setattr(active.model, "forward", forward)
    assert active.step() == expected


@pytest.mark.parametrize("missing", [False, True])
def test_accumulation_weights_variable_lengths_like_one_batch(tmp_path, missing):
    torch.manual_seed(8)
    m = manifest(tmp_path, lengths=(32, 96, 80, 128))
    if missing:
        first = replace(m.records[0], sources={"low": m.records[0].sources["low"]}, additive=False)
        m = MultitrackManifest((first, *m.records[1:]), source_order=m.source_order)
    crop = AudioCropConfig(8192, 2, 128, hop_length=16, crops_per_record=1, mode="fixed")
    train = MultitrackDataset(m, "train", crop)
    base = model(m, dropout=0)
    cfg = SeparationRunConfig(total_steps=2, warmup_steps=0, batch_size=1, accumulation_steps=2,
                              fft_sizes=(32,), weight_decay=0, max_grad_norm=1000, mixture_weight=0 if missing else .1)
    small = SeparationTrainingRun(copy.deepcopy(base), train, config=cfg)
    large = SeparationTrainingRun(copy.deepcopy(base), train, config=replace(cfg, batch_size=2, accumulation_steps=1))
    a, b = small.step(), large.step()
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-7)
    for x, y in zip(small.model.parameters(), large.model.parameters()):
        torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6)


def test_evaluation_group_weighting_silent_sources_and_modes(tmp_path):
    m = manifest(tmp_path, lengths=(128, 128, 64, 128))
    active = run(m)
    active.model.train()
    active.model.encoder.eval()
    modes = [x.training for x in active.model.modules()]
    first, second = active.evaluate(), active.evaluate()
    assert first == second and modes == [x.training for x in active.model.modules()]
    assert first["micro_loss"]["model"] == pytest.approx(first["micro_loss"]["equal_split"], abs=1e-7)
    for key in first["macro_group_loss"]:
        assert first["macro_group_loss"][key] == pytest.approx(sum(g[key] for g in first["groups"].values())/2)
    assert first["windows"][0]["sources"]["model"]["silent"]["snr_db"] is None
    json.dumps(first, allow_nan=False)
    with pytest.raises(ValueError, match="held-out"):
        evaluate_separation(active.model, active.training)


def test_cli_prepare_train_evaluate_and_resume_at_192khz(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    m = manifest(tmp_path, rate=192000)
    assignments = {"source_order": m.source_order, "tracks": [
        {"track_id": r.track_id, "mixture": r.mixture.path, "sources": {
            name: None if asset is None else asset.path for name, asset in r.sources.items()},
         "group_id": r.mixture.group_id, "split": r.mixture.split,
         "additive": r.additive, "alignment": r.alignment} for r in m.records]}
    source = tmp_path/"assignments.json"
    source.write_text(json.dumps(assignments))
    manifest_path, output = tmp_path/"manifest.json", tmp_path/"run"
    root = Path(__file__).resolve().parents[1]
    def execute(arguments):
        result = subprocess.run([sys.executable, "-m", "experiments.train_multitrack", *arguments],
                                cwd=root, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
    execute(["prepare", str(source), "--output", str(manifest_path)])
    options = ["train", str(manifest_path), "--output", str(output), "--steps", "3",
               "--updates-per-invocation", "1", "--batch-size", "1", "--accumulation-steps", "2",
               "--window-seconds", ".0005", "--hop-length", "16", "--bands", "8", "--dim", "8",
               "--fft-sizes", "32", "64", "--mixture-weight", ".1"]
    execute(options)
    execute(options+["--resume", str(output/"checkpoint.pt")])
    for step in (1, 2):
        report = json.loads((output/f"evaluation-step-{step}.json").read_text())
        assert report["sample_rate"] == 192000 and len(report["groups"]) == 2
        assert set(report["micro_loss"]) == {"model", "equal_split", "mixture_copy", "silence"}
    configuration = json.loads((output/"configuration.json").read_text())
    configuration["manifest"] = "different-corpus"
    (output/"configuration.json").write_text(json.dumps(configuration))
    checkpoint_before = (output/"checkpoint.pt").read_bytes()
    rejected = subprocess.run([sys.executable, "-m", "experiments.train_multitrack", *options,
                               "--resume", str(output/"checkpoint.pt")], cwd=root, capture_output=True, text=True, timeout=90)
    assert rejected.returncode != 0 and "output directory configuration differs" in rejected.stderr
    assert (output/"checkpoint.pt").read_bytes() == checkpoint_before
