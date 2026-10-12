"""Reusable study lifecycle, verified caches and resumable native-rate workflows."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import create_percussion_study, PercussionStudy, load_separator_bundle
from wavelet2vec.feature_cache import source_fingerprint


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def project(tmp_path, rate=8000):
    data = tmp_path/"audio"
    data.mkdir()
    t = np.arange(round(rate*.2))/rate
    a = .1*np.sin(2*np.pi*180*t)*((t % .1) < .05)
    b = .05*np.sin(2*np.pi*900*t)*(((t-.025) % .1) < .04)
    for name, value in (("only_a", a), ("only_b", b), ("a", a), ("a+b", a+b)):
        sf.write(data/f"{name}.wav", np.stack((value, -.7*value), -1).astype(np.float32), rate, subtype="FLOAT")
    path = tmp_path/"study.json"
    value = create_percussion_study(data, path, bpm=1200, bars=2, train_bars=1,
                                     beats_per_bar=2, group_id="generated-family", additive=True)
    base = {**value["variants"]["baseline"], "steps": 3, "window_seconds": .02,
            "crops_per_mixture": 1, "batch_size": 1, "bands": 8, "q": 1., "dim": 8}
    value["variants"] = {"baseline": base, "balanced": {**base, "balance_sources": True}}
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, data


def test_discovery_geometry_and_originals_are_explicit(tmp_path):
    path, data = project(tmp_path, rate=192000)
    value = json.loads(path.read_text())
    assert value["body_samples"] == 38400 and value["train_end"] == 19200
    assert [s["name"] for s in value["sources"]] == ["a", "b"]
    with pytest.raises(FileExistsError):
        create_percussion_study(data, path, bpm=1200, bars=2, train_bars=1, group_id="x", beats_per_bar=2)
    sf.write(data/"a+unknown.wav", np.zeros((38400, 2), dtype=np.float32), 192000, subtype="FLOAT")
    with pytest.raises(ValueError, match="unknown"):
        create_percussion_study(data, tmp_path/"bad.json", bpm=1200, bars=2, train_bars=1, group_id="x", beats_per_bar=2)


def test_audit_cache_avoids_reanalysis_and_detects_corruption(tmp_path, monkeypatch):
    path, _ = project(tmp_path)
    study = PercussionStudy(path, tmp_path/"work")
    first = study.audit()
    assert first["cache_misses"] == 2 and first["cache_hits"] == 0
    import wavelet2vec.study as module
    def forbidden(*args, **kwargs):
        raise AssertionError("cached analysis should not execute")
    monkeypatch.setattr(module, "analyze_drum_track", forbidden)
    again = study.audit()
    assert again["cache_hits"] == 2
    assert [r["analysis"] for r in first["sources"]] == [r["analysis"] for r in again["sources"]]
    cache = next((study.output/"analysis_cache").glob("*.json"))
    value = json.loads(cache.read_text()); value["result"]["repeat_groups"] += 1
    cache.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        study.audit()
    assert not (study.output/".study.lock").exists()


@pytest.mark.parametrize("rate", [8000, 192000])
def test_resume_preparation_reuse_evaluation_cache_and_audio(tmp_path, rate):
    path, data = project(tmp_path, rate=rate)
    originals = {p: source_fingerprint(p) for p in data.glob("*.wav")}
    study = PercussionStudy(path, tmp_path/"work")
    first = study.train(updates=1)
    prepared = {p: (source_fingerprint(p), p.stat().st_mtime_ns) for p in (study.output/"prepared").glob("*.wav")}
    directory = study.output/first["directory"]
    with (directory/"history.jsonl").open("a") as log:
        log.write('{"update_step":999}\n{incomplete')
    second = PercussionStudy(path, study.output).train(updates=2)
    assert second["previous_step"] == 1 and second["update_step"] == 3 and second["complete"]
    assert [json.loads(line)["update_step"] for line in (directory/"history.jsonl").read_text().splitlines()] == [1, 2, 3]
    assert prepared == {p: (source_fingerprint(p), p.stat().st_mtime_ns) for p in prepared}
    evaluation = study.evaluate()
    assert evaluation["interval"] == [round(.1*rate), round(.2*rate)] and not evaluation["cache_hit"]
    (study.output/"report.html").unlink()
    cached = study.evaluate()
    assert cached["cache_hit"] and (study.output/"report.html").exists()
    for row in evaluation["rows"]:
        for name in row["audition_hashes"]:
            if name.endswith(".wav"):
                info = sf.info(study.output/row["audition"]/name)
                assert info.samplerate == rate and info.subtype == "FLOAT" and info.frames == round(.2*rate)
    assert originals == {p: source_fingerprint(p) for p in originals}
    other = PercussionStudy(path, tmp_path/"uninterrupted")
    full = other.train(updates=3)
    a = load_separator_bundle(directory/"separator.w2v").model
    b = load_separator_bundle(other.output/full["directory"]/"separator.w2v").model
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    changed = study.output/evaluation["rows"][0]["audition"] / "source_000.wav"
    sf.write(changed, np.zeros((round(.2*rate), 2), dtype=np.float32), rate, subtype="FLOAT")
    with pytest.raises(ValueError, match="audition"):
        study.evaluate()


def test_workspace_input_and_lock_guards(tmp_path):
    path, data = project(tmp_path)
    study = PercussionStudy(path, tmp_path/"work")
    lock = study.output/".study.lock"; lock.write_text("other process")
    with pytest.raises(FileExistsError):
        study.audit()
    assert lock.read_text() == "other process"
    lock.unlink()
    value = json.loads(path.read_text()); value["variants"]["baseline"]["q"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs"):
        PercussionStudy(path, study.output)
    sf.write(data/"only_a.wav", np.zeros((1600, 2), dtype=np.float32), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="changed"):
        study.audit()


def test_profile_is_disposable_and_preserves_training_checkpoint(tmp_path):
    path, _ = project(tmp_path)
    study = PercussionStudy(path, tmp_path/"work")
    progress = study.train(updates=1)
    checkpoint = study.output/progress["directory"]/"checkpoint.pt"
    before = source_fingerprint(checkpoint)
    rng = torch.get_rng_state().clone()
    profile = study.profile(iterations=1, warmup=0)
    assert profile["median_seconds"] > 0 and profile["operators"]
    assert (study.output/profile["trace"]).is_file()
    assert before == source_fingerprint(checkpoint) and torch.equal(rng, torch.get_rng_state())


def test_analysis_and_evaluation_budgets_stop_large_requests(tmp_path, monkeypatch):
    path, _ = project(tmp_path)
    value = json.loads(path.read_text())
    value["max_analysis_bytes"] = value["max_evaluation_bytes"] = 1
    path.write_text(json.dumps(value))
    study = PercussionStudy(path, tmp_path/"work")
    with pytest.raises(ValueError, match="max_analysis_bytes"):
        study.audit()
    study.train(updates=1)
    def forbidden(*args):
        raise AssertionError("budget must be checked before reading full development audio")
    monkeypatch.setattr(study, "_read", forbidden)
    with pytest.raises(ValueError, match="max_evaluation_bytes"):
        study.evaluate()
    assert not (study.output/".study.lock").exists()


def test_cli_runs_and_resumes_project(tmp_path):
    path, _ = project(tmp_path)
    def execute(command):
        result = subprocess.run([sys.executable, "-m", "wavelet2vec.study_cli", *command], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    work = tmp_path/"work"
    first = execute(["run", str(path), "--output", str(work), "--updates", "1"])
    assert first[0]["cache_misses"] == 2 and first[1]["update_step"] == 1
    second = execute(["run", str(path), "--output", str(work), "--updates", "1"])
    assert second[0]["cache_hits"] == 2 and second[1]["previous_step"] == 1 and second[1]["update_step"] == 2
