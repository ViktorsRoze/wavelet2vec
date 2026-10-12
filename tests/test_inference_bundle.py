"""Portable inference identity, archive validation, precision and CLI integration."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import (HybridConfig, SynthesisConfig, MusicContextConfig, HybridSeparator, SourceRenderConfig,
    save_separator_bundle, inspect_separator_bundle, load_separator_bundle, MultitrackRecord, MultitrackManifest,
    MultitrackDataset, AudioCropConfig, SeparationTrainingRun, SeparationRunConfig)


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def network(backend="compact_wavelet", context=True, dtype=torch.float32):
    torch.manual_seed(23)
    front = HybridConfig(sample_rate=192000, backend=backend, n_bands=12, q=1, hop_length=16, n_fft=64, padding_samples=32)
    synth = SynthesisConfig(dim=8, source_names=("first", "second"))
    ctx = MusicContextConfig(dim=8, heads=2, local_layers=1, global_layers=1, local_tokens=4) if context else None
    model = HybridSeparator(front, synth, ctx).to(dtype=dtype)
    with torch.no_grad():
        model.decoder.mask_head.weight.normal_(0, .01)
        model.decoder.residual_head.weight.normal_(0, .01)
    return model


def export(tmp_path, **options):
    model = network(**options)
    path = tmp_path/"separator.w2v"
    identity = save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32, 32), metadata={"fixture": True})
    return model, path, identity


def rewrite(path, transform):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    transform(manifest, entries)
    manifest.pop("bundle_id", None)
    manifest["bundle_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundtrip_architecture_weights_predictions_modes_and_rng(tmp_path, backend, dtype):
    model = network(backend, dtype=dtype)
    model.train()
    model.encoder.eval()
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    path = tmp_path/"model.w2v"
    identity = save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32, 32))
    assert torch.equal(rng, torch.get_rng_state()) and modes == [m.training for m in model.modules()]
    inspected = inspect_separator_bundle(path)
    loaded = load_separator_bundle(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert inspected["weights_verified"] and inspected["bundle_id"] == loaded.bundle_id == identity
    assert all(not m.training for m in loaded.model.modules())
    assert all(not p.requires_grad for p in loaded.model.parameters())
    model.eval()
    audio = torch.randn(1, 2, 257, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(model(audio).waveform, loaded.model(audio).waveform, rtol=0, atol=0)
    assert next(loaded.model.parameters()).dtype == dtype
    assert all(p.requires_grad for p in load_separator_bundle(path, trainable=True).model.parameters())


def test_stable_content_id_optional_context_and_existing_path(tmp_path):
    model, path, identity = export(tmp_path, context=False)
    second = tmp_path/"second.w2v"
    assert save_separator_bundle(second, model, render_config=SourceRenderConfig(128, 32, 32), metadata={"fixture": True}) == identity
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32, 32))
    assert path.read_bytes() == before
    save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32, 32), overwrite=True)
    assert load_separator_bundle(path).model.context is None


@pytest.mark.parametrize("problem", ["checksum", "shape", "dtype", "extra", "nonfinite", "oversized_architecture", "array_header"])
def test_corrupt_or_incompatible_bundles_are_rejected(tmp_path, problem):
    _, path, _ = export(tmp_path)
    def corrupt(manifest, entries):
        row = manifest["tensors"][0]
        if problem == "checksum":
            entries[row["member"]] = entries[row["member"]][:-1]+b"X"
        elif problem == "shape":
            row["shape"][0] += 1
        elif problem == "dtype":
            row["dtype"] = "float16"
        elif problem == "extra":
            entries["../unexpected.npy"] = b"bad"
        elif problem == "oversized_architecture":
            manifest["spec"]["context"]["local_layers"] = 1000000
        elif problem == "array_header":
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.zeros((1,), dtype=np.float64), version=(1, 0), allow_pickle=False)
            entries[row["member"]] = buffer.getvalue()
            row["sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        else:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.full(row["shape"], np.nan, dtype=row["dtype"]), version=(1, 0), allow_pickle=False)
            entries[row["member"]] = buffer.getvalue()
            row["sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    rewrite(path, corrupt)
    with pytest.raises(ValueError):
        inspect_separator_bundle(path)
    with pytest.raises(ValueError):
        load_separator_bundle(path)


def test_limits_producer_mismatch_and_override_contract(tmp_path):
    _, path, _ = export(tmp_path)
    with pytest.raises(ValueError, match="weight payload"):
        load_separator_bundle(path, max_weight_bytes=1)
    rewrite(path, lambda manifest, entries: manifest["producer"].update(implementation="0"*64))
    loaded = load_separator_bundle(path)
    assert loaded.implementation_matches is False
    with pytest.raises(ValueError, match="implementation"):
        load_separator_bundle(path, require_same_implementation=True)


def test_nonfinite_export_and_custom_behavior_do_not_publish(tmp_path):
    model = network()
    path = tmp_path/"model.w2v"
    handle = model.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooks"):
        save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32))
    handle.remove()
    with torch.no_grad():
        model.decoder.mask_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32))
    assert not path.exists() and not list(tmp_path.glob("*.tmp"))


def test_failed_overwrite_preserves_previous_artifact(tmp_path):
    model, path, identity = export(tmp_path)
    before = path.read_bytes()
    with torch.no_grad():
        model.decoder.residual_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_separator_bundle(path, model, render_config=SourceRenderConfig(128, 32), overwrite=True)
    assert before == path.read_bytes()
    assert inspect_separator_bundle(path)["bundle_id"] == identity
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("dtype,subtype", [(torch.float32, "FLOAT"), (torch.float64, "DOUBLE")])
def test_bundle_render_uses_saved_branch_and_native_export_precision(tmp_path, dtype, subtype):
    model = network(dtype=dtype)
    config = SourceRenderConfig(128, 32, 32)
    path = tmp_path/"model.w2v"
    save_separator_bundle(path, model, render_config=config, branches="waveform")
    bundle = load_separator_bundle(path)
    assert bundle.branches == "waveform" and bundle.render_config == config
    audio = np.random.default_rng(20).normal(0, .1, (259, 2)).astype(np.float32 if dtype == torch.float32 else np.float64)
    source = tmp_path/"input.wav"
    sf.write(source, audio, 192000, subtype=subtype)
    actual = bundle.render_file(source, tmp_path/"stems")
    assert actual.subtype == subtype and actual.frames == 259
    from wavelet2vec import separate_audio
    expected = separate_audio(model, torch.from_numpy(audio.T.copy())[None], window_samples=128,
                              overlap_samples=32, context_samples=32, branches="waveform").waveform
    for slot, output in enumerate(actual.files):
        decoded, rate = sf.read(output, dtype="float32" if dtype == torch.float32 else "float64", always_2d=True)
        np.testing.assert_array_equal(decoded, expected[0, slot].numpy().T)
        assert rate == 192000


def test_cli_inspection_native_render_and_provenance(tmp_path):
    _, path, identity = export(tmp_path)
    source = tmp_path/"input.wav"
    sf.write(source, np.random.default_rng(7).normal(0, .1, (259, 2)).astype(np.float32), 192000, subtype="FLOAT")
    def execute(args):
        result = subprocess.run([sys.executable, "-m", "wavelet2vec.inference_cli", *args],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
        return json.loads(result.stdout)
    assert execute(["inspect", str(path)])["weights_verified"]
    output = tmp_path/"stems"
    report = execute(["render", str(path), str(source), "--output", str(output)])
    assert report["sample_rate"] == 192000 and report["frames"] == 259 and report["subtype"] == "FLOAT"
    receipt = json.loads((output/"render.json").read_text())
    assert receipt["provenance"]["bundle_id"] == identity
    assert sf.info(report["files"][0]).frames == 259


def test_training_run_export_records_provenance_and_updated_weights(tmp_path):
    x = np.random.default_rng(5).normal(0, .1, (128, 2)).astype(np.float32)
    for name, value in (("mix", x), ("a", x*.3), ("b", x*.7)):
        sf.write(tmp_path/f"{name}.wav", value, 192000, subtype="FLOAT")
    record = MultitrackRecord.inspect("track", tmp_path/"mix.wav", {"first": tmp_path/"a.wav", "second": tmp_path/"b.wav"},
                                      group_id="generated", split="train", additive=True)
    manifest = MultitrackManifest([record], source_order=("first", "second"))
    dataset = MultitrackDataset(manifest, "train", AudioCropConfig(192000, 2, 128, hop_length=16))
    active = SeparationTrainingRun(network(), dataset, config=SeparationRunConfig(total_steps=2, warmup_steps=0, fft_sizes=(32,)))
    active.step()
    path = tmp_path/"trained.w2v"
    active.export_bundle(path, render_config=SourceRenderConfig(128, 32))
    bundle = load_separator_bundle(path)
    assert bundle.metadata["training"]["update_step"] == 1
    assert bundle.metadata["training"]["data"]["training"] == dataset.identity
    for original, loaded in zip(active.model.parameters(), bundle.model.parameters()):
        assert torch.equal(original, loaded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_bundle_loads_on_cuda_without_changing_precision(tmp_path):
    _, path, _ = export(tmp_path)
    cpu, gpu = load_separator_bundle(path), load_separator_bundle(path, device="cuda")
    x = torch.randn(1, 2, 129)
    with torch.no_grad():
        a, b = cpu.model(x).waveform, gpu.model(x.cuda()).waveform.cpu()
    torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    assert next(gpu.model.parameters()).dtype == torch.float32
