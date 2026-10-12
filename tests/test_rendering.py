"""Bounded source rendering, output integrity and publication failure cases."""
from dataclasses import replace
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from wavelet2vec import (HybridConfig, HybridSeparator, SynthesisConfig, SourceRenderConfig,
                         iter_separate_reader, separate_audio, separate_file, SeparationCancelled)
from wavelet2vec.rendering import _container


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(backend="compact_wavelet", dtype=torch.float32, rate=192000):
    torch.manual_seed(7)
    cfg = HybridConfig(sample_rate=rate, backend=backend, n_bands=12, q=1, f_min=40,
                       hop_length=16, padding_samples=32, n_fft=64)
    return HybridSeparator(cfg, SynthesisConfig(dim=8)).to(dtype=dtype)


def reference_render(net, audio, config, origin):
    """Independent full-size overlap accumulator to check the rolling algorithm."""
    result = audio.new_zeros(audio.shape[0], 2, 2, audio.shape[-1])
    total = audio.new_zeros(audio.shape[-1])
    net.eval()
    with torch.no_grad():
        for start in range(0, audio.shape[-1], config.window_samples-config.overlap_samples):
            end = min(audio.shape[-1], start+config.window_samples)
            left, right = max(0, start-config.context_samples), min(audio.shape[-1], end+config.context_samples)
            waveform = net(audio[..., left:right], sample_origin=origin+left).waveform[..., start-left:end-left]
            phase = (torch.arange(end-start, dtype=torch.float64)+.5)/(end-start)
            weight = torch.sin(torch.pi*phase).square().to(audio.dtype)
            result[..., start:end] += waveform*weight
            total[start:end] += weight
            if end == audio.shape[-1]:
                break
    return result/total


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
@pytest.mark.parametrize("overlap", [0, 32, 96])
def test_rolling_matches_independent_overlap_add_with_learned_heads(backend, overlap):
    net = model(backend, torch.float64)
    with torch.no_grad():
        net.decoder.mask_head.weight.normal_(0, .01)
        net.decoder.residual_head.weight.normal_(0, .01)
    audio = torch.randn(2, 2, 517, dtype=torch.float64)*.1
    config = SourceRenderConfig(128, overlap, 32)
    expected = reference_render(net, audio, config, 10**9)
    calls = []
    def reader(lo, hi):
        calls.append((lo, hi))
        return audio[..., lo:hi]
    chunks = list(iter_separate_reader(net, reader, audio.shape[-1], config, sample_origin=10**9))
    actual = torch.cat([chunk.waveform for chunk in chunks], -1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    cursor = 10**9
    for chunk in chunks:
        assert chunk.sample_origin == cursor and not chunk.waveform.requires_grad
        assert chunk.lengths.tolist() == [chunk.waveform.shape[-1]]*2
        cursor += chunk.waveform.shape[-1]
    assert cursor == 10**9+517
    assert max(hi-lo for lo, hi in calls) <= 128+64
    torch.testing.assert_close(separate_audio(net, audio, window_samples=128, overlap_samples=overlap,
                              context_samples=32, sample_origin=10**9).waveform, actual, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 16, 128, 129])
def test_short_fragments_and_last_sample_are_preserved(n):
    net = model()
    audio = torch.randn(1, 2, n)*.1
    audio[..., -1] = 1.5
    chunks = list(iter_separate_reader(net, lambda lo, hi: audio[..., lo:hi], n, SourceRenderConfig(128, 96, 32)))
    result = torch.cat([chunk.waveform for chunk in chunks], -1).sum(1)
    torch.testing.assert_close(result, audio, rtol=2e-6, atol=2e-6)


def test_iterator_restores_modes_and_grad_state_before_yield_or_failure():
    net = model()
    net.train()
    net.encoder.eval()
    modes = [module.training for module in net.modules()]
    audio = torch.randn(1, 2, 400)
    iterator = iter_separate_reader(net, lambda lo, hi: audio[..., lo:hi], 400, SourceRenderConfig(128, 32))
    first = next(iterator)
    frozen = first.waveform.clone()
    assert modes == [module.training for module in net.modules()] and torch.is_grad_enabled()
    next(iterator)
    assert torch.equal(first.waveform, frozen)
    iterator.close()
    assert modes == [module.training for module in net.modules()] and torch.is_grad_enabled()
    bad = iter_separate_reader(net, lambda lo, hi: audio[..., lo:hi]*float("nan"), 400, SourceRenderConfig(128, 32))
    with pytest.raises(ValueError, match="nonfinite"):
        next(bad)
    assert modes == [module.training for module in net.modules()]


def test_reader_shape_precision_batch_and_budget_guards():
    net = model()
    calls = []
    config = SourceRenderConfig(128, 32, max_buffer_bytes=1)
    with pytest.raises(ValueError, match="buffer estimate"):
        next(iter_separate_reader(net, lambda lo, hi: calls.append((lo, hi)), 10000, config))
    assert not calls
    with pytest.raises(ValueError, match="model precision"):
        next(iter_separate_reader(net, lambda lo, hi: torch.zeros(1, 2, hi-lo, dtype=torch.float64), 100, SourceRenderConfig(128, 32)))
    with pytest.raises(ValueError, match="batch size changed"):
        list(iter_separate_reader(net, lambda lo, hi: torch.zeros(1 if lo == 0 else 2, 2, hi-lo), 300, SourceRenderConfig(128, 32)))
    assert SourceRenderConfig.from_dict(SourceRenderConfig(128, 32).to_dict()) == SourceRenderConfig(128, 32)


@pytest.mark.parametrize("subtype,dtype", [("FLOAT", torch.float32), ("DOUBLE", torch.float64)])
@pytest.mark.parametrize("container", ["auto", "RF64"])
def test_float_file_render_preserves_format_headroom_and_receipt(tmp_path, subtype, dtype, container):
    net = model(dtype=dtype)
    net.train()
    net.encoder.eval()
    modes = [module.training for module in net.modules()]
    source = tmp_path/"input.wav"
    audio = torch.randn(1, 2, 513, dtype=dtype)*.1
    audio[..., 30:35] = 4.
    sf.write(source, audio[0].numpy().T, 192000, subtype=subtype)
    before = source.read_bytes()
    progress = []
    cfg = SourceRenderConfig(128, 32, 32)
    result = separate_file(net, source, tmp_path/"stems", cfg, subtype=subtype, container=container,
                           progress=lambda done, total: progress.append((done, total)))
    assert modes == [module.training for module in net.modules()]
    assert source.read_bytes() == before
    assert result.frames == 513 and result.sample_rate == 192000 and result.channels == 2
    assert result.container == ("WAV" if container == "auto" else container)
    assert progress[-1] == (513, 513) and all(a[0] < b[0] for a, b in zip(progress, progress[1:]))
    receipt = json.loads(result.manifest.read_text())
    assert receipt["subtype"] == subtype and receipt["frames"] == 513
    assert [row["file"] for row in receipt["sources"]] == [path.name for path in result.files]
    expected = separate_audio(net, audio, window_samples=128, overlap_samples=32, context_samples=32).waveform
    for slot, path in enumerate(result.files):
        values, rate = sf.read(path, dtype="float32" if subtype == "FLOAT" else "float64", always_2d=True)
        assert rate == 192000 and sf.info(path).subtype == subtype
        np.testing.assert_array_equal(values, expected[0, slot].numpy().T)
        assert values.max() > 1
    assert not list(tmp_path.glob(".wavelet2vec-render-*"))


@pytest.mark.parametrize("failure", ["cancel", "callback", "model", "source_changed", "nonfinite", "destination"])
def test_failed_render_never_publishes_partial_files(tmp_path, monkeypatch, failure):
    net = model()
    net.train()
    source, destination = tmp_path/"input.wav", tmp_path/"stems"
    sf.write(source, np.random.default_rng(9).normal(0, .1, (513, 2)).astype(np.float32), 192000, subtype="FLOAT")
    cancelled = False
    forward = net.forward
    calls = 0
    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and failure == "model":
            raise RuntimeError("injected model failure")
        result = forward(*args, **kwargs)
        if calls == 2 and failure == "nonfinite":
            return replace(result, waveform=result.waveform*float("nan"))
        return result
    monkeypatch.setattr(net, "forward", broken)
    def progress(done, total):
        nonlocal cancelled
        if failure == "cancel":
            cancelled = True
        elif failure == "callback":
            raise RuntimeError("injected callback failure")
        elif failure == "source_changed":
            import os
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+10000000))
        elif failure == "destination" and not destination.exists():
            destination.mkdir()
            (destination/"keep.txt").write_text("user file")
    with pytest.raises((SeparationCancelled, RuntimeError, ValueError, FileExistsError, FloatingPointError)):
        separate_file(net, source, destination, SourceRenderConfig(128, 32), cancel=lambda: cancelled, progress=progress)
    assert net.training and torch.is_grad_enabled()
    assert not list(tmp_path.glob(".wavelet2vec-render-*"))
    if failure == "destination":
        assert list(destination.iterdir()) == [destination/"keep.txt"]
        assert (destination/"keep.txt").read_text() == "user file"
    else:
        assert not destination.exists()


def test_existing_destination_rate_and_precision_are_rejected(tmp_path):
    net = model()
    source = tmp_path/"input.wav"
    sf.write(source, np.zeros((32, 2), dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="rate/channels"):
        separate_file(net, source, tmp_path/"stems", SourceRenderConfig(128, 32))
    with pytest.raises(ValueError, match="precision"):
        separate_file(net.double(), source, tmp_path/"stems", SourceRenderConfig(128, 32))
    with pytest.raises(FileExistsError):
        separate_file(net.float(), source, tmp_path, SourceRenderConfig(128, 32))


def test_large_container_selection_and_untrusted_source_labels(tmp_path):
    assert _container(192000*3600, 2, "FLOAT", "auto") == "RF64"
    with pytest.raises(ValueError, match="capacity"):
        _container(192000*3600, 2, "FLOAT", "WAV")
    net = model()
    net = HybridSeparator(net.encoder.frontend.config,
                          SynthesisConfig(source_names=("../escape", "nested/path"), dim=8))
    source = tmp_path/"input.wav"
    sf.write(source, np.zeros((17, 2), dtype=np.float32), 192000, subtype="FLOAT")
    result = separate_file(net, source, tmp_path/"stems", SourceRenderConfig(128, 32))
    assert [path.name for path in result.files] == ["source_000.wav", "source_001.wav"]
    assert all(path.parent == result.directory for path in result.files)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_file_render_uses_native_float32_and_bounded_host_transfer(tmp_path):
    net = model().cuda()
    data = np.random.default_rng(30).normal(0, .1, (1025, 2)).astype(np.float32)
    data[0] = [1.5, -1.5]
    source = tmp_path/"input.wav"
    sf.write(source, data, 192000, subtype="FLOAT")
    result = separate_file(net, source, tmp_path/"stems", SourceRenderConfig(256, 64, 32))
    recovered = sum(sf.read(path, dtype="float32", always_2d=True)[0] for path in result.files)
    np.testing.assert_allclose(recovered, data, atol=2e-6, rtol=2e-6)
    assert next(net.parameters()).device.type == "cuda"
