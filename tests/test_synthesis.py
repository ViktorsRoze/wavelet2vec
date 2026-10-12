"""Source rendering contracts; numerical tests are not perceptual benchmarks."""
from dataclasses import replace
import json

import pytest
import torch

from wavelet2vec import (
    HybridAudioFrontend, HybridConfig, HybridSeparator, HybridSourceDecoder,
    MusicContextConfig, SynthesisConfig, project_source_coefficients, separate_audio,
    SourceExample, collate_sources, separation_loss,
)


def config(backend="compact_wavelet", rate=48000):
    return HybridConfig(sample_rate=rate, backend=backend, n_bands=12, q=1,
                        f_min=40, padding_samples=32, hop_length=16, n_fft=64)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
@pytest.mark.parametrize("rate", [48000, 96000, 192000])
@pytest.mark.parametrize("n", [1, 127, 256])
def test_initialized_source_synthesis_stereo_endpoints_and_precision(backend, rate, n):
    torch.manual_seed(n)
    cfg = config(backend, rate)
    model = HybridSeparator(cfg, SynthesisConfig(dim=8)).double()
    x = torch.randn(1, 2, n, dtype=torch.float64) * .1
    x[:, 1] = -x[:, 0]
    x[..., 0] += .2
    result = model(x, sample_origin=987654321)
    torch.testing.assert_close(result.waveform[:, 0], x/2, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result.waveform.sum(1), x, atol=1e-12, rtol=1e-12)
    assert result.waveform.dtype == x.dtype
    assert result.sample_origin == 987654321 and result.sample_rate == rate
    assert result.lengths.tolist() == [n] and result.latency_samples is None


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_masks_use_physical_grid_and_partial_final_token(backend):
    cfg = replace(config(backend), hop_length=4)
    encoding = HybridAudioFrontend(cfg)(torch.randn(1, 1, 9, dtype=torch.float64), sample_origin=10**12)
    decoder = HybridSourceDecoder(cfg, SynthesisConfig(channels=1, dim=1, waveform_residual=False)).double()
    with torch.no_grad():
        decoder.mask_head.weight[::2] = 1
        decoder.mask_head.bias[::2] = -.5
    tokens = torch.tensor([[[2.], [6.], [8.5]]], dtype=torch.float64)
    predicted = decoder(tokens, encoding, return_coefficients=True).coefficients
    if backend == "compact_wavelet":
        for value, original in zip(predicted, encoding.coefficients):
            positions = torch.arange(original.shape[-1], dtype=torch.float64)*(9+2*cfg.padding)/original.shape[-1]-cfg.padding
            expected = original[:, None] * positions.clamp(2, 8.5)
            torch.testing.assert_close(value, expected.expand_as(value))
    else:
        positions = torch.arange(encoding.coefficients.shape[-1], dtype=torch.float64)
        positions = positions*cfg.hop_length if backend == "stft" else positions-cfg.padding
        expected = encoding.coefficients[:, None] * positions.clamp(2, 8.5)
        torch.testing.assert_close(predicted, expected.expand_as(predicted))


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_prediction_projection_preserves_audio_is_idempotent_and_differentiable(backend):
    torch.manual_seed(21)
    encoding = HybridAudioFrontend(config(backend))(torch.randn(1, 2, 129, dtype=torch.float64))
    decoder = HybridSourceDecoder(encoding.config, SynthesisConfig(dim=8, waveform_residual=False)).double()
    with torch.no_grad():
        decoder.mask_head.weight.normal_(0, .2)
    tokens = torch.randn(1, 9, 8, dtype=torch.float64, requires_grad=True)
    result = decoder(tokens, encoding, return_coefficients=True)
    projected = project_source_coefficients(encoding, result.coefficients)
    again = project_source_coefficients(encoding, projected)
    torch.testing.assert_close(encoding.synthesize(projected), result.waveform, atol=1e-12, rtol=1e-12)
    values = projected if isinstance(projected, tuple) else (projected,)
    repeats = again if isinstance(again, tuple) else (again,)
    for a, b in zip(values, repeats):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    sum(value.abs().square().mean() for value in values).backward()
    assert torch.isfinite(tokens.grad).all() and tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_supervised_updates_reach_encoder_context_and_both_heads(backend):
    torch.manual_seed(14)
    cfg = config(backend)
    context = MusicContextConfig(dim=8, heads=2, local_layers=1, global_layers=1, local_tokens=4, dropout=0)
    model = HybridSeparator(cfg, SynthesisConfig(dim=8), context)
    a, b = torch.randn(2, 127)*.1, torch.randn(2, 127)*.1
    batch = collate_sources([SourceExample(a+b, {"target": a, "remainder": b}, cfg.sample_rate, True)],
                            source_order=model.config.source_names)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        output = model(batch.mixture)
        loss = separation_loss(output.waveform, batch, fft_sizes=(32, 64))["loss"]
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    for module in (model.encoder, model.context, model.decoder.mask_head, model.decoder.residual_head):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0


def test_residual_can_generate_outside_zero_carrier_without_implicit_additivity():
    cfg = config("stft")
    decoder = HybridSourceDecoder(cfg, SynthesisConfig(dim=8))
    encoding = HybridAudioFrontend(cfg)(torch.zeros(1, 2, 49))
    with torch.no_grad():
        decoder.residual_head.bias.fill_(.25)
    result = decoder(torch.zeros(1, 4, 8), encoding)
    torch.testing.assert_close(result.waveform, torch.full_like(result.waveform, .25))
    assert result.waveform.sum(1).abs().min() > 0
    result.waveform.square().mean().backward()
    assert torch.isfinite(decoder.residual_head.weight.grad).all()


def test_ragged_batch_padding_is_excluded_from_model_and_gradients():
    torch.manual_seed(6)
    model = HybridSeparator(config(), SynthesisConfig(dim=8))
    with torch.no_grad():
        model.decoder.mask_head.weight.normal_(0, .05)
    x = torch.randn(2, 2, 97)
    x[0, :, 33:] = float("nan")
    x.requires_grad_()
    lengths = torch.tensor([33, 97])
    result = model(x, lengths=lengths)
    expected = model(x[:1, :, :33]).waveform
    torch.testing.assert_close(result.waveform[:1, ..., :33], expected)
    assert (result.waveform[0, ..., 33:] == 0).all()
    result.waveform.square().mean().backward()
    assert torch.isfinite(x.grad).all() and (x.grad[0, :, 33:] == 0).all()


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
@pytest.mark.parametrize("rate", [96000, 192000])
def test_chunked_stereo_identity_length_metadata_and_mode_restoration(backend, rate):
    model = HybridSeparator(config(backend, rate), SynthesisConfig(dim=8)).double()
    model.train()
    model.encoder.eval()
    before = [m.training for m in model.modules()]
    x = torch.randn(1, 2, 513, dtype=torch.float64)
    x[:, 1] = -x[:, 0]
    result = separate_audio(model, x, window_samples=128, overlap_samples=32, context_samples=32, sample_origin=160)
    torch.testing.assert_close(result.waveform.sum(1), x, atol=1e-12, rtol=1e-12)
    assert result.sample_origin == 160 and result.lengths.tolist() == [513]
    assert before == [m.training for m in model.modules()]
    x[..., 300] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        separate_audio(model, x, window_samples=128, overlap_samples=32)
    assert before == [m.training for m in model.modules()]


def test_serialization_guards_and_memory_grid_validation():
    cfg = config()
    synth = SynthesisConfig(dim=8)
    assert SynthesisConfig.from_dict(json.loads(json.dumps(synth.to_dict()))) == synth
    model = HybridSeparator(cfg, synth)
    other = HybridSeparator(cfg, synth)
    other.load_state_dict(model.state_dict())
    mismatch = HybridSeparator(cfg, replace(synth, source_names=("a", "b")))
    with pytest.raises(ValueError, match="configuration"):
        mismatch.load_state_dict(model.state_dict())
    encoding = model.encoder.frontend(torch.zeros(1, 2, 49))
    small = HybridSourceDecoder(cfg, replace(synth, max_output_bytes=1))
    with pytest.raises(ValueError, match="payload"):
        small(torch.zeros(1, 4, 8), encoding)
    with pytest.raises(ValueError, match="grid"):
        model.decoder(torch.zeros(1, 3, 8), encoding)
    with pytest.raises(ValueError, match="aligned"):
        separate_audio(model, encoding.waveform, window_samples=65, overlap_samples=16)


@pytest.mark.parametrize("rate", [48000, 96000, 192000])
def test_high_rate_near_nyquist_dc_and_weak_signal_identity(rate):
    cfg = config("compact_wavelet", rate)
    model = HybridSeparator(cfg, SynthesisConfig(source_names=("identity",), dim=8)).double()
    t = torch.arange(1025, dtype=torch.float64) / rate
    x = (.1 + 1e-7*torch.cos(2*torch.pi*(rate*.49)*t)).expand(1, 2, -1)
    y = model(x).waveform[:, 0]
    torch.testing.assert_close(y, x, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("backend", ["wavelet", "compact_wavelet", "stft"])
def test_192khz_float32_file_corpus_synthesis_training_and_export(tmp_path, backend):
    import numpy as np
    import soundfile as sf
    from wavelet2vec import AudioRecord, AudioManifest, AudioCropConfig, AudioWindowDataset
    from wavelet2vec.audio_io import save_audio, load_audio

    torch.manual_seed(19)
    rate = 192000
    original = torch.randn(2, 4097) * .05
    original[0, :4] = torch.tensor([1.5, -2.25, 1e-12, -1e-12])
    original[1] = -original[0]
    path = save_audio(tmp_path / "native.wav", original.numpy(), rate, subtype="FLOAT")
    info = sf.info(path)
    assert (info.samplerate, info.subtype, info.channels, info.frames) == (rate, "FLOAT", 2, 4097)
    loaded, loaded_rate = load_audio(path)
    assert loaded_rate == rate and loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded.view(np.uint32), original.numpy().view(np.uint32))
    manifest = AudioManifest([AudioRecord.inspect(path, group_id="fixture", split="train")])
    dataset = AudioWindowDataset(manifest, "train", AudioCropConfig(
        sample_rate=rate, channels=2, window_samples=4112, hop_length=16,
        crops_per_record=1, mode="fixed", dtype="float32"))
    audio = dataset[0].waveform.unsqueeze(0)
    assert torch.equal(audio[0], original)
    model = HybridSeparator(config(backend, rate), SynthesisConfig(dim=8))
    rendered = separate_audio(model, audio, window_samples=1024, overlap_samples=256, context_samples=128)
    assert rendered.sample_rate == rate and rendered.waveform.dtype == torch.float32
    assert rendered.lengths.tolist() == [4097]
    torch.testing.assert_close(rendered.waveform.sum(1), audio, atol=2e-6, rtol=2e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        result = model(audio)
        loss = (result.waveform[:, 0] - audio*.3).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.encoder.waveform_projection.weight.grad.abs().sum() > 0
    output = result.waveform[0, 0].detach().numpy()
    destination = save_audio(tmp_path / "source.wav", output, rate, subtype="FLOAT")
    decoded, decoded_rate = load_audio(destination)
    assert decoded_rate == rate and sf.info(destination).subtype == "FLOAT"
    np.testing.assert_array_equal(decoded.view(np.uint32), output.view(np.uint32))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_grouped_compact_decoder_matches_per_band_outputs_and_gradients(device):
    from wavelet2vec.synthesis import _interpolate
    torch.manual_seed(44)
    cfg = replace(config(), n_bands=32, q=2)
    audio = torch.randn(1, 2, 259, dtype=torch.float64, device=device, requires_grad=True)
    encoding = HybridAudioFrontend(cfg)(audio)
    decoder = HybridSourceDecoder(cfg, SynthesisConfig(dim=8, waveform_residual=False)).double().to(device)
    with torch.no_grad():
        decoder.mask_head.weight.normal_(0, .02)
    count = (audio.shape[-1]+cfg.hop_length-1)//cfg.hop_length
    tokens = torch.randn(1, count, 8, dtype=torch.float64, device=device, requires_grad=True)
    actual = decoder(tokens, encoding, return_coefficients=True)
    raw = decoder.mask_head(tokens).reshape(1, count, 2, 2, cfg.bands, 2)
    masks = torch.complex(raw[..., 0], raw[..., 1]).permute(0, 2, 3, 4, 1)+.5
    starts = torch.arange(count, dtype=torch.float64, device=device)*cfg.hop_length
    centers = starts+(audio.shape[-1]-starts).clamp(max=cfg.hop_length)/2
    expected_bands = []
    for k, coefficient in enumerate(encoding.coefficients):
        times = torch.arange(coefficient.shape[-1], dtype=torch.float64, device=device)*(audio.shape[-1]+2*cfg.padding)/coefficient.shape[-1]-cfg.padding
        expected_bands.append(coefficient[:, None]*_interpolate(masks[..., k, :], centers, times))
    expected = encoding.synthesize(tuple(expected_bands))
    torch.testing.assert_close(actual.waveform, expected, rtol=1e-10, atol=1e-11)
    for left, right in zip(actual.coefficients, expected_bands):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
    parameters = [tokens, audio, *decoder.parameters()]
    reference = torch.randn_like(expected)
    gradient_actual = torch.autograd.grad((actual.waveform-reference).square().mean(), parameters, retain_graph=True)
    gradient_expected = torch.autograd.grad((expected-reference).square().mean(), parameters)
    for left, right in zip(gradient_actual, gradient_expected):
        torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-10)
