# Trainable source synthesis — v0.32.0

**v0.33.0 update:** [Verified multitrack training](SUPERVISED_MULTITRACK.md)
now implements the aligned-stem loader, supervised optimizer/resume recipe
and held-out crop diagnostics listed as next steps below. Real-music and
full-song perceptual qualification remain open.

21 September 2026. This release implements the first synthesis step from the
[music AI capability plan](MUSIC_AI_CAPABILITY_PLAN.md): a reusable supervised
separator with complex-mask prediction, optional waveform residuals and
offline overlap-add rendering. It is an untrained reference architecture,
not a production separator, neural time stretcher or note synthesizer.

**Required high-resolution target: up to 192 kHz, 32-bit floating-point
audio.** This applies to native-rate ingestion, model inputs/outputs and
floating-point export. See [the format contract](HIGH_RESOLUTION_AUDIO.md)
for explicit WAV export and the limits of current numerical qualification.

## Model and supervised training

```python
import torch
from wavelet2vec import (
    HybridConfig, SynthesisConfig, MusicContextConfig, HybridSeparator,
    SourceExample, collate_sources, separation_loss,
)

frontend = HybridConfig(
    sample_rate=48000, backend="compact_wavelet", n_bands=32, q=2,
    hop_length=128, padding_samples=512,
)
synthesis = SynthesisConfig(source_names=("instrument", "remainder"), dim=32)
context = MusicContextConfig(
    dim=32, heads=4, local_layers=1, global_layers=1,
    local_tokens=16, dropout=0,
)
model = HybridSeparator(frontend, synthesis, context)

# Random tensors demonstrate the data contract, not instrument supervision.
instrument = torch.randn(2, 2049) * .05
remainder = torch.randn(2, 2049) * .05
batch = collate_sources([
    SourceExample(instrument + remainder,
                  {"instrument": instrument, "remainder": remainder},
                  sample_rate=48000, additive=True)
], source_order=synthesis.source_names)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
prediction = model(batch.mixture, lengths=batch.lengths)
losses = separation_loss(prediction.waveform, batch, mixture_weight=.1)
losses["loss"].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
optimizer.step()
```

Use real aligned multitrack waveforms for meaningful learning. Source order
must match the model; source names label output slots and are not text queries.
Missing annotations and known-silent targets retain the existing distinct
`SourceBatch` contracts. A batch with true lengths is processed example by
example to prevent padded tails from entering analysis or context. Its output
tails are zero. This reference loop trades throughput for isolation.

`HybridSeparator(frontend, synthesis, context=None)` omits the context network.
`branches="waveform"`, `"spectral"` or `"both"` controls encoder features.
The spectral synthesis carrier remains present even with waveform-only
features; that option is not a waveform-only end-to-end separator baseline.
No masked-SSL checkpoint is required; this model can train directly on sources.
Transferring compatible encoder/context weights from SSL is an explicit
weights migration, not an automatic part of this example.

## Decoder and signal contract

`HybridSourceDecoder(frontend, synthesis)(tokens, encoding)` accepts tokens
`[B,ceil(N/hop),D]` on the encoder's patch grid and returns `SourceSynthesis`.
Its `waveform` is `[B,S,C,N]`; `lengths`, `source_names`, `sample_rate` and
`sample_origin` describe the complete output intervals. `latency_samples=None`
means no causal/live latency guarantee. There is no resampling, normalization,
channel mixing, limiter or integer quantization at this boundary.

Each token predicts a complex mask per source/channel/band. Masks interpolate
at actual STFT frame centers, dense-wavelet samples or compact-band sample
coordinates. They are held at their nearest endpoint outside the token grid.
The shortened final token uses its actual center; absolute origins do not
enter mask interpolation through a lossy float32 subtraction.

Masks are unbounded and can represent phase changes and gains beyond one.
They cannot independently recover a source at a zero mixture coefficient.
The optional residual head predicts overlapping waveform patches of length
`2*hop`, spanning `[j*hop-hop,j*hop+hop)` for token `j`. Midpoint-Hann weighted
overlap-add with explicit normalization combines them. This allows output
independent of a zero coefficient, but does not remove source ambiguity.

Both heads initialize to zero corrections. The complex path initially emits
`mixture / source_count` through the coefficient inverse, and the residual
is zero. A one-source initialization is a numerical identity. At that exact
initialization the zero head weights block gradients to the encoder/context;
those gradients become available after the heads update. This behavior is
tested over two supervised updates.

Sources are **not** automatically forced to sum to the mixture. Use the
existing guarded mixture loss only for complete additive targets. Dry
restoration of a nonlinearly mastered mix has a different target contract.
The decoder accepts float32 or float64 when model, tokens and carriers match;
it does not silently reduce precision. General mixed-precision training is
not qualified by this release.

`max_output_bytes` guards an estimate of explicit synthesis payloads before
the decoder heads run. It is not a bound on FFT workspaces, autograd, allocator
overhead, model parameters or full-recording output storage. Frontend
coefficient limits still apply independently.

## Coefficient consistency

```python
from wavelet2vec import project_source_coefficients

latent = model.encoder(batch.mixture)
decoded = model.decoder(latent.tokens, latent.encoding, return_coefficients=True)
projected = project_source_coefficients(latent.encoding, decoded.coefficients)
spectral_audio = latent.encoding.synthesize(projected)
```

Redundant transforms permit inconsistent predicted coefficient arrays.
`project_source_coefficients` synthesizes then reanalyzes the spectral
component with the same frontend and boundary padding. The projection
preserves its synthesized waveform and is idempotent to numerical tolerance;
gradients pass through it. This is a diagnostic/building block, not an
additional default loss. Compact outputs remain ragged tuples.

Returned `coefficients` represent the spectral component **before** the
waveform residual. Reanalyze the final waveform if its complete coefficients
are required. Projection adds analysis cost and obeys the frontend's memory
guard with source and batch dimensions combined.

## Longer recordings

```python
from wavelet2vec import separate_audio

rendered = separate_audio(
    model, batch.mixture, window_samples=1024,
    overlap_samples=256, context_samples=512,
)
assert rendered.waveform.shape == batch.sources.shape
```

Window, overlap and context sizes must align to the token hop. Each model
invocation receives available real context; core predictions are blended
with normalized positive Hann weights. The final incomplete window is
retained, global sample origins travel through analysis, and all original
module train/eval modes are restored, including on failure.

Inference disables gradients and bounds model-window size. Input, complete
output and the normalization vector remain resident: this is not yet a
file-reader/file-writer streaming engine. Context and overlap reduce seams,
but do not guarantee equivalence with whole-recording inference. The model
has no persistent cross-window memory or instrument-instance assignment.
Context length remains a caller choice subject to the enabled filters and
network; no universal artifact-free setting is claimed.

## Reproducible experiment and next work

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.supervised_separation --steps 60
```

The experiment trains each backend on identical synthetic stereo examples
and evaluates independently generated examples against an equal-split
baseline. Sources occupy distinct registers and fixed stereo pans, so this
is an easy task with shortcuts. Different backend dimensions produce
different parameter counts; it is not a matched-capacity architecture study.
The report includes source SNR, common loss, parameter counts and chunk/full
differences. See [validation](benchmarks/SOURCE_SYNTHESIS_VALIDATION.md) and
[raw results](benchmarks/source_synthesis_smoke.json).

The remaining priority is a verified **supervised multitrack** manifest,
aligned crop loader, trainer/checkpoint contract and held-out full-song
evaluation. The existing corpus runner is for masked acoustic pretraining;
it does not automatically supply aligned stems. Then qualify native 192 kHz
training/rendering on real material and add stronger decoders. Variable-time
output, pitch controls, reference-note generation and carrier-free synthesis
remain separate planned extensions.
