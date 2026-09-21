# Verified multitrack training — v0.33.0

21 September 2026. The source-synthesis model now has a verified aligned-stem
dataset, supervised optimizer/checkpoint runner and held-out diagnostic
evaluation. Native stereo through **192 kHz / float32** remains a required
target. This release supplies a training path, not trained production weights
or evidence of high-quality separation on mastered music.

## Describe recordings and source provenance

Create an assignment JSON with explicit source slots and track groups:

```json
{
  "source_order": ["bass", "drums", "other"],
  "tracks": [
    {
      "track_id": "composition-a",
      "group_id": "composition-a-family",
      "split": "train",
      "alignment": "sample_aligned",
      "additive": true,
      "mixture": "audio/a/mix.wav",
      "sources": {
        "bass": "audio/a/bass.wav",
        "drums": "audio/a/drums.wav",
        "other": "audio/a/other.wav"
      }
    },
    {
      "track_id": "composition-b",
      "group_id": "composition-b-family",
      "split": "validation",
      "alignment": "sample_aligned",
      "additive": true,
      "mixture": "audio/b/mix.wav",
      "sources": {
        "bass": "audio/b/bass.wav",
        "drums": "audio/b/drums.wav",
        "other": null
      }
    }
  ]
}
```

The example paths are placeholders, resolved relative to the assignment file.
`null` explicitly means a known-silent source. Omitting a source key means
unknown/unannotated and excludes that source from supervised loss. These two
cases must not be conflated. Additive tracks require every source slot to be
present, including explicit silence.

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_multitrack prepare assignments.json --output multitracks.json
```

Each mixture/stem gets raw-file and decoded-PCM hashes. The manifest checks
rate, channel count, length and group/split agreement. Exact reused audio
assets must remain in the same declared group, even if stored in different
containers. Duplicate mixtures and duplicate track IDs are rejected. Groups
must not cross splits. Relative saved paths allow verified relocation.

`sample_aligned` is a caller assertion. Equal lengths and metadata do not prove
that stems are actually synchronized. No automatic time shifting, resampling,
downmixing, gain normalization or stem remapping occurs. Alternate renders,
time shifts, remasters and near duplicates are not identified by exact hashes;
curate provenance before making a generalization claim.

For `additive=true`, preparation and dataset construction also compare the
**entire** mixture with the sum of annotated stems in bounded float64 blocks,
using `rtol=1e-5, atol=1e-6`. This catches failures outside selected crops.
It is a numerical consistency check, not proof of source identity. Nonlinear
bus processing or dry-source restoration may not obey additive mixing;
declare that honestly and leave the mixture loss disabled.

## Native-rate loading and supervised updates

`MultitrackDataset(manifest, split, AudioCropConfig(...))` verifies every
selected asset at construction. Every crop then checks source headers and
file size/mtime while reading exactly the mixture's selected interval from
each annotated stem. This detects ordinary source changes; it is not a
defense against adversarial filesystem tampering. Known silence is created
as zero tensors and requires no duplicate silent files.

Random crop positions retain the existing source/seed/epoch/slot contract;
fixed crops deduplicate repeated starts on short files. Window identities
include source metadata as well as the mixture interval. Changing a target
therefore changes both crop identity and checkpoint provenance. Precision,
sample rate, channels, token hop and model source order must agree.

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_multitrack train multitracks.json --output outputs/separator --steps 100 --updates-per-invocation 10 --window-seconds 0.5 --batch-size 2
```

The reference CLI creates a compact-wavelet hybrid separator with local/global
context and a waveform residual. It defaults to CPU and float32; `--device
cuda` selects CUDA. It infers the native rate and channel count from metadata,
then requires all selected train/validation tracks to match. At 192 kHz it
does not substitute lower-rate audio. By default, token hop, padding and loss
FFT sizes scale approximately with sample rate to retain physical duration.
Explicit `--hop-length` and `--fft-sizes` override those choices.

`--backend` selects dense wavelet, compact wavelet or STFT; existing coefficient
and decoder allocation limits still apply. Defaults are a reference recipe,
not a recommended perceptual-quality preset. Very large dense configurations
may exceed their limits; choose the window/backend deliberately.

The supervised runner uses AdamW, warmup/cosine scheduling, finite-loss/gradient
checks, clipping and length-bucket sampling. `--accumulation-steps` combines
microbatches into one optimizer update. Waveform/complex spectral losses are
weighted by actual annotated channel-sample count; the optional mixture term
uses mixture channel-sample count. Unequal lengths or missing labels do not
give small microbatches disproportionate weight. `--mixture-weight` defaults
to zero and requires complete additive training targets when enabled.

Each example runs with its own real length and crop origin. Padding never
becomes acoustic context. The runner reads one microbatch at a time, though
dataset verification scans all selected assets. This is a single-process
reference implementation, not a distributed or mixed-precision data engine.

## Resume the exact corpus and recipe

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_multitrack train multitracks.json --output outputs/separator --steps 100 --updates-per-invocation 10 --window-seconds 0.5 --batch-size 2 --resume outputs/separator/checkpoint.pt
```

Keep the original corpus, crop, model, loss, step-budget and runtime settings.
Checkpoints include weights, optimizer state, CPU/CUDA RNG, next epoch/batch
position, both dataset identities, source definitions and implementation
identity. Config/provenance/device/precision/runtime mismatches are rejected
before model weights are changed. Same-environment CPU regressions reproduce
the next crops, dropout-dependent loss and parameter update exactly, including
when crossing an epoch boundary. Hardware migrations require intentional
weights loading and a new run; bitwise reproducibility across environments
is not promised.

The local 192 kHz CUDA smoke completed with finite training/evaluation, but
default CUDA execution was not bitwise identical after resume: maximum
parameter difference was about `9.4e-9`. Restoring RNG does not by itself
make all accelerator operations deterministic. See the validation record;
this release's exact parameter-resume regression is scoped to CPU.

Each `SeparationTrainingRun.step()` completes one accumulated optimizer update.
A failure before the optimizer runs discards gradients and restores crop/RNG
position; retry does not silently skip the failed data. An optimizer exception
may have changed parameters partially, so that run requires checkpoint
recovery before continuing or saving. Writes use an atomic replacement at
completed-update boundaries. The CLI saves initially, periodically
(`--save-every`) and at invocation end. A nonempty output directory requires
explicit resume. An interrupted process resumes from its last saved update.
An existing output directory's recorded recipe must also match; a different
corpus/configuration is rejected before its checkpoint can be overwritten.

## Held-out reports

`evaluate_separation` requires fixed validation/test crops. The training run's
validation dataset must be a fixed validation split from the same manifest;
the CLI does not consume the reserved test split. Reports restore the model's
original module modes and include:

- Common waveform/complex losses for the model, equal-split mixture,
  mixture copied to every source, and silence.
- Loss weighted by supervised samples, equal-weight group means, and each
  individual group/window result.
- Per-window/per-source waveform error, target/output/error power and
  gain-sensitive SNR. Silent targets have `snr_db: null`; their output power
  remains available to measure unwanted output. SNR caps at 300 dB using a
  target-relative floor and is not SI-SDR.
- Model/corpus/crop identity, source order and exact sample origins.

Reports are named `evaluation-step-N.json`. They cover fixed crops; overlapping
crops may count a physical sample more than once. They do not establish
full-song performance, persistent source identity or perceptual quality.
The existing `separate_audio` utility renders longer waveforms, but this
release does not turn the evaluator into a file-streaming full-song benchmark.

## Public interfaces and remaining work

`MultitrackRecord`, `MultitrackManifest`, `MultitrackWindow`, `MultitrackDataset`
and `collate_multitracks` expose the data path. `SeparationRunConfig` and
`SeparationTrainingRun(model, training, validation=None, config=...)` expose
training, checkpoint and evaluation operations. Manual/custom trainers can
continue using `SourceBatch`, `HybridSeparator` and `separation_loss` directly.

[Validation](benchmarks/MULTITRACK_VALIDATION.md) records generated-fixture
tests, including 192 kHz float32 and the complete CLI lifecycle. No existing
loop library was relabeled as independent multitrack ground truth. Next:
curated real multitracks, complete-song/boundary listening tests, native-192 kHz
resource profiles, stronger matched separation baselines and instrument-instance
tracking. Neural time/pitch-controlled synthesis remains a separate gap.
