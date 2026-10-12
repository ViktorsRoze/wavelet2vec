# Portable separator inference — v0.35.0

22 September 2026. Inference bundles reconstruct the supported separator
architecture from versioned settings and weights. They include source order,
native rate/precision, feature-branch choice, render defaults and provenance.
This removes handwritten architecture reconstruction from deployment; it does
not supply pretrained weights or improve musical quality.

## Export, inspect and load

```python
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import torch
from wavelet2vec import (
    HybridConfig, SynthesisConfig, HybridSeparator, SourceRenderConfig,
    save_separator_bundle, inspect_separator_bundle, load_separator_bundle,
)

# This example packages an initialized model, not a trained separator.
separator = HybridSeparator(
    HybridConfig(sample_rate=192000, backend="compact_wavelet", n_bands=16,
                 q=1, hop_length=32, padding_samples=64),
    SynthesisConfig(source_names=("first", "second"), dim=8),
)
recipe = SourceRenderConfig(window_samples=256, overlap_samples=64, context_samples=64)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    identity = save_separator_bundle(
        root / "separator.w2v", separator, render_config=recipe,
        metadata={"training_status": "untrained demonstration"},
    )
    manifest = inspect_separator_bundle(root / "separator.w2v")
    assert manifest["weights_verified"]
    bundle = load_separator_bundle(root / "separator.w2v", device="cpu")
    assert bundle.bundle_id == identity
    signal = np.zeros((1025, 2), dtype=np.float32)
    signal[0] = [1.5, -1.5]
    sf.write(root / "input.wav", signal, 192000, subtype="FLOAT")
    output = bundle.render_file(root / "input.wav", root / "stems")
    assert output.sample_rate == 192000 and output.subtype == "FLOAT"
```

For real projects, export the trained model into a persistent location.
`SeparationTrainingRun.export_bundle(path, render_config=recipe)` additionally
records the optimizer update count, training recipe and dataset identities.
It refuses an active update or a run needing recovery after optimizer failure.
An update count of zero remains possible and does not establish training.

Export leaves the original module modes and RNG unchanged. It supports the
configured built-in `HybridSeparator`, with or without `MusicContextEncoder`,
across dense wavelet, compact wavelet and STFT carriers. Custom module types,
instance forward overrides and hooks are rejected. Arbitrary modifications
to Python behavior are not part of this constructor-based architecture format;
do not mutate/train a model concurrently with export.

Loading defaults to CPU, eval mode and frozen parameters. `device="cuda"`
selects CUDA explicitly. Stored float32/float64 precision is preserved: there
is no automatic cast, sample-rate change, source reordering or resampling.
Use `trainable=True` for an intentional fresh fine-tuning setup; that enables
all parameters and does not restore prior freezing rules, optimizer or RNG.

## Artifact format and validation

A `.w2v` bundle is a stored ZIP archive containing `manifest.json` and dense
NumPy weight arrays. Array encoding is explicitly non-pickle. The loader
does not deserialize arbitrary Python objects or fetch code/dependencies.

The manifest describes the supported architecture/config versions, tensor
names/shapes/precision, per-array SHA-256, producer information and content
identity. `bundle_id` hashes the canonical manifest with its weight hashes;
ZIP timestamps do not affect that identity. These checks detect corruption
and incompatible content; they are not publisher signatures.

Architecture construction first uses storage-free meta tensors. The weight
payload limit defaults to **256 MiB** and can be changed explicitly through
`max_weight_bytes`. Structural limits bound module construction, including
256 sources/channels, width 16,384 and 256 total context layers. Inventory,
archive members, entry sizes, actual NumPy headers and finite weight values
are checked. Duplicate/unexpected members and unsupported formats are rejected.
All arrays are verified by both inspection and loading.

This payload limit is not total process-memory accounting: archive metadata,
temporary array buffers, model storage and runtime activations are additional.
Existing analysis, decoder and rendering budgets still apply at inference.
Inspection verifies contents without allocating model parameter storage;
loading allocates only the requested model, with no random full-size
initialization. CPU export/inspection/loading preserve the caller's RNG state.

Saving uses a sibling temporary file. New-path publication uses an exclusive
hard link on the same filesystem; explicit `overwrite=True` uses atomic
replacement. A failed export leaves the previous artifact intact. Filesystems
must support the requested publication operation; no unsafe overwrite fallback
is attempted. Loaded archives are read directly, not extracted to file paths.

## Portability versus training resume

Training checkpoints retain their strict code, corpus, runtime, device and
optimizer contracts. Inference bundles intentionally omit optimizer/data-cursor
state and can load a supported architecture on a different device or compatible
implementation revision. They cannot be used to resume a training run exactly.

`implementation_matches` reports whether producer and current package/numerical
dependency fingerprints agree. `require_same_implementation=True` rejects a
mismatch; otherwise it remains visible to the caller and in rendering receipts.
Accepted configuration schemas do not prove identical behavior across code,
dependency or hardware changes. Revalidate outputs when migrating. The CPU
round-trip regressions are exact; CPU/CUDA comparisons use numerical tolerance.

Export a bundle from a stable run in its original environment before migrating
that run's weights to a new project. This release does not silently convert
old strict training checkpoints or change their resume rules.

## Command-line rendering

The installed `wavelet2vec-separate` command supports `inspect` and `render`.
The equivalent module entry point avoids relying on shell PATH configuration:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.inference_cli inspect separator.w2v
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.inference_cli render separator.w2v input.wav --output stems
```

These filenames refer to your exported bundle/input and a new output folder.
`--device cuda` selects CUDA; `--container RF64` forces large-file WAV format.
Rendering defaults to one CPU thread (`--threads` changes it). Inspection and
rendering accept `--max-weight-mib`; rendering also accepts
`--require-same-implementation`. Ctrl+C interrupts processing with exit code
130. The existing file renderer owns cleanup and complete-folder publication.

`SeparatorBundle.render_file` uses stored feature branches and render defaults.
It selects FLOAT for float32 or DOUBLE for float64, with no integer fallback.
An explicit `render_config` override is allowed and is recorded in the receipt.
Model rate/channels must match the input. Full-recording waveforms are not
held in memory; [bounded-rendering limits](BOUNDED_RENDERING.md) still apply.

`render.json` includes the originating bundle ID/producer, implementation-match
status and the current model fingerprint. If weights are changed after loading,
the bundle ID still identifies their origin; re-export to assign the modified
model a new artifact identity. The current model fingerprint describes the
weights/configuration actually used for that render.

[Validation](benchmarks/INFERENCE_BUNDLE_VALIDATION.md) covers numerical
round-trips, native 192 kHz FLOAT/DOUBLE export, a supervised-update export,
malformed artifacts, overwrite failure, CLI rendering and CPU/CUDA loading.
These are deployment/correctness checks, not a musical-quality benchmark.
