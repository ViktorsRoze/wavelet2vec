# Inference bundle validation — v0.35.0

22 September 2026; local Windows `sound` Python environment. Tests use generated
signals and initialized/briefly updated models, not a musical dataset.

| Check | Result |
| --- | --- |
| Complete library suite | **533 passed**, 111.89 seconds; one existing brightness-test tensor-to-scalar warning |
| New bundle regressions in full suite | **22 passed**, including CPU-to-CUDA loading |
| Initial focused bundle suite | **18 passed**, 14.16 seconds, before four final validation cases |
| Runnable documentation | **24 Python examples executed** |
| Editor compatibility | **36 passed**, 15.16 seconds |
| Installed/runtime package | Both **0.35.0**, dependencies unchanged |

For dense wavelet, compact wavelet and STFT backends in both float32 and
float64, exported/reloaded models produce exactly equal CPU waveforms. These
models have nonzero mask/residual head weights and context layers, so the
comparison is not limited to an initialized equal-split identity. Export leaves
mixed module training modes and the CPU RNG unchanged. Inspection/loading
also preserve CPU RNG, and loaded models default to eval/frozen parameters.

Editor checks cover spectral, stereo, spectral-edit UI/effects, phase-layer
and library behavior with Qt offscreen. The editor working tree was unchanged;
no new editor model-management interface was added in this release.

Further cases cover a model without context, stable content identity across
separate exports, explicit trainable loading, and source/branch/render settings.
The native 192 kHz file tests verify saved waveform-only branch selection and
FLOAT/DOUBLE subtype selection, with exact storage of computed CPU outputs.

Corruption cases include weight checksums, tensor declarations, actual array
headers, precision, unexpected members, nonfinite values and excessive
architecture size. A tiny weight budget rejects loading before parameter
storage allocation. A producer-code mismatch remains visible in permissive
loading and fails strict loading. These checks do not authenticate publishers
or provide a general resource sandbox for arbitrary files.

Export refuses unsupported custom forward hooks and nonfinite model weights.
A failed explicit overwrite preserves the previous artifact bytes and leaves
no temporary file. Bundle loading reads arrays directly without extracting
archive members or deserializing arbitrary Python objects.

The supervised-run regression performs one native 192 kHz generated-source
update, exports the run, checks the update/dataset provenance and compares
every reloaded parameter exactly with its trained counterpart. It does not
claim a useful separation model from that single update.

CLI regressions run `python -m wavelet2vec.inference_cli inspect` and `render`
as separate processes. Inspection checks every array; rendering produces a
complete 192 kHz FLOAT stem directory and records the originating bundle ID
in `render.json`. The dedicated installed entry point delegates to the same
module.

On the local CUDA device, a CPU-created float32 bundle loads onto CUDA and
produces waveforms within `rtol=5e-4, atol=5e-5` of CPU results. This is a small
numerical portability check, not bitwise accelerator reproducibility or a
whole-song performance benchmark. Strict training-resume contracts remain
separate and unchanged.

No pretrained musical capability, instrument recognition, neural stretching
or note-generation quality is demonstrated by these packaging tests. The next
dataset-independent groundwork remains physical time/pitch-control contracts
and transform/boundary diagnostics.
