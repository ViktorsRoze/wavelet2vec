# Bounded source rendering — v0.34.0

**v0.35.0 update:** [Inference bundles](INFERENCE_BUNDLES.md) now reconstruct
models and render defaults automatically and expose an inspect/render CLI.
The optional `separate_file(..., provenance=...)` JSON object is recorded in
the completed render receipt; bundle rendering supplies artifact provenance.

22 September 2026. Source rendering now has a reader iterator and a file-to-file
path whose audio buffers depend on window size rather than recording duration.
This groundwork can be verified with generated signals before a musical
dataset is available. It does not improve the untrained model's separation
ability or establish musical quality.

## Configure physical windows

```python
import torch
from wavelet2vec import (
    HybridConfig, SynthesisConfig, HybridSeparator, SourceRenderConfig,
    separate_file, iter_separate_reader,
)

rate = 192000
model = HybridSeparator(
    HybridConfig(sample_rate=rate, backend="compact_wavelet", n_bands=24,
                 q=2, hop_length=128, padding_samples=512),
    SynthesisConfig(source_names=("source_a", "source_b"), dim=16),
)
render_config = SourceRenderConfig.from_seconds(
    .01, .0025, .0025, sample_rate=rate, hop_length=128,
)
```

The three durations are core window, overlap and additional real context per
side. They round upward to the token grid; rounded overlap must remain shorter
than the core window. The examples use small settings for a quick numerical
exercise, not a perceptual-quality preset. Choose real model context and
analysis support deliberately. Configuration dictionaries carry a version and
round-trip through `to_dict()` / `from_dict()`.

## Export floating-point stems without loading a whole recording

```python
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    fixture = np.zeros((4097, 2), dtype=np.float32)
    fixture[0] = [1.5, -1.5]
    sf.write(root / "input.wav", fixture, rate, subtype="FLOAT")
    rendered = separate_file(
        model, root / "input.wav", root / "stems", render_config,
        subtype="FLOAT", container="auto",
    )
    assert rendered.frames == 4097
    assert sf.info(rendered.files[0]).subtype == "FLOAT"
```

For real work, use the trained model and a fresh output directory outside the
temporary demonstration. The renderer seeks and decodes only requested input
windows, keeps rolling overlap accumulators, and writes completed intervals
immediately to staged files. It does not concatenate a recording in RAM or
VRAM. Input sample rate/channel count must match the model. There is no
resampling, downmixing, gain normalization, clipping or integer-format fallback.

`FLOAT` requires a float32 model and writes 32-bit floating-point audio;
`DOUBLE` requires an explicitly float64 model and writes 64-bit float. Native
192 kHz stereo is covered by regression tests. Values above unity remain
representable in output files.

`container="auto"` chooses ordinary WAV when its data size fits a conservative
RIFF limit, otherwise RF64. Both use `.wav` filenames. `container="RF64"`
forces RF64; forcing WAV for an oversized output is rejected. At 192 kHz,
stereo float32, one hour of one stem exceeds ordinary WAV's size range. RF64
availability is checked against the installed audio library. Small RF64 files
are actually written/read in tests; no multi-gigabyte file is generated just
to test the size decision.

## Publication, progress and cancellation

The output directory must be new. Source labels live in `render.json`; files
use safe generated names such as `source_000.wav`. Source labels cannot create
arbitrary paths. The receipt records input SHA-256, model identity, render
configuration, source mapping and output geometry/format.

Files are written into an operation-owned sibling staging directory. After
all writers close, output headers and sample counts are checked, then the
completed folder is published by directory rename. Errors, source changes or
cancellation before publication remove the staging directory; they do not
publish partial stems or overwrite an existing destination. Parent directories
created for the requested location may remain. Use an exclusive output path;
concurrent mutation of the source/model or output namespace is unsupported.

`progress(done_samples, total_samples)` runs after a completed interval is
written. `cancel()` returning true raises `SeparationCancelled` at a window
or write boundary. Callback exceptions also abort and clean staging. Cancellation
does not interrupt a running neural kernel or the initial source hashing pass.
File size/mtime checks detect ordinary changes to the input while rendering;
they are not an adversarial tamper-detection protocol.

## Custom bounded readers

```python
def read_generated(start, stop):
    positions = torch.arange(start, stop, dtype=torch.float64)
    mono = (.1 * torch.sin(2 * torch.pi * 440 * positions / rate)).float()
    return mono.reshape(1, 1, -1).expand(1, 2, -1).clone()

processed = 0
for chunk in iter_separate_reader(model, read_generated, 8193, render_config):
    assert chunk.sample_origin == processed
    processed += chunk.waveform.shape[-1]
assert processed == 8193
```

A reader returns `[batch, channels, stop-start]` in model precision, on CPU
or the model device. Batch size must stay constant. Requests include available
real context and stay within the recording. The iterator yields contiguous,
non-overlapping `SourceSynthesis` intervals, with original sample coordinates
and exact lengths. It never discards the final partial hop. Samples emit only
after every contributing window has been accumulated, including when overlap
is greater than half a window.

Model inference runs without gradients in eval mode. Individual module modes
and global grad state are restored **before every yield**, including on failure.
Closing or abandoning an iterator therefore does not leave the model in eval
mode. Each returned waveform owns its storage; later windows cannot overwrite
an earlier chunk. Keeping all chunks in a list still consumes full-recording
memory in the caller. Do not train or mutate the same model concurrently.

The existing `separate_audio` convenience function now uses the same rendering
core. It still takes a resident input tensor and allocates a complete output;
use `separate_file` or consume the reader iterator incrementally for bounded
recording-length memory. Both remain offline processing, with no causal/live
latency claim or persistent instrument-instance tracking.

## Limits and validation

`max_buffer_bytes` rejects an oversized estimate of rolling audio buffers,
reader/model waveform payloads and shift/output temporaries. A minimum batch
estimate runs before calling a reader; actual batch size is checked on return.
This is not total process RAM/VRAM accounting: model parameters, complex
carriers, FFT/neural workspaces, allocator caching and caller-retained chunks
are additional. Existing frontend/decoder limits remain independent.

Overlap weights are computed from float64 midpoint coordinates before conversion
to model precision, avoiding loss of positive edge weights on long float32
grids. Normalization and predicted audio are checked for finite values.
Generated learned-head tests compare the rolling result with an independent
full-size accumulator across all three backends and several overlaps. This
proves agreement with the configured window algorithm; it does not prove
agreement with whole-recording model inference or inaudible boundaries.

The generated 192 kHz file smoke increased duration from 0.25 to 10 seconds.
Both runs had a maximum model input of 9,216 samples and the same explicit
buffer estimate, 1,146,880 bytes. That estimate excludes model/FFT workspaces;
it is not a measured process-memory peak. Summing the initialized equal-split
stems reconstructed the input with maximum error below `5e-7`.
See [validation](benchmarks/BOUNDED_RENDER_VALIDATION.md) and
[raw measurements](benchmarks/bounded_render_smoke.json).

Further dataset-independent groundwork can address physical time/pitch-control
interfaces, transform conditioning/boundary diagnostics and portable inference
bundles. Real separation, time stretching and note-generation quality still
require task data and appropriate trained models.
