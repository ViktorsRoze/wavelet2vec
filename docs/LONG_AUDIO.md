**Long-audio encoding foundation — v0.27.0, 13 September 2026**

`SlidingHybridEncoder` connects the local hybrid encoder to long recordings.
It reads a core interval plus real left/right context, extracts hybrid tokens,
and emits only the core's tokens. Neighboring analysis windows overlap, but
output tokens have disjoint ownership: no duplication or averaging of two
incompatible windows is hidden in the output.

This is infrastructure for learning musical relationships across time. It
does not itself infer beats, repeated loops, phrases, source identities or
the overall musical idea. Those are downstream learning/evaluation tasks.
The initial research hypothesis remains that detailed local features plus
longer context can improve such tasks; this release does not establish it.

The [validation record](benchmarks/LONG_AUDIO_VALIDATION.md) covers 336
passing library tests, 36 editor compatibility tests and executed guide examples.

**Window sizes are expressed on the token grid.**

```python
import torch
from wavelet2vec import (
    HybridConfig, HybridTokenEncoder,
    SlidingHybridEncoder, SlidingWindowConfig,
)

config = HybridConfig(sample_rate=44100, backend="compact_wavelet",
                      hop_length=256, n_bands=64)
local_encoder = HybridTokenEncoder(config, channels=2, dim=128)
windows = SlidingWindowConfig.from_seconds(
    core_seconds=2.0, context_seconds=0.5,
    sample_rate=config.sample_rate, hop_length=config.hop_length,
)
encoder = SlidingHybridEncoder(local_encoder, windows)
# Rounds upward: 345 core tokens and 87 context tokens per side.
print(windows)

audio = torch.randn(1, 2, 4410) * 0.1
for batch in encoder.iter_encode(audio):
    print(batch.tokens.shape, batch.sample_starts, batch.valid_samples)
```

`core_tokens` and `context_tokens` default to 256 and 64. Durations therefore
depend on the configured sample rate and token hop; use `from_seconds` when
physical durations are more convenient. Rounding to full tokens keeps the
grid aligned. The source's first sample anchors the grid; a nonzero
`sample_origin` moves that source onto a parent timeline without redefining
its patch boundaries.

**Use the reader/file iterator for bounded extraction.**

```python
# Example for an existing 44.1 kHz stereo file:
# for batch in encoder.iter_file("track.wav"):
#     feature_store.append(batch.tokens.cpu(), batch.sample_starts.cpu())

# A reader may wrap another decoder, database or memory-mapped audio source.
def read(start, stop):
    return audio[..., start:stop]  # exactly [B,C,stop-start]

batches = encoder.iter_from_reader(read, audio.shape[-1], sample_origin=44100)
first = next(batches)
assert first.sample_starts[0] == 44100
batches.close()
```

`iter_file` seeks and reads the needed windows through soundfile. It requires
the source sample rate and channel count to match the local encoder, and
decodes in the model's explicit float32/float64 precision. It does not resample,
normalize or downmix. The file closes on exhaustion or when the iterator is
closed. Close partially consumed iterators explicitly when keeping them alive.

A custom reader returns tensors of consistent batch size, channel count and
model dtype. Reader indices are local to the source; `sample_origin` affects
returned coordinates and phase reference. There is one shared origin per
batch, so batch together sources with aligned origins or extract separately.

Input tensors/readers may remain on CPU while the local model is on CUDA.
Only the current analysis window is moved to the model device. Dtype changes
are explicit: mismatched reader/model precision raises an error.

Iterators default to `track_grad=False` and `retain_carrier=False`. Per-window
FFT/autograd working memory is therefore bounded by window configuration
rather than recording duration, assuming the consumer releases previous
outputs. This excludes caller-owned full input tensors and stored output
tokens. Collecting all batches or retaining their carriers uses additional
memory proportional to what is retained. The existing coefficient-allocation
guard applies separately to each local analysis window, including its
transform padding.

The iterator restores the caller's gradient mode before yielding, so a
suspended no-grad extractor does not silently disable gradients in its consumer.

**Every token carries physical coordinates and context availability.**

| `SlidingTokenBatch` member | Meaning |
| --- | --- |
| `tokens` | `[B,T,D]` core tokens only; these remain lossy features |
| `sample_starts` | Integer global start sample for each patch |
| `valid_samples` | Actual source samples in each patch; the final patch may be shorter than the hop |
| `times_s` | Global midpoint of each actual patch interval |
| `left_context_samples`, `right_context_samples` | Real samples supplied on either side within that analysis window |
| `has_requested_context` | Whether both sides meet the configured requested context |
| `analysis_start`, `analysis_stop` | Global half-open interval read for analysis |
| `core_start`, `core_stop` | Global half-open interval owned by this output batch |
| `encoding` | Optional retained local `HybridEncoding` |

At actual file boundaries, there is less real context. The extractor reports
that rather than manufacturing source samples. Local wavelet zero padding or
STFT padding still follows the frontend's existing contract.
`has_requested_context` is **not** a proof that every filter is unaffected by
boundaries: low-frequency/finite-support filter tails can extend beyond the
chosen context. Validity based on measured filter support remains future work.

Windowed wavelet features are not guaranteed identical to a whole-track
transform. Local FFT grids, finite filter supports and edge context can differ,
particularly in the first and last windows. This release guarantees coordinate
and ownership contracts, not whole-track spectral equivalence or artifact-free
predicted stems across windows.

**Phase references now respect the parent timeline.**

`HybridAudioFrontend(audio, sample_origin=...)` records the fragment's global
origin in its returned encoding. Spectral demodulation uses that origin, so
each window does not independently reset its center-frequency phase reference.
Token times and compact `band_times_s(index)` also use global coordinates.
The origin does not alter raw audio, analysis coefficients or the inverse.
Existing calls default to zero origin and retain their previous behavior.

For a fixed center-frequency tone, the signed spectral context agrees across
different fragment origins in the tested periodic cases. This does not make
all musical features time-shift invariant or erase genuine phase evolution
of off-center/overlapping components. Raw-waveform tokens retain the actual
waveform. Models can learn invariance or equivariance appropriate to their task.

**Retain local carriers when reconstruction is needed.**

```python
parts = []
for batch in encoder.iter_encode(audio, retain_carrier=True):
    parts.append(batch.synthesize_core())
reconstructed = torch.cat(parts, dim=-1)
torch.testing.assert_close(reconstructed, audio, rtol=2e-5, atol=2e-5)
```

`synthesize_core()` inverts the actual local coefficients, then crops to the
owned source interval. It does not return saved raw samples as a shortcut.
The unmodified cores reconstruct the source to floating-point precision.
Optional replacement coefficients follow the local frontend's dense/ragged
contract. Predicted source edits need explicit seam/consistency evaluation;
the identity test does not establish seamless arbitrary edited output.

**Collect sequences and form multiple context scales.**

```python
with torch.no_grad():
    sequence = encoder(audio)
    coarse = sequence.pooled(group_tokens=16)
    print(sequence.tokens.shape, coarse.tokens.shape)
```

`forward` returns `HybridSequence`, collecting the full sequence of tokens,
sample intervals, midpoint times and context flags. It respects the caller's
gradient mode. `pooled()` computes duration-weighted groups and preserves
their true interval lengths; a short final patch does not count as a full
patch. The fine sequence remains available in the original object. These
coarse groups are fixed temporal summaries, not detected beats or phrases.

For training across windows, `forward` or an iterator with `track_grad=True`
keeps the gradient graphs needed for the loss. Memory can then grow with
sequence length. Gradient checkpointing, truncated backpropagation and a
learned long-context model are not supplied by this extraction layer.

Sliding configuration is versioned as `sliding-v1` and stored in module extra
state. Checkpoint loading rejects a different window configuration; retain
the local encoder configuration, channel count and token dimension alongside
the state dictionary as before. A feature cache should additionally store
source identity, model weights/version, origin and window settings. A full
cache implementation is planned rather than silently reusing existing
unversioned editor embedding caches.

**Next features should serve this architecture.** See
[FEATURE_PRIORITIES.md](FEATURE_PRIORITIES.md) for priorities: stereo cues,
multi-resolution context, training/data contracts, cache provenance, and
eventual musical/source heads. Their benefits on complex music remain to be
established by task-specific tests.
