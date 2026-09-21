# Multitrack training validation — v0.33.0

21 September 2026; local Windows `sound` Python environment.

| Check | Result |
| --- | --- |
| Complete library suite | **483 passed**, 105.75 seconds; one existing brightness-test tensor-to-scalar warning |
| New multitrack suite | **16 passed**, 23.57 seconds |
| Editor compatibility suite | **36 passed**, 17.48 seconds |
| Runnable Python documentation examples | **20 executed successfully** |
| Installed and runtime package version | Both **0.33.0**, dependencies unchanged |
| 192 kHz float32 CLI lifecycle | Prepare, train, evaluate, resume and evaluate again completed on generated multitracks |
| 192 kHz float32 CUDA smoke | Finite updates/evaluation; resumed parameters differed by up to **9.42e-9** |

The source-synthesis and high-resolution cases from v0.32.0 are included in
the full 483-test suite, including float32 WAV preservation and native-rate
source rendering. These counts do not imply a trained quality benchmark.

Editor checks cover spectral, stereo, spectral-edit UI/effects, phase-layer
and library behavior with Qt offscreen. The editor remains on its existing
DSP integration; this release does not add a separator training interface.

New regressions cover relative manifest paths, immutable source mappings,
source reuse across groups, alignment metadata, full-recording declared
additivity, stem mutation, native 192 kHz crops, known silence versus missing
annotations, epoch/fixed-crop determinism, and loss-weighted accumulation
with variable lengths and annotation counts.

At both 8,192 and 192,000 Hz, CPU resume tests reproduce the next crop IDs,
origins, dropout-dependent loss and parameter values exactly. They include
an epoch transition. Data provenance/runtime mismatches fail before loading
model weights. An injected failure on the second microbatch leaves the data
position unchanged, clears partial gradients and restores RNG; retry matches
an uninterrupted update.

Evaluation tests verify reproducible held-out fixed crops, module mode
restoration, equal group weighting, the equal-split initial model baseline,
silent-source `null` SNR and finite JSON serialization. They do not validate
perceptual scoring or generalization to complete songs.

The CLI regression creates four aligned stereo float WAV mixtures with
matching stems at 192 kHz, two train groups and two validation groups. It
declares an additional known-silent source explicitly. One optimizer update
with two microbatches is followed by a separate resumed invocation for the
next update. Both evaluation reports are checked. The fixture tracks are
only 128 samples long and intentionally test contracts, not music quality
or sustained high-rate throughput.

A final targeted CLI regression also checks that a conflicting output-folder
configuration is rejected without changing its checkpoint bytes. The full
suite timing above precedes that additional assertion and CLI guard.

**Accelerator result.** On an NVIDIA GeForce GTX 1050 Ti, the same small
192 kHz fixture family completed training, checkpoint reload and evaluation
with finite loss and gradient norm. Default CUDA execution did not reproduce
weights bit-for-bit after reload; the maximum observed difference was
`9.423766655913823e-9`. Exact CPU comparisons therefore must not be generalized
to CUDA. RNG/config restoration is necessary but insufficient for bitwise
accelerator determinism. [Raw results](multitrack_accelerator.json) include
the implementation fingerprint and diagnostic losses.

After these two updates, model macro-group loss was approximately 0.0288803,
versus 0.0288371 for equal split. This check does not demonstrate a quality
gain. It verifies that the native-rate supervised path runs and emits the
diagnostics needed to discover that limitation.

No existing soundgen loops were treated as independent multitrack ground
truth, and no real-music training or listening benchmark was performed.
Next require curated aligned stems and full-recording quality/resource tests
through 192 kHz/float32, before claiming strong source separation.
