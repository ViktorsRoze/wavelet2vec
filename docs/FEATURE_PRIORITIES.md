**Feature priorities for a stronger wavelet2vec — 13 September 2026**

**Implemented v0.37.0:** [Reusable studies](REUSABLE_STUDIES.md) remove
dataset-specific orchestration and reuse verified work. Profiling led to
target-FFT reuse and grouped compact-source decoding. The measured local
gains and scoped C/C++/Rust/Zig options are documented in
[the performance strategy](PERFORMANCE_STRATEGY.md).

**Measured v0.36.0:** [Real drum-substem results](DRUM_DATASET_FINDINGS.md)
validate the native-data/training/render/loss pipeline on one performance.
Finer wavelets improve stronger sources; quiet/noisy drums remain weak.
Prioritize per-source/active-event objectives, richer local source heads,
matched multi-resolution comparisons and independent musical evaluation.

**Implemented v0.35.0:** [Portable inference bundles](INFERENCE_BUNDLES.md)
package built-in separator settings/weights, render recipes and provenance,
with validation and native float file rendering from a CLI. Remaining
dataset-independent priorities include physical time/pitch controls and
transform/boundary diagnostics.

**Implemented v0.34.0:** [Bounded source rendering](BOUNDED_RENDERING.md)
adds reader/file inference, rolling overlap buffers, float WAV/RF64 export,
cancellation and staged publication. It removes full-recording waveform
allocation from the file path. Remaining dataset-independent work includes
time/pitch-control contracts, conditioning/boundary diagnostics and inference
bundles; musical quality awaits appropriate training/evaluation data.

**Implemented v0.33.0:** [Verified multitrack training](SUPERVISED_MULTITRACK.md)
adds aligned stems, missing/silent-source contracts, sample-weighted updates,
exact-corpus resume and held-out crop diagnostics. Curated real multitracks,
full-song evaluation and production quality at 192 kHz remain open.

**Implemented v0.32.0:** [Trainable source synthesis](SOURCE_SYNTHESIS.md)
adds complex masks, a waveform residual, a supervised reference separator and
offline overlap-add rendering. Verified multitrack ingestion, a complete
supervised trainer and real-music/high-resolution evaluation remain open.

**21 September update:** [The multi-project capability plan](MUSIC_AI_CAPABILITY_PLAN.md)
maps the current v0.31.0 foundation to separation, neural time stretching,
reference-note synthesis and high-resolution qualification. Its next shared
implementation priority is trainable synthesis plus a complete supervised
separation baseline.

The core purpose is a reliable, phase-preserving audio foundation for music
models. Feature richness should improve learning, reconstruction and reuse
around that purpose. It should not turn the library into a collection of
unrelated audio tools or assume that a larger architecture is automatically
a better encoder.

| Priority | Addition | Purpose and implementation boundary |
| --- | --- | --- |
| Implemented foundation | Compact complex carriers plus raw waveforms | Retain audio information while reducing dense wavelet redundancy. |
| Implemented v0.27.0 | Sliding-window extraction, global phase/time coordinates, explicit edge context, duration-weighted coarse summaries | Prepare long recordings for models with local and longer context. Summaries do not themselves recognize musical structure. |
| Implemented v0.28.0 | Stereo adapters | Cross-spectrum, amplitude coherence, balance and side-energy features, with energy validity and requested-time support metadata. Statistical confidence is not claimed. |
| Implemented v0.28.0 | Multi-resolution features | Shared raw projection, aligned spectral views, learned gates, ablations and joint coefficient budgeting. Musical quality remains unevaluated. |
| Initial utilities implemented v0.28.0 | Source training contracts | Named waveform targets, ragged collation, missing-source/padding masks, complex reconstruction loss and guarded additive consistency. A complete supervised separator remains open; SSL training was added in v0.29.0. |
| Implemented v0.28.0 | Inference feature caches | Source/model/config/code/dependency identity, runtime metadata, checked interval coordinates and atomic array storage. Callers must include every runtime choice. |
| Initial implementation v0.29.0 | Learned context plus masked self-supervision | Local/global attention, corrupted-student/fixed-teacher prediction, masking and variance diagnostics, and a resumable reference trainer. Real-music semantic quality remains unproven. |
| Next major step | Target quality and real-music evaluation | Compare musical targets and mask strategies against constant/local baselines, then scale a reproducible corpus/training recipe. |
| Infrastructure implemented v0.31.0 | Verified real-audio corpus pipeline | Explicit split groups, exact PCM duplicate guards, deterministic crops, corpus-aware resume and group-level diagnostic evaluation. A curated semantic benchmark remains open. |
| Optional experiment v0.30.0 | Frozen target calibration | Training-only weighted statistics, checkpointed target units and matched raw/calibrated comparisons. Average toy error improves, but constant-baseline wins are not robust. |
| Research: learnable filter adaptation | Constrained bandwidth/center adaptation with checked frame conditioning and inverse | Investigate whether adapting filters helps task performance. Preserve coverage and compare with fixed filters; no novelty claim yet. |
| Research: musical hierarchy | Beat/bar-conditioned context, repeated-motif retrieval and persistent source identities | Learn relationships across phrases and sections while local carriers retain synthesis detail. Fixed average pooling is only a baseline. |
| Task modules | Instrument/instance queries, source heads, effects attribution and dry-source restoration | Build on the encoder with source supervision and explicit target definitions; avoid coupling every downstream head into the base representation. |
| Diagnostics | Resolution/support plots, reconstruction residuals, phase/stereo checks, cache inspection and performance profiles | Explain what a configuration represents and make failures reproducible. Integrate with the editor deliberately. |

For stereo features, retaining complex left/right carriers already preserves
the information. Explicit relationships would provide useful model inputs,
not new information absent from the recording. Coherence needs integration
appropriate to the band bandwidth; a short low-frequency window should not
be declared confidently coherent just because it contains one phasor.

For multi-resolution analysis, the promising idea is complementary evidence:
attack timing from short support, pitch/harmonic structure from longer support,
and relationships between events from a slower context branch. This is a
research hypothesis. Longer windows, more bands and more parameters have
costs and can also degrade localization or learning.

For training, preserve sample rate, stereo identity, source timing, gain and
processing provenance. Augmentations should declare which invariances they
teach. An invariant retrieval descriptor and a synthesis-capable representation
need not discard the same information. In-mix extraction and dry restoration
of a mastered track need different supervision and consistency objectives.

The initial adapter/data/loss/cache interfaces are implemented; see
[ADAPTERS_AND_TRAINING.md](ADAPTERS_AND_TRAINING.md). The next major work is
target/data quality and real-music validation of the initial
[context/pretraining implementation](SELF_SUPERVISED_MUSIC.md). The
[earlier gap analysis](MUSIC_REPRESENTATION_GAPS.md) explains the architecture
choices; planned complex-music tests should determine whether they help.
Continue numerical regression tests throughout; musical understanding and
perceptual superiority should remain unclaimed until task-specific evidence
exists. The earlier small synthetic separation experiment validates plumbing,
not the full long-music hypothesis.
