# Music AI capability plan

**v0.36.0 update:** [Real drum-substem trials](DRUM_DATASET_FINDINGS.md)
now provide measured evidence from one shared musical performance. The
pipeline functions, finer wavelets help some sources, and quiet drums remain
weak. Priorities now follow those per-source results rather than numerical
round-trip tests alone.

**v0.35.0 update:** [Portable inference bundles](INFERENCE_BUNDLES.md)
now package supported architectures, weights, precision and rendering defaults
for use across projects. This adds deployment groundwork without a dataset;
time/pitch-control interfaces and musical-quality validation remain open.

**v0.34.0 update:** [Bounded rendering](BOUNDED_RENDERING.md) removes the
full-recording input/output allocation requirement from file inference and
adds checked floating-point publication. These numerical/I/O improvements
are validated without a musical dataset and do not establish learned quality.

**v0.33.0 update:** [Verified multitrack training](SUPERVISED_MULTITRACK.md)
now supplies aligned source ingestion, a supervised training/checkpoint
recipe and held-out crop diagnostics, including 192 kHz float32 regressions.
Real-music quality and full-song evaluation remain unproven.

Assessment: 21 September 2026, local v0.31.0 implementation.

**v0.32.0 implementation update:** [Source synthesis](SOURCE_SYNTHESIS.md)
now supplies a complex-mask decoder with waveform residuals, a supervised
reference separator, coefficient consistency projection and overlap-add
inference. The initial 48/96 kHz checks are numerical identities; real-music
separation and native high-resolution perceptual qualification remain open.
The gap analysis below records the rationale preceding this implementation.

wavelet2vec has a useful signal and training foundation, but has not yet
demonstrated a high-quality general music representation or complete learned
systems for separation, time stretching or reference-conditioned note
synthesis. Passing numerical tests and retaining original phase do not
establish those capabilities.

## Current capabilities and boundaries

- Native waveform plus invertible complex STFT/Gaussian filter-bank carriers;
  learned tokens are lossy and are not an independently invertible codec.
- Compact multirate analysis, multiple resolutions, stereo features, physically
  aligned sliding windows and an initial local/global context network.
- Masked acoustic pretraining, verified grouped corpora, deterministic crops,
  checkpoint resume and diagnostic baselines. Useful musical semantics remain
  unproven; the fixed teacher primarily supplies acoustic targets.
- Source-target collation, validity masks, complex reconstruction losses and
  guarded additive consistency. These are building blocks, not a full
  trainable separator with a production rendering path.
- Existing `time_stretch` integrates interpolated band-phase increments;
  `pitch_shift` remaps and heterodynes bands. Both are NumPy DSP functions,
  return float32, and do not supply a differentiable learned task pipeline.
  The pitch shifter explicitly does not preserve formants.
- The current context model is bidirectional and bounded. Sliding extraction
  does not establish persistent source identity or learned memory across an
  arbitrarily long recording.

## Shared foundation still needed

The most useful next component is a **trainable synthesis interface**. The
existing inverse reconstructs supplied coefficients; it does not learn how
to produce the appropriate coefficients for a new source, pitch or duration.

Define a decoder contract consuming context, optional local signal carriers,
task controls and an explicit output sample grid. Return waveform estimates
with length, latency, valid-region and channel metadata. Supply at least a
complex-carrier prediction head and a waveform residual head, with comparable
training recipes. For generated notes, the decoder must also work without
copying a target waveform carrier. A reference note is conditioning, not the
unobserved target audio.

Separate descriptive context from reconstruction detail: the former can be
slow and abstract, while the latter preserves transients, phase and stereo
relationships. Controlled pitch and time edits require consistent *new*
phase trajectories, rather than preservation of every original phase value.
Redundant transform coefficients also need consistency checks: compare
predicted coefficients with analysis of their synthesized waveform.

Keep the shared library modular: analysis, context, synthesis, task heads,
data and evaluation. Task-specific invariances belong in task heads or
explicit controls. Instrument recognition may benefit from pitch invariance;
note synthesis must retain or explicitly receive pitch. One universal pooled
vector cannot be assumed to satisfy both contracts.

## Three target projects

| Project | Missing model/data components | Evidence required |
| --- | --- | --- |
| Stem and instrument separation | Learned source decoder, complex estimates plus waveform residuals, source queries/instance slots, unknown-source output, supervised multitrack recipe and cross-window identity tracking | Held-out songs/artists/sessions; per-source leakage and distortion, silent-source behavior, stereo preservation, chunk-boundary checks and blind listening |
| Neural time stretching | Monotone input/output time map, pitch-preserving phase evolution, separate transient/tonal/noise handling, variable-length decoder, continuous speed control and suitable paired/controlled data | Requested duration and pitch, onset sharpness, modulation/reverb behavior, stereo stability, identity at rate 1, and listening across stretch ratios |
| New notes from a reference note | Reference timbre encoder, pitch/velocity/articulation/duration controls, attack/sustain/release modeling and a conditional synthesizer with harmonic/noise/transient capacity | Unseen instruments and pitches; pitch accuracy, timbre and attack preservation, controllability, stereo consistency and listening |

Begin separation with a fixed source set and complete target stems. Then
extend to variable instrument instances and example/text queries. Two guitars
need stable instance identities, not just a shared guitar label. In-mix stem
extraction and dry-source restoration need different targets. Nonlinear bus
processing creates interactions: separately processed stems need not sum to
the processed mixture. Use additive consistency only where the target
definition warrants it; dry restoration needs paired processing provenance
or an explicit generative restoration model. Exact recovery after destructive
clipping or ambiguous mixing cannot be guaranteed.

For time stretching, a resized magnitude image or naively interpolated
embedding does not provide a phase-consistent, transient-preserving renderer.
Use controlled renders and classical algorithms as baselines. Training only
against one classical stretcher risks reproducing its artifacts. Neural
time-scale modification already has precedents, including
[Temporal Compressing Networks](https://arxiv.org/abs/2210.17152); the use of
a latent representation alone is not a novelty claim.

For note synthesis, distinguish shifting a recorded note from predicting how
the instrument would sound at another pitch or articulation. A single note
does not uniquely specify the instrument's entire range. Broad training over
instruments, pitches and velocities supplies a prior for plausible output.
[DDSP](https://arxiv.org/abs/2001.04643) is a useful precedent for explicit
pitch/loudness controls and interpretable synthesis. A harmonic model needs
additional capacity for inharmonic attacks and noisy sounds. The
[NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) illustrates
the required pitch/velocity/instrument labels, but its 16 kHz mono audio
cannot qualify native high-resolution stereo fidelity.

## High-resolution quality contract

The required high-resolution target is **up to 192 kHz, 32-bit floating-point
audio**, including stereo. It is a core requirement. Accepting a sample-rate
argument is not end-to-end qualification; qualify configurations at
44.1/48/88.2/96/176.4/192 kHz. Preserve native source
bandwidth, channel layout, gain, timing and output precision deliberately.
Specify resampling, anti-alias filtering, integer export and optional dither;
never silently convert those choices into model behavior.

Measure passband and alias residuals, numerical reconstruction, weak signals,
near-Nyquist content, impulses, phase/stereo behavior, identity edits and seam
continuity. Pair those with task quality and listening tests. A larger sample
rate or attractive spectrogram cannot establish better audible output.

Keep durations and frequencies physical: a fixed 256-sample hop represents
different time spans at 48 and 192 kHz. Model conditioning and configuration
must account for that change. Use bounded synthesis with overlap/context,
latency compensation and consistent source assignment. Feature-window
stitching alone does not provide artifact-free waveform rendering.

Qualify accelerator training separately: memory profiles, efficient decoding,
prefetch/bucketing, selective mixed precision with stable FFT/phase/loss paths,
activation checkpointing, and later distributed resume. The reference corpus
CLI is currently single-process and does not complete this scaling layer.
Publish supported configurations, serialization/migration contracts and
minimal inference/training examples for independent projects.

## Development order and acceptance gates

1. **Synthesis contract and fidelity suite.** Implement one learned complex
   decoder plus optional waveform residual; test gradients, identity,
   coefficient consistency, output timing and stereo/chunk boundaries. Begin
   with numerical tests through 192 kHz/float32, then qualify native-rate
   training, full-recording rendering and perceptual fidelity at that target.
2. **One complete supervised separation baseline.** Add a source corpus
   schema/loader, reproducible training and full-track rendering. Compare
   waveform-only, complex-STFT and compact-wavelet/hybrid variants with
   matched data, compute budgets and evaluation conditions.
3. **Controlled transformation models.** Reuse synthesis/time-grid contracts
   for time stretching and reference-conditioned note synthesis, but give
   each its own decoder controls, supervision and evaluation. Avoid forcing
   all tasks into one output head.
4. **Musical context and scale.** Evaluate beat/bar/event conditioning,
   reference retrieval and persistent source memory using held-out musical
   tasks. Compare acoustic SSL with musical/multitask targets; proceed to
   large training runs only after small controlled studies show useful gains.
5. **Deployment and editor integration.** Add stable inference bundles,
   cancelable bounded processing, latency/length metadata and A/B audition
   with original-level comparison. Select quality/latency presets from
   measured results.

Waveform/spectral hybrids already exist in
[Hybrid Demucs](https://arxiv.org/abs/2111.03600), and
[HT Demucs](https://arxiv.org/abs/2211.08553) models cross-domain context.
[BS-RoFormer](https://arxiv.org/abs/2309.02612) operates on complex spectral
inputs. These are relevant comparison architectures, not evidence that
wavelet2vec already matches them or that other systems ignore phase.

Promising research hypotheses include conditioning shared synthesis on
explicit pitch/time transformations, persistent instrument states across
windows, and learned allocation of resolution under a coefficient budget.
Any benefit must survive matched ablations. Neither the hybrid input idea
nor wavelet use alone establishes novelty or superior musical quality.
