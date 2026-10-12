**What wavelet2vec still needs for learned music representations — 14 September 2026**

**v0.29.0 update, 15 September:** An initial learned local/global context
backbone, masked fixed-target objective and resumable reference trainer are
now implemented; see [SELF_SUPERVISED_MUSIC.md](SELF_SUPERVISED_MUSIC.md).
Their leakage/gradient/resume contracts are tested. The first synthetic
learner did not beat a constant baseline, so target quality, real-music
evaluation and scalable pretraining remain open. The analysis below records
the v0.28.0 gaps and the rationale that led to this first implementation;
its statements that no context/pretraining path exists are historical.

Compared with wav2vec 2.0, the largest missing pieces are a learned temporal
context backbone and a complete self-supervised representation-learning
objective. Pretrained weights are only one output of that machinery.
wavelet2vec now provides a substantial signal front end and supporting
toolkit, but is not yet a comparable learned contextual representation model.

The local comparison covers v0.28.0 source: hybrid/compact carriers,
sliding-window extraction, stereo and multi-resolution adapters, source
training utilities and inference caches. The old wavelet convolutional
encoder, augmented-view dataset and vision experiments also exist. They
should not be overlooked, but do not complete the missing learning system.

| Capability | wav2vec 2.0 mechanism | Current wavelet2vec gap |
| --- | --- | --- |
| Learned local features | Deep temporal convolutional encoder | Trainable projections and a separate wavelet-initialized convolutional component exist; the new hybrid path lacks a substantial learned local sequence stack. |
| Contextual representations | Transformer over feature sequences | Sliding windows collect aligned tokens; current gates combine views at a token, not learned relationships across notes/phrases/windows. |
| Learning from unlabeled audio | Masked-span contrastive prediction | Source reconstruction losses and augmented views exist, but no complete masked student/context/prediction objective. |
| Target units | Learned product quantization | No learned music target vocabulary or equivalent teacher-target pipeline. Storage bit quantization is unrelated. |
| Training stability | Codebook diversity objective and negative sampling | No corresponding integrated anti-collapse/target-monitoring system for semantic pretraining. |

These mechanism descriptions come from the
[wav2vec 2.0 paper](https://arxiv.org/html/2006.11477v3). They are not a claim
that every mechanism should be copied into music modeling unchanged.

**The priority is learned context.** A feature at one time should become
different when its role changes with surrounding notes, rhythm, repeated
motifs or arrangement. Our current encoder can describe local sound and its
spatial/spectral relationships; it cannot yet learn a song-level interpretation
through its per-token fusion gate. Coarse average pooling creates a useful
baseline input, not a learned phrase representation.

A practical next model would process local tokens, produce slower context
states, and allow the two levels to exchange information. Candidate blocks
include Transformers or Conformers with explicit masks and physical/relative
positions. Windows alone do not provide cross-window state. A long recording
also does not need full sample-rate attention over the entire song. Wav2vec
2.0's contextualization is within the sequence supplied to it; it does not
automatically solve unlimited whole-song context either.

**Self-supervised training needs a precise task and target contract.** We
need span sampling, padding-aware masks, target generation, the student/context
model, a loss and a resumable training/evaluation recipe. If using contrastive
targets, negative selection matters; if using a learned codebook, monitor
utilization/collapse. A teacher-prediction alternative does not automatically
need the same quantizer or negative-sampling mechanism. These are architecture
choices to compare, not mandatory boxes for every good encoder.

Repeated musical motifs can create false negatives: a different location in
the same track is not necessarily a different semantic event. Positive/negative
construction should reflect the intended task rather than blindly applying
speech-oriented sampling assumptions.

For music, useful targets may describe harmonic/pitch structure, transient
and rhythmic events, timbre, spatial relationships and production character.
They need not all share one target vocabulary. A discrete target or semantic
embedding can deliberately be lossy while the separate audio/complex carrier
retains synthesis detail. Do not force every hidden feature to serve both
sample-accurate reconstruction and musical abstraction.

Music-focused precedents are more relevant design comparisons than speech
alone. [MERT](https://arxiv.org/abs/2306.00107) uses acoustic and musical
teachers, including RVQ-VAE and CQT-derived supervision. [MuQ](https://arxiv.org/abs/2501.01108)
uses Mel-RVQ targets for masked music representation learning. Their target
choices show that meaningful music representations require a learning task,
not merely a more detailed picture. They do not establish that our phase
features improve those tasks, or that lossless carriers are necessary for
every recognition objective.

**The hybrid design needs protection against masked-target leakage.** If a
future student predicts a hidden span, the answer must not remain available
through its raw branch, another spectral resolution, stereo moments, cached
features, a summary or the retained unmasked synthesis carrier. Wavelet and
STFT support also extends across nominal mask boundaries.

A defensible first design is to corrupt the student's waveform before its
front ends, derive every student view from that corrupted waveform, and build
teacher targets separately from clean audio with stopped gradients. Prevent
the prediction head from reading the clean carrier. Test leakage explicitly;
otherwise low pretraining loss might simply reflect a bypass route. Compare
longer/structured masks so success is not just local waveform interpolation.

**“Musical essence” should be separated into testable aspects.**

| Aspect | Example distinction a useful representation should support |
| --- | --- |
| Timbre / source identity | Same instrument across notes, while separating similar instruments or playing styles |
| Pitch / harmony | Absolute pitch/key when relevant, plus transposition-related motifs when the task calls for it |
| Rhythm / groove | Timing, articulation and swing; some tasks need tempo invariance, others need tempo sensitivity |
| Arrangement / structure | Repeated/varied phrases, entrances, fills and section roles |
| Spatial character | Delay, panning, coherence and side information without mono cancellation |
| Production character | Effects, distortion and mastering characteristics, distinct from estimated dry source identity |

These requirements conflict if reduced to one universally invariant vector.
Gain robustness may help retrieval while amplitude matters for reconstruction;
pitch invariance may help instrument recognition while damaging harmonic
analysis. Use contextual layers and task/aspect heads with explicitly chosen
invariances. The library's existing modular descriptors can be useful baselines
or auxiliary targets, not assumed ground truth for semantic meaning.

**Training/evaluation infrastructure is still partial.** We now have validity-
aware source batches, phase-aware losses and versioned caches. We still need
an unlabeled-music training runner with realistic length bucketing, attention
masks, scheduling, checkpoint/resume, gradient accumulation/checkpointing and
appropriately scoped distributed/mixed-precision support. Add layer-wise
feature extraction and lightweight downstream probes so one task does not
dictate the whole representation.

Evaluation should use held-out source groups across instrument, pitch/harmony,
rhythm, structure, retrieval and restoration/separation tasks. Reconstruction
SNR and passing DSP tests cannot establish semantic understanding. Our small
synthetic separation experiment is not such an evaluation. A corpus and
training recipe matter even if we never distribute pretrained weights.

**What to preserve from wavelet2vec's direction.** We already retain native
stereo phase and precise synthesis carriers, provide explicit sample/time
coordinates, and expose multiple physical resolutions. These are useful for
music editing/separation. Wav2vec 2.0 starts from raw waveform too; it should
not be described as universally ignoring phase, but its latent representation
is not a lossless codec. More retained information does not by itself imply
better learned semantics.

The next major development should therefore be a **music context model plus
a leakage-tested masked pretraining objective**, built on the existing front
ends. Test continuous teacher targets before committing to a complex learned
music quantizer, then compare discrete alternatives if they help. Speech
tokenizers/CTC decoding are not a priority unless lyric transcription becomes
an explicit goal. Music/audio queries or text alignment can be added later as
task heads; they are not prerequisites for the core representation.
