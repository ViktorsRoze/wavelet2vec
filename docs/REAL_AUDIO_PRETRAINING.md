**Real-audio corpus pipeline — v0.31.0, 18 September 2026**

This release connects verified recordings to masked pretraining and adds
stronger diagnostic evaluation. It implements explicit split/group manifests,
bounded native-rate crops, repeatable masks, training-only calibration and
baselines, and checkpointed data position. It does not automatically establish
that the encoder captures musical meaning, or silently turn the existing
sample library into an independent benchmark.

**Declare provenance before creating a split.** Write an assignment file such
as `assignments.json` with paths relative to that JSON file:

```json
[
  {"path": "audio/composition-a.wav", "group_id": "composition-a-family", "split": "train"},
  {"path": "audio/composition-a-remix.wav", "group_id": "composition-a-family", "split": "train"},
  {"path": "audio/composition-b.wav", "group_id": "composition-b-family", "split": "validation"},
  {"path": "audio/composition-c.wav", "group_id": "composition-c-family", "split": "test"}
]
```

The paths above are placeholders for your recordings. Choose groups according
to the intended generalization test: related recordings, original/remastered
versions, loop variants or instrument renders from one source should remain
together. Artist/session/pack grouping may be appropriate for a stricter test.
Genre folders and filenames do not prove independent source provenance.

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_music_corpus prepare assignments.json --output corpus.json
```

`AudioRecord.inspect` records rate, channels, frame count, raw-file SHA-256 and
a decoded-PCM SHA-256. PCM hashing streams float64 samples and includes format
dimensions; exact duplicate audio can therefore be detected even across
different lossless containers. Signed zero is canonicalized. The manifest
rejects duplicate decoded audio and groups spanning multiple splits. Group
labels have surrounding whitespace removed, but their musical meaning still
comes from the caller.

Hashes do not identify near duplicates, alternate edits, pitch/time changes,
resampled versions or equivalence after lower-precision conversion. Group
checks enforce declared provenance rather than discovering it. All records
must still be reviewed for a defensible benchmark. Unknown provenance should
remain explicitly unknown; do not label every file as independent by default.

**Audio loading is bounded and reproducible.** `AudioCropConfig` specifies
sample rate, channels, window samples, token hop, crops per recording, seed,
random/fixed mode and float32/float64 decoding. Windows align to the hop;
`from_seconds` rounds upward to that grid. No implicit resampling, downmixing,
peak normalization or gain changes occur.

`AudioWindowDataset` verifies hashes at construction by default, then checks
file size/mtime and headers while reading. Ordinary source changes stop the
run. The corpus trainer/evaluator refuses datasets constructed without
verification. These checks do not defend against deliberate filesystem
tampering. Manifest files use relative paths; identity excludes locations so
verified relocation remains possible.

Random crops depend on source identity, dataset seed, epoch and crop slot,
not ambient RNG or worker access order. Fixed evaluation crops cover aligned
positions across the recording and deduplicate positions when a short file
cannot supply several different crops. Only the requested interval is read.
`AudioWindow` records its source/group IDs, true length and global sample
origin. `collate_windows` preserves those origins in an `AudioBatch`.

Each physical crop has a stable `window_id` based on source PCM, interval,
precision and hop. It does not depend on other training records or manifest
ordering. Evaluation masks are keyed by this ID and the evaluation seed, so
changing batch order does not silently change the evaluation problem.

**Run a small explicit experiment.**

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_music_corpus train corpus.json --output outputs/corpus-run --steps 100 --updates-per-invocation 10 --window-seconds 2 --mask-seconds 0.25 --batch-size 2
```

The reference CLI uses a compact hybrid encoder and one local/global context
layer, runs single-process float32 training, and defaults to CPU. `--device
cuda` selects CUDA explicitly. Model initialization, crop/shuffle randomness,
training masks and evaluation masks have separate seeds. `--calibrate` enables
the optional frozen target statistics; it remains off by default.

It verifies all selected train/validation files, constructs a training-only
constant-target baseline, and optionally calibrates from deterministic epoch-0
training crops. Calibration identity includes the crop dataset and teacher
configuration. Already-fitted statistics with different provenance are rejected.
Validation examples are never used to fit the mean or calibration.

Outputs include `configuration.json`, `checkpoint.pt` and an evaluation JSON
for the completed optimizer step. Checkpoints are saved initially, every ten
updates by default (`--save-every`), and at the invocation's end. An existing
nonempty output folder requires explicit resume. Interruptions can resume from
the most recent completed checkpoint, rather than a partially finished batch.

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.train_music_corpus train corpus.json --output outputs/corpus-run --steps 100 --updates-per-invocation 10 --window-seconds 2 --mask-seconds 0.25 --batch-size 2 --resume outputs/corpus-run/checkpoint.pt
```

Keep model, corpus, crop and total-step configurations unchanged for exact
resume. The run stores epoch, next batch, baseline, calibration identity and
both dataset identities alongside the existing trainer state. Data-provenance
mismatches are checked before model weights are changed. Same-code/same-data
tests reproduce the next crop IDs, mask/dropout loss and parameter update
exactly. Intentional code/corpus changes require a deliberate new run or
weights-only migration, not silent continuation.

This CLI is a reference recipe, not a distributed data engine. Decoding and
teacher extraction loop over examples. Models and all selected recordings must
agree on rate/channels; prepare a consistent corpus explicitly if necessary.
The existing coefficient and attention limits still apply. Very short crops
with fewer than two tokens are rejected for masked pretraining.

**Evaluation compares stronger baselines in common units.**

| Report method | What it tests |
| --- | --- |
| `model` | Full trained student's masked prediction |
| `training_mean` | Constant teacher-feature mean fitted on training crops only |
| `zero_visible_audio` | Same student/mask/timing with all acoustic input zeroed |
| `global_path_disabled` | Same trained weights with the global context contribution removed |
| `visible_neighbor` | Interpolation of visible features extracted from the corrupted waveform |

The neighbor baseline does **not** interpolate clean teacher features outside
the mask: wavelet/STFT support could carry the hidden answer into those features.
It erases masked waveform samples before feature extraction. A regression
requires its predictions to remain unchanged when hidden clean samples change.

The global-path ablation skips global Transformer layers and zeros their
contribution to fine states. It is an intervention on an already-trained model,
so distribution shift can affect its result; it is not equivalent to retraining
a matched local-only architecture. If the full model wins, that alone does not
prove learned long-range musical structure. Likewise, beating a constant does
not establish superiority over visible interpolation or semantic baselines.

All methods are scored in original teacher-feature units, including when the
model trains on standardized targets. Reports record teacher and mask configs,
crop identity, window IDs/origins, RMS, target variation and model fingerprint.
Changing the teacher changes the task, so hold teacher configuration fixed
when comparing student architectures.

`micro_loss` weights by actual masked sample count. `macro_group_loss` gives
each declared group equal weight; per-group and per-window results are retained.
This avoids hiding weak groups behind recordings that supply many windows.
These remain acoustic pretext losses, not instrument-identity, harmony, groove,
arrangement or perceptual similarity scores. Validation is for development;
the explicit `test` split should remain reserved for final evaluation.

**Library interfaces.** `AudioManifest`, `AudioRecord`, `AudioCropConfig`,
`AudioWindowDataset` and `collate_windows` support custom loaders.
`fit_training_target_mean` rejects non-training datasets.
`evaluate_masked_corpus` requires verified held-out fixed crops and a matching
training baseline/teacher. It restores module train/eval modes afterward.
`CorpusPretrainingRun` connects the existing trainer, length sampler, data
cursor, calibration and evaluator. `MusicPretrainer.load_checkpoint` now accepts
an optional expected-data-state guard. `MusicContextEncoder` and the student
path expose `use_global_context=False` for the diagnostic ablation.

**What was run on existing recordings.** Two eight-second stereo synth loops
from the local soundgen examples were verified and cropped at native 44.1 kHz.
Their one-second-sized crops produced finite predictions and losses through
the masked model. Both recordings were kept in one unknown-provenance training
group. No optimizer updates or held-out quality comparison were performed on
them. [corpus_io_smoke.json](benchmarks/corpus_io_smoke.json) records the source
hashes, crop geometry and finite-output checks.

The CLI's prepare/train/evaluate/resume cycle and evaluation aggregation were
tested on generated audio fixtures with explicit disjoint groups. Meaningful
real-music evaluation still requires a curated corpus and task-specific labels
or probes; this release supplies the reproducible path to that evidence.

[Validation record](benchmarks/CORPUS_VALIDATION.md) lists the library,
editor, documentation and CLI checks, along with their limits.
