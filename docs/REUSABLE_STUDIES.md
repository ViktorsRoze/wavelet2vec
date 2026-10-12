# Reusable percussion studies — v0.37.0

27 September 2026. `wavelet2vec-study` replaces dataset-specific orchestration
for the named-solo/cumulative-mixture workflow. One reviewable project describes
the performance, source files, production metadata and model variants. A
workspace retains checked analyses, shared prepared audio, checkpoints,
evaluation results, listening exports and performance traces.

This workflow describes **one performance with a temporal development segment**.
It does not manufacture independent train/test groups from its mixture variants.
Use the separate multitrack-corpus workflow for independent compositions/groups.

## Initialize a project

The supported discovery convention is explicit: `only_NAME.wav` for solos and
`NAME+NAME.wav` for their mixtures. Labels come from filenames, not recognition.
Every solo must appear in a mixture, and unknown/repeated mixture labels fail.

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli init drum_substem_dataset_BPM128_4bars --project outputs/drum-study.json --group drum-performance-01 --bpm 128 --bars 4 --train-bars 3 --master-processing "master limiter"
```

The installed `wavelet2vec-study` executable is equivalent. Initialization
records source hashes, native rates/channels/lengths/subtypes, source order,
musical intervals, declared group and production information. It requires a
new project path and never modifies originals. All inputs must contain the
declared musical body and share native rate/channels; extra tails remain in
the originals and full-render auditions.

`--beats-per-bar` counts quarter-note beats and defaults to four. There is no
fixed 128 BPM or four-bar assumption. `--additive` is an explicit declaration;
leave it off for master-processed/non-additive targets. Full additive-target
verification still applies when enabled. Equal lengths do not themselves prove
sample alignment; exports must be aligned before preparing this study.

Review/edit the JSON before creating a workspace. Generated `baseline`, `fine`
and `balanced` variants are starting recipes, not quality guarantees. They
include backend, band/Q settings, dimension, window duration, source balancing,
seeds, batch size, crop count and total optimizer budget. Add/edit named variants
there; generated directory IDs prevent variant labels becoming arbitrary paths.

## Run and resume

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli run outputs/drum-study.json --output outputs/drum-work --variant baseline --device cuda --updates 25
```

`run` audits, prepares, trains up to the requested additional updates, exports
an inference bundle, evaluates the development segment and renders listening
files. Repeating the command resumes the saved checkpoint, reuses prepared
audio and reuses valid analysis caches. The project recipe's total step budget
remains fixed; `--updates` controls work per invocation, not a new schedule.
`--save-every` defaults to ten optimizer updates. CPU is the default device;
CUDA is explicit. `--threads` defaults to one CPU thread.

Individual stages are also available:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli audit outputs/drum-study.json --output outputs/drum-work
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli train outputs/drum-study.json --output outputs/drum-work --variant fine --device cuda --updates 25
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli evaluate outputs/drum-study.json --output outputs/drum-work --variant fine --device cuda
```

`report.html` compares available variants, names the optimizer update count,
shows per-source scores and offers original-solo/output audio players. Scores
cover only the development interval. Listening files cover the full recording,
including training bars and tails. Reports do not establish blind instrument
recognition, a clean separation model or independent musical generalization.

## Reuse and integrity

- Original audio hashes and headers are checked before use; changed inputs
  stop the workflow. The loaded project must also remain unchanged.
- Per-source event caches include audio identity, body interval, tempo/meter,
  analysis options and implementation/dependency identity. Checksums protect
  stored results. An interrupted audit retains successfully cached sources.
- Prepared float32 audio is generated once in bounded blocks and shared across
  variants. Native rate/channel layout is retained; there is no gain
  normalization. Output is FLOAT WAV or RF64 as needed. Preparation excludes
  development audio and is published only after verification.
- Training resumes from the authoritative checkpoint. Diagnostic history is
  reconciled to its saved update position, so interrupted/replayed updates are
  not counted twice. Model/run/data mismatches retain strict resume guards.
- Evaluation caches depend on the study, bundle, device and interval, and check
  numerical JSON plus rendered file hashes. A modified audition is rejected.
  Valid caches can rebuild a missing HTML report without recomputing audio.

The project and implementation are fixed for a workspace. Changing either
requires a new workspace; portable inference bundles remain a separate route
for intentional model migration. This is conservative cache invalidation,
not a promise to reuse results across arbitrary code changes.

Only one operation may write a workspace at a time. `.study.lock` records its
process ID and is removed on normal completion/errors. If the process is
forcibly killed, confirm it has stopped before removing that one stale lock.
Do not run simultaneous writers or mutate source/model state during a study.

The event audit and development evaluator process their requested intervals
in memory, with conservative configurable `max_analysis_bytes` and
`max_evaluation_bytes` payload guards, both initially 512 MiB. These exclude
some library/FFT/model overhead and are not total RAM/VRAM limits. For long
recordings, keep studies appropriately scoped and use the existing bounded
file renderer. File rendering remains window-bounded even though the current
development metric implementation materializes target intervals.

## Profile before changing language or backend

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m wavelet2vec.study_cli profile outputs/drum-study.json --output outputs/drum-work --variant fine --device cuda --warmup 2 --iterations 5
```

Profiling constructs a disposable model/trainer. It does not load or change
saved variant weights or optimizer progress, and it restores the caller's
Torch RNG. The recipe needs enough steps for warmup, measured updates and one
extra traced update. Synchronized **unprofiled** timings are recorded separately
from the traced update, since profiling adds overhead.

The output includes per-update timings, device/thread settings, Torch version,
CPU/device operator counters, a Chrome trace and peak PyTorch-allocated CUDA
bytes. That allocation metric excludes some driver/library memory. A training
status's `update_seconds` includes final checkpoint/export overhead; use the
dedicated profiler's timings for update-speed measurements.

For Python/NumPy event-analysis profiling, run the audit under `cProfile` with
a fresh workspace so cached results do not hide the computation:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m cProfile -o outputs/events.prof -m wavelet2vec.study_cli audit outputs/drum-study.json --output outputs/drum-profile-work
```

See [speed measurements and native-language strategy](PERFORMANCE_STRATEGY.md).
The current changes already reuse target FFTs in event matching and group
equal-grid compact-wavelet masks to reduce GPU kernel launches. They preserve
the native 192 kHz/float32 contracts; no lower-rate shortcut was introduced.

The Python interfaces are `create_percussion_study(...)` and
`PercussionStudy(project, output)`, with `audit`, `train`, `evaluate` and
`profile` methods. Generated 8/192 kHz fixtures and the supplied drum recordings
exercise this lifecycle; [validation details](benchmarks/STUDY_VALIDATION.md)
separate tooling correctness from musical quality.
