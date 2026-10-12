# Reusable study and performance validation — v0.37.0

27 September 2026; local Windows `sound` Python environment.

| Check | Result |
| --- | --- |
| Final complete library suite | **551 passed**, 118.21 seconds; one existing brightness-test warning |
| Focused study, synthesis and event-analysis suites | **67 passed**, 21.00 seconds |
| Study workflow suite | **8 passed**, 20.76 seconds |
| Editor compatibility suite | **36 passed**, 146.71 seconds |
| Runnable Python documentation examples | **25 executed** |
| Installed/runtime version | Both **0.37.0**, dependencies unchanged |

The initial full suite reported 550 passes and one subprocess timeout in the
existing inference CLI inspection test during concurrent verification. That
same test passed independently in 10.24 seconds without a code change. The
subsequent separate full run passed all 551 tests. Do not interpret the unusually slow concurrent validation wall
times as model-training benchmarks.

Study regressions cover explicit discovery/tempo geometry, hash-checked cache
reuse, corrupted caches, native 192 kHz FLOAT preparation/rendering, original
source preservation, prepared-file reuse, source mutation, workspace identity,
exclusive writers, resource guards and automatic checkpoint resume. Resumed
and uninterrupted CPU model parameters compare exactly. Diagnostic history
is reconciled to checkpoint position, evaluation cache hits rebuild reports,
and modified audition files are rejected.

The CLI lifecycle runs `run` twice in separate processes, verifying cache
hits, update progression and generated reports. The installed executable's
help entry point was also exercised. Profiling tests confirm that tracing a
disposable instance leaves the stored training checkpoint and caller RNG
unchanged.

CPU/CUDA grouped-decoder regressions compare all predicted compact bands,
waveforms and input/token/decoder gradients with the prior per-band algorithm.
The grouped path reduces repeated interpolation/indexing operations while
retaining actual physical band grids. Existing source-synthesis contracts
remain under the full regression suite.

## Real-data workflow check

The supplied drum dataset was initialized as a single non-additive performance
with declared master limiting. Both workflow checks advanced an example run
from one to four updates, evaluated all seven mixes and produced full-duration
audition stems. These four-update models validate tooling, not musical quality.
Original files were verified by SHA-256 throughout.

The final grouped implementation measured:

- Cold event audit: **16.817 s**; repeat audit: **0.06197 s**, eight cache hits.
- Evaluation/render work: **14.636 s**; full cached evaluation call: **0.2644 s**.
- Three synchronized unprofiled 96-band GPU updates: median **0.23525 s**,
  compared with **0.34761 s** before grouping.
- Traced CUDA launches: **4,246**, compared with **13,020** before grouping.
- Peak PyTorch-allocated CUDA memory: **95,375,872 bytes** versus **96,883,712**.

The machine used an NVIDIA GeForce GTX 1050 Ti. Timings are local samples;
profiling has separate overhead, and allocator measurements are not total
device/process memory. A hardware/architecture throughput guarantee is not
claimed.

All eight solo analyses retain identical measured event positions, repeat
groups, pitch candidates and waveform-fit values relative to the saved
pre-optimization audit. The comparison covers all **212 events** in this
performance, not arbitrary future inputs.

Evidence files:

- [Event FFT-reuse timings](workflow_performance/event_optimization.json)
- [All-source analysis parity](workflow_performance/all_source_analysis_parity.json)
- [Initial workflow measurement](workflow_performance/workflow_smoke.json)
- [Grouped-decoder/workflow measurement](workflow_performance/grouped_decoder_optimization.json)
- [Final study project](workflow_performance/drum_study.json)
- [Final smoke-test listening report](workflow_performance/workspace_grouped/report.html)

The full traces and operator tables live under each workspace's `profiles`
directory. The [performance strategy](../PERFORMANCE_STRATEGY.md) records the
evidence-based C/C++/Rust/Zig options. No native extension or compiler
dependency was introduced in this release.
