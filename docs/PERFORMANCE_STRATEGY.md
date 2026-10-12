# Performance measurements and native-code strategy

27 September 2026. Keep a Python-facing library, optimize measured work, and
introduce native kernels where they improve end-to-end cost without losing
precision, gradients or portability. C, C++, Rust and Zig are all candidates;
language choice alone is not a performance result.

## Measured and implemented

The closed-hi-hat event audit initially spent 6.51 of 7.13 profiled seconds in
1,102 event comparisons. Repeated coarse pitch searches recomputed target FFTs
and invoked separate channel correlations. Caching the target spectrum inside
each comparison and batching channels removed redundant work.

Three unprofiled trials before/after gave median times of **5.854 → 4.839 seconds**,
or **1.21× throughput / 17.4% less time** on this case. Measured onsets, repeat
groups, pitch candidates, fit scores and lags were unchanged. This is one real
track, not a universal speed guarantee. [Raw event measurements](benchmarks/workflow_performance/event_optimization.json)
and the before/after cProfile files retain the evidence.

A native-rate training profile then found **13,020 CUDA launches** in one
96-band update, including 576 complex indexing-backward kernels. Compact bands
sharing an FFT length also share their physical sample coordinates. Grouping
their decoder masks allows one interpolation per group rather than per band.

| Matched study recipe, local GTX 1050 Ti | Before grouping | After grouping |
| --- | ---: | ---: |
| Synchronized median update, three samples | 0.34761 s | 0.23525 s |
| CUDA launches in a traced update | 13,020 | 4,246 |
| Peak PyTorch-allocated bytes | 96,883,712 | 95,375,872 |

That is about **1.48× update throughput / 32.3% less time**. CPU/CUDA float64
regressions compare grouped coefficients, synthesized waveforms, input/token
gradients and decoder-parameter gradients against the per-band implementation.
Existing float32 synthesis/training checks also pass. Traced operator timings
include profiling overhead and must not be substituted for unprofiled update
times. [Raw grouping measurements](benchmarks/workflow_performance/grouped_decoder_optimization.json)
identify the code and trace location.

Workflow reuse matters as much as kernels. On the eight supplied solos, the
final study audit took **16.82 s cold versus 0.062 s cached**, still checking
original hashes. Development evaluation/render work took about **14.64 s**;
a cached call checking saved results and audition hashes took **0.264 s**.
The cold evaluation field times its evaluation/render work, while the cached
measurement includes the full API call. These are local timings with warm OS
file caches, not storage-independent benchmarks.

The study lifecycle smoke used only four updates to validate preparation,
resume, export and cache behavior. It does not replace the earlier 300-update
musical comparison or claim improved separation quality from these speedups.

## Where C/C++/Rust/Zig fit

| Component | Preferred next experiment | Reason |
| --- | --- | --- |
| GPU compact-band interpolation/indexing | Group/vectorize first; C++/CUDA custom operator if remaining launch/index costs justify it | Fits the current PyTorch tensor, device and autograd path. |
| CPU event-comparison objective | Prototype a fused interpolation + channel-gain + residual kernel in C++ or Rust | The remaining profile contains many small array allocations and repeated objective evaluations. |
| Existing audio codecs/file support | Keep libsndfile integration; add wrappers only for a concrete missing feature | File I/O already uses a mature native implementation. |
| Stable standalone DSP ABI | C interface, implemented in C/C++, Rust or Zig according to the library being reused | Keeps host-language integration independent of implementation language. |
| Training orchestration, manifests, reports | Keep Python | These are not the measured numerical bottlenecks and benefit from existing tooling. |

For PyTorch extensions, C++/CUDA is the direct integration path: register the
operator and its training behavior rather than wrapping opaque kernels that
break gradients or compilation. [PyTorch's custom-operator guide](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
describes extension building and autograd registration. A C++ wrapper that
still dispatches the same thousands of separate tensor operations will not,
by itself, eliminate those launches.

Rust is a reasonable option for a CPU comparison kernel behind the existing
NumPy-facing API. [rust-numpy](https://github.com/PyO3/rust-numpy) provides PyO3
bindings to NumPy arrays; avoid unnecessary ownership conversions/copies when
measuring a prototype. Zig can expose a C ABI for a standalone DSP component,
as described in the [official Zig documentation](https://ziglang.org/documentation/master/#Exporting-a-C-Library).
These are integration options, not measured claims that one is faster here.

There is no reason to replace existing native libraries merely to remove
Python from the package name. SoundFile already uses libsndfile/CFFI/NumPy,
documented in its [implementation](https://python-soundfile.readthedocs.io/en/latest/_modules/soundfile.html).
SciPy also supports [alternative FFT backends](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.set_backend.html)
if a representative benchmark demonstrates a benefit. Keep backend selection
explicit and preserve analysis/synthesis conventions.

## Acceptance gate for a native extension

1. Demonstrate material end-to-end improvement on representative real events
   or training windows after accounting for copies, allocation and dispatch.
2. Retain a reference implementation and compare silence, stereo phase,
   partial windows, 192 kHz/float32, float64 diagnostics and boundary cases.
3. For training operators, validate forward/backward behavior, device transfer,
   gradients and supported compiler integrations. Avoid accidental graph breaks.
4. Include the native build/backend identity in caches, checkpoints and export
   provenance. The present Python/dependency fingerprint alone is insufficient
   for a separately changing native binary.
5. Provide reproducible builds, wheels/fallback behavior and a supported-platform
   policy. Do not make a local compiler a silent requirement for basic use.

No new native extension or compiler dependency was added in v0.37.0. Measured
batching/reuse produced useful gains first. The next native prototype should
be narrow: the CPU event-fit objective, or a registered C++/CUDA operator if
remaining GPU profiling supports it. [PyTorch's profiler guide](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
provides the tracing facilities exposed by the study command.
