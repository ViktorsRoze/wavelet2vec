**Compact-carrier validation — 7 September 2026, v0.26.0**

The existing sound environment was used. The package was reinstalled locally
with `pip install --no-deps --no-build-isolation -e .`; dependency versions
were not changed. Runtime and package metadata both report `0.26.0`.

From the wavelet2vec checkout, the complete suite passed:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
```

Result: **313 passed in 70.26 seconds**. An existing brightness-model test
emitted its tensor-to-scalar warning. After tightening the shape estimator's
integer validation, the focused hybrid/compact suites were rerun:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q tests/test_hybrid.py tests/test_multirate.py
```

Result: **97 passed in 6.15 seconds**, including 41 compact-carrier tests.
Coverage includes both FFT layouts, odd/even/one-sample signals, endpoint
weights, stereo/antiphase preservation, an independent direct Fourier
evaluation of compact samples, source-axis synthesis, reanalysis consistency,
physical time coordinates, carrier-demodulated phase, numerical gradients,
silence gradients, memory guards, oversampling, configuration migration,
supervised source-head backpropagation, and CPU/CUDA parity.

The first direct-Fourier reference test exposed float32 promotion in its own
integer-bin angle expression. Explicit float64 bin coordinates corrected the
reference; reconstruction tolerances were not relaxed to hide a DSP error.

From the editor checkout after the package update:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q tests/test_spectral.py tests/test_spectral_stereo.py tests/test_spectral_edit_ui.py tests/test_spectral_effects.py tests/test_phase_layer.py tests/test_library.py
```

Result: **36 passed in 14.58 seconds**. Editor source and its existing
spectral-edit backend were not changed. All four Python examples across the
hybrid and compact guides ran successfully, including the existing checkpoint
round trip. Guide artifact links resolve and `git diff --check` passes.

The [footprint benchmark](compact_memory.json) contains the final 30-trial
matrix: five carrier/layout configurations, CPU/CUDA, and three trials each.
It was rerun after grouping and after adding the STFT comparator. The final
configuration/data/timing records are the ones summarized in the guide.
Earlier exploratory timing runs were not retained as benchmark claims.
The real-kick result includes its hash and explicit fast-layout configuration.

The [learning experiment](compact_learning.json) was rerun after grouped
power-of-two sampling became the compact default. It records ten runs:
five branch/context choices across two seeds. The earlier dense/STFT results
in `hybrid_foundation.json` remain the historical v0.25.0 comparison. The
synthetic source task is not a real-music or compute-matched quality benchmark.

Allocation peaks refer to the PyTorch CUDA allocator. They do not establish
process-RAM limits, driver/FFT-plan memory use, long-file streaming performance,
or large-model training capacity. Both guides state these limits explicitly.

