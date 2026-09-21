**Hybrid foundation validation — completed 7 September 2026**

The implementation and experiment began on 6 September; final documentation,
installation and editor compatibility checks were completed on 7 September.
Commands used the existing `C:/Users/vikto/an3/envs/sound/python.exe`
environment. The local editable wavelet2vec installation was updated to
v0.25.0 with `--no-deps --no-build-isolation`; dependencies were not upgraded.
Runtime and installed package metadata both report `0.25.0`.

From the wavelet2vec checkout:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
```

Result: **272 passed in 95.81 s**, including 56 hybrid tests. CUDA tests ran
on the available GTX 1050 Ti. One warning came from an existing brightness
model test converting a gradient-bearing tensor to a scalar.

Coverage includes independent coefficient reconstruction (with the raw
reference zeroed), odd/even/one-sample inputs, silence/DC/Nyquist/impulses,
stereo and polarity preservation, source-axis synthesis, independent
wavelet density/bandwidth, unpadded projection identities, a boundary
impulse case, demodulated signed phase features, partial-patch timing,
finite silence gradients, numerical gradient checking, invalid input and
conditioning, allocation guards, checkpoint configuration mismatch, CPU/CUDA
parity, and phase-feature ablation without changing the complex carrier.

From the existing viktor-rose-editor checkout:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q tests/test_spectral.py tests/test_spectral_stereo.py tests/test_spectral_edit_ui.py tests/test_spectral_effects.py tests/test_phase_layer.py tests/test_library.py
```

Result: **36 passed in 23.16 s** after the package update. This checks the
existing integration; editor source and its legacy inverse were not changed.
The earlier asynchronous editor run lost its process handle across a turn,
so this final completed run is the recorded compatibility result.

All three Python code examples in [the guide](../HYBRID_ENCODING.md) executed
successfully together, including checkpoint save/load in a temporary
directory. The first check encountered a Windows cleanup issue from leaving
the working directory inside that temporary directory; restoring it before
cleanup fixed the validation script. There was no product-code failure.
Guide artifact links resolve and `git diff --check` passes.

The experiment command and [20-run results](hybrid_foundation.json) are
documented in the guide. The small synthetic task verifies learning and
phase-aware synthesis plumbing, but provides no consistent advantage for
hybrid fusion or signed phase context. It is not a real-music evaluation,
compute-matched benchmark, or demonstration of novel separation quality.

