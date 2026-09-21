**Long-audio foundation validation — 13 September 2026, v0.27.0**

The full wavelet2vec suite passed with the new global-origin and sliding APIs:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
```

Result: **336 passed in 82.09 seconds**, including 23 sliding/global-origin
tests. One existing brightness-model test emitted a tensor-to-scalar warning.

Tests cover all three spectral backends, unique token ownership, exact sample
coverage and partial final patches, global timestamps, core reconstruction
from local coefficients, unchanged raw-patch tokens, phase-reference alignment
across different origins, requested-context flags, lazy bounded reader calls,
gradient-mode restoration across generator yields, training gradients,
duration-weighted coarse pooling, checkpoint/window mismatch, file-reader
equivalence, sample-rate validation and CPU-input/CUDA-window extraction.

The initial edge-context assertion assumed that only the final window could
lack right context. The 129-sample case correctly showed that a penultimate
window can also lack the requested context near EOF; the regression now
checks its exact flag pattern. No context flags were relaxed to hide missing
samples, and reconstruction tolerances were not loosened.

After the local editable installation was updated with `--no-deps
--no-build-isolation`, runtime and installed metadata both reported `0.27.0`.
Dependencies were not upgraded. The existing editor integration was checked:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q tests/test_spectral.py tests/test_spectral_stereo.py tests/test_spectral_edit_ui.py tests/test_spectral_effects.py tests/test_phase_layer.py tests/test_library.py
```

Result: **36 passed in 18.68 seconds**. Editor source was not modified.
All eight Python examples across the hybrid, compact and long-audio guides
executed successfully. Local guide links resolve and `git diff --check` passes.

This validation establishes extraction/alignment/numerical contracts. It does
not measure understanding of long music, whole-track-transform equivalence,
true filter-support validity, seamless predicted stems or trained-model
quality. The earlier compact performance benchmark is historical and was
not reused as a measured sliding-window performance claim.

