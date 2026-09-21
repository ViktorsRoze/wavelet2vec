**Adapter/training/cache validation — 14 September 2026, v0.28.0**

The final full library run passed:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
```

Result: **378 passed in 92.75 seconds**. The only warning was the existing
brightness-model test converting a gradient-bearing tensor to a scalar.
The new tests include 29 adapter cases and 13 source-training/cache cases;
all previous hybrid, compact and sliding tests also passed.

Adapter checks cover stereo equality/opposite polarity/gain imbalance,
silent channels, independent noise, delayed-tone phase convention, channel
swaps and arbitrary channel-pair metadata, short-clip support, global-origin
invariance, numeric gradients, CPU/CUDA parity and silence backpropagation.
Multi-resolution checks cover physical time alignment, primary/secondary
reconstruction, gate normalization, exact branch exclusion, skipped analysis,
joint allocation guards, checkpoint mismatch and sliding extraction.

Training/cache checks cover missing versus known-absent sources, padded tails,
ignored NaNs, phase-sensitive loss, finite exact-silence gradients, stereo
normalization, additive-target checks, detached cache round trips, key changes,
corrupt/mismatched entries, atomic final storage, canonical processing metadata,
file hashes, editable package-code identity and long-interval timestamp precision.

The package was updated locally using `pip install --no-deps
--no-build-isolation -e .`; no dependencies were upgraded. Runtime and
installed metadata both report `0.28.0`. From the editor checkout:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q tests/test_spectral.py tests/test_spectral_stereo.py tests/test_spectral_edit_ui.py tests/test_spectral_effects.py tests/test_phase_layer.py tests/test_library.py
```

Result: **36 passed in 18.36 seconds**. Editor source and its legacy transform
integration were not changed. All twelve Python guide examples executed
successfully, including composed adapters, reconstruction, loss backward and
feature-cache round trip. Local artifact links resolve; `git diff --check`
passes. Example cache/checkpoint files were confined to a temporary directory.

These results validate numerical and API behavior. No claim is made that the
new adapters improve semantic music understanding, instrument separation or
restoration on real tracks. The supplied losses are reconstruction utilities,
not a completed self-supervised pretraining system. The remaining gap is
described in [MUSIC_REPRESENTATION_GAPS.md](../MUSIC_REPRESENTATION_GAPS.md).

