**Target-calibration validation — v0.30.0, 16 September 2026**

The complete library suite passed **405 tests in 81.78 seconds**, with the
existing brightness-model tensor-to-scalar warning. Seven new calibration
cases cover weighted streaming statistics and inverse mapping, atomic failure
and gradient-mode restoration, representable scale floors, frozen query-time
behavior, retained hidden-sample isolation, checkpointed calibration, raw-v1
model state migration, and CUDA backward. The context/SSL and earlier tests
remain included in the full run.

After installation with `--no-deps --no-build-isolation`, runtime and package
metadata both reported `0.30.0`. The existing editor's focused spectral,
phase and library suite passed **36 tests in 17.28 seconds**. All sixteen
Python guide examples executed successfully; local guide links resolve and
`git diff --check` passes. Editor source was not modified.

The final [paired experiment](target_calibration_ablation.json) contains six
runs: raw/calibrated objectives for initialization seeds 17, 18 and 19, each
with 80 updates and matched data, mask and batch-order seeds. It was rerun
after the final calibration validation/migration guards; the reported numbers
were unchanged. Training statistics use only the training clips. The final
mean raw masked error is 0.001909 for raw targets and 0.001739 for calibrated
targets; the constant baseline is 0.001800. Only one calibrated run beats
the baseline. No real-music, downstream semantic or separation evaluation
was performed; this is an optional objective experiment.

Commands used the existing sound environment:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.calibration_ablation
```

Initial v0.29.0 context/SSL validation is recorded separately in
[SSL_VALIDATION.md](SSL_VALIDATION.md).

