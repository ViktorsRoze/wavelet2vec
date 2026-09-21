**Initial context/SSL validation — v0.29.0, completed 16 September 2026**

The full library suite passed: **398 tests in 74.79 seconds**, with only the
existing brightness-model tensor-to-scalar warning. The 20 new context/SSL
cases cover distant-context influence, padding/empty groups, duration-weighted
positions, attention limits, seeded masks, hidden-sample prediction and gradient
isolation across four student architectures, ragged targets, teacher detachment,
silence, CPU/CUDA backward, accumulation failure handling, checkpoint boundaries,
configuration/code guards, exact next-update resume and length bucketing.

The existing editor's focused spectral/phase/library suite passed **36 tests
in 16.62 seconds** after installing v0.29.0 locally with no dependency changes.
All 15 Python guide examples ran successfully. Artifact links resolve and
`git diff --check` passes. Temporary checkpoints/caches were confined to the
guide-check temporary directory. Editor source was not changed.

The 80-step [synthetic demonstration](masked_music_demo.json) was rerun after
adding implementation/trainer provenance to its JSON. Numerical results were
unchanged: initial validation loss 0.025142, final 0.001872, training-mean
constant baseline 0.001800. This does not demonstrate useful music semantics.
The experiment uses fixed clean acoustic targets and no pretrained teacher;
data/mask seeds and model/optimizer configurations are recorded in JSON.

Commands used the existing sound interpreter:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m pytest -q
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.masked_music_demo --steps 80
```

Later target-calibration work is recorded separately; this file describes the
initial v0.29.0 implementation and its validation.
