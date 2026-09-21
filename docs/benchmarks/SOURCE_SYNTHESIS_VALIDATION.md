# Source synthesis validation — v0.32.0

21 September 2026; local Windows `sound` Python environment.

**192 kHz requirement update:** The expanded synthesis suite now has **51
passing tests** (5.60 seconds), including native 192 kHz float32 file/corpus,
training and export checks. See [the format contract](../HIGH_RESOLUTION_AUDIO.md).
The full-suite count below records the preceding v0.32.0 run, before those
16 additional cases. Production perceptual qualification remains open.

- Full library suite: **451 passed**, one pre-existing brightness-test
  tensor-to-scalar warning, 73.47 seconds.
- New source-synthesis suite: **35 passed**, 5.06 seconds.
- Editor compatibility: **36 passed**, 14.24 seconds, covering spectral,
  stereo, spectral-edit UI/effects, phase-layer and library tests with Qt
  offscreen. Existing editor DSP paths remain compatible; no new separator
  UI was implemented in this release.
- Runnable Python documentation: **19 blocks** executed, including all three
  new source-synthesis examples and the previous sixteen examples.
- Editable package installed as **0.32.0**, without dependency changes.

The new tests cover all three synthesis backends, float64 identity at 48 and
96 kHz, one-sample/partial-hop lengths, stereo polarity and impulses, physical
mask coordinates, coefficient-projection idempotence and gradients,
supervised gradient flow through encoder/context/both heads, residual output
with zero carriers, NaN padding isolation, chunked rendering, mode restoration
on success/failure, source-name checkpoint compatibility and payload guards.
A separate compact-wavelet check covers DC plus a weak near-Nyquist tone.
These small numerical tests are not full bandwidth or perceptual qualification.

**Accelerator check.** On the local NVIDIA GeForce GTX 1050 Ti, nine float32
CPU/CUDA comparisons exercised three backends at 48, 96 and 192 kHz with
1,025-sample stereo inputs, a context network and nonzero mask/residual weights.
Maximum waveform differences were below `1e-7`, and gradients were finite.
[Raw accelerator results](source_synthesis_accelerator.json) include the
implementation fingerprint. This is not a mixed-precision, throughput or
long-recording accelerator benchmark.

**Supervised learning smoke.**

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.supervised_separation --steps 60
```

Model seed 7; 24 training examples from seed 105 and eight held-out synthetic
examples from seed 206. Each example is 1,024 stereo samples at 8,192 Hz.
The two sources occupy distinct frequency registers and fixed pans. The
experiment is intentionally easy and admits spectral/spatial shortcuts.

| Backend | Parameters | Held-out loss | Equal-split loss | Source SNR, low/high (dB) |
| --- | ---: | ---: | ---: | ---: |
| Dense wavelet | 33,136 | 0.003268 | 0.039854 | 24.89 / 23.92 |
| Compact wavelet | 33,136 | 0.003385 | 0.039854 | 23.97 / 24.34 |
| Complex STFT | 69,256 | 0.003261 | 0.039854 | 24.57 / 24.11 |

Equal-split source SNR is 3.19 / 2.78 dB. Loss is the same waveform L1 plus
complex spectral L1 for each validation run. Training additionally uses the
guarded mixture term. The differing parameter counts and one training seed
do not support a ranking of representations or a wavelet superiority claim.

The learned models' chunked outputs were finite but differed from full-clip
outputs: mean absolute differences were approximately 0.00044–0.00049 on
one validation example. Identity overlap-add passes; learned chunk invariance
is not claimed. [Raw experiment results](source_synthesis_smoke.json) retain
configurations, seeds, parameter counts, losses and measurements.

**Remaining work.** No real multitrack training, full-song held-out listening,
neural time stretching, reference-note synthesis or 96 kHz perceptual-quality
claim follows from these results. Next implement verified aligned multitrack
loading and a supervised training/resume recipe, then evaluate complete
recordings with strong baselines and declared source groups.
