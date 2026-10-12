# Bounded rendering validation — v0.34.0

22 September 2026; local Windows `sound` Python environment. No musical
dataset or trained production model was used.

| Check | Result |
| --- | --- |
| Complete library suite | **511 passed**, 115.56 seconds; one existing brightness-test tensor-to-scalar warning |
| Focused renderer + synthesis checks before adding CUDA case | **78 passed**, 11.50 seconds |
| New renderer regressions included in full suite | **28 passed**, including native 192 kHz CUDA file export |
| Runnable documentation | **23 Python examples executed** |
| Editor compatibility suite | **36 passed**, 17.56 seconds |
| Installed/runtime package version | Both **0.34.0**, dependencies unchanged |

The new tests compare the rolling overlap accumulator with an independently
implemented full-recording accumulator. They use nonzero learned head weights
on all three backends, overlaps of 0/25/75%, two-example batches and large
sample origins. Other tests cover single samples, exact/partial windows,
final impulses, stable output storage, bounded reads and malformed readers.

Editor checks cover spectral, stereo, spectral-edit UI/effects, phase-layer
and library behavior with Qt offscreen. The editor working tree remained
unchanged; this release does not add a new rendering UI.

Model mode and grad-state checks run before iterator yield and after close or
failure. File tests cover native 192 kHz FLOAT and DOUBLE WAV/RF64, values above
unity, exact storage of the computed output, source bytes unchanged, progress
accounting and receipts. Source labels containing path components cannot
escape generated output filenames.

Cancellation, callback/model exceptions, nonfinite predictions, source mutation
and a destination appearing during rendering all leave no published partial
stems or staging leftovers. Existing user files are preserved. File rate,
channel and precision mismatches are rejected. A size-only test selects RF64
for one hour of 192 kHz stereo float32 and rejects a forced undersized WAV
container; actual small RF64 exports are also checked.

**Generated file experiment.** Reproduce with:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.bounded_render_smoke
```

The initialized two-source separator splits a generated stereo signal equally.
The signal includes sinusoidal content and impulses above unity. Generation,
rendering and reconstruction comparison use bounded blocks in temporary files.
No original user recording is modified.

| Input duration | Frames | Model calls | Largest model input | Maximum summed-source error |
| --- | ---: | ---: | ---: | ---: |
| 0.25 seconds | 48,000 | 8 | 9,216 samples | 2.38e-7 |
| 10 seconds | 1,920,000 | 313 | 9,216 samples | 4.77e-7 |

Both use 192 kHz stereo float32, an 8,192-sample core window, 2,048-sample
overlap and 512 samples of real context per side. Their explicit render-buffer
estimate is the same **1,146,880 bytes**. This excludes model parameters,
carriers, FFT/neural workspaces, allocator caching and file-library buffers;
it is **not a measured RSS or VRAM peak**. File storage grows with duration.

The ten-second CPU run completed in about 6.19 seconds on this local machine.
That is one small initialized-model integration check, not a throughput
guarantee for a trained separator or arbitrary hardware. Summed-source SNR was
about 133.5 dB for both runs. [Raw results](bounded_render_smoke.json) include
configuration, implementation fingerprint, timing and geometry.

The CUDA regression renders a 1,025-sample 192 kHz float32 file using the local
GPU and checks finite reconstruction within `2e-6` tolerance. It does not
claim bitwise accelerator reproducibility or whole-song GPU memory capacity.

Overlap-add agreement is agreement with the chosen window algorithm. It does
not establish equivalence to unchunked neural inference, inaudible seams,
stable instrument identities or musical separation quality. Those remain
separate model/data evaluation requirements.
