**Compact wavelet carrier — v0.26.0, 7 September 2026**

**v0.27.0 update:** [Long-audio extraction](LONG_AUDIO.md) now wraps the
compact carrier in overlapping context windows. Encodings carry an optional
global `sample_origin`; `band_times_s` includes it. The core transform still
operates on each padded fragment, with the limits described below.

The hybrid foundation now supports a multirate complex wavelet carrier.
Instead of storing every band at the audio sample rate, it retains a finite
frequency interval per band, moves it to baseband and represents it with a
shorter complex FFT. The complete filter bank remains invertible when its
squared-window coverage is well conditioned.

The new backend is opt-in: `backend="compact_wavelet"`. The existing dense
wavelet and STFT backends keep their behavior. Raw waveform, stereo channel
identity and amplitude remain available. Pooled features and learned tokens
remain lossy, as described in [HYBRID_ENCODING.md](HYBRID_ENCODING.md).

```python
import torch
from wavelet2vec import HybridConfig, HybridAudioFrontend, HybridTokenEncoder

config = HybridConfig(sample_rate=48000, backend="compact_wavelet",
                      n_bands=64, q=8, hop_length=256)
audio = torch.randn(1, 2, 48000) * 0.1
frontend = HybridAudioFrontend(config)
print(frontend.coefficient_bytes(tuple(audio.shape)))
encoding = frontend(audio)
print(encoding.coefficient_nbytes)  # same number: final coefficient storage
restored = encoding.synthesize()   # uses compact coefficients, not raw audio

# Each band has its own complex time grid; this is a tuple, not a rectangle.
print([band.shape[-1] for band in encoding.coefficients])
times_for_band_0 = encoding.band_times_s(0)

encoder = HybridTokenEncoder(config, channels=2, dim=64)
output = encoder(encoding)
output.tokens.square().mean().backward()
```

**What is preserved, and what changed.** Gaussian frequency windows are
truncated at a configurable number of standard deviations, rounded outward
to the padded signal's DFT bins. The first and last windows still cover DC
and the high-frequency residual. `compact_truncate_sigma=4` is the default.
The squared coverage is recomputed for these finite windows; a ratio below
`1e-6` is rejected. No uncovered frequencies are silently dropped.

Truncation changes the analysis filters, so compact coefficients are not
identical to those from the full Gaussian backend. Nevertheless, the audio
is retained because the weighted inverse accounts for the new overlap.
This is an exactly invertible construction in exact arithmetic, with
floating-point reconstruction error in practice. It is not a claim that
discarding arbitrary parts of an existing coefficient image is lossless.
Finite support can introduce temporal tails and affects predicted-mask
behavior; it requires its own editing/quality evaluation.

For a padded signal of length N, a band contains FFT bins `[lo, hi)`. Its
complex sample count M is at least `hi-lo`. Those bins are shifted by an
integer carrier bin and placed at distinct positions modulo M. Because
there are no collisions, this resampling introduces no within-band spectral
aliasing. Analysis uses `(M/N) * ifft(packed_spectrum)` to preserve band
amplitudes. Synthesis reverses the packing and scale, weights by the analysis
windows and divides by their squared sum. DC/Nyquist analytic weights are
handled separately. There is no naive striding of a full-rate analytic band.

Each `CompactBand` records center frequency, Gaussian sigma, FFT interval,
carrier bin and sample count. Sample j corresponds to original-audio time
`(j*N/M - padding_samples + sample_origin)/sample_rate`. These grids are often fractional
relative to audio samples. Index j in two different bands generally refers
to different times; use metadata, not array position, to align them.

This is grounded in established nonuniform/invertible filter-bank methods;
[Holighaus et al.](https://arxiv.org/abs/1210.0084) describe the broader
invertible constant-Q and sliced-processing framework. This implementation
does not reproduce their full streaming algorithm and makes no new-transform
novelty claim.

**Three controls govern the tradeoff.**

| Setting | Effect |
| --- | --- |
| `compact_fft_mode="power2"`, default | Round band sample counts to powers of two and batch equal-length bands. Higher payload, much lower dispatch overhead in the measured pipeline. |
| `compact_fft_mode="fast"` | Use SciPy `next_fast_len` per band. Smaller payload; many unique FFT sizes can be slow on GPU. Actual lengths are retained in metadata. |
| `compact_oversampling=2` | Double minimum samples per band before FFT-size rounding. Same frequency filters and inverse; more temporal samples for context and predicted masks, at higher memory/compute cost. |

Frequency density, Q and bandwidth floor remain independent controls from
the v0.25 foundation. Oversampling a compact band does not improve the
underlying filter's time-frequency resolution. Predicting rapid masks on a
slowly sampled band can alias modulation or generate inconsistent frequency
content; the inverse retains only the band's defined spectral interval.
More temporal sampling and explicit consistency evaluation are options, not
a guarantee of artifact-free arbitrary edits.

**Hybrid tokens are aligned without expanding full-rate bands.** Each band
is demodulated to its center-frequency reference, including the padding
offset. The implementation integrates a piecewise-linear interpolation over
each raw-patch interval to obtain signed real/imaginary averages. Sampled
power is integrated similarly for an RMS estimate. Low-rate bands can span
many raw patches; fractional interval integration handles that case. Power
interpolation is an estimate of local energy, not an exact continuous-time
power integral. Prefix sums use double precision to avoid cancellation in
short intervals late in a fragment.

Equal-length bands are processed as groups for analysis, synthesis and
context. On the one-second benchmark, power-of-two planning reduces 64
distinct FFT sizes to 10 groups. The group packing tables are proportional
to compact sample counts. No `[bands, full_audio_length]` coefficient or
window tensor is constructed. The aligned tokens are still lossy; keep the
compact carriers for reconstruction.

**Measured footprint and timing.** This benchmark uses one second of stereo
noise at 48 kHz, 64 wavelet bands, Q=8, 250 ms padding per side, float32, and
a 256-sample token hop. STFT uses a 2,048-point Hann window. Timing includes
analysis, synthesis, context features and an input-gradient backward pass;
it excludes a large separation network. CPU uses one torch thread. CUDA
runs on the GTX 1050 Ti. Warm time is the median of the last two of three
trials; these are small local measurements, not p95 latency guarantees.

| Carrier | Coefficients | CPU warm | GPU warm | PyTorch GPU allocation peak |
| --- | --- | --- | --- | --- |
| Dense wavelet reference | 70.31 MiB | 428.4 ms | 124.1 ms | 362.4 MiB |
| Complex STFT comparator | 2.94 MiB | 40.3 ms | 27.0 ms | 43.4 MiB |
| Compact, minimum fast sizes | 4.57 MiB | 178.7 ms | 381.1 ms | 29.1 MiB |
| Compact, grouped powers of two | 6.75 MiB | 109.6 ms | 101.8 ms | 71.3 MiB |
| Compact, grouped, 2x sampling | 13.50 MiB | 221.3 ms | 154.2 ms | 137.4 MiB |

The recommended grouped compact setting reduces coefficient storage by
about **10.4x** and measured GPU allocation peak by about **5.1x** relative
to the dense wavelet reference. Its measured CPU pipeline is about 3.9x
faster; GPU improvement is about 1.2x. The smaller fast-size variant saves
more memory but is slower on GPU, so it is not the default.

The STFT comparator remains cheaper in this experiment. Its frequency/time
resolution differs from the constant-Q bank; these measurements do not prove
equivalent representation quality or wavelet superiority. Compact wavelet
storage is also still larger than the original waveform. This is reduced
analysis redundancy, not a compressed-audio codec.

PyTorch allocation peaks include tensors and autograd working storage visible
to its CUDA allocator. They exclude some driver/FFT-plan allocations and are
not total process RAM/VRAM guarantees. JSON separately records coefficient,
window and group-table sizes. `max_coefficient_bytes` guards final coefficient
payload only; masks, activations, metadata and optimizer state remain extra.

Synthetic float32 round trips were approximately 127–133 dB SNR across the
tested paths/devices. A real existing stereo kick (`AA viktor rose A1 2.wav`)
through the compact fast-size path reconstructed in float64 at **302.1 dB**,
with maximum error **1.55e-15**. The file was read without modification;
its hash and configuration are recorded in the benchmark.

Reproduce the footprint/gradient benchmark:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.compact_benchmark
```

Results: [compact_memory.json](benchmarks/compact_memory.json).
See the [validation record](benchmarks/COMPACT_VALIDATION.md) for 313 passing
library tests, the focused numerical rerun, 36 editor compatibility tests,
and executed guide examples.

**The learning harness supports variable-rate source heads.** Compact mode
predicts an unbounded complex mask for each band, sampling the same raw-patch
mask rule at that band's actual times. It does not interpolate all bands to
a full-rate rectangle. The following command repeats the earlier small
lower-register-source task with the grouped compact carrier:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.hybrid_foundation --steps 100 --seeds 0 1 --backends compact_wavelet --output docs/benchmarks/compact_learning.json
```

The mean held-out source SNR is **8.62 dB** for hybrid complex-context tokens,
compared with the earlier dense wavelet run's **8.61 dB**. Raw-token,
spectral-token and magnitude-context ablations remain similarly close.
The useful result is that compact carriers support training and reconstruction
without an obvious collapse on this toy task. Two seeds on 16 synthetic
validation examples do not establish equal quality on music, improved
instrument separation, or an advantage for the phase features. See the
[complete learning results](benchmarks/compact_learning.json).

**Compatibility and remaining work.** Configurations now serialize with
`hybrid-v2`. `HybridConfig.from_dict` accepts v1 waveform/STFT configurations
and supplies defaults for the new compact fields. Compact carriers require
v2 metadata. Existing model dimensions and parameter names are retained.
To synthesize multiple compact sources, provide a tuple whose tensors all
share the new leading source/batch axes and match each original band length,
complex dtype and device.

This backend still analyzes one whole padded fragment; it is not streaming,
and long-file working memory still grows with duration. Padding is unchanged
and there is no universal boundary-validity guarantee. CUDA accumulation may
vary at floating-point rounding level. Next work should evaluate real stereo
sources, hard phase cancellation and transient masks, then add sliced
processing with valid-context metadata. The editor continues using its
existing DSP path until a deliberate integration and edit-behavior comparison.
