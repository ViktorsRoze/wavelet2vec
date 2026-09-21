**Hybrid audio encoding foundation — v0.25.0, 6 September 2026**

**v0.27.0 update:** [Sliding-window extraction](LONG_AUDIO.md) adds global
sample/phase references, real context windows, unique token ownership and
coarse summaries. The local frontend accepts `sample_origin`; default-zero
calls keep the existing fragment contract. This is a foundation for future
musical-context models, not an established musical-understanding result.

**v0.26.0 update:** [Compact multirate wavelets](COMPACT_WAVELETS.md) add an
opt-in `compact_wavelet` backend with ragged complex bands, batched FFTs,
physical sample-grid metadata and aligned context features. Configurations
now write `hybrid-v2`; v1 waveform/STFT configurations still load. The dense
wavelet/STFT contracts and historical v0.25.0 experiment below remain useful
references; full-rate memory limitations below concern the dense backend.

The new foundation pairs a native-rate waveform with an invertible complex
spectral representation. A trainable encoder derives aligned waveform and
spectral tokens while retaining both signal carriers. This supports models
that learn musical context without pretending a compact embedding contains
every sample and phase relationship.

The first implementation provides waveform + Gaussian-wavelet and waveform
+ complex-STFT configurations. It is a tested research foundation, not a
pretrained separator, a compression codec, or a demonstrated best encoding.
It does not yet implement streaming analysis or long-context music
models. The default is waveform + wavelet; STFT is an explicit comparator.

**Quick start.** Input axes are always `[batch, channels, samples]`; no axis
guessing, stereo mixdown, peak normalization or resampling takes place.

```python
import torch
from wavelet2vec import HybridConfig, HybridAudioFrontend, HybridTokenEncoder

config = HybridConfig(sample_rate=44100, n_bands=64, q=8, hop_length=256)
audio = torch.randn(1, 2, 4410) * 0.1  # 100 ms stereo, float32

frontend = HybridAudioFrontend(config)
encoded = frontend(audio)
reconstructed = encoded.synthesize()  # actually inverts coefficients
print((reconstructed - audio).abs().max())

encoder = HybridTokenEncoder(config, channels=2, dim=128)
result = encoder(audio)
print(result.tokens.shape)            # [1, 18, 128]
print(result.valid_samples[-1])       # 58: final patch is partial
print(result.encoding.coefficients.shape)

# Example task loss; real training supplies source waveforms/other targets.
result.tokens.square().mean().backward()
```

`encoded.waveform` references the original input tensor, preserving its
autograd graph. Treat that tensor and the encoded coefficients as immutable.
The raw reference is not copied; clone the input yourself if another process
will mutate it. `synthesize()` uses the complex coefficients even if the raw
reference's values are replaced. It uses the reference's shape for output
length. No fallback to raw samples conceals an inaccurate inverse.

**Public interface and information contract.**

| API | Result / meaning |
| --- | --- |
| `HybridConfig` | Frozen, versioned analysis and context configuration |
| `HybridAudioFrontend(config)(audio)` | `HybridEncoding` with raw audio, complex coefficients and inverse metadata |
| `encoding.synthesize(coefficients=None)` | Reconstruct from original or predicted coefficients; supports additional source/batch leading axes |
| `encoding.context_features()` | Lossy features `[B,T,C*K*3]`, times in seconds `[T]`, valid sample counts `[T]` |
| `HybridTokenEncoder(config, channels, dim)` | Trainable branch projections and gated fusion |
| `encoder(audio_or_encoding, branches=...)` | `HybridLatents`: tokens plus the retained encoding, times and counts |
| `frontend.coefficient_bytes(shape, dtype)` | Exact size of the final coefficient tensor, before allocating it |
| `config.to_dict()` / `HybridConfig.from_dict(...)` | Configuration serialization with `hybrid-v2` validation and v1 dense/STFT migration |

For the wavelet backend, coefficients are `[B,C,K,N+2*padding]`. For STFT,
they are `[B,C,n_fft//2+1,N//hop_length+1]`. Coefficient shape, complex dtype
and device are checked when supplying replacement coefficients. Synthesis
does not require replacement leading axes to match the original batch: a
source axis can be added for multi-source predictions.

The waveform retains the original input sample values. Spectral round trips
are accurate to floating-point precision, not bit-exact archival recovery.
Learned tokens and pooled context features are **lossy**, even though their
parent object retains lossless-in-principle signal carriers. A model must
keep access to those carriers for precise synthesis. If the carrier is
discarded and only tokens are saved, the original audio is not guaranteed
recoverable. Preserve the original PCM or losslessly compressed audio when
exact archival recovery is required.

Float32 and float64 input are supported. Input precision is never silently
reduced. The trainable module uses float32 parameters by default; use
`encoder.double()` for float64 training and move module and input to the
same device explicitly. Half-precision audio FFT inputs are rejected. CUDA
carrier parity and token backpropagation have been tested; this is not a
general mixed-precision training guarantee.

**The wavelet backend has a defined inverse and independent resolution
controls.** Interior analysis bands use

`W_k(f) = exp(-0.5*((f-f_k)/sigma_k)^2)`

with geometrically spaced centers and

`sigma_k = sqrt((f_k/q)^2 + bandwidth_floor_hz^2)`.

Here `q` means center frequency divided by the Gaussian frequency standard
deviation. It is not a conventional −3 dB bandwidth Q. Center density
(`n_bands`) and bandwidth (`q`, optional bandwidth floor) are independent.
Increasing the density while retaining shared centers leaves those filters
unchanged; a regression test checks this directly. Larger `q` sharpens
frequency discrimination and broadens the temporal response. A positive
bandwidth floor limits the longest Gaussian responses at a frequency-detail
cost. Neither setting creates arbitrarily fine time and frequency resolution.

The first/last filters extend to DC/Nyquist as residual channels. They are
not ordinary admissible bandpass wavelets and should not be interpreted as
narrowband measurements. The implementation is an analytic Gaussian filter
bank with wavelet-like interior bands, not a new family of mathematical
wavelets.

Synthesis weights each coefficient FFT by its analysis window and divides
by `sum(W_k^2)`, reversing analytic positive-frequency doubling with separate
DC/Nyquist handling. This is the canonical weighted inverse of the periodic
analytic bank. With padding, that inverse is followed by cropping; it is not
a claim of an orthogonal projection for the complete constrained
zero-extension/crop operator. Tests check reconstruction, linear source
synthesis and the unpadded bank's orthogonal projection identity.

This new carrier is deliberately distinct from legacy `frame.py` and
`torch_frame.py`, whose synthesis sums unweighted bands and divides by
`sum(W_k)`. **Do not pass hybrid wavelet coefficients to the legacy inverse.**
Always keep the `HybridEncoding` and its configuration/windows together.
Changing the legacy inverse would change existing effect behavior, so the
new API uses a separate versioned contract.

Ill-conditioned filter banks are rejected before spectral FFT allocation:
the ratio of minimum to maximum squared-window coverage must be at least
`1e-6`. More centers or broader filters can repair coverage. This threshold
is a guard against disastrous gaps, not a universal error guarantee for all
possible signal amplitudes/configurations. The actual coverage is available
on the encoding for analysis.

**Boundary treatment and memory limits are explicit.** Wavelet analysis
zero-pads each side, defaulting to 250 ms, and synthesis crops to the original
length. This materially reduces the circular interaction of short fragments
in tested cases; it is not a universal 250 ms support guarantee. Gaussian
tails, especially low-frequency and residual-band responses, can require
more context. `padding_samples=0` explicitly chooses periodic analysis.
Reflective padding, validity masks, real-neighbor chunk context and
overlap-add streaming are future work.

Wavelets still use full-rate coefficients. The default byte guard rejects
final coefficient allocations above 256 MiB. It is **not a process/VRAM peak
limit**: FFT intermediates, float64 bank construction, context features,
saved autograd tensors, source predictions and optimizer state require
additional memory. A batch of 16 eight-second stereo signals at 44.1 kHz
and 128 complex64 bands already requires about 10.8 GiB before padding and
intermediates. Raising the guard cannot fix that algorithmic cost. Shorter
batches, precomputed detached carriers for fixed datasets, and a future
multirate/sliced backend are the intended approaches.

Passing a precomputed `HybridEncoding` to the encoder reuses its carrier.
Precompute under `torch.no_grad()` only when gradients to the input waveform
are not needed. Otherwise compute through the frontend in the training graph.
The current frontend rebuilds design windows and context features on calls;
it does not claim optimized window/tile caching or streaming throughput.

**The spectral tokens use signed demodulated context.** For each wavelet
band, remove the center carrier `exp(i*2*pi*f_k*t)` before averaging over a
raw-waveform patch. This avoids the earlier “coherence” feature's cancellation
when a stable sinusoid rotates through complete cycles inside a pooling bin.
The features per band/channel are

`[Re(mean(baseband))/(1+RMS), Im(mean(baseband))/(1+RMS), log(1+RMS)]`.

RMS uses a smooth machine-epsilon floor, with its silence offset removed
inside the log feature, to keep gradients finite at silence. Phase is
computed in double precision before reducing to the carrier dtype. Actual
coefficients remain untouched. There is no `angle(0)` operation and no
per-file peak normalization. Features preserve signed local phase content,
but pooling still loses within-patch detail.

For STFT, context uses bin-amplitude calibration with correct endpoint
weights, demodulates using each window's sample origin, and interpolates to
raw-patch centers. The full STFT uses a Hann window, centered constant
padding and explicit output length for inversion. `n_fft` must be even and
`hop_length <= n_fft/2`. This path follows the documented
[torch STFT](https://docs.pytorch.org/docs/2.11/generated/torch.stft.html) and
[inverse STFT](https://docs.pytorch.org/docs/2.11/generated/torch.istft.html)
contracts. Wavelet and STFT context footprints remain different; sharing
token times is not a claim of identical temporal resolution.

Raw patches and spectral features are projected separately to dimension D.
A learned sigmoid gate over both projected branches fuses them feature by
feature. `branches="waveform"` or `"spectral"` bypasses fusion for ablations.
`context_mode="magnitude"` zeros the two signed features while retaining
the same feature dimensions, coefficient carrier and decoder. That makes
phase-feature comparisons possible without silently dropping the source
phase from synthesis as well.

The final partial raw patch is zero-padded for projection; context averages
divide by its actual sample count. Returned times are the patch interval
midpoints in seconds, and `valid_samples` exposes its real length. Input
axes, channel order, amplitude and sample-rate metadata stay explicit.

**Using the carrier for source heads.** An illustrative complex head can
predict coefficients with a source axis:

```python
encoded = HybridAudioFrontend(config)(audio)
sources = torch.stack([0.25 * encoded.coefficients,
                       0.75 * encoded.coefficients], dim=1)
stems = encoded.synthesize(sources)  # [B, 2 sources, C, N]
```

This demonstrates linear synthesis, not separation. Real source heads must
learn from source audio targets. Permit complex/unbounded estimates or a
mask-plus-residual formulation because overlapping sources can cancel in a
mixture; nonnegative gains bounded by one cannot recover every source.
Predicted redundant coefficients may be inconsistent. Train/evaluate the
synthesized waveform and, when relevant, its reanalysis. Keep mixture-sum
consistency separate from source accuracy. Dry-source restoration from
nonlinear mastered mixtures needs a different target/rendering contract.

**Checkpoint metadata is checked.**

```python
torch.save({"config": config.to_dict(), "channels": 2, "dim": 128,
            "state_dict": encoder.state_dict()}, "hybrid.pt")
saved = torch.load("hybrid.pt", weights_only=True)
restored = HybridTokenEncoder(HybridConfig.from_dict(saved["config"]),
                             channels=saved["channels"], dim=saved["dim"])
restored.load_state_dict(saved["state_dict"])
```

The frontend's extra state stores the configuration and rejects mismatches
even when parameter shapes happen to match. Source audio/calibration and
dataset identifiers belong in the training manifest as well. This API does
not define a portable on-disk complex-coefficient container yet; keep the
original waveform plus configuration for reanalysis rather than treating
tokens as an archive.

**Measured first experiment.** Reproduce it without downloads:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.hybrid_foundation --steps 100 --seeds 0 1
```

The experiment trains a small complex-mask head on 32 synthetic mixtures and
evaluates 16 independently seeded mixtures. Each is 1,024 samples at 8 kHz,
mono, with overlapping harmonic sources, variable phase/amplitude/envelopes,
and a lower-register target. Both carriers reconstruct input mixtures with
maximum float32 error at most **2.09e-7** in this run. The untouched half-mix
baseline scores **2.51 dB** scale-sensitive source SNR.

| Token inputs / phase ablation | Wavelet carrier mean SNR, 2 seeds | STFT carrier mean SNR, 2 seeds |
| --- | --- | --- |
| Raw-waveform tokens | 8.71 dB | 8.38 dB |
| Complex spectral context | 8.50 dB | 8.24 dB |
| Both, gated fusion | 8.61 dB | 8.49 dB |
| Magnitude context only | 8.54 dB | 8.28 dB |
| Raw + magnitude context | 8.63 dB | 8.48 dB |

The complete configurations, individual runs, runtimes and errors are in
[benchmarks/hybrid_foundation.json](benchmarks/hybrid_foundation.json).
The [validation record](benchmarks/HYBRID_VALIDATION.md) covers 272 passing
library tests, 36 passing editor integration tests, and executed guide examples.
The experiment first ran without phase ablations, then was extended and
rerun; this table and JSON describe the final 20-run matrix.

The result is a working learning pipeline, **not evidence that hybrid input
or phase context is consistently better**. The task has a strong frequency
register cue; magnitude context performs similarly here. The two-seed
differences are too small and task-specific to establish a preferred
representation. Carrier band counts and parameter counts differ across
backends, all branch modules are allocated even when bypassed, and the
experiment is not compute-matched. “Raw-waveform tokens” still feed a
spectral-mask decoder; this is not a pure waveform-domain separator.
Future comparisons need real stereo sources, controlled cancellation,
instrument-instance labels, matched models and held-out source groups.

**Research grounding and next steps.** Waveform/spectral fusion is already
present in [Hybrid Transformer Demucs](https://arxiv.org/abs/2211.08553);
[BS-RoFormer](https://arxiv.org/abs/2309.02612) already uses complex spectra
and phase-aware source estimates. Carrier demodulation, Gaussian frames and
gated fusion are established techniques. This implementation makes no new
mathematical or state-of-the-art claim.

The hypotheses worth testing are whether independent wavelet resolution
controls and signed, physically aligned context improve learning efficiency;
whether an efficient invertible multirate carrier improves quality per unit
memory/compute; and whether local phase detail plus beat/phrase context
improves repeated-source identity and mastering-aware extraction. These
require controlled evidence, not simply sharper pictures.

Next milestones are a bounded multirate/tiled carrier with explicit valid
context, a matched real-audio representation benchmark, source-waveform
dataset contracts, and hierarchical/long-context conditioning. Mastered-mix
restoration should have separate in-mix and dry-source targets and relevant
raw/processed datasets. The editor can later expose source queries and
residuals while retaining the original mix. It currently consumes the legacy
DSP API; this release does not silently switch its edit synthesis behavior.
