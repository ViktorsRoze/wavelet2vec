**Stereo, multiple resolutions, training contracts and feature caches — v0.28.0, 14 September 2026**

**v0.29.0 update:** [Learned context and masked acoustic prediction](SELF_SUPERVISED_MUSIC.md)
now provide an initial self-supervised training path. The source reconstruction
losses below retain their separate supervised contract. The new toy learner
has not demonstrated semantic music quality or beaten its constant baseline.

This release adds composable stereo and multi-resolution token adapters,
source-waveform supervision utilities, and versioned inference caches. They
extend the hybrid/sliding foundation; they are not pretrained music models or
evidence of improved semantic understanding. The original waveform and primary
complex carrier remain available for synthesis.

The [validation record](benchmarks/ADAPTERS_VALIDATION.md) covers 378 passing
library tests, 36 editor compatibility tests and twelve executed guide examples.

**Combine spectral resolutions on one physical timeline.**

```python
from dataclasses import replace
import torch
from wavelet2vec import (
    HybridConfig, MultiResolutionTokenEncoder, StereoTokenEncoder,
    SlidingHybridEncoder, SlidingWindowConfig,
)

primary = HybridConfig(sample_rate=48000, backend="compact_wavelet",
                       n_bands=64, q=8, hop_length=256)
short = replace(primary, backend="stft", n_fft=512)
long = replace(primary, backend="stft", n_fft=4096)
base = MultiResolutionTokenEncoder(
    [primary, short, long], channels=2, dim=128,
    max_total_coefficient_bytes=64 * 1024**2,
)
encoder = StereoTokenEncoder(base)
audio = torch.randn(1, 2, 2400) * 0.1
output = encoder(audio)
print(output.tokens.shape)                 # [1, 10, 128]
print(output.base.resolution_weights.shape) # [1, 10, 4]: raw + 3 spectral views
print(output.stereo.values.shape)           # [1, 10, 64, 7]
torch.testing.assert_close(output.encoding.synthesize(), audio,
                           rtol=2e-5, atol=2e-5)
```

All resolution configs must share sample rate and token hop. Backends, FFT
lengths, wavelet Q, density and padding can differ. Each view has its own
spectral projection; the raw-patch projection is shared. A learned softmax
gate combines the views per token. These weights are debugging/ablation
signals, not explanations of instrument identity or proof of musical attention.
This combiner does not perform temporal self-attention between tokens.

The primary config is always the synthesis carrier, regardless of gate
weights. `retain_carriers=True` on the multi-resolution encoder also returns
secondary carriers in `carriers`; otherwise they are discarded as explicit
outputs after feature extraction. Autograd may still retain tensors required
for backward. The joint byte limit checks the estimated active coefficient
payload, including the primary carrier, before allocating secondary views.
For raw tensor input it runs before primary analysis as well. It does not
bound all activations, stereo moments, optimizer memory or process RAM.

Use `branches="waveform"` or `"spectral"` for broad ablations, and
`resolution_mask=(True, False, True)` to disable individual spectral views.
Disabled views are not analyzed, apart from the primary carrier which is
always retained for synthesis. Spectral-only mode rejects an all-disabled
mask. Disabling a view removes its gate weight exactly. Changing view order,
configuration or dimensions is a checkpoint compatibility change.

**Stereo features have explicit conventions and support metadata.**

`hybrid_stereo_features(encoding, StereoFeatureConfig(...))` works with dense
wavelet, compact-wavelet and complex-STFT carriers. The public name avoids a
collision with the older `wavelet2vec.stereo_features` embedding module.
The default pair is channels `(0,1)`; other pairs can be selected for
multichannel input. No downmix or destructive mid/side conversion is applied.

| Feature in `STEREO_FEATURE_NAMES` | Definition / interpretation |
| --- | --- |
| `cross_real`, `cross_imag` | Real/imaginary parts of mean `L*conj(R)` normalized by the geometric mean channel power |
| `coherence` | Magnitude of that normalized cross-vector, in [0,1] |
| `balance` | `(power_R-power_L)/(power_L+power_R)`, positive toward the right channel |
| `side_fraction` | Side energy relative to Mid+Side energy: 0 for identical channels, 1 for opposite polarity, 0.5 for a single active channel |
| `log_energy` | `log1p((power_L+power_R)/2)` using the local integration interval |
| `support_fraction` | Available real interval divided by requested integration interval |

The return object's `values` shape is `[B,T,K,7]`. It also returns
`energy_valid` `[B,T,K]`, `support_fraction` `[T,K]`, `required_seconds` `[K]`,
global `times_s`, centers and the selected channel pair. `energy_valid` means
both channels exceed the configured power floor. It does not establish that
the phase/coherence estimate is statistically reliable.

Integration is centered on each token and lasts at least the token duration,
`minimum_seconds` (default 20 ms), and `coherence_cycles/bandwidth` (default
1.5 nominal bandwidth cycles). Wavelets use nominal Gaussian sigma as the
bandwidth; STFT uses its window's equivalent noise bandwidth. These choices
are explicit heuristics, not a proof of a fixed number of independent
observations. Residual edge bands do not have ordinary narrow Gaussian support.

Moment interpolation/integration respects each compact band's physical sample
grid and does not expand every band to full audio rate. Dense wavelets use
their padded periodic grid; the last STFT moment sample is extended constantly
to cover a partial final frame. The estimates are not identical to Welch
spectral estimates. Here coherence is an amplitude ratio; SciPy's
[coherence API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html)
reports its squared counterpart. The cross convention here is `L*conj(R)`,
so positive phase means L leads R; SciPy's
[CSD convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html)
uses the conjugate of the first channel instead.

Short clips and boundaries reduce `support_fraction`. It is temporal
availability, not a calibrated confidence probability or a true filter-support
validity mask. Silence yields zero signal features and `energy_valid=False`,
while support metadata still describes the available interval. One silent
channel has undefined cross-phase, but balance/side-energy remain meaningful.
The adapter avoids phase-angle differentiation at zero and has finite tested
silence gradients.

`StereoTokenEncoder` wraps either a local `HybridTokenEncoder` or the new
multi-resolution encoder. It projects the stereo cues from the primary
carrier and learns a gate with the base tokens. `stereo=False` bypasses the
adapter exactly. The `base` result remains accessible for resolution weights
and other diagnostics. More views and stereo cues add cost; no improvement
over simpler inputs has yet been established on a music task.

**The adapters compose with long-audio extraction.**

```python
windows = SlidingWindowConfig.from_seconds(
    2.0, 0.5, sample_rate=48000, hop_length=256)
extractor = SlidingHybridEncoder(encoder, windows).eval()
with torch.no_grad():
    sequence = extractor(audio)
    ablated = extractor(audio, encoder_options={
        "stereo": False, "resolution_mask": (True, True, False)})
```

`encoder_options` also works on reader/file iterators and forwards runtime
adapter options. Reserved audio, branch and origin arguments cannot be
overridden there. Choose context for the longest enabled analysis; the
sliding layer does not infer a universal safe context length from the views.
`SlidingTokenBatch.as_sequence()` exposes a token-only view for per-window
caching or downstream context models. Batch outputs now retain sample-rate
metadata as well as global sample intervals.

**Training uses waveform targets and explicit validity.**

```python
from wavelet2vec import SourceExample, collate_sources, separation_loss

bass = torch.randn(2, 257) * 0.05
drums = torch.randn(2, 257) * 0.05
examples = [SourceExample(
    mixture=bass+drums, sources={"bass": bass, "drums": drums},
    sample_rate=48000, additive=True)]
batch = collate_sources(examples, source_order=("bass", "drums"))
# Replace this illustrative tensor with the source decoder's predictions.
prediction = (batch.sources.detach()*0.9).requires_grad_()
losses = separation_loss(prediction, batch, fft_sizes=(128, 512), mixture_weight=0.1)
losses["loss"].backward()
```

`SourceExample` contains `[C,N]` audio and a mapping from source names to
same-shaped targets. `collate_sources` requires a fixed explicit source
order, common sample rate, channel count, precision and device. It pads to
the longest example and returns `SourceBatch` with `[B,S,C,Nmax]` targets,
true lengths, source-valid flags and additive flags. `batch.to(device)`
preserves precision.

An omitted source means **unannotated**, not silent; its loss is ignored.
An explicit zero waveform means a known absent source and is supervised.
If `additive=True`, all source targets must be present and must sum to the
mixture within the declared implementation tolerance (rtol 1e-5, atol 1e-6).
Dry-restoration or non-additive production targets should leave it false.

`separation_loss` combines masked waveform L1 and multi-resolution complex
STFT L1, with optional mixture consistency. It excludes padded samples and
unannotated sources, and sanitizes ignored prediction positions before
arithmetic so unused NaNs do not contaminate the result. Invalid predictions
inside supervised samples still produce an invalid loss; they are not hidden.
Each spectral transform uses the example's actual length, with amplitude
scaling and correct DC/Nyquist weights. Contributions are weighted by valid
source sample count. Inverting source polarity incurs a spectral error even
when magnitudes alone would agree.

Mixture consistency requires complete additive targets. It is rejected for
dry or incomplete supervision because forcing restored sources to sum directly
to a mastered mix is generally the wrong objective. The loss is a reference
training utility, not a perceptual quality score or an optimized distributed
trainer; its spectral path loops over examples. It also is **not** a masked
self-supervised representation-learning objective.

**Cache inference features with provenance.**

```python
import hashlib
from wavelet2vec import FeatureCache, FeatureCacheKey

# For real files, use source_fingerprint(path) once per source instead.
source_bytes = (str(audio.dtype)+str(tuple(audio.shape))).encode()
source_bytes += audio.detach().cpu().contiguous().numpy().tobytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
processing = {"branches": "both", "encoder_options": {},
              "range": [0, audio.shape[-1]], "pooling": None}
key = FeatureCacheKey.for_model(source_hash, extractor,
    sample_rate=48000, channels=2, processing=processing)
cache = FeatureCache("feature-cache")
cache.save(key, sequence)
loaded = cache.load(key)
torch.testing.assert_close(loaded.tokens, sequence.tokens)
```

The builder requires eval mode. Model identity covers weights, buffers, JSON
extra state, module types/training modes, available wavelet2vec Python source
and torch/NumPy/SciPy versions. The key also records library version, source
hash, sample rate, channels, global origin and canonical runtime parameters.
Recompute identity when state/code changes. Hashing a large model is not free;
`model_fingerprint` can be computed once for a fixed extraction run.

The caller must include runtime choices not in model state: source ranges,
branch/resolution/stereo ablations, pooling and any external custom-module
code revision. The cache cannot infer omitted parameters. CPU/GPU-specific
reproducibility requirements should likewise be recorded explicitly.

Entries are single NumPy archives loaded with `allow_pickle=False`. Metadata,
shapes, precision, finite values, interval continuity and midpoint times are
checked. Corrupt/mismatched entries are cache misses. Writes use a temporary
file and same-directory atomic replacement; this is not a distributed cache
or transactional multi-entry dataset. Names are content hashes rather than
caller-supplied paths. The existing editor embedding cache is not reused.

Only detached inference sequences can be saved. Cached features can train a
later context head, but cannot propagate gradients back into the original
encoder. For window-wise extraction, call `batch.as_sequence()` and key each
entry by its global origin and analysis/core range. Pooling now computes
midpoints in float64 even for very long intervals, preserving half-sample
coordinates beyond float32's integer precision.

The remaining representation-learning gaps are documented in
[MUSIC_REPRESENTATION_GAPS.md](MUSIC_REPRESENTATION_GAPS.md). These utilities
make experiments easier to build; a learned music-context backbone and an
appropriate self-supervised objective remain major next steps.
