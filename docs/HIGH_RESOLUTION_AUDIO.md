# High-resolution audio contract

21 September 2026: **192,000 samples/second, 32-bit floating-point audio is a
required target** for wavelet2vec, including stereo. Lower native rates remain
supported. Choosing a different rate or precision must be explicit; the
hybrid/corpus/source-synthesis paths do not silently resample or downmix.

32-bit float means IEEE floating-point samples, not 32-bit integer PCM.
Use `torch.float32` for native model input/output and a WAV file with
`subtype="FLOAT"` for export. Float64 remains available for numerical checks
and callers that deliberately choose it. File precision does not prescribe
the precision of every future model operation: any mixed-precision training
recipe must separately demonstrate stability and output fidelity.

```python
import numpy as np
from wavelet2vec import HybridConfig, SynthesisConfig, HybridSeparator
from wavelet2vec.audio_io import save_audio, load_audio

sample_rate = 192000
# 512 samples has the same physical duration as 128 samples at 48 kHz.
frontend = HybridConfig(
    sample_rate=sample_rate, backend="compact_wavelet",
    n_bands=48, q=2, hop_length=512, padding_samples=2048,
)
model = HybridSeparator(frontend, SynthesisConfig(dim=32))  # float32 parameters

# Explicit floating-point storage. This is an I/O example, not trained audio.
audio = np.zeros((2, 4097), dtype=np.float32)
audio[0, 0] = 1.5  # float WAV preserves headroom; no clipping/normalization
save_audio("high_resolution_example.wav", audio, sample_rate, subtype="FLOAT")
restored, restored_rate = load_audio("high_resolution_example.wav")
assert restored_rate == sample_rate
np.testing.assert_array_equal(restored, audio)
```

For this contract, use a floating-point-capable container explicitly. The
legacy `save_audio` helper can fall back to integer storage for formats
without float support; that fallback does not satisfy 32-bit float export.
Its explicit `FLOAT` option is used in the example and integration tests.
The helper converts float64 input to float32: preserve original PCM separately
if a project's archival requirements exceed this format contract.

Keep model durations physical when moving across rates. Sample counts for
token hops, STFT windows, padding and rendering context must be configured
accordingly. The example above preserves hop/padding duration relative to
128/512 samples at 48 kHz; it is not a trained quality preset. Increasing the
rate changes bandwidth, workload and potentially checkpoint dimensions.

## Qualification recorded so far

The expanded source-synthesis suite has **51 passing tests**. Added checks
cover 192 kHz identity through dense wavelets, compact wavelets and STFT,
short/partial lengths, stereo overlap-add, and a weak near-Nyquist signal.
For every backend, a 192 kHz float32 WAV fixture passes:

- Exact float32 file round-trip, including values above unity and small values.
- Verified manifest/crop loading with native rate, channels and sample count.
- Float32 overlap-add identity within numerical tolerance.
- Two small supervised updates with finite gradients into the encoder.
- Exact float32 storage/reload of the resulting source waveform.

Small CPU/CUDA float32 comparisons also pass at 192 kHz for all three
backends, with maximum waveform differences below `1e-7` and finite gradients.
[Accelerator results](benchmarks/source_synthesis_accelerator.json) record
the device and implementation identity. The documentation I/O example was
executed successfully in a temporary directory.

Exact file storage does not mean transformed waveforms are bit-exact:
FFT/model arithmetic is approximate. These short fixtures do not establish
full-track resource bounds, perceptual superiority, learned native-192 kHz
separation, or time/pitch transformation quality. The development plan must
qualify those capabilities at 192 kHz/float32 as they are implemented.
