# High-resolution audio contract

**v0.34.0:** [Bounded rendering](BOUNDED_RENDERING.md) now streams native
192 kHz float32 file inputs to completed stem files, with explicit FLOAT or
DOUBLE export, no integer fallback, and RF64 selection for large outputs.

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

## DXD: 352.8 kHz integer PCM

DXD (Digital eXtreme Definition) is 352,800 samples/second (8 x 44.1 kHz)
PCM stored as **24- or 32-bit signed integers**, normally in WAV/RF64, and
sometimes in FLAC (24-bit only) or AIFF. `wavelet2vec.dxd` handles these files
without losing any bits. Model computation still uses the float contract above.

```python
from wavelet2vec import load_dxd, save_dxd, dxd_info, dxd_hybrid_config, HybridSeparator, SynthesisConfig

info = dxd_info("master.wav")                 # .is_dxd, .bits, .frames, .duration_s
audio, info = load_dxd("master.wav")          # float64 [channels, samples], exact
save_dxd("copy.wav", audio, bits=info.bits)   # identical integer samples

model = HybridSeparator(dxd_hybrid_config(), SynthesisConfig(dim=32))
```

Rules:

- **Exact load.** Samples are `integer / 2**(bits-1)`, computed from the
  stored integers. float64 is exact for both depths. float32 is exact for
  24-bit. For 32-bit it drops up to 8 bits, so `load_dxd(..., dtype=np.float32)`
  raises unless `allow_precision_loss=True`. The generic `load_audio` always
  returns float32. Use `load_dxd` when 32-bit DXD must stay exact.
- **Explicit quantization.** `save_dxd` rounds to the nearest integer code.
  Full scale is `2**(bits-1)`, so +1.0 is *not* representable (the largest
  code is `1 - 2**-(bits-1)`). Out-of-range samples raise instead of being
  clipped silently. By default, libsndfile clips on float-to-integer writes and
  truncates on 24-bit integer writes, and this module avoids both.
  `clip=True` and seeded `dither="tpdf"` are opt-in.
- **No implicit resampling.** `save_dxd` writes only 352.8 kHz. Other rates must
  be converted deliberately with `audio_io.resample`, for example
  44.1/88.2/176.4 kHz to DXD with the integer ratios 8/4/2.
- **Containers.** The container follows the file extension. A `.wav` larger
  than the conservative RIFF limit switches to RF64 automatically. FLAC
  rejects `bits=32`.
- **Frontend preset.** `dxd_hybrid_config()` uses `hop_length=1024` and
  `padding_samples=4096`: exactly 128/512 samples at 44.1 kHz, scaled by 8.
  It sets the resolution only and is not a trained preset. Wavelet2Vec
  embeddings downsample 352.8 kHz by exactly 16 to reach their 22,050 Hz
  analysis rate.
- **Rendering.** `separate_file` accepts DXD inputs directly because it
  refuses only rate/channel mismatches. For a float32 model, a PCM_32 input is
  read as float32. Stem export stays FLOAT/DOUBLE by design. To deliver
  integer DXD stems, load the rendered float stems and pass them through
  `save_dxd`, which fails if any stem exceeds full scale.

`tests/test_dxd.py` covers bit-exact WAV/FLAC/AIFF round trips at both depths,
rounding, refusal to clip, seeded dither, header validation, RF64 selection,
embeddings from a DXD file, and a native-rate separator rendering a DXD file.
The source-synthesis identity tests also run at 352,800 Hz.

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
