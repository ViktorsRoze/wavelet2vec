# wavelet2vec — API Reference

v0.37.0 adds `create_percussion_study` and `PercussionStudy`, with the
`wavelet2vec-study` CLI. [REUSABLE_STUDIES.md](REUSABLE_STUDIES.md) covers
project discovery, cache/preparation reuse, resume, evaluation and profiling.

v0.36.0 adds `detect_percussive_events`, `compare_percussive_events`,
`analyze_drum_track` and optional `source_weights` in `separation_loss` /
`SeparationRunConfig`. [DRUM_ANALYSIS.md](DRUM_ANALYSIS.md) defines their
measurement contracts; [real-data findings](DRUM_DATASET_FINDINGS.md) include
the first trained drum-substem comparison.

v0.35.0 adds `SeparatorBundle`, `save_separator_bundle`,
`inspect_separator_bundle`, `load_separator_bundle`, and
`SeparationTrainingRun.export_bundle`. [INFERENCE_BUNDLES.md](INFERENCE_BUNDLES.md)
defines portable architecture/weight artifacts and the inspect/render CLI.

v0.34.0 adds `SourceRenderConfig`, `iter_separate_reader`, `separate_file`,
`FileSeparationResult` and `SeparationCancelled`.
[BOUNDED_RENDERING.md](BOUNDED_RENDERING.md) defines rolling audio buffers,
native float WAV/RF64 export, cancellation and staged publication.

v0.33.0 adds `MultitrackRecord`, `MultitrackManifest`, `MultitrackWindow`,
`MultitrackDataset`, `collate_multitracks`, `SeparationRunConfig`,
`SeparationTrainingRun` and `evaluate_separation`.
[SUPERVISED_MULTITRACK.md](SUPERVISED_MULTITRACK.md) defines aligned source
verification, native crops, sample-weighted optimization and strict resume.

v0.32.0 adds `SynthesisConfig`, `SourceSynthesis`, `HybridSourceDecoder`,
`HybridSeparator`, `project_source_coefficients` and `separate_audio`.
[SOURCE_SYNTHESIS.md](SOURCE_SYNTHESIS.md) defines supervised source rendering,
physical coefficient grids, residual synthesis and offline window inference.

The phase-preserving model API added in v0.25.0 is documented in
[HYBRID_ENCODING.md](HYBRID_ENCODING.md): `HybridConfig`,
`HybridAudioFrontend`, `HybridEncoding`, `HybridTokenEncoder` and
`HybridLatents`. It accepts explicit `[batch, channels, samples]` torch
tensors without resampling or downmixing and retains an invertible complex
carrier alongside the original waveform. Its learned tokens are lossy.

v0.26.0 adds `backend="compact_wavelet"`, `CompactBand` metadata,
`HybridEncoding.band_times_s(index)` and `coefficient_nbytes`.
[COMPACT_WAVELETS.md](COMPACT_WAVELETS.md) defines the ragged band tuple,
inverse and memory/performance tradeoffs.

The retrieval and legacy image APIs below have separate contracts: pooled
features are not substitutes for full complex synthesis coefficients.

For long recordings, v0.27.0 adds `SlidingHybridEncoder`,
`SlidingWindowConfig`, `SlidingTokenBatch` and `HybridSequence`.
[LONG_AUDIO.md](LONG_AUDIO.md) documents file/reader extraction, global
coordinates, carrier retention and coarse summaries. The local frontend
accepts `sample_origin` without changing analysis coefficients or synthesis.

v0.28.0's composable model utilities are documented in
[ADAPTERS_AND_TRAINING.md](ADAPTERS_AND_TRAINING.md):
`MultiResolutionTokenEncoder`, `StereoTokenEncoder`, `hybrid_stereo_features`,
`SourceExample` / `collate_sources`, `separation_loss`, and
`FeatureCache` / `FeatureCacheKey`. They retain explicit timing, supervision
and cache contracts; they are not a complete self-supervised music model.

v0.29.0 adds `MusicContextEncoder`, `MaskedMusicModel`, `AudioBatch` /
`collate_audio`, `MaskedPredictionConfig`, `sample_span_mask`,
`MusicPretrainer`, `PretrainingConfig` and `LengthBucketSampler`.
[SELF_SUPERVISED_MUSIC.md](SELF_SUPERVISED_MUSIC.md) documents learned
local/global context, isolated student/teacher paths and resumable training.
The fixed acoustic objective is an initial research task, not a validated
musical semantic representation.

v0.30.0 adds `FixedTargetStatistics` and opt-in standardized masked targets.
[TARGET_CALIBRATION.md](TARGET_CALIBRATION.md) documents fitting, prediction
units, checkpoint migration and the limited synthetic evidence.

v0.31.0 adds `AudioRecord`, `AudioManifest`, `AudioCropConfig`,
`AudioWindowDataset`, `collate_windows`, `TrainingTargetMean`,
`fit_training_target_mean`, `evaluate_masked_corpus`, and
`CorpusPretrainingRun`. [REAL_AUDIO_PRETRAINING.md](REAL_AUDIO_PRETRAINING.md)
documents verified crops, split guards, resume and diagnostic baselines.

## `Wavelet2Vec`

```python
from wavelet2vec import Wavelet2Vec, Wavelet2VecConfig

embedder = Wavelet2Vec(config: Wavelet2VecConfig | None = None)
```

| Member | Description |
| --- | --- |
| `embed(waveform, sample_rate) -> np.ndarray` | Embeds `[channels, samples]` (or 1D mono) float array — torch tensors are accepted too — at any sample rate. Returns a float32 vector of length `dim`. |
| `embed_file(path) -> np.ndarray` | Loads (WAV/FLAC/AIFF/OGG via soundfile) and embeds a file. |
| `embed_folder(folder) -> dict[str, np.ndarray]` | Recursively embeds every audio file; keys are paths relative to the folder. |
| `embed_components(waveform, sample_rate) -> dict[str, np.ndarray]` | Raw, unnormalized per-section feature vectors — useful for inspecting *why* two sounds differ. |
| `dim: int` | Total embedding dimension (543 with defaults). |
| `sections: dict[str, slice]` | Slice of each section inside the embedding vector. |
| `conv_encoder: WaveletConvEncoder \| None` | The convolutional branch (None when `include_conv=False`). |

Silence (all-zero input) returns an all-zero embedding.

## `Wavelet2VecConfig`

All fields with defaults:

```python
Wavelet2VecConfig(
    analysis_sample_rate=22050,  # inputs are resampled to this rate
    n_bands=64,                  # wavelet bands, geomspaced f_min..f_max
    f_min=27.5,                  # A0
    f_max=None,                  # default: 0.45 * analysis_sample_rate
    bandwidth_scale=1.0,         # >1 = wider (smoother) filters
    envelope_points=32,          # transient envelope shape resolution
    envelope_rate=256,           # Hz, envelope downsampling for modulation FFT
    n_band_groups=8,             # wavelet-band groups for modulation features
    n_modulation_bands=8,        # log-spaced modulation bands
    mod_f_min=0.5, mod_f_max=64.0,
    n_phase_groups=8,            # band groups for instantaneous-frequency stats
    n_phase_harmonics=8,         # harmonics in the phase signature
    n_stereo_groups=8,           # band groups for the stereo image section
    include_conv=True,           # convolutional section on/off
    conv_bands=32, conv_kernel_size=1023, conv_hop=64,
    section_weights={            # per-section weight applied after L2 norm
        "spectral": 1.0, "modulation": 0.75, "transient": 1.0,
        "harmonic": 0.75, "phase": 1.0, "stereo": 0.5, "conv": 0.5,
    },
)
```

`section_dims()` returns the dimensionality of each section for the current
settings; `resolved_f_max()` returns the effective upper analysis frequency.

## Similarity helpers

```python
from wavelet2vec import cosine_similarity, pairwise_similarity

cosine_similarity(a, b) -> float                    # 0.0 for zero vectors
pairwise_similarity(embeddings) -> (names, matrix)  # dict -> (list[str], np.ndarray)
```

## `WaveletConvEncoder`

```python
from wavelet2vec import WaveletConvEncoder

encoder = WaveletConvEncoder(
    sample_rate=22050, n_bands=32, f_min=27.5, f_max=None,
    kernel_size=1023, hop=64, temporal_kernel_size=31,
    trainable=False,
)
```

A `torch.nn.Module`. `forward` maps `[batch, 1, samples]` (at
`encoder.sample_rate`) to `[batch, output_dim]`; `encode(waveform, sample_rate)`
and `encode_mono(mono)` are no-grad conveniences for numpy input.
`output_dim == 2 * n_bands * 4`. Parameters are frozen unless
`trainable=True`.

## Style transfer and blending

```python
from wavelet2vec import blend, style_transfer

waveform, sample_rate = blend(
    sounds,          # list of [channels, samples] arrays (or 1D mono)
    sample_rates,    # one rate per sound
    weights,         # one non-negative weight per sound (normalized internally)
    carrier=None,            # index of the fine-structure reference; default: largest weight
    transfer_spectrum=True,  # EQ-match wavelet band energies
    transfer_envelope=True,  # morph the temporal envelope
    max_gain_db=24.0,        # per-band / per-instant gain limit
)

waveform = style_transfer(source, source_sr, target, target_sr, amount, **kwargs)
```

`blend` returns the morphed carrier (same channel count and sample rate as the
carrier). Weights `[1, 0, ...]` with `carrier=0` return the carrier unchanged;
increasing a reference's weight moves the result monotonically toward it in
embedding space. `style_transfer` is the two-sound special case.

## Characters

```python
from wavelet2vec import (
    SoundCharacter, extract_character, average_characters,
    character_from_files, character_from_folder, apply_character,
)

char = extract_character(waveform, sample_rate)      # one sound
char = character_from_folder("guitars/")             # group average (recursive)
char = average_characters([c1, c2], weights=[2, 1])  # weighted combination
char.save("guitar.npz"); char = SoundCharacter.load("guitar.npz")

morphed = apply_character(
    source, source_sr, char,
    amount=0.6,              # scalar in [0, 1], or a per-sample array
    transfer_spectrum=True, transfer_envelope=True, max_gain_db=24.0,
)
```

A `SoundCharacter` holds the level-normalized wavelet band energy
distribution (log domain) and the time-normalized envelope shape on a fixed
analysis grid, so characters extracted from any material at any sample rate
are interchangeable. When `amount` is an array (any length; resampled to the
source), the morph is time-varying: the full morph is computed once and
crossfaded with the source — phase-coherent, so no comb artifacts.

## ADSR

```python
from wavelet2vec import ADSR, apply_adsr_volume, apply_adsr_character

adsr = ADSR(attack=0.4, decay=0.1, sustain=0.8, release=0.2)  # seconds, level in [0,1]
env = adsr.envelope(n_samples, sample_rate, gate=None)  # gate: note-off seconds
faster = adsr.replace(attack=0.05)

shaped  = apply_adsr_volume(waveform, sr, adsr)
morphed = apply_adsr_character(source, sr, character, adsr, max_amount=1.0)
```

`apply_adsr_character` drives the character morph amount with the ADSR
envelope (scaled by `max_amount`): slow attack = the sound grows into the
character; release = it falls back.

## Sequencing and automation

```python
from wavelet2vec import AutomationCurve, render_sequence

curve = AutomationCurve.from_file("clip.flp")          # .flp/.fst (pyflp), .csv, .json
curve = AutomationCurve.from_points([(0.0, 0.0), (4.0, 1.0)])
curve.value_at(seconds); curve.value_at_fraction(0.5)  # normalized position

waveform, sr = render_sequence(
    snippets, sample_rates,
    character=char,                  # optional character to develop toward
    character_adsr=ADSR(...),        # per-snippet time-varying morph (optional)
    volume_adsr=ADSR(...),           # per-snippet loudness shaping (optional)
    automation=curve,                # sampled at each snippet's position
    modulate="amount",               # amount | attack | decay | sustain | release
    max_amount=1.0, gap_seconds=0.0, output_sr=None,
)
```

Automation-value mapping: `amount` → morph depth = `max_amount * value`;
`attack`/`decay`/`release` → stage time scaled by `1 - value` (higher = faster);
`sustain` → level scaled by `value`. CSV columns: `time_seconds` (or `seconds`
or `beat` with `tempo_bpm`) and `value`; JSON mirrors the same fields.

## Performing a DAW render

```python
from wavelet2vec import perform, slice_grid, slice_cues, slice_markers, slice_onsets, read_cue_points

starts = slice_grid(n_samples, sr, bpm=140, division=8, offset_beats=0.0)
starts = slice_grid(n_samples, sr, spacing_seconds=0.25)   # fixed note distance
starts = slice_cues("stem.wav", n_samples)                 # RIFF cue markers
starts = slice_markers("markers.csv", sr, n_samples)       # time_seconds CSV or plain text
starts = slice_onsets(waveform, sr, sensitivity=0.5)       # detection fallback

result = perform(
    waveform, sr, character, starts,
    automation=curve, character_adsr=ADSR(...), volume_adsr=None,
    modulate="amount", max_amount=1.0,
    boundary_fade_seconds=0.005,
)
```

`perform` guarantees `result.shape == input.shape` at the input's native
sample rate — the round-trip property that lets the output re-import
sample-aligned. Slice starts are normalized (sorted, deduplicated, sample 0
always included); each note's ADSR is gated to its slice length.

`save_audio(path, waveform, sample_rate, subtype=None)` writes 32-bit float
WAV/AIFF by default (FLAC falls back to 24-bit); pass `subtype="PCM_16"` etc.
to override.

### DXD (352.8 kHz, 24/32-bit PCM)

```python
from wavelet2vec import load_dxd, save_dxd, dxd_info, dxd_hybrid_config

info = dxd_info("take.wav")            # DXDInfo(sample_rate, channels, frames, subtype, format)
audio, info = load_dxd("take.wav")     # [channels, samples] float64, exact integer/2**(bits-1)
save_dxd("out.wav", audio, bits=24)    # or bits=32; dither="tpdf", seed=..., clip=True are opt-in
frontend = dxd_hybrid_config()         # HybridConfig at 352800 Hz, hop 1024, padding 4096
```

`load_dxd(path, dtype=np.float64, allow_precision_loss=False, strict=True)`:
float32 is exact for PCM_24; a float32 load of PCM_32 raises unless
`allow_precision_loss=True`. `strict=False` accepts other integer-PCM files.
`save_dxd` rounds to nearest, refuses samples outside the integer range
(+1.0 included) unless `clip=True`, picks the container from the extension
(large `.wav` becomes RF64; FLAC is 24-bit only) and verifies the header.
`quantize_pcm(waveform, bits, dither=None, seed=None, clip=False)` exposes the
same quantizer, returning left-justified int32 codes.

## Lossless stereo wavelet image

```python
from wavelet2vec.wavelet_image import WaveletImageSpec, encode, decode, save_image, load_image, magnitude_view

spec = WaveletImageSpec(sample_rate=192000, n_frames=120)   # 149 semitone rows x 10 ms frames
image = encode(audio, spec)        # audio [B, 2 (L, R), 230400] -> exact complex cells, mid/side
audio_back = decode(image)         # exact inverse: ~305 dB in float64, ~130 dB in float32
save_image(image, "note_img/")     # float32 TIFF per octave group + JSON; load_image() inverts
view = magnitude_view(image)       # [B, 2, 149, 120] cell levels in dB, for display
```

`image.groups[c]` holds the cells with `c` complex samples per 10 ms:
`[B, 2, rows, frames, c]`. There is one group per upper octave, and `c`
doubles per octave. See [WAVELET_IMAGE.md](WAVELET_IMAGE.md).

Pictures at any resolution, from every sample (for display only):

```python
from wavelet2vec.wavelet_image import detail_view, stereo_view

db = detail_view(image, mode="sharp", channel="mid",        # or exact / cells; mid, side, left, right
                 time_range=(0.0, 0.3), midi_range=(40, 100), width=1600, height=700)   # [700, 1600] dB
level_db, balance, correlation = stereo_view(image, mode="sharp", width=1600, height=700)
```

- `sharp`: reassigned to each sample's instantaneous frequency and group
  delay, in proportion to how well neighbouring bands agree.
- `exact`: each band at its own time resolution.
- `cells`: the backbone's 10 ms × semitone grid.

Row 0 is the lowest pitch.

## Tools: vocal detector

`wavelet2vec.tools` holds analysis tools. `detect_vocal_parts` labels a
monophonic voice every 10 ms. It is algorithmic: no training, only signal
measurements and published formant averages.

```python
from wavelet2vec.tools import detect_vocal_parts

track = detect_vocal_parts(audio, sample_rate, scale=None)  # [samples] or [channels, samples]
track.part            # silence / plosive / sibilant / vowel / voiced / breath / transition / other
track.vowel           # on vowel frames: i (ee), e (ay), eh, a (ah), o (oh), u (oo)
track.sibilant_type   # on sibilant frames: "s" or "sh"
track.sibilance       # 0..1 share of energy above 2.5 kHz, every frame
track.segments()      # merged runs: [{"start_s", "end_s", "part", "detail"}, ...]
track.summarize(start_s, end_s)  # majority vowel, start/end vowel, parts, sibilant frames
```

How each part is decided:

| Part | Rule |
|---|---|
| `silence` | More than `floor_db` (50) below the loudest frame, or room background more than `background_db` (30) below it |
| `plosive` | A jump of ≥15 dB out of near-silence, with a flat (noise-like) spectrum |
| `sibilant` | Most energy above 2.5 kHz, and that band noise-like; "s" if centred above 4.8 kHz, else "sh" |
| `vowel` | Harmonic frame, judged against formant templates |
| `voiced` | Harmonic, but too few harmonics to judge a vowel |
| `breath` | Other noise (h, f, aspiration) |
| `transition` | Unclear frames within 50 ms of a vowel: note onsets and endings |

Vowels are judged from the harmonic spectral envelope, i.e. the levels of
the harmonics k·f0. It is fitted against templates built from published
formant averages (Peterson & Barney 1952, Hillenbrand et al. 1995). This
keeps working at sung pitches, where LPC formant picking fails.

`scale` is the vocal-tract length factor: 1.0 adult male, about 1.15–1.2
adult female. Estimate it once per speaker and pass it to every call. A
single-vowel recording can trade scale against vowel.

**Measured accuracy:**
- **Synthetic vowels** with shifted formants: 100 % up to 330 Hz f0, 85 % at
  440–660 Hz.
- **Shevannai** sung syllables: sibilants found in 123 of 126 notes that
  contain them, with the right s/sh type. False alarms in 4 of 175 notes
  that don't.

Neighbouring vowels are confused first ("e"/"i", "o"/"u", "a"/"aw"),
especially at high pitch and when singers darken vowels.

## Invertible frame + GPU backend

```python
from wavelet2vec import analyze, synthesize          # numpy, exact ~300 dB
from wavelet2vec import analyze_torch, synthesize_torch, analyze_batch

coeffs, spec = analyze(audio, sr, n_bands=128)        # [n_bands, N] complex
audio = synthesize(coeffs, spec)                      # exact inverse

# Torch backend (optional dep): device="cpu"|"cuda", precision double/single.
# Same FrameSpec, so the backends interchange freely.
coeffs, spec = analyze_torch(audio, sr, device="cuda", precision="single")
batch, spec = analyze_batch(fragments, sr)            # [B, N] -> [B, n_bands, N]
```

`precision="double"` matches numpy to machine precision; `"single"` (complex64)
round-trips ~131 dB and is the fast GPU path. Batched short-fragment analysis
is ~9× the numpy loop on CPU alone.

## Rhythm & groove

```python
from wavelet2vec import (analyze_rhythm, estimate_tempo, apply_groove,
                         swing_offsets, groove_offsets, rhythm_embedding)

bpm, confidence = estimate_tempo(loop, sr)            # no BPM needed
profile = analyze_rhythm(loop, sr, bpm=bpm)           # swing, microtiming, pattern
swung = apply_groove(loop, sr, bpm, swing_offsets(2/3, bpm))   # straight -> triplet
grooved = apply_groove(target, sr, bpm, groove_offsets(source_profile))
vec = rhythm_embedding(loop, sr, bpm=bpm)             # 89-dim, tempo-normalized
```

## Harmony, melody, structure

```python
from wavelet2vec import (track_f0, segment_notes, chromagram, estimate_key,
                         track_chords, segment_structure, find_loop_points,
                         render_loop)

track = track_f0(audio, sr); notes = segment_notes(track)      # MIDI note events
key = estimate_key(loop, sr)                          # KeyEstimate(name, confidence)
chords = track_chords(loop, sr)                       # [(start_s, end_s, "Am"), ...]
sections = segment_structure(clip, sr)                # boundaries + sections
loop = find_loop_points(audio, sr, min_loop_s=1.9)    # seamless loop; render_loop() to play
```

## Spatial analysis & design

```python
from wavelet2vec import (stereo_phase_spectrum, spatial_matrix,
                         extract_spatial_character, apply_spatial_character,
                         design_width)

phase = stereo_phase_spectrum(stereo, sr)             # per-band IPD/coherence/balance over time
character = extract_spatial_character(reference, sr)  # width-vs-frequency profile
wide = apply_spatial_character(source, sr, character) # match a reference's width; mono untouched
widened = design_width(mono_or_stereo, sr, 0.3, coherence_floor=0.2)  # phase-safe widener
```

## Decomposition & spectral editing

```python
from wavelet2vec import hpss, transient_split, shape_transients
from wavelet2vec import (attenuate_region, isolate_regions, process_regions,
                         phase_rotate_regions, phase_scramble_regions)

harmonic, percussive = hpss(audio, sr)                # exact sum == input
edited = attenuate_region(audio, sr, 0.3, 0.6, 1400, 2800, gain_db=-80)   # T-F erase
# Region-limited effects (inside processed, outside preserved by linearity):
regions = [(0.3, 0.6, 1400.0, 2800.0)]
ducked = process_regions(audio, sr, regions, lambda x: x * 0.25)
flipped = phase_rotate_regions(audio, sr, regions, 180.0)
smeared = phase_scramble_regions(audio, sr, regions)
```

## Datasets (supervision ladders)

```python
from wavelet2vec.datasets import (filter_ladder, sine_distortion_ladder,
                                  noise_mix_ladder, attack_ladder, axis_labels,
                                  lowpass_pair, render_ladder_dataset)
```

Graded, monotone-verified ladders that supervise each imagination-axis knob
directly (brightness, distortion, harmonicity, attack). `axis_labels` measures
brightness/harmonicity/attack with the same features the models condition on.

## Feature modules

Every embedding feature is importable directly for custom pipelines:

| Module | Functions |
| --- | --- |
| `wavelet2vec.filterbank` | `log_spaced_frequencies`, `morlet_scalogram` (set `return_complex=True` for analytic band signals), `analytic_band_signals`, `analytic_envelope`, `constant_q_sigmas` |
| `wavelet2vec.wavelet_features` | `spectral_features`, `modulation_features` |
| `wavelet2vec.transient_features` | `transient_features` |
| `wavelet2vec.harmonic_features` | `harmonic_features`, `autocorrelation_pitch` |
| `wavelet2vec.phase_features` | `phase_features`, `instantaneous_frequency_features`, `onset_phase_coherence`, `harmonic_phase_signature` |
| `wavelet2vec.stereo_features` | `stereo_features`, `neutral_stereo_features` |
| `wavelet2vec.rhythm` | `analyze_rhythm`, `estimate_tempo`, `apply_groove`, `swing_offsets`, `groove_offsets`, `rhythm_embedding`, `profile_embedding` |
| `wavelet2vec.harmony` | `track_f0`, `segment_notes`, `chromagram`, `estimate_key`, `track_chords` |
| `wavelet2vec.structure` | `segment_structure`, `find_loop_points`, `render_loop` |
| `wavelet2vec.decompose` | `hpss`, `transient_split`, `shape_transients` |
| `wavelet2vec.spatial` | `stereo_phase_spectrum`, `spatial_matrix`, `extract_spatial_character`, `apply_spatial_character`, `design_width` |
| `wavelet2vec.spectral_edit` | `attenuate_region`, `region_mask`, `isolate_regions`, `process_regions`, `phase_rotate_regions`, `phase_scramble_regions` |
| `wavelet2vec.torch_frame` | `analyze_torch`, `synthesize_torch`, `analyze_batch` |
| `wavelet2vec.style_transfer` | `blend`, `style_transfer` |
| `wavelet2vec.character` | `SoundCharacter`, `extract_character`, `average_characters`, `character_from_files`, `character_from_folder`, `apply_character` |
| `wavelet2vec.adsr` | `ADSR`, `apply_adsr_volume`, `apply_adsr_character` |
| `wavelet2vec.sequence` | `AutomationCurve`, `render_sequence` |
| `wavelet2vec.slicing` | `slice_grid`, `slice_markers`, `slice_cues`, `slice_onsets`, `read_cue_points` |
| `wavelet2vec.perform` | `perform` |
| `wavelet2vec.audio_io` | `load_audio`, `save_audio`, `resample`, `list_audio_files` |
| `wavelet2vec.dxd` | `DXD_SAMPLE_RATE`, `DXDInfo`, `dxd_info`, `load_dxd`, `save_dxd`, `quantize_pcm`, `dxd_hybrid_config` |

## CLI

```text
wavelet2vec embed --input PATH [--output embeddings.npz] [--pairwise matrix.csv]
                  [--query FILE [--top-k 5]]
                  [--analysis-sample-rate 22050] [--n-bands 64] [--no-conv]

wavelet2vec transfer --source FILE --target FILE --amount 0..1 --output FILE
                     [--no-spectrum] [--no-envelope] [--max-gain-db 24]

wavelet2vec blend --inputs FILE FILE [FILE ...] --weights W W [W ...] --output FILE
                  [--carrier INDEX] [--no-spectrum] [--no-envelope] [--max-gain-db 24]

wavelet2vec character --inputs PATH [PATH ...] --output character.npz

wavelet2vec sequence --inputs FILE [FILE ...] --output FILE
                     [--character PATH] [--adsr A D S R] [--volume-adsr A D S R]
                     [--automation clip.flp|.fst|.csv|.json]
                     [--modulate amount|attack|decay|sustain|release]
                     [--max-amount 1.0] [--gap 0.0] [--max-gain-db 24]

wavelet2vec perform --input FILE --output FILE [--character PATH]
                    --slice grid|cues|markers|onsets
                    [--bpm BPM --division 8 [--offset-beats 0] | --spacing SEC]
                    [--markers FILE] [--sensitivity 0.5]
                    [--automation CLIP] [--modulate ...] [--adsr A D S R]
                    [--volume-adsr A D S R] [--max-amount 1.0]
                    [--boundary-fade 0.005] [--max-gain-db 24]
```

- `embed --input`: audio file or folder (recursive); `--output` is an `.npz`
  mapping relative file paths to embedding vectors; `--pairwise` writes the
  cosine similarity matrix as CSV; `--query` prints the `--top-k` most similar
  library sounds.
- `transfer`: makes the source sound more similar to the target by `--amount`;
  the target may be a file, a folder (group character), or a saved `.npz`
  character; `--adsr` shapes the amount over time.
- `blend`: creates a new sound similar to each input by its weight; the
  `--carrier` (default: heaviest weight) provides the fine structure.
- `character`: extracts and saves the average character of files/folders.
- `sequence`: renders snippets in order while the automation clip develops the
  character morph across the sequence (see the mapping above).
