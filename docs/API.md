# wavelet2vec — API Reference

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

## Feature modules

## Feature modules

Each section's features are importable directly for custom pipelines:

| Module | Functions |
| --- | --- |
| `wavelet2vec.filterbank` | `log_spaced_frequencies`, `morlet_scalogram` (set `return_complex=True` for analytic band signals), `analytic_band_signals`, `analytic_envelope`, `constant_q_sigmas` |
| `wavelet2vec.wavelet_features` | `spectral_features`, `modulation_features` |
| `wavelet2vec.transient_features` | `transient_features` |
| `wavelet2vec.harmonic_features` | `harmonic_features`, `autocorrelation_pitch` |
| `wavelet2vec.phase_features` | `phase_features`, `instantaneous_frequency_features`, `onset_phase_coherence`, `harmonic_phase_signature` |
| `wavelet2vec.stereo_features` | `stereo_features`, `neutral_stereo_features` |
| `wavelet2vec.style_transfer` | `blend`, `style_transfer` |
| `wavelet2vec.character` | `SoundCharacter`, `extract_character`, `average_characters`, `character_from_files`, `character_from_folder`, `apply_character` |
| `wavelet2vec.adsr` | `ADSR`, `apply_adsr_volume`, `apply_adsr_character` |
| `wavelet2vec.sequence` | `AutomationCurve`, `render_sequence` |
| `wavelet2vec.slicing` | `slice_grid`, `slice_markers`, `slice_cues`, `slice_onsets`, `read_cue_points` |
| `wavelet2vec.perform` | `perform` |
| `wavelet2vec.audio_io` | `load_audio`, `save_audio`, `resample`, `list_audio_files` |

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
