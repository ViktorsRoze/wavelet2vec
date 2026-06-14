# wavelet2vec — Experiments & Visualization

A self-contained suite that visualizes every stage of the pipeline and
compares the constant-Q Morlet wavelet front end against traditional mel
spectrograms, with quantitative metrics scored against ground-truth labels.
Design rationale: [`docs/EXPERIMENTS_PLAN.md`](../docs/EXPERIMENTS_PLAN.md).

## Running

```bash
pip install -e ".[viz]"
python -m experiments.run_experiments --audio-dir /path/to/audio_library
#   --analysis-rate 88200   # match the production rate (default = max native)
#   --limit-per-folder 3    # fast smoke run
```

Outputs (figures + `metrics.json` + `REPORT.md`) land in
`experiments/output/` (gitignored). The audio library is **not** part of this
repo; point `--audio-dir` at your own folders. Files may be any bit depth,
sample rate, channel count, or length — they are normalized once up front
(decode to float, resample to a common rate, never down-sampled).

## What the suite shows

### Stage visualization (`output/stages/<folder>.png`)
One tall figure per folder: waveform → mel spectrogram → Morlet scalogram →
each of the six embedding sections (spectral, modulation, transient, harmonic
chroma with the detected pitch marked, phase, stereo) → the final 570-dim
vector with section boundaries. This is the "how a sound becomes a vector"
explainer.

### Mel vs. wavelet gallery (`output/comparison/mel_vs_wavelet.png`)
Mel spectrogram beside the Morlet scalogram for a representative snippet per
content type, on matched percentile contrast. The scalogram's constant-Q
layout resolves low-frequency pitch (bass) and high-frequency transients
(kicks) that the linear-time mel grid blurs.

### Library map (`output/library_map.png`)
PCA of the whole library, wavelet2vec vs. the mel-summary baseline, colored by
folder. wavelet2vec separates content types visibly (percussion/kicks/bass in
one region, vocals and loops in others); the mel baseline is more entangled.

### Development strip (`output/development_strip.png`)
The `spectral` section of a 120 BPM loop across beat-aligned sliding windows —
the texture evolving bar to bar, the same machinery the `perform`/`sequence`
features drive.

### Lossless audio ↔ image round trip (`output/roundtrip/`)
Audio is turned into a picture and back **without loss** by keeping the full
complex coefficients (magnitude *and* phase) at full resolution. Two invertible
transforms are shown — an exact complex STFT and a constant-Q complex Morlet
wavelet frame (`experiments/invertible.py`):

- `<folder>_wavelet.png` / `<folder>_stft.png` — the magnitude image, the
  **phase image** (cyclic colormap), the original-vs-reconstructed waveform,
  and the reconstruction error.
- `<folder>_phase_matters.png` — the same coefficients reconstructed with
  **full phase** (perfect) vs **magnitude only** (destroyed). This is the
  whole reason wavelet2vec keeps phase: a magnitude-only spectrogram cannot be
  inverted.
- `<folder>_original.wav` / `<folder>_reconstructed.wav` — listen to the round
  trip (gitignored).

As a transform (coefficients kept in memory as float64) both reconstruct to
machine precision: **~314 dB (STFT)** and **~302 dB (constant-Q wavelet
frame)**, error ~1e-15. The wavelet frame is exact because the analysis windows
cover the whole spectrum (coverage floor 1.0), so dividing by the window sum
recovers the input.

### Base case: lossless float image on disk

The coefficients are saved as a picture and inverted back to audio. The default
is a **float32 image** (`.tiff`) — a real, viewable image whose round trip is
**~160 dB**, beyond 24-bit audio's ~144 dB ceiling, i.e. perfect for any
playback chain. The image (real + imag stacked) plus a tiny JSON sidecar
(shape + wavelet parameters) is self-contained: it inverts on its own. For a
literally bit-exact store, `.npy` keeps float64 (~300 dB). Reconstruction by
storage format:

| format | reconstruction SNR | note |
| --- | --- | --- |
| float64 (`.npy`) | ~300 dB | bit-exact, machine precision |
| **float32 (`.tiff`, base case)** | **~160 dB** | perfect-grade, viewable image |
| 16-bit (`.png`) | ~84 dB | near-transparent, most compact |
| 8-bit (`.png`) | ~14 dB | destroyed — ordinary 8-bit is not enough |

Notably, the image's **time/frequency resolution does not affect fidelity** —
the transform is exact at any band count; only the stored value precision
(float vs. integer) matters.

A perfect float32 image is written **per folder** into its own subfolder,
`roundtrip/perfect_tiff/` — each `<folder>_coefficients_float32.tiff` (the
picture) with its `<folder>_original.wav` and `<folder>_from_image.wav` (audio
reconstructed from that picture). These are large (~37 MB each); they live in a
dedicated folder so they are easy to find or delete. (Float/16-bit image IO
needs `imageio` + `tifffile`; `pip install -e ".[viz]"` includes them.)

## Results on the reference library (356 files, 10 folders)

Canonical ingestion: 356 files decoded to float; 1 resampled (a 6 kHz µ-law
percussion hit up-sampled to 44.1 kHz) — confirming heterogeneous input is
handled transparently.

### Pitch recovery — mel chroma vs. wavelet harmonic section
Predicted pitch class scored against the note/key in each filename. Folders
labeled with a single played note are scored **strictly** (exact pitch class);
folders labeled with a song key are scored **tolerantly** (±1 semitone,
octave-folded), since the snippet need not voice the tonic alone.

| folder | n | scoring | mel acc | wavelet acc |
| --- | --- | --- | --- | --- |
| bass | 10 | strict | 0.80 | **0.90** |
| harmonic_stabs_one_shots | 20 | strict | 0.45 | **0.55** |
| single_note_vocal | 53 | strict | 1.00 | 1.00 |
| synth_loops_100BPM | 16 | tolerant | 0.375 | **0.438** |
| synth_loops_120BPM | 35 | tolerant | **0.343** | 0.314 |
| vocal_shots | 51 | tolerant | 0.235 | **0.275** |
| **overall (strict)** | 185 | — | 0.503 | **0.541** |

The wavelet harmonic section (autocorrelation f0 + chroma) edges out mel chroma
overall and on the clean single-note folders; both struggle on
song-key-labeled polyphonic loops, as expected (the label isn't a single
note).

### Cluster quality by content type (balanced ≤20/folder)
k-NN folder purity and silhouette over the embeddings (cosine), random
baseline = 1/11 ≈ 0.09.

| method | kNN purity | silhouette |
| --- | --- | --- |
| **wavelet2vec** | **0.60** | **0.043** |
| mel baseline | 0.409 | −0.061 |

wavelet2vec groups sounds by content type far better than the mel-summary
baseline (≈1.5× the purity; positive vs. negative silhouette). Folders are
capped to equal size because k-NN purity is biased toward large classes and
the raw folders range from 6 to 100+ files.

## Reproducibility notes
- Everything is deterministic; re-running reproduces the figures and numbers.
- `metrics.json` is the machine-readable record; `REPORT.md` the human summary.
- Re-running after a code change surfaces regressions numerically — the suite
  doubles as a development harness.
- CI-safe unit tests (`tests/test_experiments.py`) cover metadata parsing,
  windowing, the mel baseline, and the metrics on synthetic inputs, so they
  run without the audio library.
