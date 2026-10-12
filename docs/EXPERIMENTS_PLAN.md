# Visualization & Experiments Plan — Mel vs. wavelet2vec

Status: **planned**. This spec defines a self-contained experiment suite that
(a) visualizes every stage of the wavelet2vec pipeline, (b) compares the
constant-Q Morlet wavelet front end against traditional mel spectrograms, and
(c) doubles as a development harness — quantitative metrics that catch
regressions and guide tuning. It is grounded in the provided `example_audio`
library.

## 1. The material (and why it's ideal)

`example_audio` (lives in soundgen, **not** committed here — see §8) contains
356 stereo 44.1 kHz WAVs across 10 folders, with **labels embedded in names
and folders** that give us ground truth:

| Folder | n | median dur | label in name | exercises |
| --- | --- | --- | --- | --- |
| Percussive_kick | 6 | 0.29 s | — | transient, phase onset |
| kicks | 8 | 0.39 s | — | transient |
| percussion | 79 | 0.15 s | — | transient; mixed SR (6 kHz µ-law!) |
| bass | 11 | 0.81 s | note (` A#`, ` C`) | harmonic, phase, low-freq resolution |
| harmonic_stabs_one_shots | 20 | 1.11 s | key (` G`, ` A#`) | harmonic, spectral |
| single_note_vocal | 53 | 5.00 s | note+oct (`_F#1`) | harmonic, phase (vibrato) |
| vocal_shots | 102 | 1.99 s | key (`_C#m`, `_Gm`) | harmonic, formant texture |
| synth_loops_100BPM | 16 | 9.60 s | key (`_Cm`) + BPM | modulation, sliding window |
| synth_loops_120BPM | 35 | 8.00 s | key (`_E`) + BPM | modulation, sliding window |
| fx_at_120BPM | 26 | 8.00 s | BPM | modulation, texture over time |

Confirmed: every loop duration is an **exact integer number of bars** at its
stated BPM (1 bar = 4·60/BPM s), so beat-aligned windows land on musical
boundaries. Keys/notes let us score pitch recovery objectively. Mixed sample
rates (6 kHz µ-law percussion) give a free invariance stress test.

## 2. Building blocks (new module: `experiments/`)

```
experiments/
  dataset.py         # manifest, metadata parsing, sliding windows
  mel_baseline.py    # mel spectrogram + mel-chroma + mel summary vector (numpy/scipy, self-contained)
  viz.py             # plotting helpers (matplotlib)
  metrics.py         # pitch-class scoring, cluster purity, invariance, PCA
  run_experiments.py # CLI: regenerates every figure + a metrics report
  EXPERIMENTS.md     # narrative writeup with embedded result figures
  output/            # generated figures + metrics.json (gitignored)
```

### 2.0 Canonical ingestion — one format for everything
Input is heterogeneous: 16/24-bit PCM, 32-bit float, **6 kHz µ-law**, mono and
stereo, 0.03 s to 19 s. Every file is normalized once, up front, so mel and
wavelet see *identical* input and nothing is ever degraded:

- **Decode to float** (float64 internally). PCM, float, and µ-law all decode
  losslessly to float; this removes bit-depth as a variable.
- **Resample to a single canonical rate**, default **= the maximum native
  rate present in the run** (44.1 kHz for this library), so no file is ever
  *down*-sampled (the 6 kHz µ-law percussion is up-sampled to 44.1 kHz).
  Configurable via `--analysis-rate`.
- **Channel policy**: keep stereo where present (the `stereo` section needs
  it); derive a mono mixdown for mel/scalogram panels and pitch tests. Mono
  sources stay mono (the embedder already substitutes a neutral stereo vector).
- **Length**: handled by the sliding window (§2.1); sub-window files become a
  single frame. Peak-normalization is applied per analysis window so level is
  not a confound.

**Why default to the max source rate, not 88.2 kHz?** These experiments are
*linear* analysis (mel, scalogram, embedding) with no nonlinearity to alias,
so up-sampling above the source rate adds zero information and only burns
compute — the honest "higher-quality common format" here is float at the
highest rate actually present, guaranteeing no down-sampling. (This differs
from the *production* `perform` path, where oversampling to 88.2 kHz does help
because morphing applies gain curves; that path already runs at the file's
native rate.) `--analysis-rate 88200` is available if you want the experiment
ingestion to match the production rate exactly; the result is unchanged but
slower. A one-line report logs each file's native → canonical conversion so
the normalization is auditable.

### 2.1 `dataset.py` — metadata + windowing
- **Canonical ingestion** per §2.0 (`load_canonical(path, rate) -> (stereo,
  mono, rate)`), with a small report of conversions applied.
- **Parse note/key** from filename tail: note letter, optional `#`/`b`,
  optional octave digit, optional `m` (minor). Map to pitch class 0–11.
  Handles all four labeled conventions in the table.
- **Parse BPM** from folder name (`(\d+)\s*BPM`, case-insensitive).
- **Manifest**: one row per file `{path, folder, dur, sr, channels, subtype,
  pitch_class?, key_is_minor?, bpm?}`.
- **Sliding window** (the requested algorithm):
  - canonical analysis window default **2.0 s**;
  - **BPM-aware**: when a folder has a BPM, window = 1 bar, hop = 1 beat, so
    frames are beat-aligned (1 bar @120 = 2.0 s, @100 = 2.4 s);
  - **non-BPM**: fixed window 2.0 s, hop 1.0 s (50% overlap);
  - files shorter than the window → a single whole-file frame (no padding
    needed; both representations pool over available time).
  - Returns `[(start_sample, end_sample), ...]` per file.

### 2.2 `mel_baseline.py` — the comparison baseline (self-contained)
Pure numpy/scipy so wavelet2vec gains no heavy dependency:
- STFT (scipy) → power → triangular mel filterbank → log-mel.
- **mel-chroma**: fold mel/linear bins to 12 pitch classes (baseline pitch).
- **mel summary vector**: per-mel-band mean+std (a magnitude-only analogue of
  wavelet2vec's `spectral`) — the baseline embedding for clustering tests.
Defaults chosen to roughly match the wavelet band count for fairness
(n_mels≈64, n_fft tuned so HF time resolution is representative of typical
mel use — the point of D2 is to show its limit).

## 3. Stage visualization — "every stage of wavelet2vec"

`viz.stage_figure(path)` → one tall multi-panel PNG per representative snippet
(one chosen per folder), top to bottom:

1. **Waveform** (both channels).
2. **Mel spectrogram** (baseline).
3. **Constant-Q Morlet scalogram** magnitude (the wavelet front end) — same
   snippet, log-frequency axis.
4. **Section panels** of the 570-dim embedding, each labeled and individually
   normalized:
   - `spectral` — band mean curve + std band;
   - `modulation` — band-group × modulation-band heatmap;
   - `transient` — envelope-shape curve + scalar bars (attack, decay, crest…);
   - `harmonic` — 12-bin chroma bars (with the filename note marked) + scalars;
   - `phase` — IF-deviation per group, onset-coherence triple, harmonic phase
     signature (real/imag/stability per harmonic);
   - `stereo` — width/coherence/balance per band group.
5. **Final embedding** as a single color strip with section boundaries marked.

This figure is the visual "how a sound becomes a vector" explainer for the
README/docs.

## 4. Mel vs. wavelet experiments (each: figure + numbers)

- **D1 — Side-by-side gallery.** Mel vs. Morlet scalogram for a kick, a bass
  note, a vocal, and a loop. Annotated: where each representation wins.
- **D2 — Transient time resolution.** Zoom on a kick attack. Quantify the
  effective attack time-spread: mel is hop-limited (~10 ms), wavelet is
  near sample-limited at HF. Report the numbers.
- **D3 — Pitch recovery (headline quantitative result).** For every labeled
  folder, predict pitch class from (a) mel-chroma argmax and (b) wavelet2vec
  `harmonic` chroma argmax; score against the filename label. Report per-folder
  and overall accuracy, plus a confusion matrix. Tolerant scoring option
  (±1 semitone, octave-folded) since some labels are song key not note.
- **D4 — Waveshape sensitivity on real audio.** Find two stabs/bass notes
  with the *same* pitch class but different timbre; show which sections
  separate them. Confirms the `phase`/waveshape contribution beyond magnitude.
- **D5 — Invariance battery.** Deliberately *bypasses* canonical ingestion to
  feed the embedder raw files, confirming its built-in invariance: cosine
  similarity under sample rate (6 kHz µ-law perc vs 44.1 kHz), level (−20 dB),
  and time shift (±5 ms). Table of similarities; expect > 0.98 except where
  content genuinely differs. (Canonical ingestion in §2.0 makes the *other*
  experiments fair; D5 verifies we don't even need it.)

## 5. Embedding-space structure

- **E1 — Library map.** Embed all files (sliding windows mean-pooled per file),
  PCA→2D (and UMAP if `scikit-learn` present, else PCA only), colored by
  folder. Same plot for the mel baseline vector. Compare visually.
- **E2 — Cluster quality (quantitative).** k-NN folder purity and silhouette
  score for wavelet2vec vs mel baseline. Hypothesis: wavelet2vec separates
  content types more cleanly, especially transient vs tonal.
- **E3 — Pitch geometry.** Within bass + stabs, color the map by pitch class;
  look for an ordered/circular pitch manifold.

## 6. Sliding-window development view

`viz.development_strip(loop_path)` — for a synth loop, embed each beat-aligned
window and plot how `spectral`/`modulation`/`harmonic` evolve bar-to-bar as a
heatmap-over-time. Ties directly to the `perform`/`sequence` features and shows
the sliding-window algorithm in action on real loops.

## 7. Metrics report (development harness)

`run_experiments.py` writes `output/metrics.json` + `output/REPORT.md`:
pitch-recovery accuracy (mel vs wavelet), cluster purity/silhouette, invariance
similarities, transient time-spread. Re-running after a code change surfaces
regressions numerically — this is how the suite "helps with development."

## 8. Packaging, dependencies, data hygiene

- New optional extra: `wavelet2vec[viz] = matplotlib` (+ optional
  `scikit-learn` for UMAP/silhouette; PCA/silhouette have numpy fallbacks).
- `--audio-dir` CLI arg, default `../soundgen/example_audio`; nothing assumes
  a fixed path.
- **Do not commit `example_audio`** (third-party sample content) — add to
  `.gitignore`. `output/` is gitignored too; a curated handful of result
  figures may be added to `docs/` for the README story if they are derived
  visualizations (not redistributable audio).

## 9. Tests (CI-safe, no example_audio)

Synthetic-only unit tests so the suite runs in CI without the sample library:
- metadata parsing (all four key/note conventions, BPM regex);
- sliding window counts/positions (BPM-aware and fixed) on synthetic lengths;
- mel baseline sanity (sine → correct mel/chroma bin);
- PCA / silhouette / pitch-scoring helpers on toy inputs.

## 10. Build order

1. `dataset.py` + tests → 2. `mel_baseline.py` + tests → 3. `metrics.py`
+ tests → 4. `viz.py` → 5. `run_experiments.py` → 6. run on `example_audio`,
write `EXPERIMENTS.md` with real figures and numbers → 7. README/API pointers,
version bump, commit.

## 11. Open choices (sensible defaults chosen; easy to change)

- Analysis window 2.0 s and beat-aligned hop — could switch to 1-bar hop.
- Per-file pooling = mean of window embeddings (could add max/std pooling).
- Mel `n_fft`/`hop` for D2 — pick values typical of mel use so the
  resolution limit shown is representative, not strawmanned.
