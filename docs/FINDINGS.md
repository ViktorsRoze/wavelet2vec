# Verified Findings — lab notebook

**2026-09-22, v0.34.0:** [Bounded rendering](BOUNDED_RENDERING.md) avoids
retaining full input/output recordings in the file path. Generated 192 kHz
signals of 0.25 and 10 seconds use the same maximum model-window length;
initialized stem sums reconstruct within `5e-7` peak error. This validates
rendering mechanics, not separation quality. See the
[validation record](benchmarks/BOUNDED_RENDER_VALIDATION.md).

**2026-09-21, v0.33.0:** [Verified multitrack training](SUPERVISED_MULTITRACK.md)
checks aligned targets, full-recording declared additivity and reused source
assets across groups. CPU resume reproduces next crops/loss/weights at
192 kHz; failed pre-optimizer accumulation rewinds data and RNG. Evidence
uses generated fixtures, not a real-music benchmark.

**2026-09-21, v0.32.0:** [Trainable source synthesis](SOURCE_SYNTHESIS.md)
adds a reusable decoder/separator with tested source grids, gradients, stereo,
ragged batches and overlap-add. All three backends improve over equal-split
sources on the easy synthetic register/pan task. This is pipeline evidence,
not a real-music benchmark. See the
[validation record](benchmarks/SOURCE_SYNTHESIS_VALIDATION.md).

**2026-09-18, v0.31.0:** [Verified real-audio corpora](REAL_AUDIO_PRETRAINING.md) add split/group and PCM-duplicate guards, reproducible crops, corpus-aware resume and stronger diagnostic baselines. Two existing loops passed ingestion/forward checks; no held-out music-quality claim was made.

**2026-09-16, v0.30.0:** [Frozen target calibration](TARGET_CALIBRATION.md) adds training-only target scaling. The matched three-seed toy mean improved about 8.9%, but only one run beat a constant baseline; calibration remains optional and semantic music quality unproven.

**2026-09-15, v0.29.0:** [Learned context and masked acoustic pretraining](SELF_SUPERVISED_MUSIC.md) are implemented, including isolated student/teacher paths and checkpoint resume. The initial synthetic learner reduced loss but did not beat a training-mean constant target; useful music semantics remain unproven.

**2026-09-14, v0.28.0:** [Stereo/multi-resolution adapters, source training utilities and feature caches](ADAPTERS_AND_TRAINING.md) are implemented. [The wav2vec/music-model comparison](MUSIC_REPRESENTATION_GAPS.md) identifies the main remaining gap: learned temporal context and a leakage-tested self-supervised music objective.

**2026-09-13, v0.27.0:** [Long-audio extraction](LONG_AUDIO.md) adds overlapping context windows with unique token ownership, global phase/time references, bounded reader/file inference and duration-weighted coarse summaries. [Feature priorities](FEATURE_PRIORITIES.md) focus further work on stereo, multiple resolutions, training/data contracts and cache provenance. Musical understanding remains a future evaluation target.

**2026-09-07, v0.26.0:** [Compact wavelet carriers](COMPACT_WAVELETS.md) now retain phase on per-band sample grids, with grouped FFTs and hybrid training support. The measured grouped setting reduced coefficient storage 10.4x versus dense wavelets; the STFT comparator remained cheaper. Real-music separation superiority is unproven.

**2026-09-06, v0.25.0:** The new [hybrid audio foundation](HYBRID_ENCODING.md) retains raw waveform and full complex carriers, with lossy trainable tokens, a defined inverse and synthetic ablations. Earlier entries below are historical; pooled vision images are not lossless, and hybrid superiority is not established.

Every claim here was measured in this repo and is reproducible (`pytest`
covers most as regression tests; one-off commands noted). Newest last.

## Losslessness of the frame (audio ↔ image)

- Constant-Q complex frame round trip: **~302–305 dB SNR** (error ~1e-15) on
  noise, tones, transients, real kicks and FX sweeps; complex STFT baseline
  ~314 dB. The frame is exact because analysis windows cover the whole
  spectrum (coverage floor 1.0). `tests/test_invertible.py`.
- Storage sets fidelity, **resolution does not**: 32/64/128/256 bands all
  reconstruct ~300 dB. float64 `.npy` ≈ 300 dB · float32 `.tiff` ≈ 160 dB
  (beyond 24-bit audio) · 16-bit PNG ≈ 84 dB · 8-bit ≈ 14 dB.
- Magnitude-only reconstruction (phase zeroed) collapses (< 30 dB): phase is
  half the sound. `test_magnitude_only_loses_information`.

## Embedding vs. mel baseline (356-file library)

- Cluster purity (balanced ≤20/folder): **wavelet2vec 0.60 vs mel 0.41**
  (random 0.09); silhouette +0.043 vs −0.061.
- Pitch class vs filename labels: **0.541 vs 0.503** overall strict;
  bass 0.90 vs 0.80. Repro: `python -m experiments.run_experiments`.
- Saw vs phase-scrambled saw (identical magnitude spectra): every
  magnitude section ≥ 0.997 similarity, **phase section 0.86** — only the
  phase features tell waveshape apart.

## Morphing / performing (deterministic)

- Blend of pluck/piano/vocal (0.5/0.3/0.2): result closer to *every*
  reference than they are to each other (0.95/0.81/0.79 vs cross 0.78/0.72/0.67).
- 4-note vocal sequence vs guitar character under a rising FL clip:
  similarity walks 0.674 → 0.684 → 0.696 → 0.709, note by note.
- `perform` round trip is sample-exact; 88.2 kHz float I/O bit-exact.

## Groundwork (frame API, HPSS, transients, rhythm) — commit f7ec37c

- **Exact-sum decompositions**: HPSS `harmonic + percussive == input` at
  ~303 dB on real loops; transient designer at unity gains returns the input
  **bit-exactly** (the SNR check overflowed on infinity). Complementary
  masks + linear synthesis are the mechanism. `tests/test_groundwork.py`.
- HPSS layer sanity: pure tone ≥ 80 % harmonic-layer energy; click train
  lands percussive; a real synth loop split 48 % / 52 %.
- **Rhythm** (BPM given): synthetic straight loop reads swing **50 ± 4 %**,
  triplet-swung loop **66.7 ± 4 %**; kick pattern lands exactly on slots
  {0,4,8,12}; real 120 BPM synthwave loops read swing 53–63 %,
  syncopation 0.39–0.64 — musically plausible.
- **Zero-phase timing physics**: constant-Q wavelets spread energy
  symmetrically in time, so *absolute* low-band onsets lead by up to ~10 ms
  (constant per register; 60 Hz band σ_t ≈ 18 ms). Groove quantities
  (swing, microtiming) are defined relatively (median-centered per
  register), where the constant cancels. Raw `deviations_ms` keeps the bias
  and says so.
- Two bugs found by tests: onsets at sample 0 were invisible (flux
  self-prepend); percentile gate normalization collapsed on sparse-onset
  material — now max-normalized.

## Dataset tooling + spatial API — this commit

- **Ladders supervise their own knobs** (measured, monotone):
  filter-cutoff ladder → brightness strictly rises; distortion drives →
  brightness rises; noise-mix ladder → harmonicity strictly falls;
  attack ladder 1→200 ms → measured attack_ms strictly rises. Labels are
  computed by the same features models will condition on.
- In-domain lowpass pairs keep the low region intact (corr > 0.5 with the
  original) while brightness drops > 2× — clean BWE supervision.
- RBJ biquad warps near Nyquist: keep pluck cutoffs ≲ 0.3·sr (ladder at
  22.05 kHz went non-monotone at 9 kHz cutoff; fine at 44.1 kHz).
- **Coherence needs integration time ∝ 1/bandwidth**: one 10 ms frame of a
  ~1 Hz-wide band is a single phasor — it always reads "coherent". The
  stereo phase spectrum now integrates each band over ~1.5 coherence times;
  independent noise then reads < 0.4 mean coherence above 200 Hz, while a
  1 s clip's lowest bands stay honestly undecidable (~1 independent sample).
- Inter-channel delay shows as the predicted IPD (2π·f·τ) at full
  coherence; gain differences show in balance with coherence 1 — the
  Stereo Phase Spectrum separates delay-image from level-image.
`tests/test_datasets_spatial.py`.

## Tempo estimation + groove transfer (R-6, R-8) — this commit

- **Loops give away their tempo at bar scale, not beat scale.** A beat-level
  comb (lag + 2·lag) scored only 40 % exact on 77 of our BPM-labeled loops:
  melodic synth material has weak beat periodicity, and a 16th-note grid
  makes 4:3 metrical aliases score well (100 BPM loops read as 133.5 — three
  16ths mistaken for a beat). The fix is a **bar-aware comb** over 1, 2, 3,
  4, 8 and 16 beats: the 4-beat bar term carries the largest weight (loops
  repeat at bar scale) and the 3-beat term vetoes 4:3 aliases (a true beat
  also peaks at 3·lag; the alias doesn't). Result: **57 % exact overall,
  75 % (38/51) on the rhythmic loop folders** (120: 28/35, 100: 10/16).
- The FX folder stays hard (6/26): risers/ambient textures rendered *at*
  120 BPM mostly have no beat to find. Returned confidence separates hits
  from misses there (0.34 vs 0.10 mean) — callers can fall back to a
  DAW-supplied BPM below ~0.15. An early "win" of 20/26 on FX turned out to
  be an artifact: unbiased autocorrelation at near-window lags divides by a
  handful of samples and explodes, effectively always answering 120. Now the
  divisor is capped at N/4 and lags stop at 3/4 of the window.
- **Groove = swing + microtiming residual.** Full triplet swing at 120 BPM
  is a +83 ms shift — more than half a 16th slot (±62.5 ms), so swung hats
  alias into the *next* slot with negative deviations. A raw microtiming
  copy therefore cannot represent heavy swing. `groove_offsets` composes the
  two: measured swing drives the off-8th slots {2,6,10,14}, and when swing
  departs 50 % by > 8 points the aliased neighbor slots {3,7,11,15} are
  cleared; everything else keeps measured microtiming.
- `apply_groove` re-times 16th slices with raised-cosine crossfades:
  strength 0 is a **bit-exact** identity, straight→swung transfer measures
  66.7 ± 5 % swing after processing, and a straight loop grooved from an
  analyzed triplet source measures > 58 %. `tests/test_tempo_groove.py`.

## Rhythm embedding (R-7) — this commit

- **89-dim tempo-normalized groove vector** (`rhythm_embedding`): pattern
  (3 registers × 16 slots), swing, syncopation, microtiming (per-slot
  mean/std in *slot units*), register balance — every section grid-relative,
  so the same groove at 100 and 140 BPM embeds nearly identically (cosine
  > 0.9 on synthetic loops) while a hats-only pattern at the *same* tempo
  sits > 0.1 lower: groove identity beats tempo identity. Sections are
  L2-normalized with the embedder's 0.1 floor so the 48-dim pattern doesn't
  drown the scalar sections.
- Real-data groove search is honest, not magic: across 12 of the 120 BPM
  synthwave loops the off-diagonal similarity is tight (mean 0.86 — these
  loops genuinely share four-on-floor DNA), and bars 3–4 retrieve their own
  bars 1–2 in 6/10 cases (chance 1/10; the misses are loops whose second
  half is a fill/variation). Sharper groove separation is learned-model
  territory (G-2), not more hand-tuning.
- `rhythm_embedding` estimates BPM via R-6 when not given; pass the DAW's
  BPM whenever known. `profile_embedding` reuses an existing
  `RhythmProfile` so library scans don't analyze twice.
  `tests/test_tempo_groove.py`.

## Spatial character + width designer (M-4, M-5) — this commit

- **Side-only operations keep the mono downmix bit-honest.** Width is
  defined as the per-band Side-energy fraction Es/(Em+Es) (0 = mono,
  0.5 = decorrelated, 1 = antiphase). Both `apply_spatial_character`
  (profile matching) and `design_width` touch only the Side signal, so
  (L+R)/2 is preserved at reconstruction precision — measured 303 dB on
  real loops. Transferring one synthwave loop's width profile onto another
  closed the per-band profile distance from 0.145 to 0.033.
- **A constant per-band phase rotation cannot decorrelate** — algebra shows
  L/R coherence stays exactly 1 for any single-source construction
  L = M(1+αe^{iφ}), R = M(1−αe^{iφ}): it is a delay/level *image*, not
  width. Real decorrelation needs a component orthogonal to Mid over time;
  `design_width` uses a delayed Mid (15 ms default) as added Side, which
  drops coherence to the predicted 1 − 2·width (measured 0.40 ± at target
  width 0.3). The physics is frequency-honest: bands narrower than ~1/delay
  keep coherence, so bass stays tight and pure tones take only a phase
  image — exactly what a mastering-safe widener should do.
- **Overlapping constant-Q bands veto per-band sign tricks**: alternating
  the added Side's sign between adjacent bands (to avoid lateral lean)
  cancels almost everything at synthesis — adjacent Gaussian bands sit ~1σ
  apart and overlap heavily; measured width collapsed to 1e-4. One sign for
  all bands works, and the delay's own IPD rotation (2π·f·τ) prevents any
  static lean.
- The mono-compatibility guard is exact under the same math: coherence
  floor c caps width at (1−c)/2 per band. Widening grows individual-channel
  peaks (1.4× at width 0.3 on a real loop) — by design, not a bug.
  `tests/test_spatial_design.py`.

## Multichannel character modes (M-6) — this commit

- `apply_character` (and `blend`, `style_transfer`, `perform`,
  `apply_adsr_character`) now take `channel_mode`. Measured on a real
  stereo loop pulled toward another loop's character (mid-band width /
  balance movement):
  - **shared** (default): one gain set from the mixdown — image untouched
    (width moved 0.012, balance 0.029). What the library always did.
  - **per_channel**: each channel matched with its own gain curves — the
    channels' characters converge (L/R character distance drops > 30 % in
    tests) but the image can shift laterally (balance moved 0.152).
  - **mid_side**: Mid and Side matched independently — laterally symmetric
    (balance moved 0.065) with a hard invariant: **the mono downmix of the
    result is exactly the Mid processed as mono material** (asserted to
    float32 precision). Width may move because the Side is genuinely being
    matched — per-axis semantics, not a leak.
- `perform` keeps one timeline for all channels: note boundaries, ADSR and
  automation values are shared; only the morph math is per-mode.
- Found and fixed a pre-existing edge bug via the `amount=0` identity test:
  the envelope-gain smoother used a plain `'same'` convolution, whose
  kernel coverage halves at the array ends — every processed snippet's
  first/last ~5 ms were damped (masked until now by boundary fades).
  `_smooth` now normalizes by kernel coverage; amount=0 is a true identity.
  `tests/test_channel_modes.py`.

## f0 tracking + key/chord estimation (H-1, H-2) — this commit

- **`track_f0`**: batched YIN (cumulative-mean-normalized difference via
  FFT correlation) with Viterbi smoothing over a semitone state grid plus
  an explicit unvoiced state, parabolic dip refinement. Real-data
  validation against the note-labeled one-shot folders: **pitch class
  correct on 53/53 sung single notes**, 8/10 bass one-shots (misses sit at
  the 55 Hz tracking floor), 13/20 harmonic stabs — stabs are chords, so a
  single-f0 "root" at 65 % is the honest reading, not a failure.
- **`estimate_key`**: chroma from constant-Q bands placed *exactly* on
  semitones (55 Hz · 2^(k/12) — no bin-to-note rounding), log-compressed,
  correlated with Krumhansl-Schmuckler profiles. Validation against 237
  key-labeled files first looked bad (17 % exact) until the labels were
  read correctly: most files are **single notes whose letter is a pitch,
  not a key** — mode is meaningless there and the tonic is right 100 % of
  the time (53/53 vocal notes, 8/10 bass). On real *loops* (the only
  folders where "key" applies) exact-key is ~46 % on 120 BPM and ~38 % on
  100 BPM synthwave — KS template matching's honest ceiling on pad-heavy
  synth harmony; a learned axis (G-2 family) is the upgrade path, and the
  editor library should show f0-derived pitch for one-shots and KS key
  only for loops.
- `segment_notes` groups the Viterbi track into MIDI note events;
  `track_chords` runs 24 triad templates + Viterbi (switch penalty) over
  the same chroma. Synthetic checks: vibrato tracked to ±3 Hz median,
  A3→E4 split at the right boundary, Am→F chord change segmented cleanly.
  `tests/test_harmony.py`.

## Structure segmentation + loop points (H-3, H-4) — this commit

- **`segment_structure`**: Foote checkerboard novelty on the cosine
  self-similarity of windowed Wavelet2Vec embeddings. Embedding cosines
  rarely drop below ~0.5 (norm-floored sections share mass), so boundary
  novelty lives around 0.05–0.25, and an absolute threshold cannot serve
  both hard cuts and subtle ones — the default is adaptive (median +
  8·MAD of the clip's own novelty, floored). Tone→noise→tone splits are
  found exactly; joining two *same-genre* synth loops is inherently
  low-contrast (cross-similarity ~0.93) and recall there is partial by
  nature, with the adaptive threshold tuned precision-first (no spurious
  splits on the tested pairs).
- **Loop detection needed centered features**: a steady synth spectrum
  makes raw envelope cosines ~0.9 at *every* lag; subtracting each band's
  temporal mean leaves only the pattern (true bar scores 0.50 vs −0.31
  off-grid on a real loop).
- **A sample-exact loop seam often does not exist**: synthwave supersaws
  have free-running oscillators, so the waveform never repeats even when
  the music does — raw-splice mismatch sits at signal-RMS level on
  musically perfect loops. `seam_error` is therefore spectral (envelope-
  context match), and `render_loop` blends each seam with an equal-power
  crossfade (powers sum to one because the two sides carry independent
  phase). Measured on real loops with a bar-length hint (`min_loop_s ≈
  0.95·bar`): rendered seams show no spectral-flux spike (≤ 1.2× the p95
  flux) in **10/12 (120 BPM) and 11/12 (100 BPM)** files.
- Unconstrained, the detector reports the material's honest fundamental
  period — usually a couple of beats, not a full bar, because these loops
  evolve (fills, filter movement). With the BPM hint, lengths land on bar
  multiples in 8/12 and 7/12. `tests/test_structure.py`.

## GPU/torch frame backend (C-3) — this commit

- **`torch_frame` mirrors the numpy frame exactly.** `analyze_torch` /
  `synthesize_torch` reproduce the analytic-signal construction (positive
  freqs doubled, DC and Nyquist single) with torch FFTs. Same
  `FrameSpec` — so the backends interchange freely: torch-analyze →
  numpy-synthesize round-trips at 305 dB, and vice versa. Double precision
  matches numpy coefficients to **1.4e-15** (machine epsilon); single
  (complex64, the fast GPU setting) round-trips at **131 dB**, still beyond
  16-bit audio.
- **The win is batched short fragments, not single-clip latency.** The
  numpy path is a per-band Python loop; the torch path filters all bands
  (and any leading batch dims `[..., N] → [..., n_bands, N]`) as vectorized
  FFTs. On CPU, analyzing 256 × 0.5 s fragments at 128 bands: numpy loop
  11.0 s (12× realtime) vs torch batched 1.23 s (104× realtime) — **9×**,
  before any GPU. `device="cuda"` runs the identical call on the GPU for a
  further large multiple; embedding a whole sample library is the use case.
- Design choice: keep numpy `frame.analyze` the untouched default and add
  the torch backend as separate functions (`analyze_batch` defaults to
  single precision — the inference setting). Optional dependency: absent
  torch raises a clear message, and every other module stays pure-numpy.
  `tests/test_torch_frame.py` (9 tests; CUDA cases auto-skip on CPU-only
  machines).

## Vision bridge: images + synthetic ground truth (V-1, V-2) — this commit

- **One fragment → one fixed 128×128×3 image** (`coeff_image`). Any input
  length is adaptive-pooled to a fixed frame count, so the model always
  sees the whole musical object (the short-fragments principle) at a
  constant size. Channels are the roadmap encoding — log-magnitude,
  instantaneous-frequency deviation (tanh-squashed; raw phase is
  unlearnable, IF is the GANSynth lesson), and energy-weighted phase
  coherence. The coherence channel measurably tracks tonalness: it's
  higher where magnitude is high (stable tones) than in near-silent cells.
- **Boxes and the saved tensor share one coordinate system.** A subtle
  bug caught in testing: the tensor had row 0 = lowest band while
  `event_to_bbox` emitted COCO display coords (row 0 = top = highest) — a
  silent vertical flip that would have trained a detector on mislabeled
  data. `coeff_image` now emits display orientation directly, so
  `.npy` tiles and `annotations.json` agree. Result: ground-truth boxes
  sit at **3× the image-mean energy, 18/18 on a random scene**, and
  time/Hz ↔ pixel round-trips to <3 % frequency error.
- **Unlimited exact labels** (`scenes.py`). `render_scene` /
  `random_scene` synthesize sine/saw/pluck notes and kick/snare/hat
  drums with known onsets, pitches, durations and classes; a saw note's
  peak band lands within 10 % of its fundamental. Deterministic under a
  seed (audio and labels reproduce bit-for-bit). `weak_labels` turns real
  audio into approximate boxes from the existing DSP — f0 tracking →
  pitched-note boxes, rhythm registers → kick/snare/hat boxes — the
  training signal for domains the synth generator doesn't cover.
- `export_dataset` writes `clip_XXXX.npy` tensors + a COCO `annotations.json`
  (images/annotations/categories, per-box pitch & velocity attributes).
  `tests/test_vision.py` (13 tests). This is the data foundation for V-3
  (frozen-backbone probes) and V-4 (DETR event detection); batched
  analysis from C-3 renders the corpus fast.

## Deterministic imagination axes (G-2a brightness, G-9a tonality) — this commit

- **The imagination knobs work with no trained model.** `imagine_highs`
  and `imagine_tonality` are the deterministic fallbacks the roadmap names
  (G-2/G-9); the learned versions become drop-in upgrades because the API
  and the measured axis (`datasets.axis_labels`) stay identical.
- **`imagine_highs` = spectral band replication.** For each band above the
  split, an octave-below analytic signal (or the loudest mid band, if that
  source is too quiet) is heterodyned up, scaled to a −6 dB/octave rolloff
  from the loudest *real* mid band, high-passed, and added. The key insight
  from a failed first attempt: extrapolating the sound's *own* HF slope
  predicts (correctly) near-zero for a dull sound and adds nothing — that's
  EQ, not imagination. Referencing the loudest mid band instead is what
  lets it invent air. Brightness (HF-energy fraction) rises monotonically
  0 → 0.74 on a low-passed saw and 0.0002 → 0.32 on a real synth loop;
  amount 0 is bit-exact; the low end blends within a few percent across the
  split (the apron every T-F edit has).
- **`imagine_tonality` = HPSS remix.** `harmonic·(1+amount) +
  percussive·(1−amount)`, amount ∈ [−1, 1]. Because HPSS is exact-sum,
  amount 0 reconstructs the input (corr > 0.999); harmonicity rises
  monotonically from noisy (−1: 0.55) to tonal (+1: 0.65) on a real loop.
- Both are per-channel, multichannel-safe, and validated against the same
  brightness/harmonicity features a model would condition on — so a learned
  axis has a measured baseline to beat. `tests/test_imagine.py` (9 tests).

## Expression automation, deterministic (G-11a) — this commit

- **The original ask works end to end**: "each next note gets a harder
  attack and more air" is now literally two rising automation lanes.
  Measured on 8 identical dull pad notes: brightness climbs 0 → 0.26 →
  0.64 and the crest factor 2.8 → 4.7 → 5.4 across the phrase, note 1
  bit-untouched (lane value 0 = identity by contract).
- `render_expression(waveform, sr, starts, lanes)` follows the `perform`
  slice contract (exact shape/rate out, boundary fades) and samples each
  lane once per note. Lanes are uniform **0→1** (FL Studio clip range,
  parsed clips plug straight in): 0 always identity, 1 full effect, with
  per-axis mappings (brightness→imagine_highs 0..2, tonality/noisiness→
  ±imagine_tonality, attack/sustain→shape_transients gains, character→
  apply_character) overridable via `ranges`.
- Design choice: separate `tonality` and `noisiness` lanes rather than one
  bipolar lane, so every lane keeps the 0-identity convention a drawn FL
  clip expects; when both are present they net into one HPSS remix.
- When the learned axes ship (G-2/G-8/G-9), they slot behind the same lane
  names — frame-rate FiLM morphs inside a note stay G-11 (GPU tier).
  `tests/test_expression.py` (8 tests).

## Similarity index + dedup (I-4) — this commit

- **Exact search is the right call at library scale.** 570-dim cosine over
  a normalized matrix is one GEMM: 0.1 ms per query over the real loop
  folder, and FAISS `IndexFlatIP` (also exact, auto-selected above 2 k
  entries) keeps that flat into the hundreds of thousands. No approximate
  index until a library actually outgrows exactness — approximation would
  silently weaken dedup.
- Dedup = union-find over the exact pairwise matrix, blockwise (memory
  stays flat): exact copies, scaled copies (normalization eats gain
  changes) and near-copies group together at the 0.995 default; the real
  120 BPM folder correctly reports **no** duplicate groups.
- `from_folder` builds an index straight from audio (skipping unreadable
  files, logging them); ids are folder-relative paths. Persistence is one
  `.npz`. Build cost is the embedder (~3 s/file CPU) — the editor's
  library cache amortizes it, and `analyze_batch` (C-3) is the speed path
  when batching lands in the embedder.
- With I-4 shipped, **the deterministic tier of the roadmap is complete**:
  every remaining item (V-3+, G-2/G-8/G-9, learned G-11) requires GPU
  training. `tests/test_index.py` (7 tests; faiss-vs-numpy parity checked).

## DL practicality wave 1 (D-1, D-2, D-4) — this commit

- **Profile before optimizing**: the embedding hot spot was not the
  scalogram (26 %) but `instantaneous_frequency_features` (51 %) —
  `np.unwrap` over the full-rate array. The conjugate-product identity
  (`angle(c·conj(c₋₁))` ≡ diff-of-unwrapped-phase for analytic band
  signals) removes the unwrap pass; float32 temporaries with float64
  accumulators halve the memory traffic without losing sum precision.
  With batched torch scalograms + one conv forward (`embed_batch`):
  **203 → 77 ms per fragment (2.6×), parity cosine 1.0**. The remaining
  tail is per-clip numpy feature statistics (D-1b).
- **`WaveletFrontend` and the NaN-gradient trap**: `torch.angle` and
  `.abs()` have NaN gradients at exactly-zero coefficients, so one silent
  region poisons the whole backward — and `NaN > 0` being False makes it
  masquerade as "no gradient". Guards: smooth magnitude
  `sqrt(re²+im²+ε)` and near-zero increments routed through a constant
  before `atan2` (a `where` after the fact still back-propagates NaN from
  the unselected branch). Gradients now finite through leading silence.
- **Parity needs the exact frame**: plain Gaussian windows to 0.45·sr gave
  only 0.83 correlation with `coeff_image` — different center grids. The
  module now mirrors `analyze()` exactly (edge-plateau windows, 0.5·sr
  grid, DC/Nyquist not doubled): magnitude channel 1.0000. The IF channel
  reads 0.28 *globally* but **0.9998 where energy exists** — near-silent
  cells hold precision-dependent noise that carries no signal, which is
  precisely why the encoding energy-weights it.
- **Naive CQT pitch shift provably fails**: rolling coefficient rows +12
  semitones leaves a 440 Hz tone at 440 Hz — coefficients are analytic
  band signals still oscillating at the source frequency, and synthesis
  puts the energy straight back. `pitch_shift` heterodynes each band to
  its destination (`·e^{2πjΔf t}`) before the remap: 440 → 880/220/659 Hz
  for +12/−12/+7 semitones, duration preserved, formants move with pitch
  (uncorrected — a learned-tier feature). `tests/test_dl_practicality.py`.

## Sequence tokens (D-3) — this commit

- **One encoding, two views.** `sequence_tokens` emits the same 3-channel
  encoding the vision tiles use, un-pooled per time frame: each token is
  `[log-magnitude | IF-deviation | coherence] × bands` (192 dims at 64
  bands). A transformer and a ViT therefore learn from identical features
  — and `WaveletFrontend.tokens()` is the batched differentiable twin
  (magnitude parity 1.0 vs the numpy path at energetic cells).
- Measured behavior: a note onset at 0.2 s localizes to the 0.19 s token
  at 20 ms hop; hop directly controls resolution (10 ms → 100 tokens/s);
  the coherence block separates tonal from noisy content by ~4× — but
  only when read *at energetic cells*: coherence is energy-weighted by
  design, so a pure tone (one lit band) vs broadband noise (64 lit bands)
  makes global means count cells, not tonalness. The first test made
  exactly that mistake; the encoding's contract is "interpret coherence
  where magnitude lives."

## Augmentation kit (D-5) — this commit

- **Time-stretch inside the invertible frame**: per-band magnitudes and
  per-sample phase *increments* are resampled to the new length and the
  phase re-integrated — a constant-Q phase vocoder. Because increments
  (not phases) are interpolated, instantaneous frequency survives
  exactly: a 440 Hz tone stays 440.0 Hz through both 0.5× and 2×, and
  click spacing scales to within interpolation tolerance (±4 samples).
- `random_augment` is deterministic per ``(seed, index)`` — reproducible
  epochs with per-item variety — and restores the input length after
  stretching so batch shapes hold. Pitch shift (D-4), stretch, one
  pedalboard effect, and polarity/channel flips are sampled
  independently; pedalboard being absent silently skips only the effect
  stage. `crop_bars` cuts training views on musical boundaries instead
  of arbitrary windows. `tests/test_dl_practicality.py` (22 tests total
  for Epic D).

## Dataset wrappers (D-6) — this commit

- **Training data in three lines**: `data.py` wraps the synthetic tiers in
  `torch.utils.data.Dataset`s — `SceneDetectionDataset` (V-2 scenes →
  3-channel image + COCO boxes/labels/pitches/velocities),
  `LadderAxisDataset` (one G-1 ladder step → image + *measured* axis
  label via `axis_labels` — the label is computed by the same feature the
  model will condition on, so supervision is free and honest), and
  `ContrastiveViewsDataset` (two `random_augment` views per clip for
  SimCLR/BYOL, optional per-view transform).
- Everything is generated on the fly and **deterministic per (seed,
  index)** through `np.random.SeedSequence([seed, index])`: epochs are
  reproducible, no disk corpus is needed, and DataLoader workers cannot
  collide. Verified: same-(seed,index) items are tensor-equal, different
  indices differ, boxes stay inside the image frame, and
  `detection_collate` keeps variable-length targets as a list while
  stacking images (DataLoader-integration tested).
- Design note: `LadderAxisDataset` renders **only the sampled ladder
  step** (the generators accept a one-element parameter list), not the
  whole 8-step ladder — 8× cheaper per item, and the cutoff is sampled
  log-uniformly to match the geometric spacing the ladders use.
  `tests/test_dl_practicality.py` (26 tests total for Epic D).

## Embedding rigor: eval harness + a negative result (D-7) — this commit

- **The harness**: `eval.py` adds leave-one-out retrieval metrics over
  labeled vectors — `knn_purity` (k-NN majority, ties broken by the
  nearest), `recall_at_k` (singleton labels excluded: they have no
  correct answer), `retrieval_report` (adds per-label 1-NN purity so a
  strong global number can't hide a broken class). Any embedding change
  now gets a number, not an opinion.
- **Real library (356 files, 10 folders as labels)**: raw cosine scores
  **purity@1 0.916, recall@5 0.983, recall@10 1.000**. Weakest classes
  are the loop folders (synth_loops_100BPM 0.31 — loops of the same
  *genre* at a different BPM sit in other folders, so the "error" is
  partly the labeling). Note this is a different metric from the earlier
  0.60 *cluster* purity (k-means); leave-one-out 1-NN is the retrieval
  question the editor's library search actually asks.
- **Negative result, kept**: `EmbeddingStandardizer("standardize")`
  *hurts* on the real library (0.885 vs 0.916) — the 570-dim sections
  are already scale-calibrated (norm floors + per-section normalization),
  so z-scoring only amplifies rare dimensions. Whitening is catastrophic
  there (0.458): with N=356 < dim=570 the covariance is rank-deficient,
  and whitening flattens *between-class* variance along with the rest —
  measured on synthetic clusters too (0.30 whitened vs 0.97
  standardized when noise dims dominate scale). Standardization earns
  its keep only when scales are genuinely mismatched (synthetic: 0.33 →
  0.97). **Raw cosine stays the default**; the standardizer ships as a
  tested option with these caveats.
- **Version tag**: `EMBEDDING_VERSION = "w2v570-v1"` lives in
  `embedder.py`; `Wavelet2Vec().version` carries it,
  `SimilarityIndex.from_folder` stamps it, save/load round-trips it and
  warns (with "rebuild") when a loaded index was built by a different
  embedding version. Untagged pre-D-7 indexes still load.
  `tests/test_embedding_eval.py` (7 tests).

## Batched feature stats close out Epic D (D-1b) — this commit

- The post-D-1 profile showed the per-clip numpy tail was 68 %
  `phase_features` (IF deviation 29 ms, harmonic phase 13 ms, onset
  coherence 11 ms per fragment) plus `spectral_features` (10 ms).
  The three scalogram-statistics blocks now run **batched in torch inside
  the same no_grad chunk loop that builds the scalograms**, and are
  injected into `_components_from`; dtypes and operation order mirror the
  numpy paths (float32 temporaries, float64 accumulators), so per-clip
  parity is max-abs 7.7e-7 against `embed()`.
- Result: **203 → 60 ms/fragment (3.4× total)**; D-1 alone was 77 ms.
  Two algebraic savings helped marginally (w ≥ 0 makes w·|d| = |w·d|, and
  variance via one-pass E[d²]−E[d]²), but the honest finding is that the
  pipeline is now **memory-bandwidth-bound on CPU**: the remaining cost
  is irreducible elementwise passes (complex conjugate product, angle,
  divide, log1p) over ~180 MB of complex64 per chunk. The original ≥10×
  ambition needs CUDA — `_batched_scalogram_stats` runs unchanged there —
  or decimated statistics, which would change results and break parity.
- Remaining per-clip work: `harmonic_phase_signature` (14 ms; needs
  per-f0 analytic bands, so it batches poorly) and the small
  transient/harmonic/modulation stats (< 3 ms each). Epic D is complete.

## The box had a GPU all along (D-1c) — this commit

- The long-standing "no CUDA on this box" assumption was wrong: there is a
  GTX 1050 Ti (4 GB, sm_61 Pascal, driver 560.94). The real blocker was
  the env's `torch 2.11.0+cpu` wheel. Fixed with the same version from the
  cu126 index (`torch 2.11.0+cu126`) — a drop-in swap. Wheel-line lesson:
  **CUDA 12.8+ wheels dropped sm_50–sm_70**, so Pascal needs the cu126
  line (available through torch 2.13 as of today).
- Verified on-device (all ops the frame/stats path uses — batched
  rfft/ifft, complex multiply, angle, float64 accumulation — run clean on
  sm_61): `WaveletFrontend` forward+backward on an 8×1 s batch **798 →
  218 ms (3.7×)** — the training entry point; `embed_batch(device="cuda")`
  **34.5 ms/fragment vs 58.2 CPU (5.9× vs the 203 ms original)**, parity
  vs `embed()` max-abs 1.2e-7. Remaining embed time is the per-clip numpy
  tail (harmonic phase 14 ms), untouched by the device.
- ~3.1 GB VRAM free with the desktop running: enough for the learned tier
  at small batches (V-3 frozen-backbone probes, axis specialists at
  ViT-S/Swin-T scale with reduced batch); Swin-L/Co-DETR-scale training
  still belongs on a bigger card. `embed_batch` device contract: an
  explicit `device=` that fails raises — it never silently falls back to
  the slow path. Suite 204 passed (3 CUDA-gated tests now un-skipped).

## Frozen vision backbones read our tiles (V-3) — this commit

- The go/no-go question: do generic pretrained vision features understand
  3-channel coefficient images? Measured on the real library (356 files,
  10 folders, D-7 harness + 5-fold linear probe), all frozen, no audio
  training whatsoever:

  | features | dims | purity@1 | recall@5 | linear probe |
  |---|---|---|---|---|
  | embedding v1 (deterministic) | 570 | **0.916** | **0.983** | **0.927** |
  | DINOv2 ViT-S/14 | 384 | 0.888 | 0.952 | 0.899 |
  | DINOv2 ViT-S/14, magnitude-only | 384 | 0.854 | 0.975 | 0.902 |
  | Swin-T (ImageNet supervised) | 768 | 0.860 | 0.966 | 0.896 |

- **Verdict: GO for V-4.** Frozen ImageNet-domain features land within
  3 points of an embedding purpose-built from audio physics — on imagery
  they have never seen. Honest asterisk: gate 4 ("V-3 probe beats
  embedding-v1 purity") is *not* passed by frozen features — beating v1
  will need the fine-tune the gate was always about.
- **The 3-channel encoding earns its keep**: feeding DINOv2 log-magnitude
  alone drops purity 0.888 → 0.854, and the damage concentrates exactly
  where phase matters — kicks 0.875 → 0.375, Percussive_kick 1.00 → 0.83.
  The IF-deviation/coherence channels are what let a vision model tell
  percussive classes apart; magnitude looks the same across kick folders.
- Self-supervised DINOv2 transfers better than supervised Swin-T (0.888
  vs 0.860) on this out-of-domain imagery, consistent with its
  reputation. Weak classes are the same everywhere (the BPM-split synth
  loop folders — a labeling artifact noted at D-7, not a feature failure).
- Cost reality on the 1050 Ti: backbone inference is **14 ms/tile**
  (5 s for the whole library); the pipeline cost is tile *rendering*
  (~2.5 s/file, full-length analyze at 128 bands). Caches under
  `experiments/output/frozen_probe/` make re-runs seconds.

## First learned axis: brightness v1 (G-2) — this commit

- **Design law held**: the model learns only *energy*. `BrightnessNet`
  (196k params, FiLM-conditioned encoder–decoder on a fixed 64-band ×
  128-frame log-envelope grid) predicts per-band per-frame envelopes for
  the missing highs; the carrier and phase still come from
  `imagine_highs`' deterministic heterodyne, so the low end is preserved
  exactly and `amount=0` stays bit-exact. A model is a drop-in via
  `imagine_highs(..., model=..., brightness=...)`; the library still
  never requires weights.
- **Supervision was free**: a coefficient-domain lowpass scales each
  band's envelope by the same mask, so (masked → full) envelope pairs
  need no audio synthesis in the training loop. Pool: 600 synthetic
  scenes/notes + 356 one-second crops of the real library; 15 % held
  out; cutoffs sampled log-uniform 800–8000 Hz; loss weighted onto the
  hidden bands. 4000 steps ≈ 6 min on the 1050 Ti.
- **A/B on 30 held-out real clips** (log-envelope L1 above the 4 kHz
  split, all through the real API paths): model **0.0027** — 10× better
  than deterministic G-2a (0.0266) and better than the do-nothing
  lowpassed floor (0.0039). Honest reading of "G-2a worse than floor":
  G-2a is an *exaggeration* knob referenced to the loudest mid band —
  it adds bold highs regardless of what the original had, which is its
  job; reconstruction was never its objective. The learned v1 is the
  one that can *match* a target.
- **The knob is a knob, with a caveat**: measured output brightness was
  strictly monotone in the requested target on 100 % of held-out sweeps
  — but the delivered absolute brightness is tiny (1.3–2.2e-4 for
  requests 0.05–0.6). Cause: real training sounds rarely measure above
  ~0.1 brightness, so conditioning targets like 0.6 sit outside the
  label distribution and extrapolate timidly. v2 fix: augment training
  targets with artificially brightened variants (tilt in the envelope
  domain is free) so the conditioning range is actually covered.
  Val-high-L1 was still falling at 4k steps (0.0064 → 0.0047) —
  undertrained; more steps are cheap now.
- Checkpoint at `experiments/output/brightness_v1/brightness_v1.pt`
  (790 KB, gitignored; `python -m experiments.train_brightness`
  reproduces it, reusing the cached pool and holdout split).
  `tests/test_brightness_model.py` (4 tests) covers grid/label/mask
  mechanics, FiLM conditioning, save/load, and low-end preservation
  through the hook.

## Brightness v2: the knob now delivers (G-2) — this commit

- v1's weakness was diagnosed correctly: conditioning labels on real
  sounds rarely exceed ~0.1, so targets like 0.6 extrapolated timidly.
  v2 fixes it in the *data*, not the model: during training the hidden
  bands of each target are brightness-tilted (τ log-uniform in
  [0.25, 8], spectral slope in [−0.5, 1], 30 % of batches left untilted
  to keep pure reconstruction in distribution), and the condition label
  is measured from the tilted target. The visible bands stay identical —
  at synthesis the low end is preserved, so only hidden bands can
  deliver brightness, and the training pairs now respect that.
- **Same eval, same held-out clips**: reconstruction *improved* to
  **0.0024** (v1 0.0027; deterministic 0.0266; floor 0.0039) — the
  augmentation cost nothing. The sweep 0.05→0.6 now delivers measured
  output brightness **0.0022 → 0.110**: a 50× dynamic range (v1: 1.3e-4
  → 2.2e-4), monotone on 100 % of sweeps. The delivered ceiling ~0.11
  is roughly the brightness of the brightest real library material —
  i.e. "0.6" means *as bright as bright real sounds get*, bounded by the
  18 dB per-band gain cap and the untouched low end in the denominator.
- Training: 10k steps ≈ 13.5 min on the 1050 Ti (resumable — optimizer
  state checkpointed every 500 steps); val-high-L1 plateaus ~0.009
  around 7k steps on the harder tilted task. Checkpoint:
  `experiments/output/brightness_v1/brightness_v2.pt`.
- Recipe now proven end-to-end for the axis family (G-8 transient, G-9
  tonality): free supervision in the envelope domain, FiLM conditioning
  on the measured axis, deterministic carrier at synthesis, and
  label-range augmentation so the knob's full travel is trainable.

## Tonality axis: measuring beat learning, three times (G-9) — this commit

- Goal: a *calibrated* harmonic↔noisy knob — request harmonicity 0.6,
  measure 0.6 coming out. Eval: 12 held-out real clips × 5 targets,
  mean |achieved − requested|, against a per-clip 41-point oracle grid
  search over the deterministic scalar remix (the best any scalar can
  do; its own 0.167 residual is *reachability* — the HPSS-remix family's
  median response span on real clips is 0.315 — not calibration error).
- **Three learned attempts, all honestly worse**:
  1. free [2, 64] per-band gain field → **0.230**: the label→gains
     inverse is many-to-one, MSE regresses to the conditional mean of
     every profile sharing a label, and the effective gains collapse
     toward "do nothing" (sweep span 0.49→0.63);
  2. 4-param base+slope family → **0.294**: well-posed and monotone
     (1.0) but the ±31 dB log-gain cap cannot *remove* a layer, which
     the oracle's a=±1 can — shorter reach, worse score;
  3. scalar-amount head driving exactly the oracle's family → **0.249**:
     reach now identical, so the gap is pure prediction error — with
     419 training clips a 614k-param net memorizes (train MSE 0.04) and
     cannot generalize the response curve from envelopes (val 0.21).
- **The deterministic answer wins**: the response curve is nearly free
  to *measure* — h and p are already computed, so five anchor remixes
  cost five autocorrelations (~50 ms). `imagine_tonality(...,
  harmonicity=target)` measures the 5-point curve, monotone-stabilizes
  it (cummax), and inverts: **0.172 vs oracle 0.167**, monotone on
  100 % of sweeps, zero weights. Lesson for the axis family: learn only
  what cannot be measured at inference time. Brightness (G-2) qualifies
  — the missing highs cannot be measured, only imagined. Scalar tonality
  does not.
- `TonalityNet` (envelopes + FiLM → remix), `layer_envelopes`,
  `remix_layers` (unit gains reproduce the input to 1e-6 — exact-sum
  HPSS through the analysis grid) stay in the library as the substrate
  for a frequency-selective v2 (air = percussive highs, body = harmonic
  mids), which direct measurement *cannot* do cheaply — that is where
  learning re-enters, with a bigger corpus. `tests/test_tonality_model.py`
  (4 tests). Experiment: `experiments/train_tonality.py` (resumable
  rendered-label builds; curves cached).

## Transient axis: the pad gets a pluck (G-8 v1) — this commit

- The family rule predicted learning would pay here, and it did: a
  *missing* attack cannot be measured at inference, only imagined.
  Design mirrors G-2 in time instead of frequency — the carrier is
  deterministic (each band's attack is the band's **own body** segment
  copied to the onset, so the imagined attack inherits the sound's
  timbre and stays phase-coherent with what follows), and the model
  predicts only energy: per-band gain-over-body and exponential decay
  τ, FiLM-conditioned on a sharpness target. `amount=0` is bit-exact;
  the no-model fallback is a generic HF-tilted pluck (τ ∝ f^−0.4).
- **Free supervision again**: suppress the attacks of sounds that have
  them (150 ms onset ramp) and *fit* (gain, τ) per band on the
  originals — measured labels, no annotation, and a well-posed target
  (a direct function of the clip — no G-9-style inverse ambiguity).
  495 rows (280 synthetic plucks/saws/drums + 215 real one-shots from
  the percussive folders, filtered to measured sharpness > 0.25);
  build 77 s, training 50 s on the 1050 Ti (best val at ~2k steps).
- **Held-out sharpness recovery** |measured − original|: **model
  0.049** vs deterministic generic pluck 0.076 vs do-nothing 0.176 —
  the learned attack restores the original attack character 1.5× better
  than the generic pluck, 3.6× better than leaving the sound suppressed.
  The sharpness knob is monotone on 100 % of sweeps with real travel
  (measured 0.60 → 0.82 for requests 0.1 → 0.9).
- **Metric-design lesson, kept**: whole-attack-window band-envelope L1
  could not tell the methods apart (all within 2.5 %) — an attack is a
  brief, mostly-HF event drowned by body mass in any bulk T-F distance.
  Score an axis by *its own measured label*; bulk distances are for
  reconstruction tasks. Checkpoint:
  `experiments/output/transient_v1/transient_v1.pt`.
  `tests/test_transient_model.py` (4 tests: onset/labels, deterministic
  pad→pluck raises measured sharpness with the tail untouched, amount=0
  bit-exact, net bounds/FiLM/save-load).

## Event detection, small scale: drums solved, notes need a new shape (V-4) — this commit

- Pipeline proof on the 1050 Ti: Faster R-CNN (MobileNetV3-FPN, 19M
  params, ImageNet backbone) trained on D-6's `SceneDetectionDataset`
  (2400 cached scenes; rendering is the bottleneck at ~0.7 s/scene —
  cached once, resumable), 12 epochs ≈ 9 min. Eval on 150 disjoint-seed
  scenes with **transcription-style matching**: pitched events on onset
  ±50 ms + pitch-center ±1.5 semitones (mir_eval's spirit — the 128-row
  grid quantizes pitch to ~0.8 st/px, so 50-cent tolerance would be
  sub-pixel), drums on IoU ≥ 0.5 (their boxes span octaves; IoU is fair).
- **Percussion is at gate level**: kick F1 1.00, snare 0.987, hat 0.892;
  median onset error **1.8 ms** (sub-frame — timing is essentially
  solved). Overall F1 0.792.
- **Pitched notes are the honest gap** (saw 0.28, pluck 0.24): precision
  0.62–0.71 but recall 0.15–0.18 — when the detector fires it is usually
  right, but it misses ~85 % of notes. Four measured runs isolate why:
  1. stock anchors + raw ±1.5 st boxes (~4 px tall) + IoU eval → notes
     0.03: no 32-px anchor can be assigned to or scored against a 4 px
     box;
  2. tiny anchors (4–64 px) → notes unchanged (0.01), drums *degraded*
     (kick 1.0 → 0.74), and the lowest training loss of all runs —
     training loss is a poor proxy here;
  3. mid anchors + a ±5 px symmetric apron on note boxes (pitch decodes
     from the center, which padding preserves) + transcription eval →
     notes improve (saw 0.18) but drums still degraded;
  4. stock anchors + apron + transcription eval + LR decay → best of
     both regimes (this run's numbers above), notes recall-bound.
  Conclusion: the *anchor-box formulation* is wrong for thin harmonic
  combs, not the data or the metric — a note is a 4 px fundamental with
  harmonics the box does not cover. The DETR/keypoint formulation the
  roadmap always planned for scale time is the actual fix; anchors were
  worth trying because they were cheap, and the dead end is now measured.
- Side note: `random_scene`'s default pitched palette is (saw, pluck) —
  the sine class has zero support in this eval. The user will provide
  real audio datasets for the next round of training experiments
  (recorded in the roadmap); synthetic scenes stay the exactness
  baseline.
