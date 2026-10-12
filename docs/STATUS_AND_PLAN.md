# Status & Forward Plan — wavelet2vec + Viktor Rose Editor

**2026-09-27, v0.37.0:** [Reusable study workflows](REUSABLE_STUDIES.md)
add caching, shared preparation, resume and profiling. On local measurements,
event matching improves 1.21× and grouped decoder updates improve 1.48×.
These are speed/correctness results, not new separation-quality claims.

**2026-09-23, v0.36.0:** [First real drum dataset](DRUM_DATASET_FINDINGS.md)
now has event/transform measurements, three trained separator variants,
per-source development losses and listening exports. The actual input is
44.1 kHz/16-bit. Master-processed mixtures are non-additive; all files share
one performance. Finer wavelets help, but clean eight-way separation is not
established. An onset/pitch measurement confound was fixed and tested.

**2026-09-22, v0.35.0:** [Portable inference bundles](INFERENCE_BUNDLES.md)
add architecture reconstruction, checked non-pickle arrays, source/render
metadata and a native float file-render CLI. Training runs can export with
their update/data provenance; exact training resume remains a separate format.

**2026-09-22, v0.34.0:** [Bounded source rendering](BOUNDED_RENDERING.md)
adds native 192 kHz file-to-file processing with rolling buffers, cancellation,
staged publication and large-file RF64 support. Generated signals validate
correctness; no new separation-quality or editor UI claim is made.

**2026-09-21, v0.33.0:** [Verified multitrack training](SUPERVISED_MULTITRACK.md)
implements aligned-source manifests/crops, supervised accumulation, checkpoint
resume and held-out group/source reports. Native 192 kHz float32 is included
in generated-fixture checks. No real-music separation-quality claim is made.

**192 kHz clarification, 2026-09-21:** Native stereo through 192 kHz with
32-bit float input/output is a required target for future models and
rendering. [The format contract](HIGH_RESOLUTION_AUDIO.md) records expanded
synthesis and float WAV/corpus round-trip tests; perceptual qualification
remains separate.

**2026-09-21, v0.32.0:** [Trainable source synthesis](SOURCE_SYNTHESIS.md)
connects hybrid/context features to source waveforms, with optional waveform
residuals, coefficient consistency projection and offline overlap-add. The
synthetic supervised example learns; real-music fidelity remains unproven.

**2026-09-18, v0.31.0:** [Verified real-audio corpora](REAL_AUDIO_PRETRAINING.md) add split/group and PCM-duplicate guards, reproducible crops, corpus-aware resume and stronger diagnostic baselines. Two existing loops passed ingestion/forward checks; no held-out music-quality claim was made.

**2026-09-16, v0.30.0:** [Frozen target calibration](TARGET_CALIBRATION.md) adds training-only target scaling. The matched three-seed toy mean improved about 8.9%, but only one run beat a constant baseline; calibration remains optional and semantic music quality unproven.

**2026-09-15, v0.29.0:** [Learned context and masked acoustic pretraining](SELF_SUPERVISED_MUSIC.md) are implemented, including isolated student/teacher paths and checkpoint resume. The initial synthetic learner reduced loss but did not beat a training-mean constant target; useful music semantics remain unproven.

**2026-09-14, v0.28.0:** [Stereo/multi-resolution adapters, source training utilities and feature caches](ADAPTERS_AND_TRAINING.md) are implemented. [The wav2vec/music-model comparison](MUSIC_REPRESENTATION_GAPS.md) identifies the main remaining gap: learned temporal context and a leakage-tested self-supervised music objective.

**2026-09-13, v0.27.0:** [Long-audio extraction](LONG_AUDIO.md) adds overlapping context windows with unique token ownership, global phase/time references, bounded reader/file inference and duration-weighted coarse summaries. [Feature priorities](FEATURE_PRIORITIES.md) focus further work on stereo, multiple resolutions, training/data contracts and cache provenance. Musical understanding remains a future evaluation target.

**2026-09-07, v0.26.0:** [Compact wavelet carriers](COMPACT_WAVELETS.md) now retain phase on per-band sample grids, with grouped FFTs and hybrid training support. The measured grouped setting reduced coefficient storage 10.4x versus dense wavelets; the STFT comparator remained cheaper. Real-music separation superiority is unproven.

**2026-09-06, v0.25.0:** The new [hybrid audio foundation](HYBRID_ENCODING.md) retains raw waveform and full complex carriers, with lossy trainable tokens, a defined inverse and synthetic ablations. Earlier entries below are historical; pooled vision images are not lossless, and hybrid superiority is not established.

*Updated 2026-07-11. This document is mirrored in `wavelet2vec/docs/` and
`viktor-rose-editor/docs/`. Detail lives in `wavelet2vec/docs/ROADMAP.md`
(epic-level ticks), `FINDINGS.md` (measured results), and
`viktor-rose-editor/docs/DEVELOPMENT_PLAN.md` (editor phases).*

---

## 1. Where wavelet2vec stands — v0.10.0, 116 tests passing

**The entire deterministic CPU tier of the roadmap is complete**
(C-1, M-1..M-6, R-1..R-8, H-1..H-4, P-1, G-1, G-7). By capability area:

| Area | Shipped | Headline numbers |
| --- | --- | --- |
| Signal core | Invertible constant-Q complex frame; lossless audio↔image; save/load with fidelity tiers | ~300 dB exact inversion; float32 TIFF ~160 dB; 16-bit PNG ~84 dB |
| Embedding | 570-dim (spectral/modulation/transient/harmonic/phase/stereo/conv) + similarity | shift-invariant phase features; waveshape signature separates saw vs scrambled |
| Character & style | extract/apply character, multi-reference `blend`, `style_transfer`, ADSR (volume + character), FL automation (.flp/.fst), `perform` DAW round-trip | sample-exact round-trip at native rate; `amount=0` is a true identity (edge-damping bug fixed in 0.8.0) |
| Multichannel | `channel_mode`: shared / per_channel / mid_side on all character ops | mid_side: mono downmix == morphed Mid, exactly |
| Decomposition | HPSS, transient split/shaping, T-F region attenuation | exact-sum splits; unity gains bit-exact |
| Rhythm | registers/grid/swing/microtiming/pattern/syncopation; `estimate_tempo`; `apply_groove` + `swing_offsets`/`groove_offsets`; 89-dim `rhythm_embedding` | tempo 75 % exact on rhythmic loops (bar-aware comb); groove = swing + microtiming residual; cross-tempo groove sim > 0.9 |
| Harmony | `track_f0` + `segment_notes`; `chromagram`/`estimate_key`/`track_chords` | pitch class 53/53 on sung notes, 8/10 bass; loop key ~46 % (honest KS ceiling) |
| Structure | `segment_structure` (embedding novelty); `find_loop_points` + `render_loop` | rendered loop seam spectrally inaudible in 10–11/12 real loops with bar hint |
| Spatial | stereo phase spectrum (editor layer data), N-channel matrix, `SpatialCharacter` extract/apply, `design_width` | Side-only ops keep mono at 303 dB; coherence lands at predicted 1−2·width |
| Datasets | axis labels, lowpass pairs, filter/distortion/noise/attack ladders (monotone-verified) | ladders supervise their own knobs |

Since the last push, six feature commits (v0.5 → v0.10) are local-only:
R-6/R-8, R-7, M-4/M-5, M-6 + smoother fix, H-1/H-2, H-3/H-4.

**Known honest limits** (documented in FINDINGS, drive the plan below):
tempo on weak-beat FX material (confidence flags it); full-key detection on
pad-heavy synth harmony; structure recall on same-genre joins; f0 below
~55 Hz.

## 2. Where the editor stands — 70 tests passing

| Phase | Status |
| --- | --- |
| 0 — Proof shell | ✅ open/draw/play/zoom |
| 1 — Precision editor | ✅ selection arithmetic, full keyboard map, time formats, markers/regions + WAV cue round-trip, undo history, channel select/lock, themes, Pencil |
| 2 — Rust `vre-core` | ⬜ next big rock (cpal engine, symphonia, parallel peaks, scrub) |
| 3 — Spectral & stereo views | ✅ scalogram layers, RX-style see-through blend slider, spectral selection + lossless attenuate, Stereo Phase layer, async compute |
| 4 — MIDI | ◐ MIDI 1 SMF + piano roll + MIDI-2-native model done; UMP (Rust) pending |
| 5 — AI & creativity | ◐ AI Tools dock (character/transients/tonality) + Library dock (similarity search, cache) done; more below |
| 6 — Format workbench | ⬜ coefficient-image import/export panel, hybrid audio+MIDI container |

## 3. Plan — next waves, in order

### Wave 1 — Ship & surface (highest value per effort)
1. **Push wavelet2vec to GitHub** (`git push -u origin main`), tag v0.10.0,
   refresh README: rhythm/harmony/spatial/structure sections + figures.
2. **Editor: analysis metadata in the Library** — background scan writes a
   sidecar per file: BPM (+ confidence), key (KS for loops, f0-derived pitch
   for one-shots — the FINDINGS rule), loop-ability (periodicity score).
   Columns + filters in the Library dock.
3. **Editor: three new AI Tools panels**, all thin wrappers over v0.10 API:
   - *Groove* — analyze a reference loop's groove, apply to selection
     (`groove_offsets` → `apply_groove`, strength slider);
   - *Width* — per-band width curve + coherence-floor guard
     (`design_width`, `SpatialCharacter` match to reference);
   - *Loop finder* — `find_loop_points` (bar hint from BPM field) +
     `render_loop` preview + "set as loop region".
   Character panel gains a `channel_mode` selector (default mid_side).
4. **G-11 / I-4** (wavelet2vec): expression automation from deterministic
   axes — render an `AutomationCurve` from measured brightness/tonality/
   transient trajectories so a DAW clip can *follow* the analysis.

### Wave 2 — Engine & scale
5. **Phase 2 Rust `vre-core`**: cpal playback + ring buffer + transport
   clock, symphonia decode, parallel peak pyramids, maturin wheels in CI;
   Python engine stays as fallback behind `EngineAPI`. Milestone: 1 GB file,
   glitch-free playback while painting spectral layers.
6. **C-3 GPU backend** (wavelet2vec): torch FFT path for `analyze`/
   `synthesize` and the scalogram — batch embedding of a sample library and
   real-time-ish editor layer recompute.
7. **MIDI 2 UMP** parser in Rust (Phase 4 part 2), downconvert to MIDI 1;
   the internal model is already MIDI-2-native.

### Wave 3 — Learned models & formats
8. **V-1..V-3**: dataset renders of coefficient images + linear probes;
   then **G-2/G-8/G-9** learned imagination axes (filter/distortion/attack
   ladders from G-1 are the supervision). Upgrade path for key detection
   and same-genre structure boundaries lands here too.
9. **Phase 6 format workbench**: coefficient-image import/export panel with
   round-trip SNR validator; hybrid audio+MIDI container draft (Rust
   reference reader) on top of the MIDI-2-native model.
10. **Stems** (P-3/V-5 masks → stems) and per-stem editor tools.

### Standing rules (unchanged)
- Improvements land in wavelet2vec first; the editor consumes releases.
- Every feature ships with measured validation on `example_audio` and an
  honest FINDINGS entry — negative results included.
- Deterministic before learned; CPU before GPU; short fragments are the
  compute unit.
