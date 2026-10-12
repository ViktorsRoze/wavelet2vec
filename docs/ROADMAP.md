# wavelet2vec Advanced Roadmap — a framework for musical understanding & AI sound design

**v0.37.0:** [Reusable studies](REUSABLE_STUDIES.md) now automate the named
solo/mix workflow and provide profiling. [Measured batching improvements and
native-code options](PERFORMANCE_STRATEGY.md) guide further speed work without
replacing tested audio/training foundations merely to change language.

**v0.36.0:** [The first real drum experiment](DRUM_DATASET_FINDINGS.md)
exposes the next concrete gaps: quiet-source recovery, false events from
leakage, robust transformation estimation and richer source heads. Existing
data verifies the pipeline; new independent patterns and annotations will
be needed to establish generalization and detector accuracy.

**v0.35.0:** [Inference bundles](INFERENCE_BUNDLES.md) implement portable
separator artifacts and command-line rendering. Continue foundational work
on physical time/pitch controls and numerical diagnostics before task-data
training and musical-quality evaluation.

**v0.34.0:** [Bounded source rendering](BOUNDED_RENDERING.md) now supplies
reader/file inference with checked float exports. Continue dataset-independent
work on time/pitch-control interfaces, boundary diagnostics and portable
inference bundles before claiming trained musical capabilities.

**v0.33.0:** [Verified multitrack training](SUPERVISED_MULTITRACK.md) supplies
the aligned-stem loader and supervised training/resume recipe. Next: curated
multitracks and full-song quality/resource evaluation through 192 kHz,
followed by stronger separation and controlled synthesis models.

**v0.32.0:** [Source synthesis](SOURCE_SYNTHESIS.md) implements the first
trainable source decoder and offline renderer. Next: verified aligned
multitrack data, a supervised training/checkpoint recipe and held-out musical
evaluation before extending the renderer to time/pitch-controlled generation.

Goal: grow wavelet2vec from an analysis/transfer library into a **layered
framework**: lossless signal core → descriptors → musical understanding
(events, rhythm, harmony, structure, sources) → controllable generation →
the Viktor Rose editor as the front end.

The foundation now distinguishes signal preservation from learned features.

1. **Full complex carriers preserve audio; pooled images do not.** The older
   `coeff_image`, `sequence_tokens` and `WaveletFrontend` discard information.
   Their vision models do not see every sample or phase relationship.
2. **v0.25.0 adds a waveform/spectral research foundation.**
   `HybridAudioFrontend` retains native-rate multichannel audio and complex
   wavelet or STFT coefficients with explicit inverse metadata.
   `HybridTokenEncoder` learns lossy aligned features while keeping both
   carriers. Independent Q/density, zero padding, conditioning validation,
   memory guards, phase-feature ablations and CPU/CUDA tests are implemented.
3. **Synthetic supervision is useful but limited.** Source waveforms and
   production-chain metadata are needed for separation/restoration; image
   boxes and simple rendered instrument classes are insufficient by themselves.

See [HYBRID_ENCODING.md](HYBRID_ENCODING.md) for the implemented API and the
first supervised experiment. On a small synthetic task, hybrid fusion and
signed phase context did not consistently beat the alternatives. A best
encoding or novel separation result has not been established.

Short fragments remain the prototype unit, but memory does not become free:
full-rate wavelet coefficients scale with batch, channels, bands and duration,
including padding and autograd intermediates. Efficient multirate/sliced
analysis is a core model requirement. Local high-resolution carriers should
be combined with beat/phrase context for long material.

Current research sequence:

- R0 implemented: versioned hybrid carriers, differentiable synthesis,
  phase-preserving context features, trainable fusion and ablation harness.
- R1 started: synthetic supervised pipeline/ablations; real stereo,
  source-group-held-out and compute-matched comparisons remain open.
- R2 in progress, v0.26.0: compact multirate carriers, grouped FFTs,
  physical band grids, hybrid training and footprint benchmarks implemented;
  see [COMPACT_WAVELETS.md](COMPACT_WAVELETS.md). Whole-fragment processing
  remains; streaming/valid-context metadata and real-source evaluation are open.
- R3 later: instrument/instance hierarchy, repeated-loop context and separate
  in-mix extraction versus dry-source restoration objectives.

v0.27.0 adds [sliding-window extraction](LONG_AUDIO.md), global phase/time
coordinates, explicit real-context availability and coarse duration-weighted
summaries. This bounds per-window inference work and exposes local carriers
for reconstruction; it does not establish whole-track-transform equivalence,
true filter-support validity, seamless predicted edits or learned musical
understanding. [FEATURE_PRIORITIES.md](FEATURE_PRIORITIES.md) orders further
stereo, multi-resolution, training/data and cache work.

The epic entries below record the historical implementation trajectory;
this foundation contract supersedes earlier blanket lossless-image claims.

v0.31.0 adds the [real-audio corpus pipeline](REAL_AUDIO_PRETRAINING.md):
verified grouped manifests, deterministic native-rate crops, data-cursor resume,
training-only statistics and group-aware evaluation against stronger baselines.
Existing loops passed ingestion/forward smoke checks; their unknown provenance
was not promoted to an independent validation split. Semantic evaluation is open.

v0.30.0 adds optional [frozen target calibration](TARGET_CALIBRATION.md),
with training-only statistics, checkpointed target units and a paired
three-seed experiment. It remains disabled by default: only one calibrated
run beat a constant baseline. Real-music objective/evaluation work remains open.

v0.29.0 implements the initial [learned context and masked pretraining
path](SELF_SUPERVISED_MUSIC.md): local/global Transformers, corrupted-student
views, detached fixed targets, a seeded span sampler, validity-aware batches,
variance diagnostics, length bucketing and a resumable reference trainer.
The first synthetic run did not beat a constant target baseline. Real-music
target design and evaluation, learned target alternatives, and large-scale
training remain open; numerical success is not semantic validation.

v0.28.0 adds [stereo and multi-resolution adapters, source losses and feature
caches](ADAPTERS_AND_TRAINING.md). The most important remaining representation
gap is now a learned temporal context model plus masked self-supervised
training, not additional signal views alone. The
[wav2vec/music-model comparison](MUSIC_REPRESENTATION_GAPS.md) specifies those
gaps and the need to prevent clean-target leakage through hybrid branches.

---

## Layer map

```
L4  EDITOR INTEGRATION      panels, workers, model zoo
L3  GENERATION              HF imagination · neural character v2 · groove
                            transfer · spatial designer · variations (research)
L2  UNDERSTANDING           events/notes (vision) · rhythm/groove · harmony ·
                            structure · stems · roles/tags
L1  DESCRIPTORS             embedding v1 (570-d) · rhythm section · spatial
                            profiles · embedding v2 (distilled)
L0  SIGNAL CORE             invertible CQT frame · multichannel · GPU backend
```

---

## Epic C — Signal core hardening (L0)

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| C-1 ✅ | Promote `experiments/invertible.py` into `wavelet2vec.frame` (public API: `analyze/synthesize/save/load`) | S `det` | prerequisite for everything below |
| C-2 | Streaming/tiled CQT for long files (overlapping tiles, exact stitch) | M `det` | *deprioritized by the short-fragments principle*: editor convenience, not a framework prerequisite |
| C-3 ✅ | **GPU backend** (`torch_frame`): `analyze_torch`/`synthesize_torch`/`analyze_batch`, batched FFT-domain filtering, `device`, `precision` double/single, numpy default untouched, FrameSpec interchangeable both ways | M GPU | 9× on CPU vs the numpy loop (12→104× realtime, 256 frags); double = machine-precision parity, single round-trips 131 dB; CUDA batches the same call |
| C-4 | Benchmark suite: CPU vs GPU, tile sizes, memory ceilings | S | extends the experiment harness |
| C-5 | Half-precision + `torch.compile` inference paths for all models | S GPU | after first models land |

## Epic M — Multichannel (L0/L1)

Today only stereo pairs are analyzed (channels 1–2); `perform`/character use
mono-derived gains applied to all channels.

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| M-1 | N-channel ingestion: per-channel scalograms (lazy), channel-role map (L/R/C/LFE/Ls/Rs… or generic 1..N) | M `det` | 5.1/7.1/ambisonic-ready layout metadata |
| M-2 ✅ | Generalized spatial section: pairwise per-band coherence/IPD/balance matrix for N channels; stereo stays the 2-ch special case | M `det` | embedding gains `spatial` v2 |
| M-3 ✅ | **Stereo Phase Spectrum API**: per-band inter-channel phase difference + coherence *over time* (not just pooled) → the editor view's data source | S `det` | feeds Viktor Rose editor Phase 3 |
| M-4 ✅ | Spatial character: extract/apply width-vs-frequency profile (per-band M/S ratio matching) + per-band balance gain curves | M `det` | real-loop transfer closes profile gap 0.145→0.033; mono downmix untouched (303 dB) |
| M-5 ✅ | Phase-safe width designer: per-band delayed-Mid decorrelation with mono-compatibility guard (coherence floor per band) | M `det` | coherence lands at 1−2·width as predicted; bass stays coherent by physics |
| M-6 ✅ | Multichannel `perform`/`blend`: `channel_mode` shared / per_channel / mid_side with shared timing | M `det` | mid_side downmix invariant: mono of result == morphed Mid, exactly |

## Epic R — Rhythm, groove, tempo (L2) `det`

(From the v1 roadmap, broken into tasks.)

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| R-1 ✅ | `rhythm.py`: band-grouped onset curves (low/mid/high registers) at 5 ms frames | S | reuses scalogram flux |
| R-2 ✅ | Grid quantization with given BPM: onset↔slot assignment + deviations (ms) | S | folders already carry BPM |
| R-3 ✅ | Swing % per register (off-8th/off-16th position; 50 = straight, 66 = triplet) | S | verify on straight synth loops ≈ 50 % |
| R-4 ✅ | Microtiming profile (push/drag per slot, mean ± std) + syncopation score | S | the groove fingerprint |
| R-5 ✅ | Pattern matrix (registers × 16 slots per bar) + bar-to-bar diffing | S | drum-structure display in editor |
| R-6 ✅ | **Tempo estimation when BPM absent** (onset autocorrelation + bar-aware comb), meter guess | M | 75 % exact on rhythmic loop folders; weak-beat FX material stays hard — DAW-supplied BPM remains first choice |
| R-7 ✅ | `rhythm` embedding section (tempo-normalized) + similarity | S | 89-dim grid-relative groove vector; cross-tempo same-groove sim > 0.9 |
| R-8 ✅ | **Groove transfer**: apply profile A to loop B via 16th-slice re-timing (`apply_groove`) | M | flagship creative payoff; groove = swing + microtiming residual |

## Epic V — Vision models on coefficient images (L2) GPU

The user's DINO/Co-DETR direction. Two distinct "DINO"s, both used:
**DINOv2** (self-supervised ViT features) for representation; **DINO/Co-DETR**
(detection transformers) for events. Input encoding for all vision tasks:
3-channel tiles `[log-magnitude, IF-deviation, energy-weighted coherence]`
(wrapped phase needs careful parameterization; IF is the GANSynth lesson — we already compute it).
Short-fragments principle: **one snippet = one fixed-size image** (128 bands ×
a fixed frame count, e.g. 2 s time-pooled) — no tiling, the model always sees
the complete musical object.

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| V-1 ✅ | `vision.py`: fixed-size 3-channel image `[log-mag, IF-deviation, energy-weighted coherence]`, Event↔box mapping, `CocoBuilder`/`export_dataset` | M `det` | one snippet = one 128×128×3 image; exact time/Hz↔pixel round-trip; ground-truth boxes sit 3× above image-mean energy |
| V-2 ✅ | `scenes.py`: `render_scene`/`random_scene` (sine/saw/pluck + kick/snare/hat, exact onsets/pitches/labels) + `weak_labels` (f0 notes + rhythm-register drums on real audio) | M `det` | our unfair advantage; unlimited labeled data, deterministic |
| V-3 ✅ | **Frozen-backbone probes** (`experiments/frozen_probe.py`): DINOv2 ViT-S/14 + Swin-T on coefficient tiles vs embedding v1, D-7 harness + 5-fold linear probe on the real library | M GPU | **GO**: frozen DINOv2 hits purity@1 0.888 vs v1's 0.916 with *zero* audio training (Swin-T 0.860); mag-only ablation drops to 0.854 and collapses on kicks (0.875→0.375) — the IF/coherence channels carry the percussive discrimination. Frozen does **not** beat v1 (gate 4 needs a fine-tune), but −3 pts from generic features is a strong case for V-4; extraction is 14 ms/tile on the 1050 Ti |
| V-4 🟡 | **Event detection** (`experiments/train_detector.py`): Faster R-CNN mobilenet-FPN on D-6 scenes (small-scale pipeline proof; DETR-family at scale time as planned), transcription-style eval (notes: onset ±50 ms + pitch ±1.5 st; drums: IoU 0.5) | L GPU | **Percussion at gate level**: kick F1 1.00, snare 0.987, hat 0.892, median onset error 1.8 ms; overall 0.792. **Pitched notes are the documented gap** (saw 0.28/pluck 0.24 — precision 0.6–0.7, recall 0.15–0.18: finds only easy notes): four measured runs show the anchor-box formulation, not the pipeline, is wrong for ~4 px-tall harmonic combs (tiny anchors made drums worse without fixing notes). Gate 4 met for percussion; notes wait for the DETR/keypoint formulation + user datasets |
| V-4b | **Scale-up groundwork (decided direction)**: move event detection to **MMDetection DINO with a Swin-L backbone** — not just MobileNet-class models. Groundwork proceeds **now, on CPU where necessary** (slow is acceptable: config, data adapters from D-6/COCO export, inference paths, eval harness), so the swap is ready when the hardware is | L GPU-later | this box gets a **larger-GPU upgrade**, and the library/package will run on **many machines with stronger hardware** — build for that target, prove on this one |
| V-5 | Instance segmentation of source regions → **masks on complex coefficients invert to audio**: detection→separation bridge, lossless outside masks | L GPU | stems research; requires source-waveform and phase-aware validation |
| V-6 | Self-supervised DINOv2-style pretraining on a large unlabeled coefficient-image corpus (sample libraries) → audio-native backbone; distill into **embedding v2** | L GPU | research-tier; only after V-3/V-4 prove the domain |

## Epic H — Harmony, melody, structure (L2)

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| H-1 ✅ | Frame-wise f0 track (YIN CMND + Viterbi over semitone states) → note segmentation | M `det` | pitch-class ID: 100 % on 53 sung notes, 80 % bass, 65 % stab roots |
| H-2 ✅ | Chord/key estimation: semitone-aligned chroma, KS profiles, triad templates + Viterbi | M `det` | tonic 100 % on single notes; full-key ~46 % on synth loops (honest KS ceiling) |
| H-3 ✅ | Structure segmentation: Foote novelty on windowed-embedding self-similarity, adaptive threshold | M `det` | strong on distinct content; same-genre loop joins are inherently low-contrast |
| H-4 ✅ | Loop-point detection + `render_loop` (equal-power seam crossfade) | M `det` | seam inaudible (flux) in 10-11/12 real loops with bar-length hint; free-running phase makes splice-only seams impossible |

## Epic P — Stems & sources (L2)

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| P-1 ✅ | HPSS in coefficient domain (median masks time vs frequency on complex cells, exact inversion) | S `det` | drums/tonal split; feeds R-epic |
| P-2 | Demucs bridge (`[separation]` extra): stems → per-stem embeddings/characters/grooves | S GPU-opt | pragmatic SOTA now |
| P-3 | Mask-based separation via V-5 detections | L GPU | our own path, later |

## Epic E — Lossless spectral & phase editing (L1) `det`

Grew from the Viktor Rose editor's needs; all in the invertible coefficient
domain, so edits are exact outside the wavelet apron.

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| E-1 ✅ | `attenuate_region`: scale a T-F rectangle by a gain (down to erase), raised-cosine aprons | S `det` | >40 dB depth, <1 dB collateral, <1e-6 outside the apron |
| E-2 ✅ | Region toolkit: `region_mask` (polygon/brush raster), `isolate_regions` (exact inside/outside split), `process_regions` (any effect on region content only) | S `det` | `inside + outside == input`; outside preserved by linearity |
| E-3 ✅ | Phase editing: `phase_rotate_regions` (180° flips region sign, 360° identity), `phase_scramble_regions` (keep color, dissolve timing) | S `det` | full ±π scramble costs a few dB to redundant-frame cancellation — inherent |

## Epic G — Generation: the imagination axes (L3)

Organizing idea: **imagination axes**. Brightness (how much high-frequency
content), **tonality** (harmonic ↔ noisy), and **transient** (how much
attack) are three controllable axes of the *same* conditional model family:
U-Net over the coefficient image, FiLM-conditioned on the target value,
trained on graded self-supervised ladders, phase via the harmonic phase
signature + shaped noise, deterministic fallback always available. Each axis
ships as a specialist first; they converge into one multi-axis model (G-10)
once individually proven. Every axis label is **free**: brightness =
HF-energy ratio, tonality = the harmonicity feature we already compute,
transient = onset-flux/crest — the embedding measures what the models learn
to control.

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| G-1 ✅ | Dataset tooling: filtered pairs, same-patch filter-cutoff pairs (`name__cut30/70.wav`), sine→distortion ladders, **noise-mix ladders** (same patch, rising noise osc / breathiness), **attack-knob ladders** (same patch, amp-attack 0 ms → 200 ms), attack-removed pairs (fade the first 30–80 ms → original) | M `det` | graded ladders supervise every control knob directly |
| G-2a ✅ | **Brightness axis, deterministic** (`imagine_highs`): spectral band replication — heterodyne mid content up, −6 dB/oct rolloff, high-passed onto the original | S `det` | brightness rises monotonically 0→0.74 on a dull saw; amount 0 bit-exact; the no-model fallback + baseline for G-2 |
| G-2 ✅v2 | **Brightness axis (HF imagination), learned** (`brightness_model` + `experiments/train_brightness.py`): FiLM-conditioned 196k-param net predicts high-band envelopes; carrier/phase stay deterministic (heterodyne), low end preserved; `imagine_highs(..., model=…, brightness=…)` | L GPU | **First learned axis shipped.** Held-out real clips, log-env L1 above 4 kHz: model **0.0024** vs deterministic G-2a 0.0266 (11×) and the do-nothing floor 0.0039 (G-2a > floor is *expected* — it's an exaggeration knob, not a reconstructor). v1's caveat (knob monotone but delivery stuck at ~2e-4) fixed in v2 by brightness-tilting the *hidden* bands during training so conditioning labels cover the knob range: sweep 0.05→0.6 now delivers measured brightness **0.0022→0.110** (50× span, monotone on 100 % of sweeps; ceiling ≈ the brightest real material, with the 18 dB gain cap and the preserved low end bounding it). 10k steps ≈ 13.5 min on the 1050 Ti |
| G-7 ✅ | **Transient shaping `det`**: transient/sustain split in the coefficient domain (onset-flux-gated transient layer vs. residual, exact inversion) + multiband transient designer — attack gain and sustain gain per band group | M `det` | the classic tool, but per-band and lossless outside the edit; also the no-model fallback for G-8 |
| G-8 ✅v1 | **Transient axis (creation)**: `imagine_transient(audio, sr, amount, sharpness=…, model=…)` — pad gets a pluck. Carrier = each band's own body copied to the onset (timbre-inherited, phase-coherent); `TransientNet` (647k) predicts per-band (gain-over-body, decay τ) conditioned on sharpness; deterministic generic pluck as the model-free fallback; free supervision = suppress real attacks, fit labels on the originals | L GPU | **Learning pays here** (the family rule: a missing attack cannot be measured): held-out sharpness recovery **model 0.049** vs deterministic pluck 0.076 vs do-nothing 0.176; knob monotone on 100 % of sweeps with real travel (measured 0.60→0.82 for requests 0.1→0.9). Metric lesson recorded: whole-window envelope L1 cannot see an attack (all methods within 2.5 %) — score the axis's own label. Reference-transient conditioning (borrow a specific kick's click) stays open for v2 |
| G-9a ✅ | **Tonality axis, deterministic** (`imagine_tonality`): HPSS-layer remix, `h·(1+amount) + p·(1−amount)`, amount ∈ [−1,1] | S `det` | harmonicity moves monotonically noisy→tonal on real loops; amount 0 reconstructs the input; the fallback the roadmap names |
| G-9 ✅ | **Calibrated tonality axis**: `imagine_tonality(..., harmonicity=target)` — measures the HPSS-remix response at 5 anchor amounts (the layers are already in hand; ~5 autocorrs) and inverts the monotone curve; no model required | L | **Deterministic beat learned, measured**: anchored inversion 0.172 mean \|achieved−requested\| vs the 41-point per-clip oracle's 0.167 (its residual is family *reachability* — median response span 0.315 on real clips — not calibration), monotone on 100 % of sweeps. Three learned variants tried and recorded: free [2,64] gain field 0.23 (conditional-mean collapse), 4-param base+slope 0.29 (±31 dB cap can't remove a layer), scalar-amount head 0.25 (419 clips too few to out-predict direct measurement). `TonalityNet` + `imagine_tonality(model=…)` kept as the substrate for a frequency-selective v2 with a bigger corpus |
| G-3 | **Neural character transfer v2**: same U-Net family conditioned on a target character embedding — *nonlinear* timbre transfer (creates harmonics; "crunchy like guitar" for real) | L GPU | the axes are scalar knobs; this is the vector version |
| G-10 | **Unified multi-axis model**: one FiLM vector = [brightness, tonality, transient, …]; joint training on all ladders; enables *coupled* moves (brighter *and* noisier = riser) | L GPU | after ≥ 2 specialists ship |
| G-11a ✅ | **Expression automation, deterministic** (`render_expression`): lanes {brightness, tonality, noisiness, attack, sustain, character} over a note grid, 0→1 FL-clip values (0 = identity), AutomationCurve or per-note sequences, multiple simultaneous lanes, `perform` slice contract | M `det` | "each next note: harder attack + more air" measured working — brightness 0→0.64 and crest 2.8→5.4 across 8 notes from two rising lanes |
| G-11 | **Expression automation, learned**: frame-rate FiLM interpolation for continuous morphs inside one fragment once axis models ship; same lane names | M GPU | drop-in behind G-11a's surface |
| G-4 | Groove transfer (= R-8) surfaced as a generation op | S | timing is an expression lane too — G-11 unifies timbre + timing |
| G-5 | Spatial designer (= M-5) surfaced as a generation op | S | |
| G-6 | Variations (research): small diffusion/flow in coefficient domain conditioned on embedding — "give me 5 takes of this hit" | L GPU | park until G-2/G-8/G-9 ship |

## Epic I — Infrastructure & integration (L4)

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| I-1 | Training harness: config-driven loops, checkpoints, seeds, run cards | M | shared by V/G epics |
| I-2 | Model zoo: versioned weights + model cards, download-on-demand, deterministic fallback if absent | M | library never *requires* a model |
| I-3 | Evaluation expansion: event F1, note transcription metrics, separation SDR, BWE log-spectral distance — all in the experiment harness with regression tracking | M | keeps claims honest |
| I-4 ✅ | `SimilarityIndex`: exact cosine search (numpy GEMM; FAISS IndexFlatIP auto above 2 k entries — also exact), union-find dedup groups, save/load, `from_folder` | S | 0.1 ms queries over real loops; exactness kept deliberately — approximate search only when a library outgrows it |
| I-5 | Optional CLAP bridge for text↔sound search (`[semantics]` extra) | S GPU-opt | closes the "what it's not" gap when wanted |
| I-6 | Packaging: extras `[gpu] [vision] [separation] [semantics]`; CI wheels; docs site | M | |
| I-7 | Editor panels (Viktor Rose Phase 5): rhythm/pattern display, spatial designer, imagine-highs knob, per-stem tools — all through worker threads with undo | M | consumes M-3, R-5, G-2… |

## Epic D — Deep-learning practicality (L1) `det`

From the 2026-07 practitioner audit: what a music-DL pipeline hits first when
building on wavelet2vec. Findings that motivated it: single embed = 203 ms/s
(≈5× realtime) with no batch API; only two representation endpoints (one
global vector or one image); naive CQT row-shift pitch augmentation
*provably fails* (coefficients are analytic band signals at source
frequency); the torch frame is differentiable end-to-end but undocumented
and unused by the embedder.

| ID | Task | Effort | Notes |
| --- | --- | --- | --- |
| D-1 ✅ | **`embed_batch`**: profile-guided — conjugate-product IF features (unwrap removed), float32 compute/float64 accumulation, batched torch scalograms + one conv forward | M | 203 → 77 ms/fragment (2.6×), parity cosine 1.0; remaining tail = per-clip numpy feature stats (D-1b: port stats to torch batched) |
| D-1b ✅ | **Batched scalogram stats**: spectral, IF-deviation and onset-coherence sections computed in torch on the whole chunk while the coefficients are hot (`_batched_scalogram_stats`), injected into `_components_from`; one-pass variance + `w·\|d\|=\|w·d\|` reuse | S | 77 → 60 ms/fragment (**3.4× total vs the 203 ms baseline**), parity max-abs 7.7e-7; now memory-bandwidth-bound on CPU — the ≥10× ambition needs CUDA (the same torch ops run unchanged) or decimated stats (would break parity); remaining per-clip tail is `harmonic_phase_signature` (14 ms, needs per-f0 bands) |
| D-1c ✅ | **GPU device support**: `embed_batch(device="cuda")` runs scalograms + batched stats on the GPU (explicit device never silently degrades); verified on the box's GTX 1050 Ti with `torch 2.11.0+cu126` | S | 34.5 ms/fragment (**5.9× vs 203 ms**), parity max-abs 1.2e-7; remaining time is the untouched per-clip tail; `WaveletFrontend` fwd+bwd 3.7× vs CPU |
| D-2 ✅ | **`WaveletFrontend(nn.Module)`**: the 3-channel image computed *inside the graph*, batched, buffers move with `.to(device)`, NaN-safe gradients (smooth magnitude + masked angle) | M | magnitude parity 1.0000 vs `coeff_image`; IF/coherence 0.99+ where energy exists (silent cells are precision noise by design); gradients finite through silence |
| D-3 ✅ | **Sequence tokens**: `sequence_tokens(audio, sr, hop_ms) → [T, 3·bands]` + `WaveletFrontend.tokens()` (batched, differentiable twin, parity 1.0) | M | same 3-channel encoding as the vision tiles, un-pooled per frame; onset localized to ±1 frame at 20 ms hop |
| D-4 ✅ | **Correct `pitch_shift`**: per-band heterodyne row remap on semitone-spaced frame (naive roll leaves content at source pitch — measured) | S | 440→880/220/659 Hz for ±12/+7 st, duration preserved; integer semitones v1 |
| D-5 ✅ | Augmentation kit (`augment`): `time_stretch` (constant-Q phase vocoder — pitch exact at 440 Hz through 0.5×/2×), `effect_chain` (pedalboard shorthands), `crop_bars`, channel ops, seeded `random_augment` (deterministic per (seed, index), batch shapes held) | M | composes with D-4 pitch_shift and the G-1 ladders |
| D-6 ✅ | **`torch.utils.data` wrappers** (`data`): `SceneDetectionDataset` (scenes → image + COCO boxes/labels/pitches, `detection_collate` for variable box counts), `LadderAxisDataset` (ladder step → image + *measured* axis label), `ContrastiveViewsDataset` (two `random_augment` views, optional transform) — all deterministic per (seed, index), generated on the fly | M | training data in three lines; DataLoader-verified |
| D-7 ✅ | **Embedding rigor** (`eval`): `EmbeddingStandardizer` (standardize/whiten, save/load), `knn_purity`/`recall_at_k`/`retrieval_report` (leave-one-out, singleton-safe), `EMBEDDING_VERSION` tag through `Wavelet2Vec` and `SimilarityIndex` save/load (stale-index warning) | M | real library (356 files, 10 folders): **raw purity@1 0.916, recall@10 1.000**; standardize 0.885 (embedding scales already calibrated), whiten 0.458 (N < dim; flattens discriminative variance) — raw cosine stays the default, the harness is the deliverable |

---

## Sequencing (dependency-honest)

```
Now      C-1, C-3, M-2..M-6, R-1..R-8, H-1..H-4, P-1, G-1, G-7  (det; C-3 adds GPU/torch core)
Next     learned tier only: V-3+ probes, G-2/G-8/G-9 axes, G-11 FiLM — all GPU; deterministic tier complete
Then     V-3 (frozen-backbone probes; V-1/V-2 data done)  →  G-2, G-8, G-9 (imagination axes)
Scale    V-4 (DETR events) → V-5/P-3 (masks→stems) → G-3, G-10
Research V-6 (audio-native backbone → embedding v2), G-6 (variations)
```

Note G-11 lands twice: the automation plumbing ships early against the
*deterministic* axes (transient designer, HPSS remix, EQ tilt) so expression
rendering works before any model exists; learned axes drop in behind the
same lanes when they arrive.

Milestone gates (each demo-able in the editor):
1. **"It understands the beat"** — R-1..R-5 on the 100/120 BPM folders; swing
   report + pattern display.
2. **"It shapes transients"** — G-7 deterministic multiband transient
   designer: pluck→pad and pad→pluck A/B, lossless outside the edit.
3. **"It runs on the GPU"** — C-3 benchmark: full-library embedding scan
   minutes → seconds.
4. **"It sees notes"** — V-3 probe beats embedding-v1 purity; V-4 small-scale
   F1 > 0.9 on synthetic scenes.
5. **"It imagines"** — G-2/G-8/G-9 A/B on held-out ladders; every knob 0→1
   audibly and measurably monotone in its own label (brightness, transient
   sharpness, harmonicity).
6. **"It performs expression"** — G-11: one FL automation clip drives
   brightness + transient + groove lanes across a rendered vocal phrase.
7. **"It hears sources"** — P-1 + V-5 masks invert to convincing stems.

## Honest constraints

- ~~The current env's torch is CPU-only~~ **Resolved 2026-07-18**: the box
  has a GTX 1050 Ti (4 GB, sm_61 Pascal) and env "sound" now runs
  `torch 2.11.0+cu126` (cu126 is the Pascal-supporting wheel line; cu128+
  dropped sm_50–70). Verified: frontend fwd+bwd 3.7× vs CPU,
  `embed_batch(device="cuda")` 34.5 ms/frag (5.9× vs the 203 ms original),
  parity 1.2e-7; ~3.1 GB VRAM free in practice. Small-model training
  (probes, ViT-S/Swin-T-scale experiments at reduced batch) is now local;
  Swin-L/Co-DETR-scale training still wants ≥ 24 GB VRAM elsewhere.
- **Hardware trajectory (declared 2026-07-19)**: this computer will be
  upgraded with a larger GPU, and the package/library will run on **many
  machines, including much stronger hardware**. Consequences: (1) large
  models are a first-class target, not a footnote — for detection
  specifically, **MMDetection DINO with Swin-L backbones**, not only
  MobileNet-class nets; (2) groundwork for the large-model path proceeds
  **now, on CPU if that is what this box can run** — slow is acceptable
  when it de-risks the future (configs, D-6/COCO data adapters, inference
  and eval paths wired and tested at tiny scale); (3) nothing in the
  library may assume the small-GPU case — `device` parameters, batched
  paths, and checkpoint formats stay hardware-agnostic.
- Vision backbones expect ~square RGB statistics; our tiles are long and
  physically meaningful — V-3 exists precisely to measure how much domain
  gap remains before investing in V-4 fine-tunes.
- Deterministic fallbacks are a design law: any panel works without weights,
  just with less magic.
- **Pending: user-provided audio datasets.** Viktors will supply larger
  audio datasets for accurate training and inference experiments
  (2026-07-18). The current corpus (356-file library + synthetic) is the
  main limiter on the learned tier — G-9's scalar net lost to direct
  measurement partly on 419 training clips. When the datasets land:
  extend the `experiments/train_*.py` corpus builds, re-run the recorded
  evals, and log before/after numbers in FINDINGS (G-2 v3, G-9
  frequency-selective v2, G-8 reference-transient v2, V-4+ all want
  this data).
