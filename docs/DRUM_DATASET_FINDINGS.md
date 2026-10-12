# First real drum dataset: findings and next steps

22–23 September 2026, wavelet2vec v0.36.0.

**The complete real-audio path works, but the initial separator is not yet a
clean eight-source separator.** Verified loading, native-rate hybrid analysis,
supervised updates, saved inference bundles, full-file rendering and waveform
loss measurement all ran on the supplied recordings. Finer wavelet resolution
improved the first learned result. Quiet/noisy drums remain weak.

Open the [listening report](benchmarks/drum_dataset_2026-09-22/LISTEN_AND_RESULTS.html)
to audition each original solo against all three rendered models. It also
contains waveform/event, pitch and separation plots. All original recordings
were left unchanged; their SHA-256 values are recorded and rechecked.

## What the files establish

- Eight solo tracks and seven cumulative mixes, labeled by filenames.
- Stereo **44.1 kHz / 16-bit PCM**, not native 192 kHz/float32 recordings.
- Four 4/4 bars at 128 BPM are **7.5 seconds / 330,750 samples**.
- File lengths differ because release tails extend beyond the four bars.
  Original frame counts range from 330,965 to 339,148.
- Every mixture's best global alignment with its named solo sum is zero
  samples within the tested ±20 ms range.
- Solo sums are not exact mixtures: sum-reference SNR spans approximately
  **22.8–30.1 dB**, with peak residuals up to about **0.074**.

The user reports a master limiter. The measurements verify non-additivity;
they do not identify all limiter settings or attribute every residual uniquely
to that effect. Solos are useful corresponding waveform references, but this
is not an exact linear mixture dataset. No mixture-sum constraint was imposed
on training. Comparing cumulative mixes by subtraction also leaves errors,
especially for quiet sources; that method additionally requires another mix
and is not separation from one recording.

Training uses derived **float32** copies of the first three bars, preserving
the original sample values/rate. There is no resampling, normalization or
attempt to turn 16-bit source recordings into newly acquired high-resolution
content. Different tails are excluded from numerical target comparisons;
full-file listening exports retain the original mixture's complete duration.
The library's 192 kHz/float32 requirement remains covered by separate numerical
tests, not by this 44.1 kHz dataset.

## Solo events, repetition and transformations

New `detect_percussive_events`, `compare_percussive_events` and
`analyze_drum_track` APIs analyze the isolated annotated tracks. Instrument
names come from the filenames. These results do not demonstrate blind drum
classification directly from the mixture.

| Solo | Detected events | Main observations |
| --- | ---: | --- |
| Kick | 16 | Highly repeatable after alignment; fitted pitch and gain are essentially stable; centered stereo energy. |
| Snare | 8 | Pitch stable in reference fits; small waveform/gain differences; slightly left-weighted energy. |
| Clap | 9 | Centered energy; small raw level changes; several hits are not strongly explained by one reference waveform. |
| Open hi-hat | 16 | Right-weighted energy; variable waveform/level; 11/16 fit the chosen reference strongly without measured pitch change. |
| Tom | 15 | Highly repeatable shape, essentially stable fitted pitch; clear left/right balance changes. |
| Closed hi-hat | 64 | Clear pitch, level and stereo-balance modulation; 55/64 events strongly fit a transformed reference. |
| Bongos | 42 | Multiple repeat candidates and strong balance/level differences; one chosen waveform does not support reliable pitch estimates for other hits. |
| Cowbell | 42 | Highly repeatable, stable fitted pitch and nearly constant level/pan; some comparison windows are exactly equal PCM. |

For the **closed hi-hat**, accepted playback-rate-equivalent shifts span about
**−1.6 to +6.6 semitones relative to the selected reference** (8.2 semitones
total). Matched gain spans about **3.64 dB**; raw onset-window RMS spans
5.30 dB. Stereo energy balance varies from **−0.219 to +0.405**, where −1 is
left-only, zero is equal energy and +1 is right-only. This is not the DAW's
pan-knob value.

Bongo balance spans about −0.184 to +0.583; tom balance spans −0.236 to +0.205.
The corresponding event-level L/R dB ratios and timestamps are available in
[events.csv](benchmarks/drum_dataset_2026-09-22/events.csv). Raw measurements
and reference fits are preserved in [audit.json](benchmarks/drum_dataset_2026-09-22/audit.json)
and the individual `events_*.json` files.

**A measurement bug was found and fixed.** The preliminary first-kick estimate
suggested a one-semitone pitch change because the alignment search allowed
only 8 ms while the detected onset offset was about 9 ms. A 20 ms nuisance
timing search resolves the same hit at essentially zero pitch/gain change.
A regression now checks that an onset offset does not become a false pitch or
volume change. The kick's raw 0.78 dB onset-window RMS span is therefore not
evidence of MIDI velocity automation; its aligned fitted-gain span is only
about 0.001 dB. Final report files supersede that preliminary estimate.

The detector pools channel power separately, so anti-phase stereo does not
cancel into silence. Wavelet-flux analysis may resample to 22.05 kHz internally;
event positions/measurements retain native coordinates, and model training
remains at 44.1 kHz. Actual activity at a clip start is labeled separately;
the slicer's unconditional zero-time boundary is not treated as a detected hit.

Pitch estimates fit playback rate, timing and channel gains to a reference
snippet. They are not absolute F0 estimates for inharmonic/noisy drums.
The 0.95 fit threshold is a heuristic, not a calibrated probability; low-fit
events remain unresolved. Comparisons cover up to 60 ms and may shorten for
closely spaced notes. An exact PCM flag applies only to that comparison window,
not necessarily the complete note/tail. Candidate repeat groups use a 0.995
gain/timing fit and may separate different pitches of the same sample. Neither
group counts nor waveform variability prove synthesizer versus sample origin.
Measured gain and stereo balance also do not uniquely recover MIDI velocity
or a particular pan law.

## Separation experiments

All files remain in one declared performance group. Models trained on bars
1–3, `[0,248062)` samples, and were compared on bar 4, `[248062,330750)`.
The fourth bar is a **development segment**, used to inspect failures and
choose variants. It is not an untouched independent test set. Repeated sounds,
rhythm and shared stems also prevent a musical generalization claim.

Each model received 300 optimizer updates, batch size two, 8,192-sample
training crops, the same seeds/data policy, waveform plus complex spectral
losses, and no additive-consistency penalty. The prepared training PCM was
verified identical across all three variants.

1. **Initial baseline:** compact wavelets, 32 bands, Q=2, 356,416 parameters.
2. **Source-balanced loss:** same architecture/data; positive source weights
   fitted from training-only RMS with a 0.002 RMS floor. This emphasizes quiet
   targets without changing audio gain or evaluation units.
3. **Finer wavelets:** 96 bands, Q=8, original unweighted objective, 436,288
   parameters. This changes capacity as well as resolution, so it is not a
   parameter-matched proof of a wavelet-design advantage.

The Q convention is center frequency divided by Gaussian frequency standard
deviation, as defined by the hybrid frontend.

**Eight-source mix, fourth-bar gain-sensitive waveform SNR:**

| Source | Initial | Balanced loss | Finer wavelets | Reference-informed Q8 masks |
| --- | ---: | ---: | ---: | ---: |
| Kick | 4.88 dB | 4.97 dB | 5.29 dB | 15.61 dB |
| Snare | −0.32 dB | 0.03 dB | −0.19 dB | 3.86 dB |
| Clap | −3.36 dB | −0.62 dB | −0.91 dB | 1.81 dB |
| Open hi-hat | −0.81 dB | −0.02 dB | 0.08 dB | 1.55 dB |
| Tom | 2.47 dB | 2.02 dB | 4.39 dB | 14.24 dB |
| Closed hi-hat | −0.75 dB | 0.17 dB | 0.37 dB | 5.35 dB |
| Bongos | 3.06 dB | 2.62 dB | 4.67 dB | 12.13 dB |
| Cowbell | 4.24 dB | 2.09 dB | 7.93 dB | 16.30 dB |

Silence scores 0 dB. Negative scores are worse than silence under this metric,
and near-zero positive scores may still describe an almost-muted output.
The finer model is a better prototype for the stronger sources, but quiet
drum recovery remains poor. The source-balanced objective helps some quiet
outputs while hurting others; it stays optional rather than becoming a default.

On this complete mix, common raw loss falls from **0.010367** to **0.007744**
between initial and finer models, approximately 25.3%. Equal split scores
0.019623 and silence 0.014781. These aggregate improvements must not hide the
weak per-source results. Both individual source scores and all seven mixture
comparisons are retained in each run's `results.json`.

The reference-informed masks use the target solo coefficients to distribute
mixture coefficients. They are **privileged diagnostics**, not a trained
separator or a theoretical upper bound. Finer wavelets improve their scores
substantially over the Q=2 setting. STFT mask comparators are competitive:
the 2,048-point STFT gives about 14.87 dB kick / 13.51 dB tom / 16.70 dB cowbell,
versus 15.61 / 14.24 / 16.30 dB for Q8 wavelets. This does not establish universal
wavelet superiority. See [resolution comparison](benchmarks/drum_dataset_2026-09-22/oracle_resolution_comparison.json).

Running the event detector on the finer model's outputs also exposes leakage:
within the fourth-bar interior, its snare and clap outputs each yield 15
candidates for two solo-derived reference events. These are diagnostic
agreements with our automatic solo detector, not annotated precision/recall.
High recall or closed-hi-hat timing agreement does not establish correct
source identity or clean audio when other drums share the rhythm.

## Delivered artifacts and reproducibility

- [Interactive listening/report page](benchmarks/drum_dataset_2026-09-22/LISTEN_AND_RESULTS.html)
- [Per-event CSV](benchmarks/drum_dataset_2026-09-22/events.csv), [full audit](benchmarks/drum_dataset_2026-09-22/audit.json)
- [Initial run](benchmarks/drum_dataset_2026-09-22/separation/results.json)
- [Balanced run](benchmarks/drum_dataset_2026-09-22/separation_balanced/results.json)
- [Finer-wavelet run](benchmarks/drum_dataset_2026-09-22/separation_q8/results.json)
- [Output-event diagnostic](benchmarks/drum_dataset_2026-09-22/separated_event_agreement.json)

Every run directory contains eight full-render WAV stems, a checkpoint, an
inference bundle and prepared training files. Bundle/source identities travel
with the render receipts. Full-render players include training bars as well as
bar 4; the score table above uses only bar 4 with isolated model context.

Run from the repository with the local Python environment:

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.drum_dataset_audit
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.drum_substem_training --steps 300 --device cuda --output outputs/drum-baseline
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.drum_substem_training --steps 300 --device cuda --balance-sources --output outputs/drum-balanced
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.drum_substem_training --steps 300 --device cuda --bands 96 --q 8 --output outputs/drum-q8
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.drum_report
```

Training outputs must be fresh; `--evaluate-existing` re-evaluates the saved
bundle in an existing run directory. `drum_report` defaults to the delivered
benchmark folders; use `--results` when those folders are elsewhere. Artifact
producer/code identities are preserved even when later analysis-only fixes
change the package fingerprint. Inference bundles remain loadable through the
explicit portable loader; strict optimizer resume still checks original code.

The first experiment completed training/export but its report writer confused
absolute rendered paths with a relative result directory. Resolving the output
directory fixed that bug; the saved model was re-evaluated without retraining.
Its initial elapsed time/history was not preserved and is not fabricated.
The balanced and finer runs persist their histories and completed end to end.

## What to improve next

1. Improve quiet-source objectives and event coverage without rewarding only
   silence; compare active-event errors, false outputs on absent sources and
   waveform/complex/magnitude terms separately.
2. Strengthen local spectral processing and source heads. The large gap to
   reference-informed masks shows room beyond merely retaining phase.
3. Compare multi-resolution wavelets and complex STFT with matched budgets,
   several seeds and new patterns. Do not change global defaults from one loop.
4. Add annotated/MIDI onset and pitch controls when available to measure true
   detector/transform accuracy, and distinguish dry-target restoration from
   limited in-mix extraction with explicit production metadata.
5. Evaluate independent performances/instruments and blind listening before
   claiming general drum recognition or high-quality substem separation.

Validation: the full library suite passed **541 tests**; the final event-fit
default was then checked with **six focused tests**, including the timing/pitch
regression. Source-weighted losses and accumulation match independently computed
weighted objectives. This is an initial engineering result, not a finished
musical separator.

The editor compatibility suite also passed **36 tests**, and **25 runnable
Python documentation examples** executed successfully. No editor UI change
was needed for this experiment. The source files remain unchanged.
