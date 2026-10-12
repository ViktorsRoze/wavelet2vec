# KeyboardNet v1: a whole keyboard from one or two notes

8 October 2026. First model trained on [note_dataset](../note_dataset/README.md).
Given one or two notes of a sound, it generates the sound's other notes.

**Result:** on 10 sound families never seen in training, the predicted
envelopes have **half the error of plain pitch-shifting** (7.4 dB vs 14.6 dB
mean absolute error over the audible part of each note). Two examples beat
one; a blend of two shifted examples is a much weaker baseline than the
model. Rendered audio also improves on pitch-shifting in every demo, but by
less than the predictions do: the renderer is now the bottleneck. Listening
has not been done yet.

## How it works

1. **Envelope.** A note becomes log band energies on a semitone grid, one
   band per MIDI note from MIDI 12 (16.35 Hz) to 135 (19.9 kHz), in 10 ms
   frames: 124 × 120 values per 1.2 s note. It is computed with the
   invertible frame at 48 kHz.
2. **Carrier.** The nearest example is pitch-shifted to the target key by
   heterodyning in the same frame. Because the bands sit on semitones, a
   shift by *d* semitones moves the envelope by exactly *d* bands, so
   training works on shifted envelopes and needs no audio processing.
3. **Learned correction.** A FiLM-conditioned 2-D CNN (486k parameters, a
   U-Net over bands × time) receives:
   - the shifted example envelopes and their validity masks;
   - a distance-weighted blend of the examples;
   - the target note and the intervals.

   It outputs a dB correction to that blend: how this sound's brightness,
   decay, level and spectral shape change at the target key. With two
   examples it also sees the difference between them, i.e. the sound's own
   trend.
4. **Audio.** The shifted carrier is reshaped band by band, frame by frame,
   toward the predicted envelope (gains capped at ±30 dB) and resynthesized.

Code: [src/wavelet2vec/keyboard.py](../src/wavelet2vec/keyboard.py). Data:
[experiments/keyboard_dataset.py](../experiments/keyboard_dataset.py).
Training and evaluation:
[experiments/train_keyboard.py](../experiments/train_keyboard.py).

## Data and training

- **Notes:** 14,785 from 151 files in 67 sound families.
  - Included: the renders, the Korg recordings (on their fitted grids) and
    the five own-voice takes.
  - Excluded, following the dataset README: silent and cut slots, the notes
    above MIDI 127 that only repeat MIDI 127, Korg notes from MIDI 124 up,
    and the arpeggio/sequence patches.
- **Split by family:** 10 families are held out entirely:
  - Harmor Rhodes, Serum preset 1, Massive X 3 and 6;
  - Repro-1 presets 1, 3, 9, 17 and 20;
  - the author's own sung takes.

  The other 57 families train the model.
- **Episodes:** pick a family, then a file, a target note and one or two
  example notes from the same file. Examples sit within 12 semitones 60 % of
  the time, within 24 semitones 30 %, and within 48 semitones 10 %.
- **Loss:** L1 in dB, with weight 1 on the audible part (within 70 dB of the
  note's peak) and 0.1 elsewhere.
- **Training run:** on the CPU (16 cores, as requested). 4,000 steps of
  batch 32, AdamW with a one-cycle learning rate up to 2e-3: **62 minutes**
  at ~0.93 s per step. Training loss fell from 9.0 to 5.2 dB and was still
  falling.

## Results on the held-out families

Mean absolute dB error of the target envelope, audible part only, 150
episodes per cell:

| Examples | Distance | KeyboardNet v1 | Pitch-shift nearest | Blend of both |
|---|---|---|---|---|
| 1 | 1–5 semitones | **5.37** | 7.82 | — |
| 1 | 6–12 | **7.43** | 12.76 | — |
| 1 | 13–24 | **9.70** | 18.86 | — |
| 1 | 25–48 | **13.21** | 29.09 | — |
| 2 | 1–5 | **3.92** | 6.24 | 5.28 |
| 2 | 6–12 | **4.83** | 8.72 | 7.22 |
| 2 | 13–24 | **6.57** | 13.20 | 10.82 |
| 2 | 25–48 | **8.16** | 19.90 | 17.41 |
| All | | **7.40** | 14.57 | 13.66 |

Per family, the model beats pitch-shifting on all ten. Its worst family is
the own voice: **11.3 dB** vs 16.7 dB for pitch-shifting. The training data
holds only two other vocal sources, both library recordings. The best is
Repro-1 preset 9: 4.6 vs 12.9 dB.

**Rendered audio** (demos at 48 kHz, whole keyboards; one global level offset
removed because demo clips are peak-normalized):

| Demo | Pitch-shift only | KeyboardNet v1 |
|---|---|---|
| Own voice "A", from MIDI 65 | 10.3 dB | **8.5** |
| Own voice "A", from MIDI 61 + 67 | 11.7 | **9.8** |
| Harmor Rhodes, from MIDI 70 (targets up to 58 semitones away) | 24.0 | **20.1** |
| Harmor Rhodes, from MIDI 41 + 99 | 12.8 | **10.8** |
| Serum preset 1, from MIDI 70 | 24.1 | **17.0** |
| Serum preset 1, from MIDI 41 + 99 | 11.7 | **10.6** |

Demo files are in `experiments/output/keyboard/demos/` (git-ignored). For
each, `__original`, `__pitch_shift_only` and `__keyboardnet_v1` play the same
notes in the same order, 1.2 s each.

## What these numbers do and do not show

- They measure **spectral envelopes**: per semitone band, per 10 ms. Phase,
  fine harmonic structure and noise texture come from the shifted example
  and are not scored. No listening test has been done yet.
- The model learns **systematic changes across the keyboard**: brighter or
  duller, shorter or longer decays, level trends. It cannot invent content
  the shifted example lacks, e.g. a different multisample zone, or harmonics
  above the example's own top band after a downward shift.
- **The renderer loses much of the predicted gain.** Gains are capped at
  ±30 dB, the frame's Gaussian bands overlap (reshaping one band leaks into
  its neighbours), and gain cannot create energy where the carrier has none.
- **Overlapping notes stay in the targets.** In the 89 held/ringing files
  the slot contains the previous note's release. The model is scored on it
  like the rest of the slot.
- **The test split is random:** 5 of the 10 held-out families are Repro-1
  presets.
- **Resolution:** 48 kHz, up to 20 kHz. The dataset's 192 kHz content is not
  modelled yet.

## Next steps

1. **Renderer:** iterate the gains (apply, re-measure, correct) and fill
   carrier-empty bands, so the audio reaches the predicted envelope. Score
   the rendered audio, not only the prediction.
2. **Listening:** check the demos against pitch-shifting.
3. **Train longer:** the loss was still falling at 4,000 steps. A run of
   about 3 hours on the CPU is feasible.
4. **192 kHz tier:** extend the bands above 20 kHz for the 47 files with
   real ultrasonic content.
5. **More voices:** the own-voice error is the highest. More vocal takes
   (and Shevannai and Exhale in training) are the most direct fix.

## Reproduce

```
PYTHONPATH=src python experiments/keyboard_dataset.py   # envelope cache (23 min measured; now one thread per worker, should be faster)
PYTHONPATH=src python experiments/train_keyboard.py     # train + evaluate + demos (~75 min on 16 CPU cores)
```

# KeyboardNet v2: Swin-L backbone, deep head, rich wavelet features

Started 8 October 2026 for 50 epochs on the CPU. **Stopped after 6 epochs**
(10 h) for an in-depth evaluation; see below. Resume with `--resume`.

## Evaluation after 6 epochs (best.pt = epoch 6)

From [experiments/evaluate_keyboard_v2.py](../experiments/evaluate_keyboard_v2.py),
on the 10 held-out test families. Values are envelope error in dB; lower is
better.

**Accuracy: about equal to v1, not better yet.**

| Examples | 1–5 st | 6–12 | 13–24 | 25–48 | v1 (same episodes) | Pitch-shift |
|---|---|---|---|---|---|---|
| 1 | 4.57 | 7.50 | 9.27 | 12.61 | 4.36 / 7.41 / 9.14 / 12.32 | 6.9 / 12.8 / 18.7 / 28.3 |
| 2 | 4.30 | 5.93 | 8.00 | 10.93 | 4.12 / 5.68 / 7.62 / 10.67 | 6.5 / 10.6 / 17.1 / 24.7 |
| 3 | **3.99** | **5.45** | **6.69** | 11.44 | — (v1 takes at most 2) | 5.8 / 10.6 / 15.1 / 26.0 |

- With 1–2 examples, v2 is 7.89 dB overall against v1's 7.67: v1 is 0.1–0.4
  dB better in every cell.
- Three examples help most within two octaves.
- Both models roughly halve the error of plain pitch-shifting.
- Own voice: 9.8 dB (pitch-shift 15.7).

**Head: works, but delivers only part of the needed correction.**
- **Correction:** the head applies 60 % of the needed correction (slope 0.60;
  correlation 0.86 between applied and needed). The mean needed is 14.1 dB,
  the mean applied 9.5 dB, and 7.6 dB remains.
- **Auxiliary outputs:** clearly better than copying the nearest example.
  Phase coherence error 0.082 vs 0.225, stereo width 0.12 vs 0.22, IF
  deviation 0.023 vs 0.036.
- **Error by region:** largest above 8 kHz (9.5–12.2 dB) and below 100 Hz
  (8.1–8.4 dB); smallest at 0.1–2 kHz (5.8–6.9 dB).
- **Ablations** (full model 7.71 dB):

  | Path switched off | Error |
  |---|---|
  | High-resolution stem skip | 11.98 dB |
  | Conditioning | 11.41 dB |
  | Swin backbone path | 10.85 dB |

  **The convolutional stem + decoder carries more of the result than the
  223M-parameter backbone.**

**Backbone: learning, but its capacity is barely used.**
- **Utilization:** no block is idle. Stage 3's later blocks change their
  input by only 1–5 % (blocks 9 and 15 by about 1 %).
- **Attention:** it sharpens with depth; normalized entropy is 0.68, 0.42,
  0.24 and 0.03 for stages 1–4.
- **Effective rank of the pooled stage features is very low: 3.3, 4.3, 5.9
  and 5.0** out of 192–1,536 dimensions. The representation uses only a
  handful of directions: a strong sign the model is far larger than 13,006
  notes from 52 families can fill.
- **Probes** (fitted on training sounds, R² on held-out sounds), comparing
  the untrained network with epoch 6:

  | What | Input | Untrained stage 4 | Epoch 6 stage 4 | Epoch 6 head |
  |---|---|---|---|---|
  | Target pitch | 0.61 | 0.44 | 0.96 | 0.99 |
  | Brightness | 0.66 | 0.42 | 0.81 | 0.86 |
  | Decay length | 0.07 | -0.05 | -0.02 | 0.07 |

  Training taught the backbone pitch and brightness; decay length is not
  encoded in pooled features. The vox/synth and envelope-class probes are
  not informative: their accuracy falls below the majority class, because
  the test families are almost all synths.

**Reading:** after 6 epochs, the large model matches v1 and adds the
three-example scenario, but its backbone is underused. Training was still
improving (validation 8.93 → 6.91 dB), so continuing may close the gap. The
evidence, however, points at data, not model size, as the limit.

## What changes from v1

**Richer input.** For each example, the model sees 7 planes per semitone band
per 10 ms instead of the envelope alone. Spectral, phase and wave
information come together, using the library's encodings
([src/wavelet2vec/keyboard_v2.py](../src/wavelet2vec/keyboard_v2.py)):

| Plane | Information |
|---|---|
| Log energy | spectral envelope (as in v1) |
| Instantaneous-frequency deviation from the band centre | where partials sit inside each band (vision encoding) |
| Phase coherence of the demodulated phasor | steady partial (~1) vs noise (low) |
| Demodulated baseband, real + imaginary | phase relative to each band's carrier (hybrid context encoding); exactly unchanged by the heterodyne pitch shift |
| Modulation depth within each frame | beating, unison, tremolo |
| Stereo width (side/mid) | spatial spread per band |

**One, two or three examples.** Examples are sorted nearest-first.
Conditioning carries the target note, the example count and every interval.
Training episodes use 1, 2 or 3 examples equally often. Evaluation and demos
report all three scenarios, including "three notes given, generate the rest".

**Swin-L backbone** (written in the library; no `timm`):
- 192-dim embedding, depths 2/2/18/2, heads 6/12/24/48: 223M parameters.
- The input is padded to 128 × 128 and cut into 4 × 4 patches; attention
  runs in 8 × 8 windows, shifted on alternate blocks.
- Every block is adaLN-Zero conditioned on the target note and intervals, so
  the deep stack starts as the identity.
- Stochastic depth 0.2.

**Deep dense head (UPerNet-style)**, 21M parameters:
- pyramid pooling on the last stage;
- an FPN over all four stages, fused at 1/4 resolution;
- a FiLM-conditioned refinement decoder to full resolution, with skips from
  a high-resolution convolutional stem (so 10 ms attacks survive the 4 × 4
  patches).

It outputs the envelope correction plus three auxiliary maps: the target's
IF deviation, phase coherence and stereo width, at loss weight 0.25.
**Total: 244M parameters.**

**Training:**
- **Optimizer:** AdamW at 2e-4, weight decay 0.05, warm-up then cosine decay,
  batch 16, gradient clipping 1.0.
- **Augmentation:** a random ±12 dB gain shared by examples and target.
- **Epoch:** as many episodes as there are training notes (13,006), i.e. 812
  steps.
- **Split:** test families are v1's ten. Validation uses five more
  single-file families (Sakura string, Serum preset 4, Iris 2 preset 2,
  Repro-1 preset 13, Zeta+ preset 1), scored every epoch.
- **Speed:** ~7.3 s per step on the Ryzen 7 5800X, i.e. about 100 minutes
  per epoch and **about 3.5 days for 50 epochs**.

## Training tools: checkpoints, probes, dashboard

- **Half-epoch checkpoints:** `checkpoints/1.5_epoch.pth`, `2_epoch.pth`,
  … (older runs' `epoch_NNN.pt` were renamed to `N_epoch.pth`). `last.pt`
  stores the step inside the epoch, so `--resume` continues from a half
  epoch.
- **Training probes,** written to `metrics.jsonl`:
  - every 50 steps (`--log-every`): loss parts; learning rate; gradient norm
    and update/weight ratio per module group (stem, conditioning, patch and
    merging, stages 1–4, head); per-stage block update ratios and attention
    entropy;
  - every checkpoint: validation by example count and distance, and ridge
    probes of input, stage and head features (fitted on training families,
    scored on validation families; test families are never used during
    training).
- **Dashboard:** `python experiments/training_dashboard.py` serves
  <http://localhost:6006>. It shows run status, progress and ETA, and charts
  for:
  - loss;
  - validation against the baselines and by example count;
  - learning rate;
  - gradient norms and update ratios;
  - block utilization and attention entropy;
  - probes over training.

  It also lists checkpoints and evaluation reports, and shows the log tail.
  It refreshes every 30 s and only reads files.
- **In-depth evaluation:** `python experiments/evaluate_keyboard_v2.py
  --checkpoint best.pt` (17 min). It writes `evals/eval_*.json`, which the
  dashboard displays.

## The running job

- **Process:** detached, so it survives the terminal. Its PID is in
  `experiments/output/keyboard/v2/train_pid.txt`.
- **Log:** `experiments/output/keyboard/v2/train_log.txt`. Every 50 steps it
  prints the envelope loss; at each epoch's end it prints the validation
  score against both baselines.
- **Per-epoch history:** `experiments/output/keyboard/v2/history.json`.
- **Checkpoints** in `experiments/output/keyboard/v2/checkpoints/`:
  - `epoch_NNN.pt`: weights and metrics, one per epoch (~1 GB each);
  - `best.pt`: the best validation epoch;
  - `last.pt`: the full resumable state.
- **Stop:** `Stop-Process -Id <pid>`.
- **Resume** from the last finished epoch:
  `PYTHONPATH=src python experiments/train_keyboard_v2.py --epochs 50 --resume`.
- **Finish:** at the end, the best epoch is scored on the test families
  (1, 2 and 3 examples, against v1's baselines). Demos from 1, 2 and 3 given
  notes are written to `experiments/output/keyboard/v2/demos/`.

**Expectation, stated before results:** 244M parameters against 13,006 notes
from 52 families is a large model for this much data. Overfitting to the
training sounds is the main risk; the per-epoch validation score will show
it. If validation stops improving early, `best.pt` keeps the best epoch, and
a smaller backbone (`--size base` or `tiny`) is the comparison to run.

```
PYTHONPATH=src python experiments/keyboard_dataset.py --features rich   # rich-feature cache (86 min measured)
PYTHONPATH=src python experiments/train_keyboard_v2.py --epochs 50 --size large
```

# KeyboardNet v3: lossless 192 kHz stereo wavelet image, warm-started from v2

Started 8 October 2026, 21:55: 10 epochs on the CPU, about 4.1 h per epoch
(~1.7 days). Watch it on the dashboard (<http://localhost:6006>, run `v3`).
**It stopped after epoch 1** (9 October, when the PC went to sleep).
`--resume` continues at epoch 2.

## What changes from v2

**Representation:** the [lossless stereo wavelet image](WAVELET_IMAGE.md).
- 192 kHz, 149 semitone rows (16 Hz to 96 kHz) by 10 ms frames.
- Exact complex mid and side cells.
- Round trip about 130 dB in float32 and about 305 dB in float64.

v2 saw lossy 48 kHz summaries; v3's input and output are the complete
signal.

**Warm start:** v3 contains v2's full network, loaded from v2's `best.pt`
(epoch 6):
- stem, Swin-L backbone, UPerNet head and refinement decoder (244.4M
  parameters);
- the backbone is generalized to rectangular grids and runs on 160 × 128.
  v2's first 124 rows are v3's first 124 rows; rows 124-148 (20-96 kHz) are
  new;
- the refactor reproduces v2's outputs bit for bit, and all v2 weights load.

v2's seven feature planes are computed from the exact cells, so the
backbone sees familiar inputs.

**Exact key change:** the nearest example is moved to the target key inside
the image. Every row shifts by the interval with the exact heterodyne phase
ramp, and is band-limited-resampled to its destination cell size. The
shifted harmonics land on the target note's harmonics (checked: 521.7 Hz
and 783.3 Hz for middle C).

**New lossless path (0.2M parameters, zero-initialized, so v3 starts as
v2):**
- **Input:** the carrier's exact cells, embedded per octave group and joined
  into the full-resolution decoder.
- **Output, per cell:**
  - a mid gain (from v2's envelope prediction);
  - a side gain (from v2's predicted stereo width);
  - a phase rotation (new);
  - an additive residual per octave group (new), for content the carrier
    lacks.

  These are applied to the carrier's exact cells and decoded exactly to
  192 kHz stereo audio.

**Losses** (all masked above each note's source Nyquist):

| Term | Weight |
|---|---|
| Mid cell level | 1.0 |
| Side cell level | 0.5 |
| Phase-aware complex error | 0.2 |
| Multi-resolution log-spectral error of the decoded audio | 0.5 |
| v2's IF deviation and coherence maps | 0.1 |

**Training:**
- **Learning rates:** 1e-4 for the warm-started weights, 5e-4 for the new
  path.
- **Schedule:** 200 warm-up steps, then cosine decay over 10 epochs.
- **Data:** batch 16 with ±12 dB gain augmentation. Notes come from the exact
  192 kHz note store, encoded on the fly (about 2 s of each 18 s step).
- **Splits:** v2's (the same test and validation families).

**First steps** (timing run; 12 validation episodes, so a rough sample):

| After 6 steps | v3 | Shifted carrier | Blend |
|---|---|---|---|
| Cell-level error, mid | **9.2 dB** | 17.3 | 13.5 |
| Ultrasonic rows (192 kHz sources) | **5.2 dB** | 8.7 | |
| Side channel | **11.5 dB** | 18.9 | |
| Decoded-audio spectral error | 9.5 dB | **7.9** | |

The warm start brings v2's knowledge in immediately: a tiny untrained v3
scored 15.4 dB. The decoded-audio spectral error is still worse than the
carrier's, because v2's level-only corrections boost some cells in ways that
decode less cleanly. The spectral loss targets exactly this; it is the
first thing to check on the dashboard.

```
PYTHONPATH=src python experiments/keyboard_v3_dataset.py      # exact 192 kHz stereo note store (27.3 GB)
PYTHONPATH=src python experiments/train_keyboard_v3.py --epochs 10 [--resume]
```

Code:
- [src/wavelet2vec/keyboard_v3.py](../src/wavelet2vec/keyboard_v3.py);
- [src/wavelet2vec/wavelet_image.py](../src/wavelet2vec/wavelet_image.py);
- [experiments/train_keyboard_v3.py](../experiments/train_keyboard_v3.py).
