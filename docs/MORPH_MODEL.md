# MorphNet: vox–synth hybrids in the lossless wavelet image

9 October 2026. The second purpose of
[note_dataset](../note_dataset/README.md#second-purpose-voxsynth-hybrids):
new short notes heard as **partly vocal and partly synth**, about 50/50 for
vox/synth judges, that are **neither a plain mix nor a filter**. A synth
shaped by the voice's envelope (a vocoder), or the reverse, does not count.

Code:
- [src/wavelet2vec/morph.py](../src/wavelet2vec/morph.py): views, judges,
  generator, baselines;
- [experiments/train_morph.py](../experiments/train_morph.py): training;
- tests: [tests/test_morph.py](../tests/test_morph.py).

## Lessons from KeyboardNet v1–v3, applied

| Lesson | What this run does |
|---|---|
| A two-class "50 %" is reached by noise or broken audio too | Two heads per judge, **vox present** and **synth present**, trained against generated noise (both absent); a presence loss wants both high |
| Classifiers learn shortcuts: sample rate, high band, level, stereo, key range, envelope, amount | The judges see only the **mid channel**, **below 18 kHz**, **levels relative to the note's peak, floored at −60 dB** (random −60…−45 dB floors in training), in **MIDI 36–82**. Synth is sampled **per envelope class and family**, vox **per singer**. |
| Generators fool the classifier they train against | Two judges with different designs (2-D CNN, temporal TCN). Their inputs are randomly shifted and low-passed. A **third judge never used in training** (Wavelet2Vec embedding + logistic regression) scores validation. A realism critic is added. |
| Two clean voices only | Leave-one-singer-out check of the judge; validation uses **held-out takes** (Shevannai "tze", "koh"; your "EH" take) |
| Validation must not share sources with training | 8 synth families held out (single-file families) |
| The v2 Swin-L backbone was underused (effective rank 3–6) | A moderate model: about 10 M parameters |
| Lossless 192 kHz wavelet image, exact decode | The generator reads and writes exact complex cells; the output is 192 kHz stereo |
| Per-cell gains make 10 ms steps | Gains are interpolated inside each cell, per sample |
| abs/angle give NaN gradients at zero | All views use power-based features |
| CPU busy (KeyboardNet v3) | Runs on the GPU (GTX 1050 Ti, ~2.6 GB peak) |

## Judges

**View** (`judge_view`): `[4, 120 rows, 120 frames]` per note. The planes
are:
- level re peak (dB, floored at −60, scaled to ±1);
- phase-increment coherence (re, im), which shows how steady and
  where-in-the-band each partial is;
- modulation depth (std/mean of sample powers in each cell).

The last three fade to zero at the floor. The view is invariant to stereo
width and loudness (tested).

- **A, JudgeCNN** (1.4 M): thin full-resolution stem, residual CNN,
  attention pooling.
- **B, JudgeTCN** (0.8 M): every frame's spectrum as one vector, dilated
  temporal convolutions.

Both produce a 128-dim embedding and two logits: vox present and synth
present. P(vox) = sigmoid(logit_vox − logit_synth).

**Training data:**

| Class | Notes in batch | Labels (vox present, synth present) |
|---|---|---|
| Clean vox (both singers equally) | 40 % | (1, 0) |
| Synth (envelope class, then family) | 40 % | (0, 1) |
| Noise (white, pink, brown, band-pass, clicks; random envelopes) | 20 % | (0, 0) |

Exhale (processed vox) and the Harmor vox presets are **references only**.
A trustworthy judge places them between the classes.

**Reports** go to `metrics.jsonl` (`judge` events) and the dashboard:
- accuracy and AUC on held-out takes and families;
- P(vox) of the references;
- noise presence;
- leave-one-singer-out recall;
- the held-out Wavelet2Vec judge.

## Generator

```
vocal note ─┐                              ┌─ synth cells
            ├─ shared encoder ─┐           ├─ vocal cells
synth note ─┘  (planes + every │ fusion    ├─ synth fine structure × vocal magnitude
               exact sample    ├─transformer├─ vocal fine structure × synth magnitude
               per cell)       │ (α)       └─ new content (residual)
                               └─ decoder ──── per-cell complex gains → exact decode → 192 kHz
```

1. **Encoder** (shared by both notes).
   - Five feature planes on all 136 frames: level re peak, side re mid,
     coherence re/im, modulation.
   - Row and frame coordinates.
   - An embedding of **every exact complex sample** of every cell, per
     octave group (normalized by the cell's RMS).
   - Conv stages go from 160 × 144 down to 20 × 18 tokens.
2. **Fusion.** Six transformer blocks over vocal and synth tokens jointly
   (720 tokens). Each attends to the other, with adaLN-Zero conditioning on
   the vocal amount α.
3. **Decoder.** Upsampling with skips from both encoders, back to full
   resolution.
4. **Output.** A complex gain (level and phase) per cell for each of four
   materials, interpolated per sample inside each cell:

   | Material | What it can carry |
   |---|---|
   | Synth cells | Dense, stable harmonics, unison, filter motion, ultrasonics |
   | Vocal cells | The voice itself |
   | Synth fine structure × the voice's sample-level magnitude | Formant motion, consonant onsets, breath envelope on synth partials |
   | Vocal fine structure × the synth's magnitude | Vibrato, glide, jitter, breath phase with a synth's spectral shape |

   A residual per octave group (bounded, relative to the local level)
   synthesizes content neither note has. The hybrid image is decoded exactly
   to 192 kHz stereo.

At initialization every material is at −12 dB and the residual is zero, so
the network starts close to an even blend. The losses push it away.

## Losses

| Loss | Weight | Meaning |
|---|---|---|
| balance | 1.0 | Both judges: logit_vox − logit_synth → logit(α), on randomly shifted / low-passed views |
| presence | 1.0 | softplus(2 − logit) for **both** detectors: each must be confidently on (P ≈ 0.88). A sound that is neither must not pay |
| between | 1.0 | In each judge's embedding, the hybrid is no farther from the vocal note, nor from the synth note, than the two notes are from each other |
| novelty | 1.0 | In each judge's embedding, the hybrid must stay at least ¼ of the vocal–synth distance from **the plain mix, the vocoder with synth carrier and the vocoder with vocal carrier**, computed for every pair |
| realism | 0.3 (ramped in over steps 200–400) | Hinge-GAN critic (spectral norm) against real notes of every class: synth, vox, Exhale, Harmor vox |
| loudness | 0.5 | Hybrid energy near the inputs' mean |
| anchor | 0.5 | Harmonic rows 1–4 of the key not more than 15 dB below the stronger input (pitch kept) |
| residual | 1.0 | New content at most half the hybrid's energy |
| smooth | 0.02 | Gain level changes over time |

α is 0.5 for half the pairs, otherwise uniform in 0.2–0.8, so α also works
as a "how vocal" knob.

### Attempt 1: the "neither" solution (stopped after 300 steps)

The first run gave presence a weight of only 0.1. In about 100 steps the
generator found sounds that **neither** detector recognized: P(vox present)
and P(synth present) were both near 0, and the two-way vote landed near
0.55. The balance loss fell from 12 to 0.6. Every other metric looked fine,
and the novelty margin was met trivially, since an unrecognized sound is far
from every baseline.

This is the README's "uncertain is not hybrid" pitfall happening in
practice. Attempt 2 adds three changes:
- the presence constraint above, at full weight;
- the "between" constraint;
- the critic's ramp, which now starts at step 200.

The dashboard plots both detectors' probabilities for the training hybrids.
Attempt 1's logs are in `v1/attempt1/`.

## Judge results (9 October 2026)

| Set | A (CNN) P(vox) | B (TCN) P(vox) | J (held out) P(vox) |
|---|---|---|---|
| Vox, held-out takes (70 notes) | 1.00 | 1.00 | 0.99 |
| Synth, held-out families | 0.00 | 0.00 | 0.01 |
| Exhale (processed vox, reference) | 0.02 | 0.05 | 0.08 |
| Harmor vox presets (reference) | 0.00 | 0.00 | 0.02 |
| Noise: P(vox present) / P(synth present) | 0.00 / 0.00 | 0.00 / 0.03 | – |

AUC on held-out takes vs held-out families: 1.0 for A and B.

**Leave one singer out** (CNN, trained on one voice, scoring all notes of the
other):

| Trained on | Scored on | Recognized as vox |
|---|---|---|
| Shevannai | your voice (126 notes) | 57 % |
| your voice | Shevannai (326 notes) | 44 % |

**What this means:**
- **The judges' "voice" means a clean, unprocessed voice like these two.**
  Exhale and the Harmor vox presets both land on the synth side, not in
  between. So 50 % by these judges sits halfway between a clean voice and a
  synth. That is **more vocal than Exhale**, which fits the design intent
  (voice as the anchor).
- **Voice only partly transfers between the two singers** (44–57 %). With
  both singers in training, the judges also know these particular voices and
  recordings. More singers would make "voice" more general.

## Validation, checkpoints, listening

**Every half epoch:** checkpoint `{epoch:g}_epoch.pth` plus `last.pt`, and
validation on 40 held-out pairs at α = 0.5, plus 8 at α 0.25/0.75. The pairs
combine held-out vocal takes with held-out synth families.

For the hybrid and each baseline (mix, both vocoders) the validation
reports:
- P(vox) from A, B and the held-out J;
- presence;
- the critic's realism gap;
- the hybrid's distance to each baseline.

**Score** (lower is better; `best.pt` is chosen by it):

    mean |P(vox) − α| over A, B, J  +  0.5 × novelty shortfall  +  0.1 × realism gap

**Every epoch:** listening demos in `demos/epoch_NN/`, as 192 kHz float WAV
files of the hybrid, the vocal, the synth, the mix and the vocoder. There are
6 held-out pairs plus 2 pairs from your "Piuw" take. The dashboard plays
them: open the run "morph/v1", section "Vox–synth morph".

## Running

```
PYTHONPATH=src python experiments/train_morph.py --epochs 40 --steps-per-epoch 1000 --batch 4
PYTHONPATH=src python experiments/train_morph.py --epochs 40 --resume     # continue from last.pt
```

Outputs go to `experiments/output/morph/v1/`. The judges are trained once
and saved to `judges.pt`; resuming reuses them. Judge views are cached in
`experiments/output/morph/judge_cache.pt`: 6,515 notes plus 512 noise notes,
built in about 1 minute on the GPU.

**Speed:** about 0.9 s per step with batch 4 on the GTX 1050 Ti, so roughly
17 minutes per epoch including validation, and ~11–12 hours for 40 epochs.

## Result of the small run (v1, 40 epochs, 10 October 2026)

| Held-out pairs | Epoch 0.5 | Epoch 20.5 | Epoch 40 |
|---|---|---|---|
| Both detectors on (judge A, mean of the weaker one) | 0.04 | 0.30 | 0.50 |
| Balance error, judge A | 0.50 | 0.40 | 0.35 |
| Balance error, held-out judge J | 0.33 | 0.21 | 0.28 |
| Critic realism gap (higher = easier to spot) | 1.9 | 7.8 | 6.6 |
| Distance to the plain mix (required ≥ 0.25) | 0.13 | 0.41 | 0.37 |

The hybrids became more genuinely two-sided and stayed away from the mix,
but the critic separated them from real notes easily.

Two flaws:
- **`best.pt` points to epoch 0.5.** The score's critic-gap term grows as
  the critic gets stronger, so the score is not comparable across training.
- **The model was small: 12 M parameters,** sized for the 4 GB GPU.

## MorphNet-L (v2): pretrained Swin-L encoder, on the CPU

Code: [src/wavelet2vec/morph_large.py](../src/wavelet2vec/morph_large.py),
[experiments/train_morph_large.py](../experiments/train_morph_large.py),
tests: [tests/test_morph_large.py](../tests/test_morph_large.py). Output:
`experiments/output/morph/v2/`, dashboard run "morph/v2". Started 10 October
2026, about 16 s per step, 8000 steps (~1.5 days). **Stopped the same day at
step 530 of epoch 1**, to free the machine. It can be resumed from the
half-epoch state (`checkpoints/last.pt`).

| v1 weakness | MorphNet-L |
|---|---|
| 12 M parameters, untrained encoder | **433 M**. The shared encoder is KeyboardNet v3's Swin-L backbone (217 M), warm-started and fine-tuned at learning rate 2e-5, with v3's stems and exact-sample embedding. The decoder is v3's pretrained UPerNet head (pyramid pooling, FPN, full-resolution refinement). |
| Fusion at one coarse scale | **Fusion transformers at three scales** (8/16/32 semitones × 80/160/320 ms tokens; depths 4/8/2, 188 M new parameters). Vocal and synth tokens attend to each other, conditioned on α. The finest scale and the stems are mixed by α-conditioned 1×1 convolutions that start as the average. |
| Judges overconfident (logits ±12) with mutually exclusive heads | **Calibrated judges:** wider; label smoothing 0.05 (logits stay within ±3); a supervised contrastive term on the embedding. The class centres define a **vocality axis** (synth 0, vox 1), a graded target that does not saturate. |
| α only a conditioning input | **Two α per pair** (0.5 and a random one) and a **ranking loss**: a higher α must give a higher vocality |
| One critic, easily ahead | **Two critics:** the judge view, and a stereo view (mid and side) up to 20 kHz. **Feature matching** pulls the hybrids' critic features toward real notes' statistics. |
| Selection by a drifting critic | **`best.pt` by fixed metrics:** see below |

**Selection score** (lower is better):

    J balance error + ½ (vocality error A + B) + ½ novelty shortfall
      + ¼ (1 − fraction with both detectors on) + ¼ min(KID / KID of the plain mix, 4) / 4

KID is the kernel distance between the hybrids' Wavelet2Vec embeddings and
266 real held-out notes of every class.

**Losses per hybrid:**

| Loss | Weight |
|---|---|
| vocality → α | 1.0 |
| logit balance | 0.2 |
| presence (both detectors ≥ logit 1) | 0.5 |
| novelty | 1.0 |
| between | 1.0 |
| feature matching | 1.0 |
| critics (hinge) | 0.3, ramped in over steps 100–300 |
| loudness | 0.5 |
| anchor | 0.5 |
| residual | 1.0 |
| smooth | 0.02 |
| α ranking (per pair) | 1.0 |

**Robustness:** the process keeps Windows awake and enables Python's fault
handler, so a native crash leaves a traceback in `train_err.txt`.
KeyboardNet v3 died silently on 9 October after its epoch-1 checkpoint.

### Stable pitch (added 10 October, before the run that counts)

Listening to the v1 demos showed unsteady notes. The spectrograms in the
dashboard show it too: v1 hybrids are smeared, with no clean harmonic lines.
There were two causes, both in the output stage:
- **Per-cell phase rotation.** A phase that changes from cell to cell is a
  frequency change, so it added warble.
- **Complex sums of the synth's and the voice's cells.** Two slightly
  detuned partials in one band beat against each other.

MorphNet-L now renders every hybrid with **one pitch source**:

    sample = (1 − m) · A · U  +  m · N

- **U, the pitch source's unit phasor.** The pitch source is the synth or
  the vocal note, a control input like α; training uses each half the time.
  Every partial keeps that note's own phase, so the hybrid's pitch is exactly
  as stable as the source's. Where the source is more than 30 dB quieter
  than the other note, the other note's phase takes over with a smooth
  crossfade.
- **A, the learned magnitude.** A time-varying combination of both notes'
  sample-level **magnitudes**, plus a phase-locked residual magnitude.
  Magnitudes add without interference, so there is no beating.
- **N, the noise path.** Both notes' cells with their own phases, plus a
  learned residual. It is used where the source cells are noise-like
  (breath, consonants, noisy synths), so noise is not "pitched".
- **m, from the source cells' phase-increment coherence:** 0 for tonal
  cells (coherence ≥ 0.75), 1 for noise-like cells (≤ 0.4).

**Pitch jitter** measures the energy-weighted spread, in cents, of the
instantaneous frequency of harmonics 1–3. Test signal (tests:
`test_hybrid_pitch_is_exactly_as_stable_as_its_pitch_source`):

| Signal | Jitter |
|---|---|
| Steady synth | 0.3 ct |
| Vocal with vibrato | 13.7 ct |
| Plain mix (beats) | 7.1 ct |
| Hybrid, synth pitch | **0.3 ct** |
| Hybrid, vocal pitch | **13.7 ct** |

Validation reports jitter for both hybrids, the two sources and the mix. The
dashboard plots it. The demos hold both versions of every pair:
`pairNN_hybrid.wav` (synth pitch) and `pairNN_hybrid_vocal_pitch.wav`.

The first MorphNet-L launch, without stable pitch, was stopped at step 80.
Its log is in `v2/attempt_unstable_pitch/`.

### Syllables and vowels: kept, and controllable over time (added 10 October)

The voice's articulation should stay recognizable in the hybrid, and at
inference the vowels should be automatable, like a DAW automation lane.
Code: [src/wavelet2vec/articulation.py](../src/wavelet2vec/articulation.py),
tests: [tests/test_articulation.py](../tests/test_articulation.py).

**Targets.** Every 10 ms frame of each vocal note gets a distribution over 10
classes, from the algorithmic vocal detector (`tools/vocal_detector.py`):
- silence;
- the vowels i, e, eh, a, o, u;
- sibilant, plosive, breath.

Vowel frames get a softmax over the detector's template-fit errors, because
neighbouring vowels are confused first. Each singer gets one vocal-tract
scale, the median of its notes' own estimates. The targets are cached in
`morph/articulation_cache.pt`.

**Articulation judge.** A frame-wise classifier on the judge view. It
convolves over frequency, pools over frequency, and keeps full time
resolution through a dilated temporal stack. It is trained on the training
takes and reports its accuracy on held-out takes. It reads the articulation
of any sound, hybrids included.

**Conditioning.** MorphNet-L takes an optional articulation curve
`[10 classes, 120 frames]`, embedded at every scale. It enters the vocal
encoder features before fusion, the fused features, and the full-resolution
features of the output heads. All of it starts at zero, so a checkpoint
without articulation loads unchanged (tested).

**Training targets per pair:**

| Share | Target |
|---|---|
| 50 % | The voice's own articulation (keep its syllables) |
| 25 % | Another vowel, sung on the voice's vowel frames |
| 25 % | A glide between two random vowels across the voice's vowel frames |

Consonants and silences stay those of the source. The articulation loss
(weight 0.5) is the soft cross-entropy between the judge's reading of the
hybrid and the target, on voiced frames. It is weighted by 0.5 + α, so a more
vocal hybrid is held more strictly.

**Validation:**
- **Vowels kept:** for the hybrid and for the voice itself (the ceiling).
- **Vowels controlled:** a different vowel is requested.

Both enter the selection score. Demos add **"hybrid, vowels a→o"**, a glide
across each pair's vowel frames.

**Inference** (vowel automation):

```
PYTHONPATH=src python experiments/morph_generate.py --checkpoint experiments/output/morph/v2/checkpoints/best.pt \
    --vocal voice.wav --synth synth.wav --midi 60 --alpha 0.5 --pitch synth --vowels "a@0,o@0.8" --out hybrid.wav
```

| `--vowels` value | Effect |
|---|---|
| empty | Keep the voice's own syllables and vowels |
| `"a"` | Sing "a" on the voice's vowel frames |
| `"a@0,o@0.8"` | Glide from "a" at 0 s to "o" at 0.8 s |

`--vowel-amount` (0–1) blends the curve with the voice's own vowels.
In Python: `wavelet2vec.morph_large.generate_hybrid(model, vocal, synth, midi, alpha, pitch_source, vowels)`.
`vowels` also accepts weight mixtures per point, e.g. `{"a": 0.5, "o": 0.5}`.

**Limits:**
- The detector's vowel labels are noisy at high pitch: Shevannai's "mu" is
  labelled partly "i".
- There are two singers.
- Vowels can be steered only where the voice actually sings a vowel.
- The vocal amount α is still one value per note; it cannot be automated
  within a note yet.

### Checkpoints every quarter epoch

`{epoch:g}_epoch.pth` (0.25, 0.5, 0.75, 1.0, …) and a resumable `last.pt` are
written every quarter epoch, about every 55 minutes at 800 steps per epoch.
Stopping loses at most that. Each checkpoint is about 1.7 GB, and `last.pt`
is about 5 GB because it holds the optimizer state.

`--init-from <checkpoint>` starts a new run from a MorphNet-L checkpoint.
Parts the checkpoint lacks start at zero. This is how the articulation
version continues from the stable-pitch run.

### Pause and continue exactly where it stopped (added 10 October)

Code: [experiments/training_control.py](../experiments/training_control.py).
Used by MorphNet-L and KeyboardNet v3.

**Pausing.**
- **Request:** press **Pause** in the dashboard (live run), or run
  `python experiments/training_control.py pause experiments/output/morph/v2`.
  Either creates a `PAUSE` file in the run folder.
- **What the loop does:** it checks for that file before every step. It then
  saves its complete state to `checkpoints/paused.pt`, logs a "paused" event
  and exits.
- **What the state holds:**
  - model and critic weights;
  - both optimizers and the learning-rate schedule;
  - every random generator: the numpy generator (data order, vowel targets),
    the torch generator (augmentation) and torch's global generator
    (stochastic depth);
  - the critics' running real-note statistics;
  - the running epoch's step records and time;
  - the position.

**Resuming.**
- **Request:** press **Resume training** in the dashboard (paused run), or run
  `python experiments/training_control.py resume experiments/output/morph/v2`.
- **What it does:** it starts the recorded command (`launch.json`: Python,
  arguments, folder, threads) with `--resume`. The script loads the newer of
  `paused.pt` and `last.pt` and continues at the next step.

The quarter-epoch `last.pt` holds the same complete state, so resuming after
a crash is exact too, from the last quarter. Checkpoints written before this
change lack the generator states. Resuming from them continues at the right
step, but augmentation and stochastic depth draw new random numbers.

**Check:** run A trained 4 steps straight. Run B paused after step 2,
resumed, and paused after step 4. Both must end with identical weights,
optimizer state, random-generator states and training records (see below).

### Dashboard

- **Header:** a model-system dropdown (KeyboardNet, MorphNet), then
  versions / experiments. A live run is listed first, opened by default and
  marked by a green pulsing **LIVE training** pill. When you look at an older
  run, a green **Go to live training** button jumps back.
- **Live status:** a run counts as live only if its PID is a Python process
  running a training script. Windows reuses process IDs, which once made the
  dead v3 run look alive.
- **Listening:** every audio player has a spectrogram of that file under it
  (sharp or exact-samples picture, mid channel, 27 Hz–21 kHz, hoverable). The
  server renders it from the WAV's exact wavelet image at the canvas's pixel
  size.
- **Links:** `?run=morph/v1` (or any run name) opens a specific run.

## A diffusion approach

[MORPH_DIFFUSION.md](MORPH_DIFFUSION.md): stage 0 tests whether a diffusion
model with composable vox/synth guidance gives more realistic, more two-sided
hybrids than MorphNet-L, with per-frame α. It has go/no-go criteria. It trains
a 245 M-parameter Swin-L denoiser on the CPU and is paused after epoch 3 of 10.

## Limits to keep in mind

- **"Voice" rests on two voices.** The leave-one-singer-out check says how
  far the judges' notion of voice transfers between them. More singers would
  make the target sharper.
- **Judges only steer.** Whether a hybrid is new and interesting is decided
  by listening; the demos are there for that.
- **GAN training can oscillate.** Watch the realism gap and the critic loss.
  The novelty and balance terms keep the generator from collapsing onto the
  mix, but they do not guarantee musicality.
- **The novelty margin is a design choice** (¼ of the vocal–synth distance
  in judge embeddings), not a perceptual measurement.
