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
