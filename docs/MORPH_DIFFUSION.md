# MorphDiffusion, stage 0: is diffusion worth it for vox–synth hybrids?

A separate experiment, prepared on 10 October 2026. **Training started 10 October at 20:10**, on the CPU, with
a ~245 M-parameter Swin-L denoiser. **Paused on 12 October after 6,063 of 20,000 steps** (epoch 3 of 10; the
best check so far is epoch 3). It continues exactly from `checkpoints/paused.pt`.

**The question:** does a diffusion model give vox–synth hybrids that are
**more realistic** and **more often genuinely two-sided** than MorphNet-L?
MorphNet-L trains a generator against judges and critics. Stage 0 answers
the question before a pair-conditioned diffusion model is built.

Code:
- [src/wavelet2vec/morph_diffusion.py](../src/wavelet2vec/morph_diffusion.py)
- [experiments/train_morph_diffusion.py](../experiments/train_morph_diffusion.py)
- [experiments/eval_morph_diffusion.py](../experiments/eval_morph_diffusion.py)
- tests: [tests/test_morph_diffusion.py](../tests/test_morph_diffusion.py)

Run folder: `experiments/output/morph/diffusion_s0/`; in the dashboard,
model system **MorphNet**, version **diffusion_s0**.

## Why diffusion might help

| What MorphNet runs showed | What diffusion offers |
|---|---|
| The realism critic separated hybrids from real notes easily (v1: gap 6–9) | No adversarial critic. The model learns what real notes look like, so its samples land among real notes by construction |
| "Both detectors on" was rare; one attempt found "neither" sounds that split the vote | **Composition:** sampling with vox guidance and synth guidance together asks for notes likely under both, which means "both present" by definition |
| Optimizing against the judges risks fooling them | Labels steer the sampling; judges only evaluate |
| One output per pair | A new seed gives a new variation |
| α is one value per note | Guidance weights can change **per frame**: α automated within a note |

Costs:
- **Generation is slower:** 30–40 network evaluations per note, against
  one for MorphNet-L.
- **Training needs more steps.**
- **Memorization risk:** only 458 clean vocal notes from 2 singers, so the
  model may copy them.

## What stage 0 trains

**Data:** the cached feature image of all 6,515 notes in MIDI 36–82
(`morph/critic_cache.pt`), shape `[5 planes, 122 semitone rows (≤ 20 kHz),
120 frames of 10 ms]`. The planes:
- mid level and side level, relative to the note's peak (floor −60 dB);
- phase-increment coherence, real and imaginary parts;
- modulation depth.

The judges' view is an **exact slice** of this image (tested: max difference
0.0). So MorphNet-L's calibrated judges A and B score generated notes
directly, without rendering audio.

**Classes:**

| Class | Train | Held out |
|---|---|---|
| vox | 382 | 70 (takes "tze", "koh", "EH") |
| synth | 5,069 | 376 (8 families) |
| Exhale (processed vox) | 448 | 70 (presets 14, 15) |
| Harmor vox (synth vox) | 94 | none (only 2 presets) |

Held-out splits are the same as MorphNet's.

**Batches:** class-balanced. vox 30 %, synth 40 % (per envelope class and
family), Exhale 15 %, Harmor vox 15 %.

**Augmentation:**
- pitch shift by ±2 rows (the key label moves with it);
- time shift of −2…+4 frames;
- a random noise floor.

**Model:** `SwinDenoiser`, **244.8 M parameters** (default). It is
KeyboardNet v3's Swin-L backbone and UPerNet head, warm-started from
`keyboard/v3/checkpoints/best.pt`, used as the denoiser on a 160 × 128 grid.
- **Input:** a 1×1 adapter maps the feature image and row/frame coordinates
  to v3's 28 input planes. These feed the pretrained patch embedding and
  stems.
- **Conditioning:**
  - diffusion time and class embeddings are added to v3's adaLN
    conditioning vector (zero-initialized, so training starts from v3's
    behaviour);
  - the key enters through v3's own key embedding.
- **Output:** a new 5-plane output layer on the full-resolution head.
- **Training:**
  - learning rates 5e-5 for the pretrained parts, 2e-4 for the new ones;
  - cosine schedule, v-prediction;
  - classifier-free guidance (15 % of labels dropped to "null");
  - EMA 0.9995.
- **Run size:** 10 epochs × 2,000 steps × batch 16, so 320k notes seen.

**Speed** (measured): **10.2–10.9 s per step on the CPU** (16 threads),
about **2.4 days** in total. Half-epoch checks sample on the GPU (about
1.5 minutes each, 2.4 GB).

**Alternative:** the small `FeatureUNet` (`--arch unet`, 23–45 M) trains
5–7× faster on the GPU: `--arch unet --device cuda --chans 48,96,192,256
--batch 8 --epochs 20`, 0.68 s per step, about 7–8 hours. On the CPU it was
not faster per note than the Swin-L.

**Checkpoints:**
- Every quarter epoch, the progress is saved.
- Every half epoch, a quick check also runs (about 45 s). It samples 12 vox,
  12 synth and 12 "and α 0.5" notes and records their vocality, how often
  both detectors are on, and KID. `best.pt` is chosen by that check.

**Exact pause / resume:** like MorphNet-L. The saved state includes the GPU
random generators.

## Morphing at sampling time (no paired training)

| Mode | Guidance |
|---|---|
| `"and"` | v = v_null + g·α·(v_vox − v_null) + g·(1 − α)·(v_synth − v_null). α may be a curve over the 120 frames |
| `"interp"` | One class embedding: α·vox + (1 − α)·synth |
| `"label"` | One class, for checking fidelity |

The sampler is DDIM (40 steps by default), clamped to each plane's range at
every step. It runs the denoiser in chunks of 16 to fit 4 GB.

**Listening:** `render_features` imposes a generated feature image on a
synth note at the same key:
- the synth keeps its phase, so the pitch is stable;
- noise-like cells get a random-phase copy of it;
- rows above 20 kHz follow the top rows.

## Evaluation and go / no-go

`experiments/eval_morph_diffusion.py` runs in about 20 minutes on the GPU.
Add `--morphnet <checkpoint>` to compare with MorphNet-L on the CPU. It
writes:
- `evals/eval_<checkpoint>_<time>.json`, shown in the dashboard's
  Evaluations with the verdict;
- renders in `demos/eval_<checkpoint>/`, shown in the dashboard's
  Listening.

It measures:
1. **Class fidelity:** vocality and P(vox) of vox, synth, Exhale and
   Harmor-vox samples.
2. **Realism:** KID in judge-A embeddings against real held-out notes. The
   reference is the KID between real training and held-out notes.
3. **Memorization:** the distance from generated vox notes to their nearest
   training note, compared with the same distance for real held-out vox
   notes.
4. **Composition grid:** "and" / "interp" × α 0.25/0.5/0.75 × guidance
   2/4/6. For each: vocality and its error, both detectors on (A, B), KID
   against all real held-out notes, and diversity.
5. **References** on 24 held-out pairs, in the same metrics: the plain mix,
   the vocoder, and optionally MorphNet-L.
6. **α within the note:** α rising 0 → 1 across the note; the vocality of
   the first third against the last third.
7. **Renders** of 6 pairs:
   - "and α=0.5", "interp α=0.5";
   - α automated 0 → 1;
   - a vox sample and a synth sample;
   - the synth carrier itself.

**Go to stage 1 only if all pass:**

| Criterion | Pass if |
|---|---|
| fidelity | vox samples vocality_A ≥ 0.7 and synth samples ≤ 0.3 |
| realism | generated-vs-held-out KID ≤ 3× the real training-vs-held-out KID, for vox and synth |
| composition | best "and" setting at α 0.5: both detectors on (A) ≥ 0.5 and \|vocality − 0.5\| ≤ 0.2 |
| references | that setting is on for both detectors at least as often as the mix and vocoder (and MorphNet-L if given), with a KID no higher than the mix's |
| automation | with α 0 → 1, vocality of the last third minus the first third ≥ 0.3 |
| memorization | generated vox notes are no closer to training notes than half the real held-out distance |

**Stage 1 (if go):** a pair-conditioned diffusion model.
- It is conditioned on the vocal note, the synth note, α (per frame), pitch
  source and articulation.
- It is warm-started from KeyboardNet v3's Swin-L: its blocks already take a
  conditioning vector, and the noise level joins it.
- It generates the hybrid's feature image, which MorphNet-L's stable-pitch
  renderer turns into 192 kHz audio.
- It is compared with MorphNet-L at matched compute.

Either way, the stage-0 model can also polish and vary MorphNet-L outputs:
partly noise a hybrid's feature image, then denoise it toward realistic
notes.

## How to run

```
# train (CPU, Swin-L 245 M, ~2.4 days; checks on the GPU)
PYTHONPATH=src python experiments/train_morph_diffusion.py

# pause / continue exactly (or the dashboard buttons)
python experiments/training_control.py pause  experiments/output/morph/diffusion_s0
python experiments/training_control.py resume experiments/output/morph/diffusion_s0

# evaluate the best checkpoint (add --morphnet experiments/output/morph/v2/checkpoints/best.pt to compare)
PYTHONPATH=src python experiments/eval_morph_diffusion.py
```

## Checks done while preparing (not the experiment)

**Unit tests** (5, all pass):
- the feature-image layout and the judge slice;
- the schedule and v-prediction inversion;
- denoiser gradients and interpolated classes;
- guided sampling: deterministic, in range; "and" with α = 1 equals "vox";
  α curves work;
- `render_features` reproduces a carrier from its own features
  (SNR > 15 dB).

**Smoke run** (scratch folder):
- 8 training steps at the real size: 0.68 s per step, 3.2 GB.
- One pause after step 4 and a resume: it continued at step 5.
- Checkpoints, quick checks, and the full evaluation with tiny settings.
  The report, verdict, references and renders were all produced. The
  verdict on an untrained 8-step model was "no-go", as expected.

**Sizing:**
- On the GPU, batch 16 needed 6.5 GB and spilled out of the 4 GB card at
  27 s per step; batch 8 fits.
- On the CPU, the conv UNet runs at 1.6–2.4 notes per second whatever its
  size, and the Swin-L denoiser at 1.6 notes per second. So the Swin-L gives
  about 5–10× the parameters for the same time.
- Channels-last layout gave no speed-up on this CPU.

**Swin-L checks:**
- a test with a tiny Swin config: shapes, the zero-initialized output, and
  sampling;
- a 2-step CPU smoke run: checkpoints and a GPU check;
- the evaluation script on the Swin-L checkpoint: report, verdict and
  renders.

## Limits

- Stage 0 tests the **idea** (a learned note prior plus composition). Its
  renders impose generated features on a synth carrier. They are not yet
  the pair-conditioned hybrids MorphNet-L makes, so its listening examples
  are indicative only.
- KID here is measured in judge-A embeddings (fast, on the GPU).
  MorphNet-L's validation uses Wavelet2Vec space. The `--morphnet` option
  puts MorphNet-L into the same space for a fair comparison.
- 458 vocal notes from 2 singers. Watch the memorization criterion and
  listen.
