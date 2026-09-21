**Optional training-only target calibration — v0.30.0, 16 September 2026**

The first masked-acoustic experiment did not beat a training-mean constant
target. One plausible problem is that the fixed teacher's bands/features have
very different scales. This release adds an optional experiment in balancing
those targets, while preserving the clean/student boundary.

`MaskedPredictionConfig(standardize_targets=True)` enables a frozen per-feature
mean and standard deviation fitted on training data. It remains **off by
default**: three small synthetic runs show a modest average improvement, but
only one beats the constant baseline. This is not yet a robust music result.

The [validation record](benchmarks/CALIBRATION_VALIDATION.md) covers 405
library tests, 36 editor compatibility tests and sixteen executed guide examples.

```python
import torch
from wavelet2vec import (
    HybridConfig, HybridTokenEncoder, MusicContextConfig,
    MaskedPredictionConfig, MaskedMusicModel,
    collate_audio, MusicPretrainer, PretrainingConfig,
)

analysis = HybridConfig(sample_rate=8192, backend="compact_wavelet",
                        n_bands=8, f_min=64, f_max=2048, q=2,
                        hop_length=16, padding_samples=32)
model = MaskedMusicModel(
    HybridTokenEncoder(analysis, channels=2, dim=8),
    context_config=MusicContextConfig(dim=8, heads=2, local_layers=1,
        global_layers=1, local_tokens=4, dropout=0.0),
    prediction_config=MaskedPredictionConfig(
        span_tokens=2, standardize_targets=True, target_scale_floor=0.01),
)
training_batch = collate_audio(
    [torch.randn(2,128)*0.1, torch.randn(2,111)*0.1], sample_rate=8192)
summary = model.fit_target_statistics(
    [training_batch], calibration_id="guide-training-split-v1")

trainer = MusicPretrainer(model, PretrainingConfig(warmup_steps=0, total_steps=3))
trainer.step(training_batch)

query = collate_audio([torch.randn(2,128)*0.1], sample_rate=8192)
model.eval()
with torch.no_grad():
    output = model(query, generator=torch.Generator().manual_seed(7))
    raw_predictions = model.from_target_space(output.predictions)
    raw_targets = model.from_target_space(output.targets)
print(summary, raw_predictions.shape)
```

**The fit is global, weighted, frozen and explicit.** Teacher features are
accumulated in float64 with weighted Welford updates. Real sample counts weight
each token, including partial final tokens. Every feature is transformed as
`(feature - training_mean) / max(training_std, scale_floor)`. The floor limits
amplification of nearly constant dimensions. It is an objective hyperparameter,
not a perceptual calibration or new information extracted from the audio.

The operation commits buffers only after consuming the calibration iterator
successfully. An invalid batch, empty fit or unrepresentable scale leaves the
module unfitted. A second fit is rejected: create a new model for a different
calibration experiment. Training forward passes require a fitted calibration
when enabled. Queries never update statistics. `encode()` still produces
contextual representations without computing clean teacher targets.

Only training data should be passed to `fit_target_statistics`. The caller's
`calibration_id` is hashed and stored for provenance; it does not prove that
the supplied data belong to the correct split. Prefer an actual training
manifest identifier and keep validation/test examples out of calibration.
Statistics must not be recomputed from the current clean example: that would
change the objective and risk exposing example-specific hidden information.

`FixedTargetStatistics` is also available separately. It accepts chunks of
feature tensors and integer duration weights, with explicit device/dtype
matching. Mean, observed standard deviation, floored scale, sample count,
fitted flag and calibration identifier are checkpointed buffers, not trainable
parameters. They move with the model and participate in feature-cache model
identity. The standardization does not change student waveform inputs or
the original invertible carrier.

**Prediction units are explicit.** With calibration enabled, `predictions`
and `targets` in the masked output are in standardized target space.
`to_target_space` and `from_target_space` apply/invert the frozen mapping.
Without calibration they are identity functions. Compare models in common
raw teacher-feature units as well as their training objective; normalized
loss values cannot be directly compared with unnormalized ones.

**A matched experiment tests the hypothesis.**

```powershell
& C:\Users\vikto\an3\envs\sound\python.exe -m experiments.calibration_ablation
```

Each configuration uses 80 updates, the same 24 training and eight validation
clips, the same training/validation masks and batch order, and three matched
initialization seeds. Calibration uses the training clips only. The clips
are short synthetic harmonic textures (roughly 47–94 ms at 8.192 kHz), not
long or complex music. All comparisons below use the original unnormalized
masked objective, including the original phase/magnitude weights.

| Initialization seed | Raw-target model | Calibrated-target model | Constant training-mean baseline |
| --- | --- | --- | --- |
| 17 | 0.001872 | 0.001551 | 0.001800 |
| 18 | 0.001879 | 0.001839 | 0.001800 |
| 19 | 0.001976 | 0.001829 | 0.001800 |
| Mean | 0.001909 | 0.001739 | 0.001800 |

Calibration improves mean error by about 8.9% versus the matched raw-target
models, but only **one of three** calibrated runs beats the constant baseline.
The apparent advantage is not robust across initializations. The baseline is
the training mean, not an optimized best-possible constant under every loss.

As an additional diagnostic, removing all visible acoustic input from the
calibrated students gives mean raw error about 0.001902, versus 0.001739 with
the actual input. This suggests some acoustic conditioning on this split;
it does not isolate long-context benefits or establish musical semantics.
All model/data/mask seeds, calibration summaries and results are recorded in
[target_calibration_ablation.json](benchmarks/target_calibration_ablation.json).

The first single-seed calibrated probe is retained separately as historical
exploration; the six-run matched matrix is the reported comparison. No
hyperparameter sweep or real-music test was performed. More realistic targets,
longer masks, larger datasets and strong constant/local baselines remain
necessary before recommending a training recipe.

**Compatibility.** Prediction configuration writes `masked-prediction-v2`.
The reader accepts v1 raw-target configurations and supplies disabled
standardization defaults. Raw v0.29 model state can migrate through the model
state reader without new calibration buffers. Exact trainer resume remains
strict about implementation identity; use a deliberate weights-only migration
across code versions. New calibrated checkpoints restore fitted statistics
into a freshly constructed calibrated model, without refitting from queries.
