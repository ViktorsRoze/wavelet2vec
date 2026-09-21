# Corpus pipeline validation — v0.31.0

Validation completed on 21 September 2026 with the local Windows `sound`
Python environment. This record covers reproducibility and integration;
it is not a benchmark of learned musical understanding.

| Check | Result |
| --- | --- |
| Full library suite, `python -m pytest -q` | 416 passed; one existing tensor-to-scalar warning in the brightness test |
| Corpus regression suite | 11 passed |
| Editor compatibility suite | 36 passed |
| Runnable Python examples in six foundation/training guides | 16 executed successfully |
| Runtime and installed distribution version | Both `0.31.0` |
| CLI prepare/train/evaluate/resume lifecycle | Completed on four generated stereo recordings with explicit disjoint groups |
| Two existing stereo recordings | Verified, bounded native-rate loading and finite masked-model outputs |

The editor checks covered `test_spectral.py`, `test_spectral_stereo.py`,
`test_spectral_edit_ui.py`, `test_spectral_effects.py`, `test_phase_layer.py`
and `test_library.py`, using Qt's offscreen platform. The editor working tree
remained clean. These checks establish compatibility with its existing DSP
paths; they do not establish adoption of the new corpus trainer in its UI.

The documentation execution covered `HYBRID_ENCODING.md`,
`COMPACT_WAVELETS.md`, `LONG_AUDIO.md`, `ADAPTERS_AND_TRAINING.md`,
`SELF_SUPERVISED_MUSIC.md` and `TARGET_CALIBRATION.md` in a temporary directory.
The new corpus guide's commands use placeholder recording paths; equivalent
commands were exercised by the generated-fixture CLI lifecycle check.

Corpus regressions cover split/group isolation, cross-container exact PCM
duplicates, manifest relocation, deterministic random/fixed crops, source
mutation detection, short-recording crop deduplication, and refusal of
unverified datasets. Resume tests compare the next crop IDs, loss and parameter
update, and require incompatible data provenance to fail before loading model
weights. Evaluation tests cover repeatable masks, group aggregation, module
mode restoration, global-path ablation and hidden-audio isolation in the
visible-neighbor baseline.

The CLI check prepared four generated float WAV files at 8,192 Hz, used two
training and two validation groups, performed one optimizer update, then
resumed for a second update. Both step-specific evaluation reports were
created. This verifies the command lifecycle, not useful representation
learning on a realistic corpus.

For the existing recordings, [corpus_io_smoke.json](corpus_io_smoke.json)
records exact source hashes and crop geometry. Both inputs were eight-second,
44.1 kHz stereo synth loops. Each supplied a 44,288-sample crop, rounded to
the 256-sample hop, yielding 173 context tokens and finite predictions/loss.
Their provenance was unknown, so both remained in one training group.
There were no optimizer updates or held-out comparisons on these recordings.

Next, curate composition/session/pack groups with known provenance and keep
related renders, remasters and loop variants together. Run matched training
seeds and compare held-out group results against the training mean,
visible-neighbor, zero-audio and disabled-global diagnostics. Then add
task-specific probes for instrument identity, rhythm, harmony, loop retrieval
and source separation. Acoustic pretext loss alone cannot validate those
capabilities, or perfect recovery of instruments from a mastered mix.
