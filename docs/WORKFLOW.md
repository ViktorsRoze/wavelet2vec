# Inference Workflow — FL Studio Round-Trip

Status: **implemented** (v0.4.0) — `perform` and the slicing strategies below
are available in `wavelet2vec.perform` / `wavelet2vec.slicing` and as the
`wavelet2vec perform` CLI subcommand. MIDI-based slicing remains future work.

## 0. Audio quality

The processing chain is built for high-resolution material:

- input at any bit depth (16/24-bit PCM, 32-bit float) loads losslessly;
  all internal math is float64;
- the morph runs at the file's **native sample rate** — an 88.2 kHz render
  stays 88.2 kHz end to end, with no resampling or lowpass in the signal
  path (spectral gains above the analysis band hold their edge value rather
  than cutting off);
- output WAV/AIFF is written as **32-bit float** by default (FLAC falls back
  to 24-bit), so nothing is quantized on the way out;
- the character analysis grid covers ~27.5 Hz to ~19.8 kHz, so character
  above 10 kHz (air, brightness, crunch) transfers too. Only the
  *embedding/character analysis* is band-limited; the audio itself never is.

## 1. The problem

`render_sequence` assumes you already have separate snippet files. In real
production the material lives in the DAW: e.g. **16 bars of repeated 8th
notes** rendered as one WAV, plus an **automation clip** drawn over those
16 bars. The tool needs to:

1. find where each note starts in the long render,
2. apply the character morph per note, with the automation clip controlling
   the development (amount, or ADSR attack/decay/sustain/release),
3. reassemble everything at the **exact original length**, so the result can
   be dropped back into FL Studio at bar 1, sample-aligned with the project.

The round-trip property (output length == input length, no time
displacement) is the core requirement — it makes the tool behave like an
offline effect rather than a generator.

## 2. Key insight: one timeline

The render and the automation clip come from the same project, so they share
one timeline. Automation must therefore be sampled at each note's **start
time in seconds within the render** (mapped through
`AutomationCurve.value_at_fraction(start / total_duration)`), not at a
per-snippet index. A clip drawn over 16 bars then lines up with the notes
automatically, whatever the note count.

## 3. Slicing strategies (layered, in order of preference)

| Strategy | Input | When to use |
| --- | --- | --- |
| **Grid** | `--bpm 140 --division 8 [--offset-beats 0]` | Quantized FL patterns — exact and dependency-free. 16 bars of 8ths at 140 BPM = 256 slices of 60/140/2 s each. **Recommended default.** |
| **MIDI notes** | `--notes pattern.mid` | The pattern's MIDI export; note-on times = slice starts. Handles swing, gaps, and uneven rhythms. FL exports MIDI from any pattern. |
| **WAV cue markers** | `--slice cues` | Your marker idea: Edison (and some export paths) can embed cue points in the WAV's RIFF `cue ` chunk. Readable with the Python stdlib (no new dependency). |
| **Marker file** | `--markers markers.csv` | A plain `time_seconds` list — escape hatch for any external marker source. |
| **Onset detection** | `--slice onsets [--sensitivity 0.5]` | Unquantized material (live takes, vocals). Reuses the scalogram onset-flux machinery already in `transient_features`. Least exact; offered as fallback. |

About audio markers specifically: they work, but they're the most manual
path (placing/maintaining markers per note) and FL's plain WAV export does
not always carry them. For grid-locked patterns the BPM+division grid is
strictly easier and exact; markers stay supported (`cues` / `markers.csv`)
for cases where the grid doesn't apply.

## 4. Per-note processing chain

For each slice `[start, next_start)`:

1. sample the automation at the slice start → modulated ADSR / amount
   (same mapping as `render_sequence`: `amount` = depth, `attack`/`decay`/
   `release` time × (1 − value), `sustain` × value);
2. apply the character morph with the ADSR envelope, **gate = slice
   duration** so the release completes before the next note;
3. optional volume ADSR;
4. the processed slice keeps its exact length.

Boundary handling: slices are cut hard at note starts, so a note's ring-out
lives inside the next slice and gets that slice's processing. A short
(~5 ms) crossfade at each boundary hides any gain discontinuity. This is the
honest v1; full overlap-add of ringing tails is listed under future work.

Output: one WAV, same sample rate, channel count, and sample count as the
input. Re-import at bar 1.

## 5. API and CLI

```python
from wavelet2vec import perform, slice_grid, slice_cues, slice_markers, slice_onsets

starts = slice_grid(n_samples, sample_rate, bpm=140, division=8)   # -> sample indices
starts = slice_grid(n_samples, sample_rate, spacing_seconds=0.25)  # fixed note distance
starts = slice_cues("performance.wav", n_samples)                  # embedded WAV markers
result = perform(
    audio, sample_rate, character, starts,
    automation=curve, character_adsr=ADSR(0.4, 0.1, 1.0, 0.15),
    modulate="attack", max_amount=1.0, volume_adsr=None,
    boundary_fade_seconds=0.005,
)
```

```text
wavelet2vec perform --input performance.wav --output performed.wav
    --character guitars/ (or .npz / single file)
    --automation clip.flp|.fst|.csv|.json
    --slice grid --bpm 140 --division 8 [--offset-beats 0] | --spacing 0.25
  | --slice cues
  | --slice markers --markers markers.csv
  | --slice onsets [--sensitivity 0.5]
    [--adsr A D S R] [--volume-adsr A D S R]
    [--modulate amount|attack|decay|sustain|release] [--max-amount 1.0]
    [--boundary-fade 0.005]
```

## 6. The FL Studio session, step by step

1. Arrange the pattern (e.g. vocal vowel on repeated 8ths, 16 bars).
2. Draw the automation clip over the same 16 bars (any parameter — only its
   shape is used).
3. Export: the stem as WAV (no master FX), and the automation by saving the
   project (`.flp`) or exporting the curve to CSV.
4. `wavelet2vec perform --input stem.wav --character guitars/ --automation
   project.flp --slice grid --bpm 140 --division 8 --adsr 0.4 0.1 1.0 0.15
   --modulate attack --output stem_performed.wav`
5. Drag `stem_performed.wav` into the playlist at bar 1.

For material that isn't arranged yet, the existing `sequence` command
(separate one-shot files, automation across the sequence) already covers the
folder-based workflow.

## 7. Implementation status

Implemented and tested (`tests/test_perform.py`): `slicing.py` (grid by
BPM/division or literal spacing, marker CSV/text, RIFF `cue ` chunk reading,
onset detection via scalogram flux), `perform.py` (per-note loop,
automation-by-time sampling, boundary fades, sample-exact output asserted),
and the `perform` CLI subcommand. Verified at 88.2 kHz: output shape equals
input shape, 32-bit float round trip is bit-exact, and a rising automation
clip produces monotone development toward the character in embedding space.
Still future: `slice_midi` (note-on parser — soundgen's `music/score.py` has
one to port).

## 8. Open questions / future work

- **Tail overlap-add**: process each note with its natural ring-out and
  overlap-add, instead of hard cuts (better for long releases, more complex).
- **Per-note pitch awareness**: slices could reuse their own f0 for
  pitch-synchronous character application.
- **Live preview**: a small watch-folder mode so FL's "edit and re-export"
  loop feels interactive.
- Whether FL's WAV export reliably preserves Edison cue points on your
  setup — to verify with a real export; grid/MIDI paths don't depend on it.
