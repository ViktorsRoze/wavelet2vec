# note_dataset: every note of 150 synth sounds, C1 to B10

By Viktors Roze. **144 files were rendered** in FL Studio 2026 (30 September
to 4 October 2026). **6 files are hardware recordings** of a Korg monologue
analog synth (6 October 2026). Each file is one sound (synth patch plus
effects) playing every chromatic note from FL Studio's **C1 to B10**, one
note per beat at **50 BPM**. Five more files are the author's **own sung
vocals**, recorded along with the same score (see
[Sung vocal take](#sung-vocal-take-own-voice)).

## Purpose

This is a **sound-design dataset** with two goals:

1. **Generate a whole keyboard from one or two example notes** (described
   below).
2. **Create vocal–synth hybrid sounds:** split the data into vox and synth
   sounds, and generate new sounds that a vox/synth classifier rates 50/50.
   See [Second purpose: vox–synth hybrids](#second-purpose-voxsynth-hybrids).

It records how a sound changes from note to note across the whole keyboard.
For the first goal, it is meant for training a generative model that:

- receives one or a few example notes of a sound, for example two notes;
- infers what the sound is, and how it already changes between the examples
  (brightness, decay length, unison spread, distortion character, formants);
- **generates the remaining notes as high-quality audio**, at the 192 kHz /
  32-bit float resolution the dataset was rendered in.

The model is a generator, not a note detector. Note numbers are known from
the FL Studio score ([c1_till_b10.fsc](c1_till_b10.fsc)) and serve as
conditioning inputs ("which key to generate").
The pitch measurements in this document exist only to verify that every slot
contains what the grid says it does.

**Verdict: usable for that goal.** Every file is a complete 120-note
trajectory. The renders have a sample-exact grid; the recordings have a
measured one (see [Hardware recordings](#hardware-recordings-korg-monologue)). That gives ground truth for every
"given these notes, generate that note" question. Before training, filter a
known set of slots and patches (see [Known issues and nuances](#known-issues-and-nuances)).
The main limitations are:

- **Timbre diversity:** 150 files reduce to about 70 independent base sounds.
- **Overlapping notes:** in 89 files a note's release runs into the next
  note's slot.
- **Mixed sample rates:** 51 files are 44.1 kHz and 6 are 96 kHz.
  The 96 kHz recordings carry no real content above about 24 kHz.

All measurements come from
[experiments/note_dataset_audit.py](../experiments/note_dataset_audit.py).
The per-file results are in [dataset_audit.csv](dataset_audit.csv). The audio
was only read. One file was renamed with the author's approval (issue 19).

## At a glance

| | |
|---|---|
| Files | 150 grid WAVs (144 renders + 6 recordings), 22.3 GiB, about 6.0 hours; plus 5 sung vocal takes (157 s) |
| Format | All **32-bit float, stereo**. Renders: 93 at **192 kHz**, 51 at **44.1 kHz**. Recordings: 6 at **96 kHz** |
| Layout | 120 slots of 1.2 s. Slot *k* (0-based) is MIDI note **12 + k**. Exact in renders, fitted in recordings |
| Note range | FL C1 to B10 = MIDI 12 to 131 = **16.35 Hz to 15.8 kHz** fundamentals |
| Slots with sound | 15,360 of 18,000 (85 %) |
| Sounds | Plucks, basses, bells and leads, plus vocal patches. **No pads** |
| Chromatic steps | 86 % of consecutive notes are exactly +1 semitone apart (±10 cents) |
| Levels | Not normalized; file peaks −23.5 to 0 dBFS; no sample exceeds full scale |
| Base sounds | About 70 families; variants of one patch count once |

FL Studio names MIDI 60 "C5". So FL C1 is MIDI 12 (scientific C0, 16.35 Hz)
and FL B10 is MIDI 131 (about 15.8 kHz). Standard MIDI stops at 127 (FL G10).

## How the files were made

Every file was made from the same piano roll, saved as the FL Studio score
[c1_till_b10.fsc](c1_till_b10.fsc). The renders played it through plugins;
the recordings play the same notes on the Korg monologue (author). It is the
authoritative note list.
Decoded (`fl_score_notes` in the audit script):

| Field | Value in all 120 notes |
|---|---|
| Keys | MIDI 12 to 131, ascending by one semitone |
| Position | Note *k* starts at beat *k* (96 ticks per beat), so key = 12 + beat |
| Length | 96 ticks = **exactly one beat**: each note-off coincides with the next note-on |
| Velocity | 100 (FL scale 0–128); release velocity 64 |
| Pan, fine pitch, mod X/Y | Centre, no detune, defaults; no slides or portamento |

The author adds:

- **Note length:** for every sound the note fills (almost) the whole beat.
  Most sounds end earlier because their filter envelope decays.
- **Sounds:** presets are usually plucks, basses, bells or leads. No pads
  were used.
- **Arpeggiated patches:** some patches are arpeggiated.
- **Two-note patches:** some sounds contain two notes at once because of
  detuned oscillators.
- **Patch details:** for some synths the filename lists the sound design.
  For example, `Harmor_synth_playing_simple_saw_pluck_40_cent_vibrato_250ms_speed`
  is Harmor with a simple saw, a pluck filter envelope and 40-cent vibrato
  with a 250 ms period.

## File naming

`from_C1_till_B10_<synth>_<patch description>.wav`

| Synth token | Plugin | Files | Rates |
|---|---|---|---|
| `Harmor` | Image-Line Harmor | 47 | 44 × 192 kHz, 3 × 44.1 kHz |
| `kontakt_shalanaja` | Best Service **Shevannai** in Kontakt: real human voice (see below) | 13 | 44.1 kHz |
| `kontakt_vox` | Output **Exhale** in Kontakt: heavily processed vocals (see below) | 15 | 12 × 44.1 kHz, 3 × 192 kHz |
| `kontakt_piano` | A Kontakt piano library ("piano gigant") | 1 | 44.1 kHz |
| `repro-1` | u-he Repro-1 | 20 | 192 kHz |
| `Serum` | Xfer Serum | 16 | 192 kHz |
| `massiv_x` | Native Instruments Massive X | 12 | 44.1 kHz |
| `iZotope_Iris2` | iZotope Iris 2 | 8 | 44.1 kHz |
| `Sakura` | Image-Line Sakura | 4 | 3 × 192 kHz, 1 × 44.1 kHz |
| `synthplant` | Sonic Charge Synplant | 3 | 2 × 192 kHz, 1 × 44.1 kHz |
| `T_synth`, `zeta+`, `avangers` | as named (Avangers presumably Vengeance Avenger) | 2, 2, 1 | 192 kHz |
| `Korg_monolog_analog_synth` | **Korg monologue** analog synth, recorded through an Audient interface into Sound Forge Pro 13 | 6 | 96 kHz |

The patch description lists the sound design, in signal-chain order where
several effects are named:

- **Oscillator and filter:** `simple_saw`, `simple_square`, `pluck` (pluck
  filter envelope), `slower_filter`, `comb_filter`, `high_resonanse_filter`,
  `hl6plus`. Serum wavetables: `4088_waveform`, `acid_waveform`,
  `Analog_BD_Sin_waveform`, `BAsic_Mg_waveform_and_Default_waveform_minus_one_octave`.
- **Modulation and voicing:** `N_cent_vibrato` (depth), `_250ms_speed`
  (vibrato period), `N_voice_unison`, `unison_2`, `N_procent_prism` and
  `minus_N_procent_prism` (Harmor prism), `blur_trem`, `harmonizer_max`.
- **Effects:**
  - Dynamics: `burning_compressor`, `burning_compressor_max`,
    `multiband_compressor`.
  - Distortion: `bitcrush`, `bitjam`, `classic`, `cube`, `flathat`, `hill`,
    `log`, `robber` and `soft` `_distortion`.
  - Modulation: `chorus`, `flanger`, `phaser_cascading`.
  - Ozone exciter modes: `ozone_exciter_analog`, `_tape`, `_triode`, `_tube`.
  - Reverb: `reverb`, `convolutional_reverb`, `convolutional_reverb_wet`,
    `serumfx_reverb_fx`, `fx2`.
- **Presets:**
  - `preset_N` / `presetN`: presets without further description; plucks,
    basses, bells or leads.
  - `shalanaja_<syllable>`: "shalanaja" is the author's short name for
    [Shevannai](https://www.bestservice.com/en/shevannai.html) (Best
    Service). Each file is one sung syllable (`from`, `koh`, `le`, `leh`,
    `lih`, `mu`, `ni`, `no`, `sa`, `shu`, `tus`, `tze`). According to the
    publisher, the soprano is Lara Ausensi, recorded **without vibrato** and
    without added effects, as mono 44.1 kHz / 24-bit samples. This explains
    why these files are 44.1 kHz and dual-mono.
  - `vox_N`: presets of [Exhale](https://output.com/products/exhale)
    (Output, "Modern Vocal Engine"). Per the author, its vocals are **heavily
    processed with LFO and distortion**.
  - `rhodes`, `piano_gigant`, `organ`, `string` and `bass` name the
    instrument.

`category_from_name` in the CSV records only what a filename states:
58 pluck, 30 vox, 2 bass, 2 keys, 1 organ and 1 string. That leaves 50
unlabelled presets. Of those, 36 hold through the beat, 10 ring past it and
4 decay within it. That fits mostly leads and basses, plus a few
plucks/bells, but which is which has not been checked.

## Timing grid and embedded metadata

At 50 BPM one beat is 1.2 s: **230,400 samples at 192 kHz**, **115,200 at
96 kHz** and **52,920 at 44.1 kHz**.

**Renders (144 files):** every file starts exactly on beat 0. Most are
144.012 s long; 55 run longer (up to 162.1 s) because the last note's release
continues after slot 119. Slot *k* = samples `[k·beat, (k+1)·beat)`, note
MIDI `12 + k`. This matches the score exactly, and is
`wavelet2vec.slicing.slice_grid(n, rate, bpm=50, division=4)`.

**Recordings (6 files):** trimmed by hand, so the grid is measured, not
assumed. Slot *k* starts at `grid_origin_ms + k · beat_ms` (both columns in
the CSV). Per-slot start samples are in the audit's `notes.csv` as
`slot_start`. See [Hardware recordings](#hardware-recordings-korg-monologue).

Take note numbers from the score or the grid. Do not use the embedded cue
markers (issue 1). Onset detection is only needed to fit the recordings'
grid.

FL Studio wrote cue markers into the renders:

- **130 files:** 120 markers lie exactly on the grid. FL's `tlst` chunk
  stores a MIDI note-on per marker, equal to 12 + beat index.
- **2 files have no markers:** `Harmor_synth_playing_simple_saw_pluck` and
  `Harmor_synth_playing_simple_square_pluck` (renamed). Their audio follows
  the grid.
- **12 `kontakt_shalanaja_*` files** have wrong markers (issue 1).
- **Generic chunks:** `smpl` (unity note 60) and `acid` (root 60, 50 BPM)
  are the same in every render and carry nothing per note. Each marker's text
  label is the plugin name, e.g. "Serum" or "Repro-1".
- **The recordings** carry Sound Forge markers instead: start and end
  points, plus one marker labelled "best note" in preset 5. None of them
  give note positions.

## Hardware recordings (Korg monologue)

Six takes of a Korg monologue (an analog, monophonic synth). The author
played the same notes as the renders and recorded them through an Audient
audio interface, which limits their quality. All are 96 kHz / 32-bit float,
edited in Sound Forge Pro 13.

| File (`Korg_monolog_analog_synth_…`) | Sounds at | Grid start | Beat | Onset jitter (median / max) | Slots cut by the trim |
|---|---|---|---|---|---|
| `preset_1_bass` | octave below the key | −79 ms | 1199.85 ms | 3.7 / 36.5 ms | C1 |
| `preset_1_bass_attempt2` (second take of preset 1) | octave below the key | −43 ms | 1199.88 ms | 2.9 / 26.0 ms | C1 |
| `preset_2_bass_with_modulation` | the key | −356 ms | 1200.57 ms | 10.4 / 38.4 ms | C1 |
| `preset_3_sequance` | a sequence: 0 / +4 / +7 semitones | −36 ms | 1200.15 ms | 11.7 / 41.3 ms | C1, B10 |
| `preset_4_bass` | octave below the key | −27 ms | 1200.01 ms | 12.9 / 37.3 ms | C1, B10 |
| `preset_5_bass` | the key | −24 ms | 1200.01 ms | 13.0 / 25.9 ms | C1, B10 |

"Grid start" is where beat 0 (C1) would begin, relative to the first sample:
every take starts just after the C1 attack. "Sounds at" for presets 2 and 5
comes from note-to-note tracking; their pitch estimate is ambiguous by a 3rd
harmonic (CSV shows −19 / +19). Issues 23–31 list what this means for
training.

## Sung vocal take (own voice)

`from_C_real_Vocal_recording_melodyne.wav` is the author singing along with
the piano score: 96 kHz / 32-bit float, stereo with near-identical channels
(L−R −114 dB), 88.2 s. The pitch was stabilized afterwards in Melodyne. It
is **not** laid out on the C1..B10 slot grid, so its notes are listed in a
sidecar instead:

- [from_C_real_Vocal_recording_melodyne.notes.csv](from_C_real_Vocal_recording_melodyne.notes.csv):
  one row per sung note. Columns: start/end in seconds and samples, sung
  MIDI note (scientific and FL names), cents, pitch spread, level, the score
  note playing at the time and the interval to it.
- The same 71 notes are **embedded as named regions in the WAV** (cue + LIST
  `adtl` chunks), e.g. "C2 (FL C3) pass 1", so Sound Forge shows them.
  The audio samples were verified unchanged.

Produced by [experiments/annotate_sung_notes.py](../experiments/annotate_sung_notes.py):
pYIN pitch tracking, held-note segmentation, and a score grid fitted to the
chromatic pass.

**What was sung.** Note names are scientific (C4 = middle C); FL names are
one octave higher.

| Pass | Time | Notes | Relation to the piano |
|---|---|---|---|
| 1 | 0.3–49.4 s | **C2 → E5 chromatic**: all 41 pitches, one per beat (42 regions; a held D2 is split in two) | Exactly on the piano note from D2 (3.5 s) to E5. The first three notes run a semitone ahead while settling in |
| 2 | 50.4–51.8 s | E4, F4 | Too high: dropped an octave, one beat late (13 semitones under the piano) |
| 3 | 52.6–63.9 s | F#2 A#2 C3 D3 D#3 F3 G#3 A#3 B3 C#4 | Restart low; rises faster than the piano, not chromatic |
| 4 | 64.7–71.2 s | F#2 B2 C3 D3 E3 F3 | Restart low |
| 5 | 73.1–85.5 s | A2 C3 D#3 F3 G3 G#3 A#3 C4 C#4 D4 D#4 | Restart low |

The piano was 27.7 s into the score (FL B2) when the take started. Passes
3–5 sing under very high piano notes (FL G6–A#8), which explains why they
stop following it. Every note is labelled with the pitch **sung**, not the
piano note.

**Totals:** 71 notes covering 41 distinct pitches (MIDI 36–76, C2–E5). 20
pitches were sung more than once (F#2–F4), giving repeated takes of the same
note.

**Syllable: every note is sung as "Piuw"** (author; column `syllable`). Each
note is a small gesture, not a steady vowel:
- **A consonant onset ("P") before the pitch starts.** The 80 ms before each
  detected pitch onset already carry sound (median −25 dBFS, against
  −50 dBFS in the gaps).
- **A vowel glide from bright "ee" (i) to dark "oo" (u/w).** The
  upper-formant band (1.8–3.5 kHz) falls by a median of **13 dB** from the
  start to the end of the note, in 64 of 71 notes. This moving formant is
  one of the cues listeners rely on most to hear a voice, so it is the part
  of this take worth carrying into hybrids.

**Nuances:**
- **Lightly processed by design.** Melodyne stabilized the pitch: notes sit
  within ±10 cents of the semitone, with 1–17 cents of remaining wobble, so
  natural intonation is mostly removed. The author welcomes this: the end
  goal lies between voice and synth, and an unusual, slightly processed
  source suits that.
- **Clean, isolated notes.** Sung notes last a median 0.48 s (0.2–1.3 s),
  with gaps of median −49.5 dBFS between them. Unlike the held synth
  presets, nothing overlaps. `start_sample` marks where the **pitch**
  begins, so start each slice about 100 ms earlier to keep the "P"; end it
  at `end_sample` plus a short margin. Never use the 1.2 s grid.
- **Band-limited.** Energy above 8 kHz is −48 dB and above 20 kHz −86 dB of
  the total, so the 96 kHz file carries nothing real above about 20 kHz.
- **Peaks** reach −3.0 dBFS; nothing clips.
- **Not in `dataset_audit.csv`,** which covers only the 150 grid files.

### Four single-vowel takes

The author also sang one vowel per take, along with the same score, over
the part of the range their voice reaches. The filename gives the sound and
the first note; every take starts on that note in scientific octave 3. All
are 96 kHz / 32-bit float, Melodyne-stabilized, one note per beat, with
regions embedded in the WAV and a notes table beside it.

| File (`…_real_Vocal_recording_melodyne_…`) | Sound | Notes | Range | Gaps |
|---|---|---|---|---|
| `from_A_…_sound_A_` | A | 10 | A3 → A4 | A#3, B3 and E4 not detected |
| `from_E_…_sound_EH` | EH | 21 | E3 → C#5 | F#3 not detected |
| `from_F#_…_sound_O_` | O | 16 | F#3 → A4 | — |
| `from_G#_…_sound_E_` | E | 10 | G#3 → F4 | — |

Own voice in total: **128 notes in 5 takes, MIDI 36–76.** The vowel takes
cover MIDI 52–73, and every single-vowel take has several notes there.

### Detected vowels and parts

Every notes table also has the output of the
[vocal detector](../docs/API.md#tools-vocal-detector)
(`wavelet2vec.tools.detect_vocal_parts`). Its columns:
- `vowel_detected` and `vowel_share`: the note's majority vowel and its
  share of the frames;
- `vowel_start` and `vowel_end`: the vowel of the note's first and last
  thirds;
- `sibilant_frames` and `plosive_frames`: counts of those frames.

The tables use voice scale 1.04, which best fits the four labelled takes;
per-file estimates drift because each take holds one vowel.

| Take | Sound | Detected (notes) | Reading |
|---|---|---|---|
| O | O | **o** 15/16 | Correct |
| EH | EH | **eh** 11/21; u/o 8 | Partly correct; the misses fall on darker notes |
| E | E | **i** 10/10 | Correct acoustically: the sound has "ee"'s low F1 and high F2 (≈2.5 kHz), like the English letter name "E" |
| A | A | o 9/10 | A dark, covered "ah": flat 300–700 Hz, falling steeply above 1 kHz (F2 ≈ 1 kHz). Singers often darken "ah" like this; the generic "a" template expects F2 ≈ 1.2–1.4 kHz |
| Piuw | Piuw | **i** at the start 53/71; "P" bursts before 7 notes | The "ee"→"oo" glide is in the audio (measured above), but the detector seldom labels the short, quiet "oo" ending |

No sibilants were sung, and at most 3 notes per take show a few sibilant
frames, from breathy onsets. To make the detector fit this voice better, the
next step is **speaker enrollment**: measure templates from these labelled
takes instead of the published averages.

## Known issues and nuances

Each item lists what was measured, which files are affected, and what to do.
The CSV column holding each per-file value is named where one exists.

**Metadata and naming**

1. **Wrong cue notes in the shalanaja files.** The 12 `kontakt_shalanaja_*`
   files (all except `sa_`) have 76 markers starting at beat 44, and their
   stored notes are beat + 4. Both the score and the audio contradict this:
   sound occurs only in the slots for MIDI 57 to 82, at the pitch the grid
   predicts. → Use the score / grid.
2. **Two files have no markers** (listed above). → The grid applies.
3. **Generic `smpl` and `acid` chunks.** Root note 60 and tempo 50 are the
   same in every file. → Ignore them.
4. **Spelling in names:** `procent` (percent), `resonanse`, `massiv_x`,
   `avangers`, `synthplant` (Synplant), `BAsic`, `monolog` (monologue),
   `sequance`. → Harmless; the audit
   script matches these spellings.

**Which slots contain a usable note**

5. **Partial key ranges.** Sampled instruments cover only part of the range;
   the remaining slots are digital silence (below −100 dBFS). All other files
   use all 120 slots.

   | Files | Slots with sound |
   |---|---|
   | `kontakt_vox_*` (15, Exhale) | MIDI 48 to 78–91 (FL C4 to F#6–G7), 31–44 notes |
   | `kontakt_shalanaja_*` (13, Shevannai) | MIDI 57 to 81–82 (FL A4 to A6/A#6, 220–932 Hz: the soprano's range), 25–26 notes |
   | `kontakt_piano_gigant` | MIDI 21 to 111 (FL A1 to D#9), 91 notes |
   | `Harmor_synth_preset_old_vox` | MIDI 12 to 94, 78 notes (gap at MIDI 84 to 88) |
   | `massiv_x__preset_11` | MIDI 12 to 99, 88 notes |
   | `synthplant_preset` | 81 notes, gaps in the low range |
   | `avangers_preset_1` | MIDI 12 to 116, 105 notes |
   | `Harmor_synth_preset_rhodes` | MIDI 12 to 128, 117 notes |

   → Never use a silent slot as a target. Treat range limits as part of the
   instrument when sampling context and query notes.
6. **The top four notes repeat in 46 files.** MIDI 128 to 131 (FL G#10 to
   B10) sound identical to MIDI 127: each step measures 0 cents. Those
   plugins stop at MIDI 127. Harmor, an FL native plugin, renders them
   correctly. → Drop these 184 slots, or relabel them as MIDI 127. Column:
   `clamped_above_midi127`.

**Note boundaries**

7. **Releases spill into the next slot.** In the score every note-off falls
   exactly on the next note-on. Any sound still playing at note-off releases
   into the next slot:
   - **61 files decay within the slot** (49 of them labelled plucks): each
     slot holds one complete, isolated note.
   - **28 files ring** into the next slot: at the slot end they are 20 to
     60 dB below their peak.
   - **61 files are held** through the slot: at the slot end they are still
     within 20 dB of their peak. These are leads, basses, vox and keys.

   In those 89 files, a slot starts with the previous note's release, one
   semitone lower, and loses its own release. An isolated note with its full
   release cannot be cut from these files. → Columns: `envelope`,
   `tail_at_slot_end_db`. See [Overlapping notes](#overlapping-notes) below.
8. **Slow attacks.** Most onsets are under 1 ms after the grid. The
   exceptions:
   - `Sakura_*`: up to 245 ms.
   - `synthplant_preset`: 305 ms.
   - Shevannai (`shalanaja`) sung syllables: 8–30 ms.

   → Keep onset timing as part of the sound; never re-align notes to their
   onset.

**Pitch behaviour, all part of the sound design**

9. **Arpeggiated patches.**
   - `massiv_x__preset_5` steps +2 to +4 semitones within 39 % of its slots.
     Across slots it cycles a 5-step pattern (+250, −50, +200, 0, +100 cents)
     that nets one semitone per note.
   - `massiv_x__preset_8` changes pitch by intervals unrelated to the played
     key (+480, −300, +500 cents where +600 is played).
   - `massiv_x__preset_4` barely follows the keyboard: notes 6 semitones
     apart produce the same spectrum.

   Other arpeggiated patches the author knows of should be added here. → The
   arpeggio pattern depends on time and is not determined by the note
   number. Exclude these three from note-to-note generation, or treat
   arpeggios as a separate task.
10. **Two-note and layered sounds.** Some patches sound two notes at once.
    Some are octave layers that decay at different rates. Others are detuned
    oscillators (author's note). Their louder component switches within a
    slot. This shows as ±12/±13 semitone jumps in 15 files with movement in
    at least 10 % of slots, for example:
    - Repro-1 presets 6, 9, 14 and 18;
    - Serum `4088`, `acid` and `BAsic_Mg..._minus_one_octave`;
    - `iZotope_Iris2__preset_8`;
    - Exhale `kontakt_vox_4` and `kontakt_vox_14`, from its processing.

    `massiv_x__preset_10` is perfectly chromatic, but its pitch estimate
    alternates between two notes. → Keep these sounds: reproducing such
    layers is part of what the generator must do. Their "pitch" is not one
    number. Columns: `moving_slots`, `main_jump_semitones`.
11. **Octave-transposed presets.** 43 files sound whole octaves away from the
    played key: 25 at −12, 18 at −24 and 5 at +12. A few show fifths or other
    intervals, from layering. The offset is constant within a file. → The
    generator should learn it from the examples. Column:
    `sounding_offset_semitones`.
12. **Tuning and intonation.**
    - Most patches are within ±6 cents.
    - `T_synth` presets sit about −16 cents flat.
    - Shevannai drifts ±15–20 cents from step to step: natural intonation
      of a voice recorded without vibrato.
    - Exhale `kontakt_vox_12` and `kontakt_vox_6` contain whole-tone pitch
      glides, consistent with its LFO processing.
    - The 30/40/100-cent vibrato files move pitch by design.

    → All of this is part of the sound. Do not "correct" it.
13. **The lowest octave.** Fundamentals of 16–31 Hz are below most monitors.
    Many patches behave unusually there, and spectral measurements are least
    reliable. → Weight the top and bottom octaves separately in evaluation.

**Resolution and signal**

14. **Mixed sample rates.** 93 files are 192 kHz and 51 are 44.1 kHz. → See
    [Resolution](#resolution-train-for-192-khz-output).
15. **Real ultrasonic content.** Of the 93 files at 192 kHz:
    - **47** contain real content above 24 kHz (more than −60 dB of note
      energy). Distortion, exciters and unison FX add it, as do several
      Repro-1 and Serum presets.
    - **28** are below −90 dB above 24 kHz. The clean Harmor patches are
      among them; Harmor appears band-limited near 20 kHz.

    → These 47 files are the only supervision for generating content above
    24 kHz. Columns: `hf_24_48k_db`, `hf_48_96k_db`.
16. **The top of the keyboard needs the high rate.** At 44.1 kHz, the notes
    of the top two octaves keep only a few harmonics, and at B10 only the
    fundamental remains below Nyquist. At 192 kHz, up to six harmonics of
    B10 fit.
17. **Levels are as rendered.** Peaks run from −23.5 to 0 dBFS per file, and
    level changes along the keyboard. For example, the Serum saw pluck falls
    about 3.5 dB from MIDI 12 to MIDI 126, while its decay shortens from
    800 ms to 225 ms. → Do not normalize per note; that would erase key
    tracking. Per-file gain is acceptable.
18. **54 files are dual-mono** (side level below −150 dB): Harmor 23,
    Kontakt 13, Repro-1 8, the 6 Korg recordings and a few others. The rest have real width. →
    Generate stereo; mono examples should give mono output. Column:
    `stereo_side_db`.

**Variants and duplicates**

19. **Renamed file.** `Harmor_square_playing_simple_saw_pluck` was a square
    wave: no even harmonics, matching the square-pluck files. On 4 October
    2026 it was renamed to `Harmor_synth_playing_simple_square_pluck.wav`,
    with the author's approval.
20. **Near-duplicates and families.** No two files are bit-identical, but
    several are very close:
    - Plain Harmor saw pluck vs ±10 % prism vs `slower_filter`: −16 to
      −20 dB waveform difference.
    - 1–2 cent vibrato vs none: statistically indistinguishable.
    - `log` vs `robber` distortion, and `classic` vs `soft`.

    → Keep each family together when splitting. Column: `family`.

    | Family | Files |
    |---|---|
    | Harmor simple saw pluck | 38 |
    | Exhale (Kontakt) | 15 |
    | Shevannai (Kontakt) | 13 |
    | Serum saw/pluck patch | 11 |
    | Harmor simple square pluck | 6 |
    | T_synth preset 1 | 2 |
    | Korg monologue preset 1 bass (two takes) | 2 |
    | 63 other presets | 1 each |
21. **Harmor "simple saw" spectrum.** Its even harmonics sit about 30 dB
    below the odd ones; a textbook saw would be about −5 dB. It is a real
    property of the patch, but worth knowing when it is described as a saw.

**Performance**

22. **One velocity, one articulation.** Every note uses velocity 100,
    centre pan and no pitch offset, and the notes never overlap or glide
    (from the score). The dataset therefore shows how a sound changes with
    **key only**, not with velocity or playing style. → Generated notes are
    "velocity 100" notes. Velocity response would need further renders.

**Hardware recordings (Korg monologue)**

23. **Hand-trimmed starts and ends.**
    - Every take starts 24–356 ms after the C1 attack, so the C1 slot is
      unusable.
    - Presets 3–5 end before the B10 slot completes.
    - Presets 1 and 3 end while the last note still sounds (−31 and
      −19 dBFS in the final 100 ms), so that release is cut.

    → Never use a cut slot. Column: `cut_slots`; `cut` in `notes.csv`.
24. **Measured, not exact, timing.**
    - The fitted beat runs 1199.85–1200.57 ms instead of 1200, i.e. up to
      65 ms of drift over a take.
    - Individual onsets scatter around the fitted grid by a median of
      3–13 ms and at most 41 ms. This is MIDI and analog timing, not the
      fit.

    → Slice from the fitted grid with a margin (e.g. start each slot 50 ms
    early), or at each note's detected onset. Never compare a recording with
    a render sample by sample. Columns: `grid_origin_ms`, `beat_ms`,
    `onset_jitter_ms`, `onset_jitter_max_ms`.
25. **Mono.** Left and right are bit-identical (the monologue has a mono
    output). → Like the other 48 dual-mono files.
26. **Analog pitch tracking.** Pitch drifts flat as the keys rise:
    - Presets 1 and 4 go from +4/+6 cents in octaves 1–3 to −18/−26 cents
      at the top, about −3 cents per octave.
    - Preset 2 sits about −15 cents in its upper octaves.
    - Preset 5 stays within ±4 cents.

    The two takes of preset 1 agree to a median of 2 cents and 0.0 dB in
    level, so the drift is systematic and repeatable, not random. → It is
    part of the analog sound; keep it.
27. **Bandwidth.** The files are 96 kHz, but above 24 kHz they hold only
    −83 to −87 dB of the note energy. This is the interface limit the author
    mentions. The 20–24 kHz band varies by patch (−29 to −90 dB). → Treat
    them as band-limited to about 24 kHz.
28. **Noise and hum.**
    - No mains hum: the 50/100/150 Hz lines do not rise above the local
      floor.
    - The takes contain no silence, so the noise floor cannot be measured
      directly. The quietest moments (end tails of presets 2, 4 and 5) reach
      −58 to −65 dBFS, an upper bound.
    - Peaks are −3.5 to −6.7 dBFS, with no clipping and no DC offset.
29. **Sequence patch.** `preset_3_sequance` plays the monologue's step
    sequencer: each slot steps through 0, +4 and +7 semitones (a major
    triad). → Same as issue 9: exclude it from note-to-note generation, or
    treat it as a separate task.
30. **Top range.** Preset 1 (both takes) repeats MIDI 127 for MIDI 128 to
    131, like issue 6. In presets 2–5 pitch steps become irregular above
    about MIDI 123 (FL D#10). → Drop MIDI 124 and above in the recordings.
31. **Sound Forge `.sfk` files.** Each recording has a `.sfk` file beside it.
    It is Sound Forge's waveform-overview cache, not audio; Sound Forge
    recreates it, so it is safe to delete. The repository's `.gitignore`
    only excludes audio files, so these would be committed with the folder.

## Using it to train the generator

### The task

- **Context:** K example notes from one file, each with its note number.
  K = 1 or 2 at first, more later.
- **Query:** a note number.
- **Output:** that note as stereo audio at the native rate: 1.2 s, i.e.
  230,400 samples per channel at 192 kHz.

The model sees only the examples. It has to infer the sound's identity and
its note-to-note behaviour. With two examples it can read the trend between
them, for example how much the decay shortened or the brightness grew over
those semitones.

The dataset supervises this directly. Every file gives ground truth for every
query:

- 120 one-example tasks per file;
- 7,140 two-example context pairs per file, each with about 118 targets.

### Episodes

1. Pick a file, then pick K context notes among its live slots.
2. Use the other live slots as targets. Mix three kinds of query:
   - **Interpolation:** the query lies between the examples.
   - **Near extrapolation:** up to 12 semitones beyond them.
   - **Far extrapolation:** 24 or more semitones beyond them.
3. Exclude as targets:
   - silent slots (issue 5);
   - the repeated top notes (issue 6);
   - the arpeggiated `massiv_x__preset_4`, `_5` and `_8` (issue 9), and the
     Korg `preset_3_sequance` (issue 29);
   - slots cut by a recording's trim (issue 23).
4. Keep everything else, including layered, detuned, vibrato and
   octave-transposed sounds (issues 10–12). That variety is what the model
   should learn to reproduce.
5. The 38-file Harmor saw family shows one source through many effect
   chains. It can also train effect behaviour across the keyboard, but it
   must not dominate sampling. Sample per family, not per file.

### Resolution: train for 192 kHz output

The goal is high-quality samples, so train the final model to generate at
**192 kHz / float32**, following wavelet2vec's
[high-resolution contract](../docs/HIGH_RESOLUTION_AUDIO.md). That contract
forbids implicit resampling.

- **192 kHz files (93):** native targets. The 47 with real ultrasonic content
  teach the top band. The 28 without it teach the model to stay band-limited
  when the examples are.
- **44.1 kHz files (51):** explicitly upsample them to 192 kHz (ratio
  640/147). Train with a loss that ignores content above about 20 kHz for
  these files, because they say nothing about it. They then add their sounds'
  low-band behaviour without teaching the model that high bands are empty.
- **96 kHz recordings (6):** explicitly upsample them 2:1 to 192 kHz, and
  ignore the loss above about 24 kHz, where the interface recorded nothing
  real (issue 27). They add real analog behaviour: tuning drift, take-to-take
  variation and analog noise.
- **Prototyping:** 48 kHz (an exact 4:1 decimation of 192 kHz) is fine for
  quick architecture search. It is not the target resolution.

### Overlapping notes

For the 61 files that decay within the slot, every slot is a clean,
isolated note. **Start training and evaluation with these** (mostly plucks).

For the 89 held or ringing files there are three options:

- Train on the slot as it is. Give the model the previous slot as an extra
  input, so the leftover release is explained rather than imagined.
- Down-weight the first ~100 ms of the loss, where the previous release
  overlaps the new attack.
- Best: re-render these sounds with a silent beat between notes (see below),
  so every note keeps its full release.

### Splits and evaluation

- **Unseen sounds (the real goal):** hold out whole families, so test sounds
  never appear in training in any variant.
- **Unseen notes of a known sound:** hold out note ranges inside a family.
- **Report results by:**
  - distance from the nearest example (interpolation, near and far
    extrapolation),
  - envelope class and category,
  - source sample rate,
  - octave.
- **Baselines to beat:**
  1. `wavelet2vec.imagine.pitch_shift` of the nearest example: whole
     semitones, duration kept, formants shifted.
  2. Resampling the nearest example, as a sampler does; this changes
     duration.
  3. For two examples: a crossfade between the two shifted examples.

  The model must win on what these cannot do:
  - key-tracked decay and brightness;
  - multisample zone changes (Kontakt, Iris 2);
  - fixed vocal formants;
  - how unison, chorus and distortion change with pitch;
  - correct content above 20 kHz.
- **Metrics:**
  - multi-resolution spectral distance over the full 0–96 kHz band, and
    separately above 20 kHz;
  - envelope and decay error;
  - pitch accuracy of the generated note in cents (the output must land on
    the requested note);
  - stereo width error;
  - listening.
- **Dataset size:** 70 families is small for generalizing to unseen sounds.
  More distinct sounds are worth more than more effect variants of the same
  patch. Bells are the least clearly represented category.

### Suggestions for future renders

- **Held sounds** (leads, basses, anything that lasts the whole note): use a
  second score that places the same one-beat notes every two beats, so each
  release lands in silence (issue 7).
- **Velocity:** to learn velocity response, re-render a few families from
  copies of the score at other velocities (e.g. 40 and 127) (issue 22).
- **Non-FL plugins:** stop at MIDI 127 (FL G10) (issue 6).
- **192 kHz:** only gains anything when the synth produces ultrasonic
  content, so enable oversampling and avoid 20 kHz band-limited modes
  (issue 15).
- **Sidecar file:** list per file the category (pluck / bass / bell / lead),
  arpeggiator on or off, oscillator count, detune and voices, and the octave
  or transpose setting.
- **Variety:** prefer new distinct sounds over effect variants of one patch.

## Second purpose: vox–synth hybrids

**Goal (author):**
1. Split the dataset into a vocal (vox) part and a synthesizer (synth) part.
2. Train a classifier to tell them apart.
3. Generate new sounds that the classifier rates **50 % vox / 50 % synth**.

The aim is vocal–synth hybrids that are immediately heard as something new.

**Design intent (author).** The target is a unique, interesting sound that
stands out in sound design:
- **Voice is the anchor.** People recognize the human voice easily and are
  very sensitive to it, so a sound that is heard as partly vocal immediately
  catches attention.
- **The rest should be synthetic:** complex, rich and clearly synth-like,
  rather than a plain processed vocal.
- **Unusual sources are welcome.** Lightly processed, out-of-the-ordinary
  material, like the Melodyne-stabilized "Piuw" take, is good raw material.

In practice, the hybrid should keep the few cues that make a sound heard as
voice and take the rest from a synth:

| Heard as voice (keep from the vox note) | Heard as synth (take from the synth note) |
|---|---|
| Formant motion, e.g. the "ee"→"oo" glide of "Piuw" | Dense, stable harmonics; unison and detune |
| Consonant onsets ("P"), breath and noise bursts | Resonant or moving filters; distortion, bit-crush |
| Small pitch movements (where not removed) | Exact pitch, perfect repeats, the very high and very low range |
| Vocal-tract resonances that stay fixed while pitch changes | Ultrasonic sheen (192 kHz sources only) |

### The split

Column `source_type` in [dataset_audit.csv](dataset_audit.csv):

| `source_type` | Files | Notes with sound | Independent sources | What it is |
|---|---|---|---|---|
| `vox` | 13 + 5 | 332 + 128 | **2** | Shevannai (`kontakt_shalanaja_*`): one soprano, real voice, no vibrato, no effects; 13 sung syllables. Plus the author's own [sung takes](#sung-vocal-take-own-voice): "Piuw", A, EH, O, E (pitch-corrected, MIDI 36–76; listed in their notes tables, not the audit CSV) |
| `processed_vox` | 15 | 557 | 1 | Exhale (`kontakt_vox_*`): vocals heavily processed with LFO and distortion |
| `synth_vox` | 2 | 198 | 2 | Harmor `old_vox` and `robot_vox`: synthesized voice imitations |
| `synth` | 120 | 14,273 | 66 | Every other sound, including the 6 Korg recordings. "Synth" here means "not vocal", so the sampled `kontakt_piano_gigant` is included |

**Train the classifier on the clean ends only: `vox` vs `synth`.** The other
two classes are already part-way hybrids, from opposite directions:

- **Exhale (`processed_vox`)** is voice pushed toward synth by processing.
- **Harmor vox (`synth_vox`)** is synth pushed toward voice.

Keep both out of classifier training and use them as reference points:

1. A trustworthy classifier should place them between the classes, not at
   the extremes.
2. Generated hybrids should be compared with them. The goal is "new", so a
   hybrid that just sounds like an Exhale preset falls short.

Labelling Exhale as `vox` instead would teach the classifier that heavy
synth-style processing still counts as voice. Hybrids would then need more
synth character to reach 50 %, and the target would become harder to
interpret.

### Shortcuts the classifier would learn instead of "voice"

As rendered, the clean classes differ in ways that have nothing to do with
voice. A classifier will use whichever difference is easiest to detect:

| Difference | `vox`: Shevannai | `vox`: own voice | `synth` | Neutralize by |
|---|---|---|---|---|
| Sample rate | 13 of 13 at 44.1 kHz | 96 kHz | 90 of 120 at 192 kHz | Give the classifier one common rate |
| High band | −113 dB at 20–24 kHz (none) | −86 dB above 20 kHz (none) | −56 dB at 20–24 kHz | Low-pass the classifier input at about 18 kHz |
| Key range | MIDI 57 to 82 (soprano) | MIDI 36 to 76 | Full range | Train the classifier on the vox range (MIDI 36 to 82) for both classes |
| Peak level | −15.7 dBFS (median) | −3.0 dBFS | −3.8 dBFS (median) | Loudness-normalize each classifier input |
| Envelope | Short syllables, decaying within the beat | ~0.5 s notes with silence between | Mixed: plucks, leads, basses | Balance envelope classes |
| Stereo | Mono recordings (dual-mono) | Near-identical channels | 80 of 120 stereo | Feed the mid channel, or randomize width |
| Amount | 332 notes | 71 notes | 14,273 notes (35×) | Sample per family and per class, not per note |
| Pitch | No vibrato (recorded without) | Melodyne-stabilized | Some patches have vibrato | Know that steady pitch is part of what the vox class looks like |

These normalizations apply **to the classifier's input view only**. The
generator still works on the native audio, at up to 192 kHz.

The biggest limit is **voice diversity: two clean voices.** The clean vox data
is one soprano (Shevannai, 13 syllables) plus the author's own sung take.
With two voices, **leave-one-singer-out** is now possible: train on one voice
and test on the other. Their ranges overlap at MIDI 57–76, and that overlap
is the fairest place to compare them. Two voices are still one test in each
direction, and the two also differ in recording chain (library vs.
Melodyne-corrected home recording). A classifier could therefore still learn
"these two recordings".

Testing on Exhale says something too, but its processing confounds the
result.

More clean vocal sources would make "voice" measurable, ideally at 192 kHz:
- other singers, male and female;
- other libraries, or more own takes;
- sung vowels, choirs and spoken voice.

### Why "50 % confidence" needs care

1. **Uncertain is not the same as hybrid.** A two-class classifier also
   says 50/50 for noise, silence or broken audio, because it cannot tell.
   Better targets are:
   - two separate detectors, "vox present" and "synth present", each trained
     against other sounds and noise, with **both** high for a hybrid; or
   - an embedding position between the two class centres that still scores
     as a real sound.
2. **Gradient-driven generators fool classifiers.** If the generator is
   optimized directly against the classifier's gradient, it finds small,
   inaudible changes that move the score; adding a little energy above
   20 kHz to read as "synth" is one example. Mitigations:
   - let the generator move only musically meaningful controls (blend
     weights, formant/spectral envelope, excitation mix) rather than raw
     samples;
   - score with several classifiers built on different features, for
     example the deterministic `Wavelet2Vec` 570-dim embedding with a linear
     probe, plus a learned network;
   - confirm results on a **held-out judge classifier** never used during
     generation.
3. **The classifier only steers.** Whether a hybrid "sounds new and
   interesting" is decided by listening. Plan listening checks: do listeners
   call it vox, synth or hybrid, and do they prefer it?

### A baseline that already exists, and its limit

`wavelet2vec.style_transfer.blend` makes a sound between references. One
reference (the carrier) keeps its fine structure (pitch, waveshape), and is
reshaped toward the weighted average spectral envelope (formants) and
temporal envelope of all references. The result moves monotonically with the
weight.

**Limit:** `SoundCharacter` holds one **time-averaged** spectrum plus an
amplitude envelope. Blending a "Piuw" note this way transfers the average of
"ee" and "oo" and **loses the glide**, which is the very cue that makes it
vocal. A blend is fine for a first calibration experiment. The step that
fits this goal is a **time-varying** spectral-envelope transfer: impose the
vox note's frame-by-frame formant envelope on the synth carrier (classic
cross-synthesis, as in a vocoder), with the morph amount controlling how
much formant motion is imposed. wavelet2vec does not provide this yet; it
would extend `SoundCharacter` from one spectrum to a sequence of spectra
over time.

The dataset gives **pitch-aligned pairs**: every vox note has synth notes at
the same key. Shevannai alone gives 26 keys (MIDI 57 to 82) × 13 syllables ×
120 synth files. Exhale adds MIDI 48 to 91 when partly processed voice is the
starting point. A first experiment needs no new model:

1. Blend a synth note and a vox note at the same key: synth carrier with the
   vox envelopes, and the reverse.
2. Measure the classifier's vox score at a few blend weights.
3. Invert that monotone curve to find each pair's 50 % weight. This is the
   anchored calibration method of the tonality knob (`imagine_tonality`,
   roadmap G-9).
4. Listen.

The learned version can then combine both purposes: create a hybrid at one
or two notes, and let the keyboard generator produce the rest of its notes.

### The learned version: MorphNet

MorphNet started training on 9 October 2026; see
[docs/MORPH_MODEL.md](../docs/MORPH_MODEL.md). It applies the points above:
- judges see a confound-controlled view: mid channel, below 18 kHz,
  normalized level, MIDI 36–82;
- "vox present" and "synth present" are separate heads, trained against
  noise;
- a third judge is held out from training;
- a leave-one-singer-out check;
- every hybrid must stay away from the plain mix and from vocoders in both
  directions;
- a realism critic;
- listening demos every epoch.

## Still unknown

1. Which category (pluck, bass, bell, lead) each of the 50 unlabelled presets
   is.
2. Which other patches use an arpeggiator, beyond Massive X 4, 5 and 8.
3. Do any Iris 2 presets use vocal samples? That would make them vox or
   hybrid references.
4. Is the class choice right? The current choice: Exhale and the Harmor vox
   presets are hybrid references (`processed_vox`, `synth_vox`), not
   training data for either class.

## Reproducing the audit

```
PYTHONPATH=src python experiments/note_dataset_audit.py
PYTHONPATH=src python experiments/annotate_sung_notes.py note_dataset/from_C_real_Vocal_recording_melodyne.wav --syllable Piuw --vowel-scale 1.04
PYTHONPATH=src python experiments/annotate_sung_notes.py note_dataset/from_A_real_Vocal_recording_melodyne_sound_A_.wav --syllable A --vowel-scale 1.04
PYTHONPATH=src python experiments/annotate_sung_notes.py note_dataset/from_E_real_Vocal_recording_melodyne_sound_EH.wav --syllable EH --vowel-scale 1.04
PYTHONPATH=src python experiments/annotate_sung_notes.py "note_dataset/from_F#_real_Vocal_recording_melodyne_sound_O_.wav" --syllable O --vowel-scale 1.04
PYTHONPATH=src python experiments/annotate_sung_notes.py "note_dataset/from_G#_real_Vocal_recording_melodyne_sound_E_.wav" --syllable E --vowel-scale 1.04
```

The annotation commands regenerate each vocal take's notes table, with
pitch, notes and detected vowels and parts (about 40 s for the long take).
Adding `--embed-markers` writes the named regions into the WAV; the script
refuses if the file already has markers.

The script first decodes `c1_till_b10.fsc` and prints whether its notes sit
one per beat. It then reads every WAV once, using 8 worker processes (about
5 minutes on this machine). It writes:

- `note_dataset/dataset_audit.csv`: one row per file.
- `experiments/output/note_dataset_audit/notes.csv`: one row per slot, with
  its start sample, a cut flag, peak, onset, decay, tail, side level, centroid, pitch offset, cents, the
  within-slot jump and the shift to the next note.
- `experiments/output/note_dataset_audit/files.json`: header facts, band
  energy and the cue-marker check.

Columns of `dataset_audit.csv`:

| Column | Meaning |
|---|---|
| `synth`, `family`, `category_from_name` | Plugin; base-patch group for splits; category stated by the filename (empty = unlabelled preset) |
| `source_type` | `vox` (Shevannai, clean voice), `processed_vox` (Exhale), `synth_vox` (Harmor voice imitations) or `synth` (all non-vocal sounds), for the hybrid purpose |
| `origin` | `render` (FL Studio, exact grid) or `recording` (fitted grid) |
| `grid_origin_ms`, `beat_ms` | Start of beat 0 relative to the first sample, and beat length (0 and 1200 for renders) |
| `onset_jitter_ms`, `onset_jitter_max_ms` | Median and largest onset distance from the fitted grid (recordings) |
| `cut_slots` | Slots cut off by the file start or end |
| `live_notes`, `live_from`, `live_to` | Slots with sound (FL note names) |
| `clamped_above_midi127` | MIDI 128 to 131 repeat MIDI 127 |
| `semitone_steps_ok` | Fraction of consecutive notes shifting 100 ± 10 cents |
| `sounding_offset_semitones`, `offset_agreement` | The preset's dominant pitch offset, and the share of mid-range notes agreeing with it |
| `moving_slots`, `main_jump_semitones` | Share of slots whose spectrum jumps ≥ 150 cents within the slot, and the most common jump |
| `envelope`, `tail_at_slot_end_db` | Median level in the last 50 ms of a slot, relative to the note peak |
| `stereo_side_db` | Median side/mid energy ratio |
| `hf_20_24k_db`, `hf_24_48k_db`, `hf_48_96k_db` | Share of note energy in each band |
