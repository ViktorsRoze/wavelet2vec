# Percussive-event diagnostics — v0.36.0

23 September 2026. These APIs analyze annotated isolated percussion and compare
waveform snippets. They do not automatically classify every instrument in a
mixture. [The real dataset findings](DRUM_DATASET_FINDINGS.md) demonstrate the
current results and limitations on the supplied eight-source drum performance.

```python
import numpy as np
from wavelet2vec import analyze_drum_track

sample_rate = 8000
time = np.arange(1200) / sample_rate
hit = np.sin(2*np.pi*180*time) * np.exp(-35*time)
audio = np.zeros((2, sample_rate))
audio[:, :len(hit)] = hit
audio[:, 4000:4000+len(hit)] = .5 * hit
analysis = analyze_drum_track(audio, sample_rate, bpm=120)
assert len(analysis["events"]) == 2
```

`detect_percussive_events` uses multichannel wavelet-flux novelty plus native
energy refinement. Channels contribute power without a cancellation-prone
mono sum. It declares analysis-only resampling, defaults to 5 ms wavelet frames
and 70 ms minimum spacing, and labels activity at the start separately from
ordinary onset candidates. Silence returns no events. Event times estimate
audible attacks, not exact MIDI trigger times; the reported frame resolution
is not an error bound. Rolls faster than the minimum spacing need a deliberately
different configuration and independent validation.

`compare_percussive_events` estimates playback-rate-equivalent semitone shift,
time offset and independent channel gains between two `[channels,samples]`
snippets. Declare their common onset preroll using `preroll_samples`. Defaults
search ±12 semitones, ±20 ms timing, and up to 60 ms of signal; available length
can shorten the comparison. Pitch is returned only when the fit exceeds 0.95
and is not at the pitch-search boundary. The score is a waveform-fit heuristic,
not a probability or a general fundamental-frequency estimator. Re-synthesis,
time-preserving pitch shifts, stereo processing or random excitation may not
fit this playback-rate model.

The output distinguishes raw pitch candidates from accepted estimates,
zero-pitch fit, fitted level change, channel gains and exact PCM equality in
the integer-aligned comparison window. Exact equality of a short prefix does
not prove equality of the whole note. A strong match does not identify the
instrument's generator or distinguish a synthesizer from a sample player.

`analyze_drum_track` detects events, chooses the loudest measured event as its
reference, estimates transformations, and groups candidates using a 0.995
zero-pitch gain/timing fit. Different pitches of one sample can form different
groups. It also records native sample/time positions, optional 4/4 beat/bar
coordinates, first-up-to-60-ms RMS, raw level differences, stereo energy
balance and L/R dB ratio. Gain and pan are measurements under explicit models;
neither is an inverse of unknown MIDI velocity curves or DAW pan laws.

The first real audit exposed an onset/pitch confound: an 8 ms timing bound
misclassified a roughly 9 ms onset offset as a semitone change. The default
timing allowance and a regression now handle that case. Raw onset-window RMS
also changed despite stable fitted gain, so inspect both rather than treating
every RMS change as volume automation.

## Source-balanced supervised losses

`separation_loss(..., source_weights=...)` and
`SeparationRunConfig(source_weights=...)` accept fixed, finite positive weights
in the model's exact `source_order`. Waveform and complex spectral terms use
weighted annotated sample counts; missing targets stay ignored. Accumulation
uses the same weighting so variable-size microbatches reproduce a combined
weighted objective. The mixture-consistency term retains its own unweighted
mixture-sample normalization and additive-target guard.

Weights do not change input/target audio, synthesized output gain or rate.
Training-only statistics can supply weights; never fit them on held-out targets.
The standard held-out evaluator continues reporting common unweighted units.
The drum experiment's inverse-RMS weighting helped some quiet sources while
hurting strong ones, so the option remains disabled by default.
