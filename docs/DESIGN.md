# wavelet2vec — Design and Theory

This document explains the design decisions behind `wavelet2vec`: why the
representation looks the way it does, and the math behind the phase features.

## 1. The problem

We want a fixed-size vector for a short audio snippet (0.2–5 s) such that
cosine distance behaves like perceptual distance, for use in sound design:
similarity search, clustering, interpolation targets, generative-model rewards.

Constraints:

- **Deterministic** — same input, same output, no training, no model weights.
- **Invariant** to things that don't change how a sound sounds: playback level,
  input sample rate, small time offsets.
- **Sensitive** to things that do: spectrum, texture, attack, pitch, waveshape.

The last item — waveshape — is where almost every existing representation fails.

## 2. Why magnitude spectra are not enough

A sound's short-time magnitude spectrum discards phase. Two signals with the
same harmonic amplitudes but different harmonic phases — a saw wave and a
"phase-scrambled saw" — have identical spectrograms, mel spectrograms, MFCCs,
and CQT frames, yet different waveforms and (especially at low pitch and for
transients) audibly different character.

The obvious fix — keep the raw phase, or keep raw waveform samples — fails
because phase is **shift-variant**. Delay a signal by τ and every spectral
component's phase rotates by

```
φ_f  →  φ_f − 2π f τ
```

A 1 ms offset (inaudible, and inevitable in any real library) completely
scrambles raw phase and decorrelates raw waveforms. An embedding built on them
makes nearest-neighbor search meaningless.

So the design question is: **which functions of phase are invariant to time
shifts but still carry the perceptual information phase encodes?**

## 3. The analysis front end: constant-Q complex Morlet filterbank

Everything starts from a constant-Q filterbank of complex Morlet (Gabor)
wavelets, implemented as Gaussian windows on the positive half of the FFT
spectrum (`filterbank.py`). For band center *f* with bandwidth σ ∝ *f*
(constant-Q, geometric spacing), the inverse FFT of the windowed positive
spectrum yields the **analytic band signal**

```
z_b(t) = A_b(t) · exp(j φ_b(t))
```

whose magnitude `A_b(t)` is the band envelope and whose angle `φ_b(t)` is the
band phase. Constant-Q log-spaced wavelets match both hearing (≈ constant
bandwidth in octaves) and music (semitones are geometric): fine time
resolution at high frequencies where transients live, fine frequency
resolution at low frequencies where pitch lives.

Cost: one FFT plus one inverse FFT per band — milliseconds per snippet on CPU.

## 4. The six sections

### 4.1 `spectral` — first-order wavelet statistics

Per-band time-mean and time-std of `log(1 + A_b(t))`. The mean is the
multi-scale spectral envelope (timbre, brightness, body); the std measures how
unsteady each band is.

### 4.2 `modulation` — second-order (scattering-style) features

Band envelopes are downsampled, level-normalized, and Fourier-transformed; the
modulation energy is pooled over log-spaced modulation bands (0.5–64 Hz) and
groups of wavelet bands. This is the second order of a wavelet scattering
transform: it captures *texture* — roughness, tremolo, grain density, flutter —
which first-order spectra average away.

### 4.3 `transient` — envelope morphology from the waveform

The Hilbert envelope of the full-band waveform, summarized shift-robustly:
envelope shape resampled to 32 points, log attack time (10%→90% of peak),
decay slope, temporal centroid, crest factor, zero-crossing rate, and onset
flux statistics from the scalogram. This is the perceptual content people
usually mean when they say "we must keep the phase": attack, punch, decay.

### 4.4 `harmonic` — pitch content

Chroma (energy per pitch class, from the FFT for fine frequency resolution),
harmonicity and f₀ via autocorrelation, and spectral flatness. Two sounds
playing the same note or chord agree here regardless of octave or timbre.

### 4.5 `phase` — shift-invariant phase information

Three quantities, each provably or approximately shift-invariant:

**(a) Instantaneous frequency.** `IF_b(t) = (1/2π) dφ_b/dt`. A delay
translates the IF trajectory in time but does not change its values, so its
energy-weighted statistics are invariant. Deviation of IF from the band center
separates steady tones (≈ 0), vibrato (oscillating), and noise (large,
erratic) inside each band.

**(b) Cross-band onset phase coherence.** For an impulse at t₀, every analytic
band's phase passes through alignment at t₀ — broadband phase locking is the
defining signature of a true transient (this underlies classical
phase-deviation onset detectors). We measure `C(t) = |mean_b exp(jφ_b(t))|`
over active bands and keep its max, mean, and time-above-threshold. A delay
moves the alignment instant; the statistics don't change. This distinguishes a
genuine percussive attack from a mere loudness bump.

**(c) Harmonic phase signature — the waveshape invariant.** For a pitched
sound with fundamental f₀, define for each harmonic k:

```
sig(k) = φ_{k·f₀}(t) − k · φ_{f₀}(t)
```

Under a delay τ, the first term rotates by −2π·k·f₀·τ and the second by
−k·2π·f₀·τ — **exactly cancelling**. (More generally, any combination
Σ aᵢ φ_{fᵢ} with Σ aᵢ fᵢ = 0 is delay-invariant; sig(k) is the harmonic
special case.) For a stationary pitched tone, sig(k) is constant in time and
equals the relative phase offset of harmonic k within one period — i.e. the
**waveshape**. We estimate f₀ by autocorrelation, extract analytic signals at
exact harmonic multiples (narrow Morlet filters at k·f₀), and store the
amplitude-weighted mean phasor `E[exp(j·sig(k))]` per harmonic: its real and
imaginary parts encode the angle; its magnitude measures stability. Unpitched
sounds (low harmonicity) yield zeros.

This is the feature that separates a saw from its phase-scrambled twin
(section similarity 0.86) while every magnitude-based section reads ≥ 0.997.

### 4.6 `stereo` — spatial image

Stereo input is analyzed twice: the mono mixdown feeds every timbral section,
and the left/right pair feeds the stereo section. All stereo quantities
compare the channels against each other — side/mid energy (width),
inter-channel coherence `|E[L·R*]| / sqrt(E|L|² E|R|²)` (decorrelation), and
energy balance (pan) — globally and per band group, so frequency-dependent
images (wide hats over a mono kick) are captured. A common delay or gain
affects both channels equally and cancels, preserving shift and level
invariance. Mono input receives the canonical centered vector (width 0,
coherence 1, balance 0), keeping dimensions fixed.

### 4.7 `conv` — wavelet-initialized convolutional encoder

A two-layer 1D CNN (`conv_encoder.py`):

1. **Quadrature analysis layer.** A strided `Conv1d` with 2N kernels: for each
   band, a Gaussian-windowed cosine and sine pair at the band center (a real
   implementation of a complex Morlet). The modulus √(cos² + sin²) of each
   pair is a translation-stable band envelope — convolution is translation-
   equivariant, and the modulus + pooling make the output approximately
   translation-invariant. This mirrors what the first layer of trained
   waveform networks (wav2vec, SoundStream) converges to anyway.
2. **Depthwise temporal layer.** Four filters shared across bands — Gaussian
   smoother, derivative, and 16/64 Hz Gabors — capture local envelope dynamics
   per band. Responses are rectified and pooled (mean + std over time).

Frozen by default, the encoder is deterministic and the whole embedding stays
training-free. Construct with `trainable=True` to fine-tune (e.g. with a
contrastive objective on your own library): the wavelet initialization means
optimization starts from a meaningful filterbank rather than random noise.

## 5. Normalization and similarity

Inputs are resampled to a fixed analysis rate (22.05 kHz default) and
peak-normalized, making the embedding invariant to source sample rate and
level by construction. Each section is L2-normalized and multiplied by a
configurable weight before concatenation, so the cosine similarity of two full
embeddings decomposes (up to normalization) into a weighted sum of per-section
cosine similarities. This makes the space *steerable*: zero out `harmonic` to
compare timbre across pitches; keep only `phase` + `transient` to compare
attack character; and so on.

The `conv` section defaults to weight 0.5 because, at initialization, it
partially overlaps with `spectral`; raise it after fine-tuning.

## 6. From analysis to synthesis: blending and style transfer

The same analysis that produces the embedding can steer a morph. `blend`
takes several references with weights; the heaviest-weighted one (the
*carrier*) keeps its fine structure — pitch, waveshape, micro-detail — while
two character dimensions are pulled toward the weighted blend of all
references:

- **spectral envelope**: the carrier's wavelet band energies are EQ-matched to
  the weighted geometric mean of the references' (level-normalized) band
  energy distributions, applied as a smooth log-interpolated gain curve in the
  FFT domain;
- **temporal envelope**: the carrier's Hilbert envelope is morphed toward the
  weighted average of the references' time-normalized envelope shapes via a
  bounded, smoothed time-varying gain.

Because the morph preserves the carrier's phase exactly, a time-domain
crossfade between source and morphed signal interpolates the filter smoothly
without comb artifacts — which is what makes time-varying character control
(ADSR-driven morphing) artifact-free.

Both transfers are bounded (±24 dB by default) and verified by the embedding:
similarity to a reference increases monotonically with its weight.

### 6.1 Characters as first-class objects

The transferable part of a sound — its level-normalized wavelet band energy
distribution and time-normalized envelope shape — is factored out as a
`SoundCharacter` on a fixed analysis grid (22.05 kHz, 64 bands, 27.5 Hz to
~9.9 kHz, 256 envelope points). Characters can be extracted from one sound or
averaged over a group (geometric mean of energies, arithmetic mean of
envelope shapes), saved, and applied to any other material at any sample
rate. A group character captures what its members share (e.g. the harmonic
density and pick attack of an electric guitar library) while individual
quirks average out — this is what makes "vocal, but crunchy like my guitars"
different from applying a distortion effect.

### 6.2 Time-varying character: ADSR and automation

`apply_character` accepts a per-sample amount array. The implementation
computes the full morph once and crossfades it with the source; both signals
share the source's phase, so the crossfade interpolates the underlying filter
smoothly with no comb artifacts. An `ADSR` envelope is the musical interface
to this: attack = how fast the sound grows into the character, release = how
it falls back. `render_sequence` then samples an external automation curve
(FL Studio `.flp`/`.fst` via pyflp, or CSV/JSON) at each snippet's position
in the sequence and lets it modulate either the morph amount or one ADSR
stage per snippet — texture development over a phrase, driven from the DAW.

## 7. Known limitations

- Signal-level, not semantic: no notion of instrument names or genres.
  Concatenating a CLAP embedding adds that axis if needed.
- Global summary: temporal structure beyond the envelope shape (e.g. a melody
  inside a 5 s clip) is averaged out. Frame-wise embedding + sequence pooling
  would be the extension.
- The harmonic phase signature requires a detectable fundamental
  (autocorrelation harmonicity ≥ 0.15) and is zero for noise-like sounds —
  by design, since "waveshape" is undefined without a period.
- Polyphonic input: f₀ estimation picks the dominant pitch; the signature then
  describes the dominant voice only.

## 8. Relation to prior work

- **Wavelet scattering transform** (Mallat; Andén & Mallat): sections 4.1–4.2
  are a lightweight engineered scattering front end (first- and second-order
  coefficients of a Morlet filterbank).
- **GANSynth** (Engel et al.): demonstrated that instantaneous frequency is
  the learnable form of phase; section 4.5a uses it as statistics rather than
  images.
- **Phase-deviation onset detection** (Bello et al.): the basis of 4.5b.
- **Bispectrum / quadratic phase coupling**: sig(k) is a pitch-synchronous,
  filterbank-domain relative of bispectral phase invariants, specialized to
  harmonic audio where it has a direct interpretation as waveshape.
- **wav2vec / SoundStream / EnCodec**: motivate the quadrature conv layer and
  the trainable upgrade path in 4.6.
