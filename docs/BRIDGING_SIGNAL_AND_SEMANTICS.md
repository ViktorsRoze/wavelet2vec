# Bridging Lossless Complex Wavelet Signal Processing and Invariant Semantic Audio Understanding

## 1. Executive Summary: The Fundamental Dualism in Audio Representation

A central open problem in modern audio representation is the mathematical tension between **exact, phase-preserving signal reconstruction** and **invariant, abstract semantic understanding**:

1. **High-Fidelity Invertible Signal Processing (`WaveletImage`)**:
   - Requires retaining exact local carrier phase $\phi(k, t)$, sub-sample timing offsets, and linear superposition across all bands so that the inverse transform recovers the 192 kHz stereo waveform to machine precision (~305 dB SNR in `float64`, ~132 dB SNR in `float32`).
   - However, raw complex wavelet coefficients covary rapidly with nuisance transformations: shifting a 110 Hz note by just 2.7 ms rotates its complex carrier phase enough to flip the cosine similarity of its raw coefficients to **$-0.2910$**, and a $90^\circ$ phase rotation makes the raw complex vector strictly orthogonal ($\cos = 0.0000$).
   - Simultaneously, constant-$Q$ semitone filters obey the **Heisenberg-Gabor uncertainty bound** $\sigma_t(f) \cdot \sigma_f(f) = \frac{1}{2\pi}$. At sub-bass and bass frequencies ($16.35\text{ Hz} \le f \le 165\text{ Hz}$), narrow semitone bandwidths ($\sigma_f \approx 0.4\text{ Hz} - 4\text{ Hz}$) force temporal impulse responses to ring across **$40\text{ ms} - 250\text{ ms}$**, smearing sharp percussive attacks across dozens of 10 ms frames (**The Low-Frequency Heisenberg Trap**).

2. **Invariant Semantic Audio Understanding**:
   - Requires representations that are invariant to global phase rotations and small time translations, yet hyper-sensitive to **harmonic ridge structure**, **sub-frame transient attack timing**, **tonal-vs-stochastic texture**, **spectro-temporal modulation trajectories** (upward vs. downward pitch glides, formant transitions, vibrato rates), and **stereo spatial coherence**.

Because offline embedding generation, neural judge training, and interactive inspection in the audio editor can afford rich, exact spectral computation, we do not need to compromise between reconstruction and semantics. Instead, we introduce a **Dual-Layer Architecture** implemented in `src/wavelet2vec/semantic_bridge.py`:

- **Layer 1 — Lossless Complex Carrier (`WaveletImage`)**: Retains the exact multirate complex coefficients $z_{k, m} \in \mathbb{C}$ across 149 semitone bands ($c \in \{1, 2, 4, \dots, 256\}$ samples per 10 ms frame) for bit-accurate synthesis and phase-locked editing.
- **Layer 2 — Differentiable Reassigned & Scattering Semantic Bridge (`semantic_bridge.py`)**: Extracts exact spectral derivatives ($s_0, s_1, s_2$) via band-level FFTs in PyTorch, resolves the low-frequency `c = 1` cell degeneracy from first principles, performs differentiable 2D synchrosqueezed reassignment, and applies 2D Joint Time-Frequency Scattering (JTFS) to produce:
  - `semantic_planes(image)` $\to$ `[B, 12, 149, 120]` differentiable feature planes for neural backbones, critics, diffusion models, and audio editor overlays.
  - `semantic_wavelet_embedding(audio_or_image)` $\to$ `[B, 252]` shift-invariant, phase-rotation-invariant, timbre- and trajectory-discriminative vector embeddings.
  - `harmonic_noise_gate(image)` $\to$ `[B, 149, T]` scale-consistent harmonic-vs-stochastic gate across all 149 bands.
  - `semantic_structure_loss(pred, target)` $\to$ fully differentiable physical + perceptual training objective backpropagating cleanly to 192 kHz stereo waveforms.

---

## 2. Deep Codebase Audit: Critical Scientific Findings

During our line-by-line inspection of `wavelet_image.py`, `morph.py`, `morph_large.py`, `morph_diffusion.py`, and `keyboard_v3.py`, we uncovered four foundational bottlenecks that prevented the neural models from exploiting the full information inside `WaveletImage`.

### Finding 1: The `c = 1` Multirate Cell Degeneracy Bug (MIDI 12–69, $16.35\text{ Hz} - 440.0\text{ Hz}$)

In `WaveletImageSpec()`, each semitone band $k$ stores $c_k = 2^{\lceil \log_2(\text{bw}_k \cdot T_{\text{total}}) \rceil / T_{\text{total}}}$ complex samples per 10 ms frame. For the **lowest 58 semitone bands** (MIDI 12 at $16.35\text{ Hz}$ all the way up to Concert A4, MIDI 69 at $440.0\text{ Hz}$), the sample count per 10 ms frame is:
$$c_k = 1 \quad \forall k \in \{0, 1, \dots, 57\}$$

In `morph.py` (`_cell_stats`), `morph_large.py` (`critic_view` and `_render`), and `keyboard_v3.py` (`cell_features`), per-cell phase coherence and amplitude modulation depth are computed by reducing exclusively along `dim=-1` (the within-cell axis of length $c_k$):

```python
# Legacy reduction along dim=-1 (size c = 1 for rows 0..57):
u = inc.sum(-1) / (torch.sqrt(power(inc) + TINY).sum(-1) + 1e-20)
mod = torch.sqrt(p.var(-1, unbiased=False) + TINY) / (mean_p + 1e-20)
```

**Mathematical Consequence when $c = 1$**:
1. The sum of a single complex phase increment divided by its own magnitude is a **unit vector for any non-zero signal**:
   $$|u| = \frac{|\text{inc}_1|}{|\text{inc}_1|} \equiv 1.0000$$
   Even on **pure white noise**, legacy phase coherence evaluates to **`1.0000`** (100% tonal saturation) across every single cell of the lowest 58 semitone bands!
2. The variance of a 1-element tensor `p.var(dim=-1, unbiased=False)` is **identically zero**:
   $$\text{mod} \equiv 0.0000$$
   Even on **pure white noise** or sharp bass transients, modulation depth collapses to **`0.0000`** below 440 Hz!
3. **Downstream Impact on `MorphNet-L` (`morph_large.py`, line 351)**:
   `MorphNet-L` splits each band into a coherent harmonic carrier path weighted by $m$ and a stochastic filtered-noise path weighted by $(1 - m)$:
   $$m = \sigma(8.0 \cdot (\text{coh} - 0.55) - 3.0 \cdot \text{mod})$$
   Because $\text{coh} \equiv 1.0$ and $\text{mod} \equiv 0.0$ for all 58 bands below 440 Hz, the gate is permanently locked at:
   $$m = \sigma(8.0 \times 0.45 - 0) = \sigma(3.6) \approx 0.9734 \implies 1 - m \approx 0.0266$$
   **Result**: The stochastic noise synthesis pathway in `MorphNet-L` is 97.3% disabled across the entire sub-bass, bass, and lower-midrange registers ($16.35\text{ Hz} - 440\text{ Hz}$), and `KeyboardNet v3` / `MorphCritic` cannot distinguish a pure 110 Hz sine wave from sub-bass rumble or breath noise using their coherence/modulation planes.

### Finding 2: The Low-Frequency Heisenberg Trap in Neural Planes

In `wavelet_image.py`, `detail_view(mode="sharp")` implemented Auger-Flandrin time-frequency reassignment in NumPy for visual inspection only. Meanwhile, every PyTorch neural view (`judge_view` in `morph.py`, `critic_view` in `morph_large.py`, `cell_features` in `keyboard_v3.py`) fed **raw, un-reassigned 10 ms cell powers** to the convolutional backbones.
- In the bass register ($41.2\text{ Hz} - 164.8\text{ Hz}$), a sharp impulse has an RMS temporal spread of **38.30 ms**, and only **37.74%** of its energy lands within $\pm 10\text{ ms}$ of the true onset frame.
- Consequently, neural encoders and losses operating on raw cell power see blurry horizontal streaks in the bass rather than crisp vertical attack lines.

### Finding 3: First-Order Marginal Blindness to Spectro-Temporal Trajectories

Standard pooling over time (mean/std across frames) discards joint spectro-temporal orientation. An upward pitch glide ($110\text{ Hz} \to 880\text{ Hz}$) and its time-reversed downward glide ($880\text{ Hz} \to 110\text{ Hz}$) have the **exact same 1D spectral marginal** (cosine distance $= 10^{-6}$). Separating upward vs. downward glides, formant transitions, and vibrato rates requires **Second-Order 2D Joint Time-Frequency Scattering (JTFS)** filters oriented in the `(row, frame)` plane.

---

## 3. Mathematical Formulation & Proofs of the Semantic Bridge

### 3.1 Exact Spectral Derivatives ($s_0, s_1, s_2$)

Let $S_0(f)$ be the narrowband spectrum of semitone band $k$ centered at $f_k$ with Gaussian width $\sigma_k$ and FFT carrier frequency $f_{\text{car}, k}$. Within each sample-count group $c_k$, $s_0(t)$ has $M_k = c_k \cdot T_{\text{total}}$ complex samples. Via exact per-band FFT/IFFT on the GPU, `reassigned_primitives` computes two auxiliary complex signals $s_1(t)$ and $s_2(t)$:

$$S_1(f) = i 2\pi (f - f_{\text{car}, k}) S_0(f), \qquad S_2(f) = -\frac{f - f_k}{\sigma_k^2} S_0(f)$$

### 3.2 Theorem: First-Principles Resolution of the `c = 1` Degeneracy via Cohen-Mandel Instantaneous Bandwidth & Group-Delay Dispersion

Write the complex band signal in polar form $s_0(t) = A(t) e^{i \phi(t)}$. Because $s_1(t) = \frac{d}{dt} s_0(t) = \left(\dot{A}(t) + i \dot{\phi}(t) A(t)\right) e^{i\phi(t)}$, the hermitian product $s_1(t) s_0^*(t)$ splits into orthogonal physical quantities in its real and imaginary parts:

$$s_1(t) s_0^*(t) = \underbrace{A(t) \dot{A}(t)}_{\mathrm{Re}(s_1 s_0^*)\text{: Instantaneous AM}} + \, i \underbrace{\dot{\phi}(t) |s_0(t)|^2}_{\mathrm{Im}(s_1 s_0^*)\text{: Instantaneous FM}}$$

Similarly, from the frequency derivative $s_2(t)$, the local group-delay offset $\hat{\tau}_k(t)$ is given by:
$$\hat{\tau}_k(t) = \frac{\mathrm{Im}(s_2(t) s_0^*(t))}{2\pi |s_0(t)|^2}$$

**Proposition (Scale-Invariant Stationary-Phase Discrimination at $c = 1$)**:
Define the two dimensionless, scale-normalized quantities using the band's Gaussian bandwidth $\sigma_k$:
$$\tilde{m}_k(t) = \frac{|\mathrm{Re}(s_1(t) s_0^*(t))|}{2\pi \sigma_k |s_0(t)|^2} = \frac{|\dot{A}(t)|}{2\pi \sigma_k A(t)}, \qquad \tilde{\tau}_k(t) = 2\pi \sigma_k |\hat{\tau}_k(t)| = \frac{\sigma_k |\mathrm{Im}(s_2(t) s_0^*(t))|}{|s_0(t)|^2}$$

1. **Pure Stationary Sinusoid**: For any pure sinusoid $x(t) = A_0 \cos(2\pi f_0 t + \phi_0)$ at any frequency $f_0$ (including sub-bass $c = 1$ bands), $A(t)$ is constant ($\dot{A}(t) \equiv 0$) and $S_0(f)$ is supported at $f = f_0$, where $-\frac{f_0 - f_k}{\sigma_k^2}$ is a real scalar constant. Hence $s_2(t) s_0^*(t)$ is purely real, yielding:
   $$\tilde{m}_k(t) \equiv 0, \qquad \tilde{\tau}_k(t) \equiv 0$$
2. **Broadband Gaussian Noise**: By Rice's envelope theorem and the Heisenberg scaling identity $\sigma_k \cdot \tau_{\text{coh}, k} = \frac{1}{2\pi}$, white noise filtered through any Gaussian constant-$Q$ band $k$ has envelope derivative $\dot{A}/A \propto 2\pi \sigma_k$ and group-delay dispersion $|\hat{\tau}_k| \propto (2\pi \sigma_k)^{-1}$. Thus both $\tilde{m}_k(t)$ and $\tilde{\tau}_k(t)$ have **scale-invariant $\mathcal{O}(1)$ distributions across all 149 semitone bands**, even when a cell contains only $c_k = 1$ sample!

Combining multi-frame demodulated phase-step consistency $C_{\text{step}}(k, t)$ with the stationary-phase spectral factor:
$$\mathcal{C}(k, t) = C_{\text{step}}(k, t) \cdot \exp\left(-0.85 \left(\overline{\tilde{m}}_k^2(t) + \overline{\tilde{\tau}}_k^2(t)\right)\right)$$
yields a scale-consistent phase coherence $\mathcal{C}(k, t) \in [0, 1]$ that evaluates to $\approx 1.000$ on bass sinusoids while dropping to $\approx 0.598$ on bass white noise.

### 3.3 Differentiable 2D Soft-Bilinear Reassignment (Synchrosqueezing)

To bring `detail_view(mode="sharp")` into end-to-end neural training, we compute cross-band harmonic ridge agreement $\mathcal{A}_f(k, t) \in [0, 1]$ and cross-band transient time agreement $\mathcal{A}_t(k, t) \in [0, 1]$:
$$\mathcal{A}_f(k, t) = \exp\!\left(-\frac{\min(|\hat{M}_k - \hat{M}_{k-1}|, |\hat{M}_{k+1} - \hat{M}_k|)^2}{\Delta_M^2}\right), \quad \mathcal{A}_t(k, t) = \exp\!\left(-\frac{\min(|\hat{n}_k - \hat{n}_{k-1}|, |\hat{n}_{k+1} - \hat{n}_k|)^2}{(\alpha \sigma_{t,k})^2}\right)$$
Each cell $(k, t)$ moves to continuous reassigned coordinates:
$$r^*(k, t) = k + \mathcal{A}_f(k, t) \cdot \Delta M_k(t), \qquad t^*(k, t) = t + \mathcal{A}_t(k, t) \cdot \Delta n_k(t)$$
and splats its linear Mid power $P_{\text{mid}}(k, t)$ onto the four surrounding grid cells $(\lfloor r^* \rfloor, \lfloor t^* \rfloor), \dots, (\lfloor r^* \rfloor + 1, \lfloor t^* \rfloor + 1)$ via bilinear weights (`_soft_bilinear_scatter_2d`). Because the four bilinear weights sum to $1$, **total energy is conserved exactly** ($\sum_{k,t} P_{\text{sharp}}(k, t) = \sum_{k,t} P_{\text{mid}}(k, t)$) and gradients propagate smoothly with respect to both power and reassigned coordinates.

---

## 4. The 12-Plane Unified Semantic Tensor (`semantic_planes`)

`semantic_planes(image)` returns a tensor of shape `[B, 12, 149, 120]` where silent regions below $-60\text{ dB}$ are cleanly gated to prevent noise-floor phase randomness from polluting gradients or embeddings:

| Plane | Name | Range | Physical & Semantic Role |
| :---: | :--- | :---: | :--- |
| **0** | `mid_db` | $[-1, 1]$ | Raw cell Mid power in dB re clip peak (smooth energy anchor) |
| **1** | `side_rel_db` | $[-1, 1]$ | Side-to-Mid power ratio in dB (stereo width per band/frame) |
| **2** | `sharp_db` | $[-1, 1]$ | **Differentiable Reassigned (Synchrosqueezed) Mid power in dB** (de-smeared bass attacks & 1-pixel harmonic ridges) |
| **3** | `phase_coherence` | $[0, 1]$ | **Scale-consistent phase & stationary-phase coherence** (fixes `c = 1` bug down to 16.35 Hz) |
| **4** | `vel_cos` | $[-1, 1]$ | Demodulated phase velocity cosine weighted by coherence |
| **5** | `vel_sin` | $[-1, 1]$ | Demodulated phase velocity sine (sub-semitone pitch offset) |
| **6** | `group_delay` | $[-1, 1]$ | Normalized local group-delay offset $\tanh(\hat{\tau}_k / 3)$ (sub-frame onset pointer) |
| **7** | `harmonic_agreement` | $[0, 1]$ | Cross-band instantaneous frequency consensus $\mathcal{A}_f$ ($1$ on resolved harmonics) |
| **8** | `transient_agreement` | $[0, 1]$ | Cross-band group-delay consensus $\mathcal{A}_t$ ($1$ on broadband/bass impacts) |
| **9** | `mod_depth` | $[0, 1]$ | **Scale-consistent amplitude modulation depth** (fixes `c = 1` bug via Cohen-Mandel AM) |
| **10** | `jtfs_tilt` | $[-1, 1]$ | **2D Joint Time-Frequency Scattering directional tilt** ($+1$ upward glide, $-1$ downward glide) |
| **11** | `stereo_corr` | $[-1, 1]$ | Inter-channel L/R phase correlation ($+1$ mono, $0$ wide, $-1$ anti-phase) |

---

## 5. Reproducible Experimental Validation (`experiments/semantic_bridge_experiments.py`)

All five experiments were executed using `experiments/semantic_bridge_experiments.py` on 192 kHz stereo signals (`WaveletImageSpec()`), with exact results saved in `experiments/semantic_bridge_results.json`.

### Experiment 1: Resolution of the `c = 1` Multirate Cell Degeneracy ($16.35\text{ Hz} - 440.0\text{ Hz}$)

We compared legacy `_cell_stats` (`morph.py`) against `reassigned_primitives` and `harmonic_noise_gate` (`semantic_bridge.py`) across all 58 bands in the `c = 1` group on pure bass sinusoids (E1 = 41.2 Hz, A2 = 110 Hz, A3 = 220 Hz) vs. white noise:

| Metric (Rows 0..57, $c = 1$) | Legacy (`_cell_stats` / `morph_large`) | Semantic Bridge (`semantic_bridge.py`) | Scientific Impact |
| :--- | :---: | :---: | :--- |
| **E1 (41.2 Hz) Tone Coherence** | `1.0000` | `0.9932` | Preserves sub-bass tonal lock |
| **A2 (110 Hz) Tone Coherence** | `1.0000` | `1.0000` | Exact tonal lock |
| **A3 (220 Hz) Tone Coherence** | `1.0000` | `1.0000` | Exact tonal lock |
| **White Noise Coherence ($c=1$)** | **`1.0000`** *(bug!)* | **`0.5975`** | **Fixes false 100% tonality on noise** |
| **A2 (110 Hz) Modulation Depth** | `0.0000` | `0.0006` | Near-zero AM on steady tone |
| **White Noise Modulation Depth ($c=1$)** | **`0.0000`** *(bug!)* | **`0.8411`** | **Restores envelope roughness in bass** |
| **A2 (110 Hz) Harmonic Gate $m$** | `0.9734` | `0.9734` | Full harmonic synthesis on bass tone |
| **White Noise Harmonic Gate $m$ ($c=1$)** | **`0.9734`** *(bug!)* | **`0.2795`** | **Re-enables 72% stochastic noise path** |

### Experiment 2: Breaking the Low-Frequency Heisenberg Trap ($41.2\text{ Hz} - 164.8\text{ Hz}$)

We evaluated a sharp impulse at $t = 0.50\text{ s}$ (frame 50) and a double impulse separated by $60\text{ ms}$ ($t = 0.47\text{ s}$ and $t = 0.53\text{ s}$) in the bass register:

| Bass Transient Metric | Raw Wavelet Power (`mid_power`) | Reassigned Sharp Power (`sharp_mid_power`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Energy Fraction within $\pm 1$ Frame ($30\text{ ms}$) of Impulse** | `37.74%` (`0.3774`) | **`99.37%`** (`0.9937`) | **2.63$\times$ concentration** |
| **RMS Temporal Spread $\sigma_t$ in Bass ($41 - 165\text{ Hz}$)** | `38.30 ms` | **`4.00 ms`** | **9.58$\times$ sharper onset** |
| **60 ms Double-Impulse Peak-to-Valley Contrast** | `2.31 dB` | **`5.35 dB`** | **+3.04 dB valley separation** |
| **Global Energy Conservation Ratio** | `1.000000` | **`1.000000`** | **Exact conservation** |

### Experiment 3: Second-Order 2D Joint Time-Frequency Scattering (JTFS)

We tested signal pairs with **identical 1D frequency marginals and identical temporal envelopes**:
1. **Upward vs. Downward Exponential Chirp ($110\text{ Hz} \leftrightarrow 880\text{ Hz}$)**:
   - 1D time-pooled `judge_view` cosine distance: **`0.000001`** (blind to direction).
   - 2D `jtfs_modulation_spectrum` cosine distance: **`0.1661`** (>166,000$\times$ higher separability).
   - Plane 10 (`jtfs_tilt`) mean on active ridge: **`+0.3961`** (upward) vs. **`-0.4153`** (downward).
2. **Fast Vibrato ($8\text{ Hz}$) vs. Slow Pitch Drift ($1.5\text{ Hz}$) at $220\text{ Hz}$**:
   - 1D time-pooled `judge_view` cosine distance: **`0.000906`**.
   - 2D `jtfs_modulation_spectrum` cosine distance: **`0.0628`** (~69$\times$ higher separability).

### Experiment 4: Time-Shift & Phase Invariance vs. Physical Sensitivity

We compared the raw complex wavelet carrier in the $110\text{ Hz}$ band against `semantic_wavelet_embedding` (`[B, 252]`):

| Transformation / Comparison | Raw Complex Carrier Cosine | `semantic_wavelet_embedding` Cosine | Desired Property |
| :--- | :---: | :---: | :--- |
| **Sub-Frame Time Shift ($\Delta t = 2.7\text{ ms}$)** | `-0.2910` *(phase flip)* | **`1.00000`** | Shift-invariant |
| **Multi-Frame Time Shift ($\Delta t = 43.0\text{ ms}$)** | `-0.1218` *(decorrelated)* | **`1.00000`** | Shift-invariant |
| **Global Carrier Phase Rotation ($\Delta \phi = 90^\circ$)** | `0.0000` *(orthogonal)* | **`1.00000`** | Phase-rotation-invariant |
| **110 Hz Sine vs. 110 Hz Sawtooth** | — | **`0.8365`** | Distinguishes harmonic timbre |
| **110 Hz Sine vs. 110 Hz Square** | — | **`0.9059`** | Distinguishes odd/even harmonics |
| **Mono vs. Anti-Phase Stereo** | — | **`-0.1854`** | Distinguishes spatial phase |

### Experiment 5: Direct Autograd Waveform Optimization via `semantic_structure_loss`

Optimizing a 192 kHz stereo waveform directly from a noisy, detuned initialization ($116\text{ Hz}$ + white noise) toward a plucked $110\text{ Hz} + 220\text{ Hz}$ bass note over 25 Adam steps using `semantic_structure_loss`:
- `semantic_structure_loss` decreased smoothly from **`4.5363` to `3.8853` ($-14.35\%$)** with zero gradient instabilities.
- `semantic_wavelet_embedding` cosine similarity to the target rose from **`0.3300` to `0.4529`**.

---

## 6. Recommended Next-Stage Applications & Experiments

1. **Upgrading `MorphNet-L` and `MorphDiffusion` with `semantic_planes` & `harmonic_noise_gate`**:
   - Replace the 7-plane `critic_view` in `morph_large.py` with the 12-plane `semantic_planes` (or its 20 kHz slice) and replace line 351's `_cell_stats` gate with `harmonic_noise_gate(image)`. This immediately unlocks realistic sub-bass/mid-bass breath, bow noise, and percussive attack synthesis below 440 Hz.
2. **Interactive Audio Editor Multi-Layer Inspection**:
   - Render `semantic_planes` channels 2 (`sharp_db`), 3 (`phase_coherence`), 7 (`harmonic_agreement`), 8 (`transient_agreement`), and 10 (`jtfs_tilt`) as toggleable HSV/RGBA overlays in the audio editor, enabling users to visually select and edit **only coherent harmonics**, **only reassigned transient attacks**, or **only upward/downward pitch glides** while synthesizing back through the lossless `WaveletImage` carrier.
3. **Contrastive / Self-Supervised Audio Foundation Embeddings**:
   - Use `semantic_wavelet_embedding` (252-D) as a deterministic teacher target (`FixedTargetStatistics`) or input token representation for `MaskedMusicModel` and `Wavelet2Vec`, combining analytical deformation stability with learned high-level musical semantics.
