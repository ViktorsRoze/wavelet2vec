from __future__ import annotations

import math
import torch

from wavelet2vec.morph import _cell_stats
from wavelet2vec.semantic_bridge import (
    SPEC,
    harmonic_noise_gate,
    jtfs_modulation_spectrum,
    reassigned_primitives,
    semantic_planes,
    semantic_structure_loss,
    semantic_wavelet_embedding,
)
from wavelet2vec.wavelet_image import encode


def _stereo(mono: torch.Tensor) -> torch.Tensor:
    return torch.stack([mono, mono], dim=0)[None, :, :]


def test_c1_degeneracy_resolved():
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr

    # 110 Hz pure bass tone (MIDI 45, row 33, inside c=1 group rows 0..57)
    tone = 0.5 * torch.sin(2.0 * math.pi * 110.0 * t)
    torch.manual_seed(42)
    noise = 0.25 * torch.randn(n, dtype=torch.float32)

    batch = torch.cat([_stereo(tone), _stereo(noise)], dim=0)
    img = encode(batch, SPEC)

    # Confirm legacy _cell_stats suffers from c=1 degeneracy (|u| == 1, mod == 0 even on white noise)
    _, _, u_re, u_im, legacy_mod = _cell_stats(img.groups[1])
    legacy_coh = torch.sqrt(u_re ** 2 + u_im ** 2)
    assert torch.allclose(legacy_coh[1], torch.ones_like(legacy_coh[1]), atol=1e-4)
    assert torch.allclose(legacy_coh[0, 33], torch.ones_like(legacy_coh[0, 33]), atol=1e-4)
    assert torch.allclose(legacy_mod[1], torch.zeros_like(legacy_mod[1]), atol=5e-3)

    # Verify semantic_bridge resolves c=1 degeneracy
    prim = reassigned_primitives(img)
    sl = slice(SPEC.pad_frames + 10, SPEC.pad_frames + SPEC.n_frames - 10)
    row_110 = 33  # MIDI 45 = 110 Hz

    tone_coh = prim.phase_coherence[0, row_110, sl].mean().item()
    noise_coh = prim.phase_coherence[1, :58, sl].mean().item()
    assert tone_coh > 0.95, f"Expected high bass tone coherence, got {tone_coh}"
    assert noise_coh < 0.65, f"Expected low bass noise coherence, got {noise_coh}"

    tone_mod = prim.mod_depth[0, row_110, sl].mean().item()
    noise_mod = prim.mod_depth[1, :58, sl].mean().item()
    assert tone_mod < 0.05, f"Expected near-zero modulation on steady tone, got {tone_mod}"
    assert noise_mod > 0.25, f"Expected non-zero modulation on bass noise, got {noise_mod}"

    gate = harmonic_noise_gate(img, core_only=True)
    tone_gate = gate[0, row_110, 10:-10].mean().item()
    noise_gate = gate[1, :58, 10:-10].mean().item()
    assert tone_gate > 0.85, f"Expected high harmonic gate on 110 Hz tone, got {tone_gate}"
    assert noise_gate < 0.45, f"Expected low harmonic gate on bass noise, got {noise_gate}"


def test_low_frequency_heisenberg_reassignment():
    n = SPEC.audio_length
    click = torch.zeros(n, dtype=torch.float32)
    center_sample = int(0.50 * SPEC.sample_rate)  # frame 50 in core
    # Short broadband pulse
    win = torch.hann_window(64)
    click[center_sample - 32 : center_sample + 32] = 0.9 * win

    img = encode(_stereo(click), SPEC)
    prim = reassigned_primitives(img)

    # Energy conservation of soft bilinear scatter
    raw_total = prim.mid_power.sum()
    sharp_total = prim.sharp_mid_power.sum()
    assert torch.isclose(raw_total, sharp_total, rtol=1e-2)

    # Check temporal concentration in low-frequency rows (rows 20..50, ~26 Hz to 147 Hz)
    target_frame = SPEC.pad_frames + 50
    bass_raw = prim.mid_power[0, 20:50, :]
    bass_sharp = prim.sharp_mid_power[0, 20:50, :]

    raw_peak_ratio = (
        bass_raw[:, target_frame - 1 : target_frame + 2].sum() / (bass_raw.sum() + 1e-30)
    ).item()
    sharp_peak_ratio = (
        bass_sharp[:, target_frame - 1 : target_frame + 2].sum() / (bass_sharp.sum() + 1e-30)
    ).item()

    assert sharp_peak_ratio > raw_peak_ratio * 1.35, (
        f"Expected sharp reassigned bass energy concentration ({sharp_peak_ratio:.3f}) "
        f"to exceed raw blurred concentration ({raw_peak_ratio:.3f})"
    )


def test_jtfs_chirp_direction_asymmetry():
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr
    # Instantaneous frequency sweeping exponentially from 110 Hz to 880 Hz (3 octaves in 1 s)
    f0, f1 = 110.0, 880.0
    k = math.log(f1 / f0)
    phase = 2.0 * math.pi * f0 * (torch.exp(k * t) - 1.0) / k
    env = torch.sin(math.pi * t) ** 2
    chirp_up = 0.5 * env * torch.sin(phase)
    chirp_down = torch.flip(chirp_up, dims=[0])

    batch = torch.cat([_stereo(chirp_up), _stereo(chirp_down)], dim=0)
    img = encode(batch, SPEC)
    planes = semantic_planes(img, core_only=True)

    # Plane 10 is jtfs_tilt; active cells should have opposite mean tilt signs
    active_up = planes[0, 2] > -0.5
    active_down = planes[1, 2] > -0.5
    tilt_up = planes[0, 10][active_up].mean().item()
    tilt_down = planes[1, 10][active_down].mean().item()
    assert tilt_up * tilt_down < 0.0, (
        f"Expected opposite JTFS tilt signs for up ({tilt_up:.4f}) vs down ({tilt_down:.4f})"
    )
    assert abs(tilt_up - tilt_down) > 0.25

    jtfs_vec = jtfs_modulation_spectrum(planes)
    cos_sim = torch.nn.functional.cosine_similarity(jtfs_vec[0:1], jtfs_vec[1:2]).item()
    assert cos_sim < 0.85, f"Expected JTFS spectrum to separate up/down chirps, got cos={cos_sim:.4f}"


def test_differentiability_and_silence_stability():
    n = SPEC.audio_length
    sr = SPEC.sample_rate
    t = torch.arange(n, dtype=torch.float32) / sr

    audio_a = _stereo(0.3 * torch.sin(2.0 * math.pi * 220.0 * t)).requires_grad_(True)
    audio_b = _stereo(0.3 * torch.sin(2.0 * math.pi * 440.0 * t))

    img_a = encode(audio_a, SPEC)
    img_b = encode(audio_b, SPEC)
    loss = semantic_structure_loss(img_a, img_b)
    loss.backward()

    assert audio_a.grad is not None
    assert torch.isfinite(audio_a.grad).all()
    assert audio_a.grad.abs().max().item() > 0.0

    # Exact silence stability check
    silent = torch.zeros((1, 2, n), dtype=torch.float32, requires_grad=True)
    img_silent = encode(silent, SPEC)
    planes_silent = semantic_planes(img_silent)
    emb_silent = semantic_wavelet_embedding(img_silent)
    assert torch.isfinite(planes_silent).all()
    assert torch.isfinite(emb_silent).all()

    silent_loss = planes_silent.sum() + emb_silent.sum()
    silent_loss.backward()
    assert silent.grad is not None
    assert torch.isfinite(silent.grad).all()


def test_shift_invariance_and_timbre_discrimination():
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr

    # Windowed 220 Hz sine note vs shifted by 37 ms (3.7 frames -> non-integer frame shift)
    env1 = torch.exp(-0.5 * ((t - 0.45) / 0.12) ** 2)
    env2 = torch.exp(-0.5 * ((t - 0.487) / 0.12) ** 2)
    sine1 = 0.5 * env1 * torch.sin(2.0 * math.pi * 220.0 * t)
    sine2 = 0.5 * env2 * torch.sin(2.0 * math.pi * 220.0 * (t - 0.037))

    # Sawtooth wave at same fundamental 220 Hz and envelope env1
    saw = torch.zeros_like(t)
    for h in range(1, 12):
        saw = saw + ((-1) ** (h + 1)) * (1.0 / h) * torch.sin(2.0 * math.pi * 220.0 * h * t)
    saw = 0.35 * env1 * saw

    batch = torch.cat([_stereo(sine1), _stereo(sine2), _stereo(saw)], dim=0)
    emb = semantic_wavelet_embedding(batch, SPEC)

    sim_shifted = (emb[0] @ emb[1]).item()
    sim_timbre = (emb[0] @ emb[2]).item()
    assert sim_shifted > 0.995, f"Expected high shift invariance, got {sim_shifted:.5f}"
    assert sim_timbre < 0.92, f"Expected clear sine vs sawtooth separation, got {sim_timbre:.5f}"
