"""Reproducible scientific experiments validating the Semantic Bridge on 192 kHz WaveletImage.

Runs 5 controlled experiments comparing legacy cell-level views (`_cell_stats`, `judge_view`,
`critic_view`) against `wavelet2vec.semantic_bridge` (`scale_consistent_cell_stats`,
`reassigned_primitives`, `semantic_planes`, `jtfs_modulation_spectrum`,
`semantic_wavelet_embedding`, `harmonic_noise_gate`, and `semantic_structure_loss`).

Saves structured results to `experiments/semantic_bridge_results.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from wavelet2vec.morph import _cell_stats, judge_view
from wavelet2vec.semantic_bridge import (
    SPEC,
    harmonic_noise_gate,
    jtfs_modulation_spectrum,
    reassigned_primitives,
    semantic_planes,
    semantic_structure_loss,
    semantic_wavelet_embedding,
)
from wavelet2vec.wavelet_image import band_layout, encode


def _stereo(mono: torch.Tensor) -> torch.Tensor:
    return torch.stack([mono, mono], dim=0)[None, :, :]


def experiment_1_c1_degeneracy() -> Dict[str, Any]:
    """Experiment 1: Sub-bass & mid-bass (16.35..440 Hz, c=1) tonality vs. noise separation."""
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr
    bands = band_layout(SPEC)

    # Pure tones at E1 (41.2 Hz, MIDI 28, row 16), A2 (110 Hz, MIDI 45, row 33), A3 (220 Hz, MIDI 57, row 45)
    tone_e1 = 0.5 * torch.sin(2.0 * math.pi * bands[16].center_hz * t)
    tone_a2 = 0.5 * torch.sin(2.0 * math.pi * bands[33].center_hz * t)
    tone_a3 = 0.5 * torch.sin(2.0 * math.pi * bands[45].center_hz * t)

    torch.manual_seed(101)
    white_noise = 0.25 * torch.randn(n, dtype=torch.float32)

    batch = torch.cat(
        [_stereo(tone_e1), _stereo(tone_a2), _stereo(tone_a3), _stereo(white_noise)],
        dim=0,
    )
    img = encode(batch, SPEC)

    # Legacy _cell_stats on c=1 group (rows 0..57)
    _, _, u_re, u_im, leg_mod = _cell_stats(img.groups[1])
    leg_coh = torch.sqrt(u_re ** 2 + u_im ** 2)
    leg_gate = torch.sigmoid(8.0 * (leg_coh - 0.55) - 3.0 * leg_mod)

    # New scale-consistent primitives and harmonic-noise gate
    prim = reassigned_primitives(img)
    gate = harmonic_noise_gate(img, core_only=False)
    sl = slice(SPEC.pad_frames + 12, SPEC.pad_frames + SPEC.n_frames - 12)

    results = {
        "c1_band_count": 58,
        "c1_freq_range_hz": [round(bands[0].center_hz, 2), round(bands[57].center_hz, 2)],
        "legacy": {
            "tone_E1_coherence": round(leg_coh[0, 16, sl].mean().item(), 4),
            "tone_A2_coherence": round(leg_coh[1, 33, sl].mean().item(), 4),
            "tone_A3_coherence": round(leg_coh[2, 45, sl].mean().item(), 4),
            "white_noise_c1_coherence": round(leg_coh[3, :58, sl].mean().item(), 4),
            "tone_A2_mod_depth": round(leg_mod[1, 33, sl].mean().item(), 4),
            "white_noise_c1_mod_depth": round(leg_mod[3, :58, sl].mean().item(), 4),
            "tone_A2_harmonic_gate": round(leg_gate[1, 33, sl].mean().item(), 4),
            "white_noise_c1_harmonic_gate": round(leg_gate[3, :58, sl].mean().item(), 4),
        },
        "semantic_bridge": {
            "tone_E1_coherence": round(prim.phase_coherence[0, 16, sl].mean().item(), 4),
            "tone_A2_coherence": round(prim.phase_coherence[1, 33, sl].mean().item(), 4),
            "tone_A3_coherence": round(prim.phase_coherence[2, 45, sl].mean().item(), 4),
            "white_noise_c1_coherence": round(prim.phase_coherence[3, :58, sl].mean().item(), 4),
            "tone_A2_mod_depth": round(prim.mod_depth[1, 33, sl].mean().item(), 4),
            "white_noise_c1_mod_depth": round(prim.mod_depth[3, :58, sl].mean().item(), 4),
            "tone_A2_harmonic_gate": round(gate[1, 33, sl].mean().item(), 4),
            "white_noise_c1_harmonic_gate": round(gate[3, :58, sl].mean().item(), 4),
        },
    }
    return results


def experiment_2_heisenberg_trap() -> Dict[str, Any]:
    """Experiment 2: Low-frequency Heisenberg transient localization & 40 ms double-click resolution."""
    sr = SPEC.sample_rate
    n = SPEC.audio_length

    # 1. Single impulse at t = 0.50 s (frame 50 in core)
    click1 = torch.zeros(n, dtype=torch.float32)
    c0 = int(0.50 * sr)
    win = torch.hann_window(64)
    click1[c0 - 32 : c0 + 32] = 0.95 * win

    # 2. Double impulse separated by 60 ms (t = 0.47 s and t = 0.53 s -> frames 47 and 53)
    click2 = torch.zeros(n, dtype=torch.float32)
    c_left = int(0.47 * sr)
    c_right = int(0.53 * sr)
    click2[c_left - 32 : c_left + 32] = 0.95 * win
    click2[c_right - 32 : c_right + 32] = 0.95 * win

    batch = torch.cat([_stereo(click1), _stereo(click2)], dim=0)
    img = encode(batch, SPEC)
    prim = reassigned_primitives(img)

    f_center = SPEC.pad_frames + 50
    f_left = SPEC.pad_frames + 47
    f_right = SPEC.pad_frames + 53
    hop_ms = SPEC.hop_ms

    # Bass rows: MIDI 28..52 (41.2 Hz .. 164.8 Hz, rows 16..40)
    bass_sl = slice(16, 41)
    raw_bass = prim.mid_power[0, bass_sl, :]
    sharp_bass = prim.sharp_mid_power[0, bass_sl, :]

    # Energy within +/- 1 frame (30 ms window) of true impulse
    raw_conc = (raw_bass[:, f_center - 1 : f_center + 2].sum() / (raw_bass.sum() + 1e-30)).item()
    sharp_conc = (sharp_bass[:, f_center - 1 : f_center + 2].sum() / (sharp_bass.sum() + 1e-30)).item()

    # RMS temporal spread in ms around f_center
    frames = torch.arange(SPEC.total_frames, dtype=torch.float32)[None, :]
    dt_ms = (frames - f_center) * hop_ms
    raw_rms_ms = torch.sqrt((raw_bass * (dt_ms ** 2)).sum() / (raw_bass.sum() + 1e-30)).item()
    sharp_rms_ms = torch.sqrt((sharp_bass * (dt_ms ** 2)).sum() / (sharp_bass.sum() + 1e-30)).item()

    # Double click (60 ms apart) peak-to-valley ratio in bass register (rows 28..46, 82 Hz..233 Hz)
    dbl_sl = slice(28, 47)
    raw_dbl = prim.mid_power[1, dbl_sl, :].mean(dim=0)
    sharp_dbl = prim.sharp_mid_power[1, dbl_sl, :].mean(dim=0)

    raw_peak = 0.5 * (raw_dbl[f_left] + raw_dbl[f_right])
    raw_valley = raw_dbl[f_center]
    raw_p2v_db = (10.0 * torch.log10((raw_peak + 1e-30) / (raw_valley + 1e-30))).item()

    sharp_peak = 0.5 * (sharp_dbl[f_left] + sharp_dbl[f_right])
    sharp_valley = sharp_dbl[f_center]
    sharp_p2v_db = (10.0 * torch.log10((sharp_peak + 1e-30) / (sharp_valley + 1e-30))).item()

    energy_conservation = (prim.sharp_mid_power[0].sum() / (prim.mid_power[0].sum() + 1e-30)).item()

    return {
        "bass_register_hz": [41.2, 164.8],
        "single_impulse_30ms_energy_fraction": {
            "raw_wavelet_power": round(raw_conc, 4),
            "reassigned_sharp_power": round(sharp_conc, 4),
            "improvement_factor": round(sharp_conc / max(raw_conc, 1e-6), 2),
        },
        "single_impulse_rms_spread_ms": {
            "raw_wavelet_power_ms": round(raw_rms_ms, 2),
            "reassigned_sharp_power_ms": round(sharp_rms_ms, 2),
        },
        "double_impulse_60ms_peak_to_valley_db": {
            "raw_wavelet_power_db": round(raw_p2v_db, 2),
            "reassigned_sharp_power_db": round(sharp_p2v_db, 2),
        },
        "energy_conservation_ratio": round(energy_conservation, 6),
    }


def experiment_3_jtfs_chirp_and_modulation() -> Dict[str, Any]:
    """Experiment 3: 2D JTFS discrimination of time-reversed chirps, vibrato rates, and harmonic vs noise."""
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr
    env = torch.sin(math.pi * t) ** 2

    # Pair A: Upward vs Downward exponential chirp (110 Hz <-> 880 Hz)
    f0, f1 = 110.0, 880.0
    k = math.log(f1 / f0)
    phase = 2.0 * math.pi * f0 * (torch.exp(k * t) - 1.0) / k
    chirp_up = 0.5 * env * torch.sin(phase)
    chirp_down = torch.flip(chirp_up, dims=[0])

    # Pair B: Fast vibrato (8 Hz) vs Slow pitch sway (1.5 Hz) around 220 Hz (+/- 0.6 semitone)
    depth_hz = 220.0 * (2.0 ** (0.6 / 12.0) - 1.0)
    vib_fast = 0.5 * env * torch.sin(
        2.0 * math.pi * 220.0 * t - (depth_hz / 8.0) * torch.cos(2.0 * math.pi * 8.0 * t)
    )
    vib_slow = 0.5 * env * torch.sin(
        2.0 * math.pi * 220.0 * t - (depth_hz / 1.5) * torch.cos(2.0 * math.pi * 1.5 * t)
    )

    batch = torch.cat(
        [_stereo(chirp_up), _stereo(chirp_down), _stereo(vib_fast), _stereo(vib_slow)],
        dim=0,
    )
    img = encode(batch, SPEC)

    # 1D time-pooled spectral marginal from legacy judge_view
    j_view = judge_view(img)  # [4, 4, JUDGE_ROWS, 100]
    marginal_1d = F.normalize(j_view[:, 0].mean(dim=-1), dim=-1)  # time-averaged level spectrum

    # Our 12-plane semantic representation & JTFS spectrum
    planes = semantic_planes(img, core_only=True)
    jtfs_vec = F.normalize(jtfs_modulation_spectrum(planes), dim=-1)
    full_emb = semantic_wavelet_embedding(img)

    active_up = planes[0, 2] > -0.5
    active_down = planes[1, 2] > -0.5

    return {
        "up_vs_down_chirp": {
            "marginal_1d_cosine_distance": round(1.0 - (marginal_1d[0] @ marginal_1d[1]).item(), 6),
            "jtfs_cosine_distance": round(1.0 - (jtfs_vec[0] @ jtfs_vec[1]).item(), 4),
            "full_embedding_cosine_distance": round(1.0 - (full_emb[0] @ full_emb[1]).item(), 4),
            "mean_jtfs_tilt_up": round(planes[0, 10][active_up].mean().item(), 4),
            "mean_jtfs_tilt_down": round(planes[1, 10][active_down].mean().item(), 4),
        },
        "fast_8hz_vs_slow_1p5hz_vibrato": {
            "marginal_1d_cosine_distance": round(1.0 - (marginal_1d[2] @ marginal_1d[3]).item(), 6),
            "jtfs_cosine_distance": round(1.0 - (jtfs_vec[2] @ jtfs_vec[3]).item(), 4),
            "full_embedding_cosine_distance": round(1.0 - (full_emb[2] @ full_emb[3]).item(), 4),
        },
    }


def experiment_4_invariance_vs_sensitivity() -> Dict[str, Any]:
    """Experiment 4: Time-shift & phase-rotation invariance vs. waveshape & stereo sensitivity."""
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr

    def make_note(shift_s: float, phase_rad: float = 0.0, waveform: str = "sine") -> torch.Tensor:
        tau = t - (0.45 + shift_s)
        env = torch.exp(-0.5 * (tau / 0.12) ** 2)
        f0 = 110.0  # A2 in c=1 bass register
        if waveform == "sine":
            sig = torch.sin(2.0 * math.pi * f0 * tau + phase_rad)
        elif waveform == "saw":
            sig = torch.zeros_like(tau)
            for h in range(1, 16):
                sig = sig + ((-1) ** (h + 1)) * (1.0 / h) * torch.sin(2.0 * math.pi * f0 * h * tau + phase_rad)
        elif waveform == "square":
            sig = torch.zeros_like(tau)
            for h in range(1, 16, 2):
                sig = sig + (1.0 / h) * torch.sin(2.0 * math.pi * f0 * h * tau + phase_rad)
        else:
            raise ValueError(waveform)
        return _stereo(0.4 * env * sig)

    ref = make_note(0.0, 0.0, "sine")
    shift_subframe = make_note(0.0027, 0.0, "sine")   # 2.7 ms (0.27 frames)
    shift_multiframe = make_note(0.043, 0.0, "sine")  # 43.0 ms (4.3 frames)
    phase_rot_90 = make_note(0.0, math.pi / 2.0, "sine")
    saw_note = make_note(0.0, 0.0, "saw")
    square_note = make_note(0.0, 0.0, "square")

    # Anti-phase stereo version of ref
    anti_stereo = ref.clone()
    anti_stereo[:, 1, :] = -anti_stereo[:, 1, :]

    batch = torch.cat(
        [ref, shift_subframe, shift_multiframe, phase_rot_90, saw_note, square_note, anti_stereo],
        dim=0,
    )
    img = encode(batch, SPEC)
    emb = semantic_wavelet_embedding(img)

    # Raw complex coefficient cosine similarity in group c=1 (row 33 = 110 Hz)
    z = img.groups[1][:, 0, 33].reshape(7, -1)
    z_real_vec = F.normalize(torch.cat([z.real, z.imag], dim=-1), dim=-1)

    return {
        "shift_2p7ms_similarity": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[1]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[1]).item(), 5),
        },
        "shift_43ms_similarity": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[2]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[2]).item(), 5),
        },
        "phase_rotation_90deg_similarity": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[3]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[3]).item(), 5),
        },
        "timbre_and_spatial_discrimination_cosine": {
            "sine_vs_sawtooth_110hz": round((emb[0] @ emb[4]).item(), 4),
            "sine_vs_square_110hz": round((emb[0] @ emb[5]).item(), 4),
            "mono_vs_antiphase_stereo": round((emb[0] @ emb[6]).item(), 4),
        },
    }


def experiment_5_gradient_optimization() -> Dict[str, Any]:
    """Experiment 5: Direct autograd optimization of a waveform using semantic_structure_loss."""
    sr = SPEC.sample_rate
    n = SPEC.audio_length
    t = torch.arange(n, dtype=torch.float32) / sr

    # Target: percussive 110 Hz + 220 Hz plucked bass note with onset at t = 0.30 s
    onset = (t >= 0.30).float()
    decay = torch.exp(-6.0 * torch.clamp(t - 0.30, min=0.0)) * onset
    target_mono = decay * (
        0.55 * torch.sin(2.0 * math.pi * 110.0 * (t - 0.30))
        + 0.30 * torch.sin(2.0 * math.pi * 220.0 * (t - 0.30))
    )
    target_audio = _stereo(target_mono)
    target_img = encode(target_audio, SPEC)

    # Initialize candidate as low-level noise + detuned 116 Hz tone
    torch.manual_seed(7)
    init_mono = 0.08 * torch.randn(n) + 0.15 * torch.sin(2.0 * math.pi * 116.0 * t)
    cand_audio = _stereo(init_mono).clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([cand_audio], lr=1.5e-2)
    initial_loss = None
    final_loss = None
    initial_emb_sim = None

    for step in range(25):
        opt.zero_grad()
        cand_img = encode(cand_audio, SPEC)
        loss = semantic_structure_loss(cand_img, target_img)
        if step == 0:
            initial_loss = loss.item()
            with torch.no_grad():
                e_cand = semantic_wavelet_embedding(cand_img)
                e_targ = semantic_wavelet_embedding(target_img)
                initial_emb_sim = (e_cand[0] @ e_targ[0]).item()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    with torch.no_grad():
        cand_img = encode(cand_audio, SPEC)
        e_cand = semantic_wavelet_embedding(cand_img)
        e_targ = semantic_wavelet_embedding(target_img)
        final_emb_sim = (e_cand[0] @ e_targ[0]).item()

    return {
        "optimization_steps": 25,
        "initial_semantic_loss": round(float(initial_loss), 4),
        "final_semantic_loss": round(float(final_loss), 4),
        "loss_reduction_pct": round(100.0 * (initial_loss - final_loss) / initial_loss, 2),
        "initial_embedding_cosine": round(float(initial_emb_sim), 4),
        "final_embedding_cosine": round(float(final_emb_sim), 4),
    }


def run_all_experiments() -> Dict[str, Any]:
    results = {
        "experiment_1_c1_degeneracy": experiment_1_c1_degeneracy(),
        "experiment_2_heisenberg_trap": experiment_2_heisenberg_trap(),
        "experiment_3_jtfs_chirp_and_modulation": experiment_3_jtfs_chirp_and_modulation(),
        "experiment_4_invariance_vs_sensitivity": experiment_4_invariance_vs_sensitivity(),
        "experiment_5_gradient_optimization": experiment_5_gradient_optimization(),
    }
    out_path = Path(__file__).resolve().parent / "semantic_bridge_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_all_experiments()

