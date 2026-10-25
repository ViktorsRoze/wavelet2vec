"""Experiments on wavelet2vec.semantic_bridge (192 kHz WaveletImage).

1-5 use synthetic signals (tones, white noise, clicks, chirps). 6 uses real notes from note_dataset (the
prepared 192 kHz store of experiments/train_morph.py); it is skipped when that data is not present or with
--skip-real. Compared against: the per-cell statistics of the existing views (``_cell_stats``, ``judge_view``,
``critic_view``) and MorphNet-L's actual tonal/noise gate (``morph_large.noise_mask``).

Saves results to `experiments/semantic_bridge_results.json`; `plot_semantic_bridge_figures.py` draws them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from wavelet2vec.morph import _cell_stats, judge_view
from wavelet2vec.morph_large import harmonic_row_mask, noise_gate_inputs, noise_mask
from wavelet2vec.semantic_bridge import (
    SPEC,
    fit_embedding_standardizer,
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
    # MorphNet-L's tonal share (1 - noise share m) in those rows: as trained ("cell") and opt-in ("scale")
    rows1 = torch.as_tensor(list(img.rows[1]))
    v1 = img.groups[1]
    tonal_cell = 1.0 - noise_mask(v1, v1, 1, rows1)
    tonal_scale = 1.0 - noise_mask(v1, v1, 1, rows1, noise_gate_inputs(img, img, "scale"))

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
            "tone_A2_morphnet_l_tonal_share": round(tonal_cell[1, 33, sl].mean().item(), 4),
            "white_noise_c1_morphnet_l_tonal_share": round(tonal_cell[3, :58, sl].mean().item(), 4),
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
            "tone_A2_morphnet_l_tonal_share": round(tonal_scale[1, 33, sl].mean().item(), 4),
            "white_noise_c1_morphnet_l_tonal_share": round(tonal_scale[3, :58, sl].mean().item(), 4),
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
        "sine_vs_sawtooth_110hz": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[4]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[4]).item(), 4),
        },
        "sine_vs_square_110hz": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[5]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[5]).item(), 4),
        },
        # anti-phase stereo has no mid signal: the raw mid carrier is all zeros (cosine reported as 0)
        "mono_vs_antiphase_stereo": {
            "raw_complex_carrier_cosine": round((z_real_vec[0] @ z_real_vec[6]).item(), 4),
            "semantic_embedding_cosine": round((emb[0] @ emb[6]).item(), 4),
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


def _auc(pos, neg) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    values = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]                 # average rank of each value, ties shared
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def experiment_6_real_notes(per_class: int = 48, families: int = 60) -> Dict[str, Any]:
    """Experiment 6: real notes from note_dataset (needs the prepared store; see train_morph.Catalog).

    a. Harmonic vs noise cells below 440 Hz (one-sample rows): AUC of the existing per-cell coherence, the
       semantic-bridge coherence and the bridge's harmonic gate. Positives: loud cells (within 30 dB of the
       note's peak) on a row of one of the first 11 partials of vocal or synth notes (MIDI 36-64). Negatives:
       loud cells of note-shaped noise bursts (white, pink, brown, band-pass, clicks).
    b. MorphNet-L's noise share m in those rows: share of cells sent to the noise path (m > 0.5).
    c. Embeddings: synth-family identification by nearest neighbour (cosine) among notes at least 7 semitones
       apart, and vocal vs synth by cross-validated logistic regression.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_morph as tm

    started = time.time()
    cat = tm.Catalog()
    rng = np.random.default_rng(3)
    core = slice(SPEC.pad_frames, SPEC.pad_frames + SPEC.n_frames)
    low = [i for i, x in enumerate(cat.notes) if 36 <= x["midi"] <= 64]
    picks = {k: rng.choice([i for i in low if cat.notes[i]["kind"] == k], per_class, replace=False).tolist()
             for k in ("vox", "synth")}
    noise = np.stack([tm.noise_note(rng) for _ in range(per_class)])
    scores = ("existing per-cell coherence", "semantic-bridge coherence", "semantic-bridge harmonic gate")
    routes = ("cell (as trained)", "scale", "scale + pitch protection")
    cells = {k: {name: [] for name in scores} for k in ("vox", "synth", "noise")}
    routed = {k: {name: [] for name in routes} for k in ("vox", "synth", "noise")}

    def measure(kind, audio, midis):
        img = encode(torch.from_numpy(audio).float(), SPEC)
        rows = torch.as_tensor(list(img.rows[1]))
        prim = reassigned_primitives(img)
        _, _, u_re, u_im, _ = _cell_stats(img.groups[1])
        values = {scores[0]: torch.sqrt(u_re ** 2 + u_im ** 2), scores[1]: prim.phase_coherence[:, rows],
                  scores[2]: harmonic_noise_gate(img)[:, rows]}
        level = 10 * torch.log10(prim.mid_power[:, rows] + 1e-30)
        mask = (level > level.amax(dim=(1, 2), keepdim=True) - 30)[..., core]
        if kind != "noise":
            mask &= harmonic_row_mask(midis, harmonics=11, reach=0.5)[:, rows][:, :, None]
        for name in scores:
            cells[kind][name].append(values[name][..., core][mask].numpy())
        v = img.groups[1]
        gates = (None, noise_gate_inputs(img, img, "scale"), noise_gate_inputs(img, img, "scale", midis))
        for name, gate in zip(routes, gates):
            routed[kind][name].append((noise_mask(v, v, 1, rows, gate)[..., core][mask] > 0.5).float().numpy())

    for kind, idx in picks.items():
        for chunk in np.array_split(np.array(idx), 6):
            measure(kind, np.stack([cat.normalized(int(i))[0] for i in chunk]),
                    [cat.notes[int(i)]["midi"] for i in chunk])
    for chunk in np.array_split(np.arange(per_class), 6):
        measure("noise", noise[chunk], [48] * len(chunk))      # pitch protection at an arbitrary key for noise
    cells = {k: {name: np.concatenate(v) for name, v in d.items()} for k, d in cells.items()}
    routed = {k: {name: np.concatenate(v) for name, v in d.items()} for k, d in routed.items()}
    part_a = {name: {"auc_vox": round(_auc(cells["vox"][name], cells["noise"][name]), 3),
                     "auc_synth": round(_auc(cells["synth"][name], cells["noise"][name]), 3),
                     "mean_harmonic_vox": round(float(cells["vox"][name].mean()), 3),
                     "mean_harmonic_synth": round(float(cells["synth"][name].mean()), 3),
                     "mean_noise": round(float(cells["noise"][name].mean()), 3)} for name in scores}
    part_b = {name: {"harmonic_cells_to_noise_vox": round(float(routed["vox"][name].mean()), 3),
                     "harmonic_cells_to_noise_synth": round(float(routed["synth"][name].mean()), 3),
                     "noise_cells_to_noise": round(float(routed["noise"][name].mean()), 3)} for name in routes}

    # c. embeddings
    fams = rng.choice(sorted({n["family"] for n in cat.notes if n["kind"] == "synth"}), families, replace=False)
    synth = []
    for f in fams:
        idx = sorted((i for i, n in enumerate(cat.notes) if n["family"] == f), key=lambda i: cat.notes[i]["midi"])
        synth += [idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, 6)]
    vox = rng.choice([i for i, n in enumerate(cat.notes) if n["kind"] == "vox"], 2 * families, replace=False).tolist()
    notes = synth + vox
    audio = np.stack([cat.normalized(i)[0] for i in notes])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sem, spectrum = [], []
    for start in range(0, len(notes), 8):
        img = encode(torch.from_numpy(audio[start:start + 8]).float().to(device), SPEC)
        with torch.no_grad():
            sem.append(semantic_wavelet_embedding(img).cpu())
            spectrum.append(judge_view(img)[:, 0].mean(-1).cpu())
    sem, spectrum = torch.cat(sem), torch.cat(spectrum)
    stats = fit_embedding_standardizer(sem)
    sem_std = F.normalize((sem - stats["mean"]) / stats["std"], dim=-1)
    w2v = np.stack([tm.w2v_features(a) for a in audio])
    midi = np.array([cat.notes[i]["midi"] for i in notes])
    fam = np.array([cat.notes[i]["family"] for i in notes])
    is_vox = np.array([cat.notes[i]["kind"] == "vox" for i in notes])

    def family_1nn(x):
        x = np.asarray(x, dtype=np.float64)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        n = len(synth)
        sims = x[:n] @ x[:n].T
        sims[np.abs(midi[:n, None] - midi[None, :n]) < 7] = -np.inf
        return round(float((fam[:n][sims.argmax(1)] == fam[:n]).mean()), 3)

    def vox_cv(x):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
        return round(float(cross_val_score(clf, np.asarray(x), is_vox, cv=5, scoring="balanced_accuracy").mean()), 3)

    def zscore(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-8)

    candidates = [("semantic embedding", sem.numpy()), ("semantic embedding, standardized", sem_std.numpy()),
                  ("Wavelet2Vec", w2v), ("Wavelet2Vec, standardized", zscore(w2v)),
                  ("mean spectrum", spectrum.numpy()), ("mean spectrum, standardized", zscore(spectrum.numpy()))]
    part_c = {name: {"dims": int(x.shape[1]), "family_1nn": family_1nn(x), "vox_vs_synth": vox_cv(x)}
              for name, x in candidates}
    return {
        "notes": {"cells_notes_per_class": per_class, "embedding_synth_notes": len(synth),
                  "embedding_synth_families": int(families), "embedding_vox_notes": len(vox)},
        "a_harmonic_vs_noise_cells_below_440hz": part_a,
        "b_morphnet_l_noise_path_below_440hz": part_b,
        "c_embeddings": {**part_c, "chance_family_1nn": round(1 / families, 3)},
        "seconds": round(time.time() - started, 1),
    }


def run_all_experiments(real: bool = True) -> Dict[str, Any]:
    results = {
        "experiment_1_c1_degeneracy": experiment_1_c1_degeneracy(),
        "experiment_2_heisenberg_trap": experiment_2_heisenberg_trap(),
        "experiment_3_jtfs_chirp_and_modulation": experiment_3_jtfs_chirp_and_modulation(),
        "experiment_4_invariance_vs_sensitivity": experiment_4_invariance_vs_sensitivity(),
        "experiment_5_gradient_optimization": experiment_5_gradient_optimization(),
    }
    out_path = Path(__file__).resolve().parent / "semantic_bridge_results.json"
    if real:
        try:
            results["experiment_6_real_notes"] = experiment_6_real_notes()
        except (FileNotFoundError, ImportError) as error:
            print(f"experiment 6 skipped (note_dataset store not available): {error}")
    if "experiment_6_real_notes" not in results and out_path.exists():   # keep the last real-note results
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if "experiment_6_real_notes" in previous:
            results["experiment_6_real_notes"] = previous["experiment_6_real_notes"]
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true", help="skip experiment 6 (real notes)")
    run_all_experiments(real=not parser.parse_args().skip_real)
