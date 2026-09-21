"""G-8: train the conditioned transient net and A/B it against the
deterministic generic pluck.

Free supervision: take sounds that *have* attacks (plucks, drums, the
percussive library folders), suppress the attack with a 150 ms onset ramp,
and fit (gain-over-body, decay τ) per band on the original — measured
labels, no annotation. The model sees the suppressed sound's envelopes +
the original's measured sharpness and predicts the fitted parameters.

Eval (held-out real one-shots, audio domain, real API paths):
- reconstruction: log band-envelope L1 in the attack window vs the
  original, for do-nothing / deterministic generic pluck / model;
- conditioning: sharpness sweep on suppressed clips must be monotone in
  the measured sharpness label.

Run:  python -m experiments.train_transient
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wavelet2vec.audio_io import list_audio_files, load_audio, resample
from wavelet2vec.frame import analyze
from wavelet2vec.imagine import imagine_transient
from wavelet2vec.scenes import render_drum, render_note
from wavelet2vec.transient_model import (
    ATTACK_S,
    MODEL_BANDS,
    MODEL_FRAMES,
    TransientNet,
    band_envelopes,
    fit_attack_params,
    onset_index,
    save_transient_model,
    sharpness_label,
)

LIBRARY = Path(r"c:\Users\vikto\PycharmProjects\soundgen\example_audio")
PERCUSSIVE_FOLDERS = ("kicks", "Percussive_kick", "percussion",
                      "vocal_shots", "harmonic_stabs_one_shots")
OUT = Path(__file__).parent / "output" / "transient_v1"
SR = 22050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SYNTHETIC = 280
STEPS = 4000
BATCH = 32
CHECKPOINT = OUT / "transient_v1.pt"


def _real_paths() -> list[Path]:
    paths = []
    for p in list_audio_files(LIBRARY):
        rel = Path(p).relative_to(LIBRARY)
        if len(rel.parts) >= 2 and rel.parts[0] in PERCUSSIVE_FOLDERS:
            paths.append(Path(p))
    return paths


def _render_clip(index: int) -> np.ndarray | None:
    if index < N_SYNTHETIC:
        rng = np.random.default_rng(5000 + index)
        kind = index % 10
        if kind < 4:
            clip = render_note(int(rng.integers(36, 85)), 1.0, SR,
                               instrument="pluck")
        elif kind < 7:
            clip = render_note(int(rng.integers(36, 85)), 1.0, SR,
                               instrument="saw")
        else:
            drum = render_drum(str(rng.choice(["kick", "snare", "hat"])), SR)
            clip = np.zeros(SR)
            clip[: drum.shape[0]] = drum
    else:
        try:
            waveform, sr = load_audio(_real_paths()[index - N_SYNTHETIC])
        except Exception:
            return None
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        mono = resample(mono[None, :], sr, SR)[0][: SR]
        if mono.shape[0] < SR:
            mono = np.pad(mono, (0, SR - mono.shape[0]))
        clip = mono
    peak = np.abs(clip).max()
    return None if peak < 1e-6 else clip / peak


def _suppress_attack(clip: np.ndarray, onset: int) -> np.ndarray:
    t = np.arange(clip.shape[0])
    ramp = np.clip((t - onset) / (0.15 * SR), 0.0, 1.0)
    return clip * ramp


def _model_envelopes(clip: np.ndarray) -> np.ndarray:
    """The exact input transform of imagine_transient's model path."""
    coeffs, _spec = analyze(clip, SR, n_bands=MODEL_BANDS, f_min=30.0)
    magnitude = np.abs(coeffs)
    hop = max(magnitude.shape[1] // MODEL_FRAMES, 1)
    pooled = magnitude[:, : hop * MODEL_FRAMES].reshape(
        MODEL_BANDS, MODEL_FRAMES, hop).mean(axis=2)
    peak = pooled.max()
    return np.log1p(pooled / peak if peak > 1e-12 else pooled).astype(np.float32)


def build_rows() -> dict:
    cache = OUT / "rows.npz"
    n_clips = N_SYNTHETIC + len(_real_paths())
    state = {"envelopes": [], "labels": [], "sharpness": [], "is_real": [],
             "clip_index": [], "n_done": 0}
    if cache.exists():
        archive = np.load(cache)
        if int(archive["n_done"]) >= n_clips:
            return {key: archive[key] for key in archive.files}
        for key in ("envelopes", "labels", "sharpness", "is_real", "clip_index"):
            state[key] = list(archive[key])
        state["n_done"] = int(archive["n_done"])
        print(f"resuming build at clip {state['n_done']}/{n_clips}")

    start = time.perf_counter()
    for index in range(state["n_done"], n_clips):
        clip = _render_clip(index)
        if clip is not None:
            sharp = sharpness_label(clip, SR)
            if sharp > 0.25:  # only sounds that actually have an attack
                onset = onset_index(clip, SR)
                coeffs, _spec = analyze(clip, SR, n_bands=MODEL_BANDS,
                                        f_min=30.0)
                labels = fit_attack_params(coeffs, SR, onset)
                suppressed = _suppress_attack(clip, onset)
                state["envelopes"].append(_model_envelopes(suppressed))
                state["labels"].append(labels)
                state["sharpness"].append(np.float32(sharp))
                state["is_real"].append(index >= N_SYNTHETIC)
                state["clip_index"].append(index)
        state["n_done"] = index + 1
        if state["n_done"] % 50 == 0 or state["n_done"] == n_clips:
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, envelopes=np.stack(state["envelopes"]),
                labels=np.stack(state["labels"]),
                sharpness=np.array(state["sharpness"]),
                is_real=np.array(state["is_real"]),
                clip_index=np.array(state["clip_index"]),
                n_done=state["n_done"])
            print(f"  clip {state['n_done']}/{n_clips} "
                  f"({len(state['labels'])} rows, "
                  f"{time.perf_counter() - start:.0f}s)")
    archive = np.load(cache)
    return {key: archive[key] for key in archive.files}


def train(rows: dict) -> TransientNet:
    rng = np.random.default_rng(6)
    n_rows = rows["labels"].shape[0]
    holdout = rng.random(n_rows) < 0.15
    envelopes = torch.from_numpy(rows["envelopes"]).to(DEVICE)[:, None]
    labels = torch.from_numpy(rows["labels"]).to(DEVICE)
    sharpness = torch.from_numpy(rows["sharpness"]).to(DEVICE)[:, None]
    train_rows = torch.from_numpy(np.flatnonzero(~holdout)).long().to(DEVICE)
    val_rows = torch.from_numpy(np.flatnonzero(holdout)).long().to(DEVICE)

    model = TransientNet(width=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TransientNet {n_params/1e3:.0f}k params on {DEVICE}: "
          f"{train_rows.shape[0]} train / {val_rows.shape[0]} val rows")

    def batch_from(pool, batch_rng):
        picks = pool[torch.from_numpy(
            batch_rng.integers(0, pool.shape[0], BATCH)).long()]
        return envelopes[picks], sharpness[picks], labels[picks]

    best_val = float("inf")
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        model.train()
        inputs, condition, target = batch_from(train_rows, rng)
        loss = ((model(inputs, condition) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                v_in, v_cond, v_tgt = batch_from(
                    val_rows, np.random.default_rng(7))
                val_loss = float(((model(v_in, v_cond) - v_tgt) ** 2).mean())
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_transient_model(model, CHECKPOINT)
                marker = "  *saved"
            if step % 500 == 0 or step == 1:
                print(f"step {step:5d}  train {loss.item():.4f}  "
                      f"val {val_loss:.4f}  "
                      f"({time.perf_counter()-start:.0f}s){marker}")
    np.save(OUT / "holdout_rows.npy", holdout)
    from wavelet2vec.transient_model import load_transient_model

    return load_transient_model(CHECKPOINT)


def _attack_window_l1(original: np.ndarray, candidate: np.ndarray,
                      onset: int) -> float:
    attack_n = int(ATTACK_S * SR)
    c_orig, _s = analyze(original, SR, n_bands=MODEL_BANDS, f_min=30.0)
    c_cand, _s = analyze(candidate, SR, n_bands=MODEL_BANDS, f_min=30.0)
    window = slice(onset, onset + attack_n)
    e_orig = np.log1p(band_envelopes(c_orig, SR)[:, window])
    e_cand = np.log1p(band_envelopes(c_cand, SR)[:, window])
    return float(np.abs(e_orig - e_cand).mean())


def evaluate(model: TransientNet, rows: dict) -> dict:
    holdout = np.load(OUT / "holdout_rows.npy")
    held_real = np.flatnonzero(holdout & rows["is_real"])[:30]
    losses = {"do_nothing": [], "deterministic": [], "model": []}
    recovery = {"do_nothing": [], "deterministic": [], "model": []}
    for row in held_real:
        clip = _render_clip(int(rows["clip_index"][row]))
        if clip is None:
            continue
        onset = onset_index(clip, SR)
        suppressed = _suppress_attack(clip, onset)
        sharp = float(rows["sharpness"][row])
        det = imagine_transient(suppressed, SR, 1.0)
        learned = imagine_transient(suppressed, SR, 1.0, model=model,
                                    sharpness=sharp)
        losses["do_nothing"].append(_attack_window_l1(clip, suppressed, onset))
        losses["deterministic"].append(
            _attack_window_l1(clip, det.astype(np.float64), onset))
        losses["model"].append(
            _attack_window_l1(clip, learned.astype(np.float64), onset))
        # Sharpness recovery: the axis-relevant question — how close does
        # the measured attack character land to the original's?
        for name, candidate in (("do_nothing", suppressed),
                                ("deterministic", det.astype(np.float64)),
                                ("model", learned.astype(np.float64))):
            recovery[name].append(
                abs(sharpness_label(candidate, SR) - sharp))
    report = {name: float(np.mean(values)) for name, values in losses.items()}
    report.update({f"sharpness_recovery_{name}": float(np.mean(values))
                   for name, values in recovery.items()})
    report["n_eval_clips"] = len(losses["model"])

    # the sharpness knob: monotone in the measured label on suppressed pads
    targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    sweeps = []
    for row in held_real[:8]:
        clip = _render_clip(int(rows["clip_index"][row]))
        if clip is None:
            continue
        suppressed = _suppress_attack(clip, onset_index(clip, SR))
        measured = [
            sharpness_label(
                imagine_transient(suppressed, SR, 1.0, model=model,
                                  sharpness=t).astype(np.float64), SR)
            for t in targets
        ]
        sweeps.append(measured)
    sweeps = np.array(sweeps)
    report["sweep_targets"] = targets
    report["mean_measured_sweep"] = [float(x) for x in sweeps.mean(axis=0)]
    report["monotone_fraction"] = float(np.mean(
        np.all(np.diff(sweeps, axis=1) >= -0.02, axis=1)))
    return report


def main() -> None:
    rows = build_rows()
    print(f"{rows['labels'].shape[0]} rows "
          f"({int(rows['is_real'].sum())} real); sharpness "
          f"{rows['sharpness'].min():.2f}–{rows['sharpness'].max():.2f}")
    if CHECKPOINT.exists() and (OUT / "holdout_rows.npy").exists():
        from wavelet2vec.transient_model import load_transient_model

        print("reusing", CHECKPOINT)
        model = load_transient_model(CHECKPOINT)
    else:
        model = train(rows)
    report = evaluate(model, rows)
    (OUT / "report.json").write_text(json.dumps(report, indent=1),
                                     encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
