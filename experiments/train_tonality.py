"""G-9: train the conditioned tonality net and score its calibration.

The G-2 recipe with one twist: harmonicity is a *waveform* feature, so
labels must be rendered — each training row is a scalar+tilt band-gain
remix of a clip's HPSS layers, synthesized and measured with
``autocorrelation_pitch``. The build is cached and resumable (it rewrites
its npz every 25 clips), so interrupted runs continue where they stopped.

Eval (held-out real clips): request harmonicity targets and measure what
arrives — the model's one-forward-pass calibration against an *oracle*
scalar remix (per-clip, per-target grid search over the deterministic
knob, the best any scalar can do), plus monotonicity of measured vs
requested.

Run:  python -m experiments.train_tonality   (repeat until it reports)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wavelet2vec.audio_io import list_audio_files, load_audio, resample
from wavelet2vec.decompose import hpss
from wavelet2vec.harmonic_features import autocorrelation_pitch
from wavelet2vec.scenes import random_scene, render_note
from wavelet2vec.tonality_model import (
    LOG_GAIN_MAX,
    MODEL_BANDS,
    TonalityNet,
    layer_envelopes,
    remix_layers,
    save_tonality_model,
)

LIBRARY = Path(r"c:\Users\vikto\PycharmProjects\soundgen\example_audio")
OUT = Path(__file__).parent / "output" / "tonality_v1"
SR = 22050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SYNTHETIC = 300
N_REAL = 200
SETTINGS_PER_CLIP = 7  # 1 identity + 6 sampled remixes
STEPS = 6000
BATCH = 64
CHECKPOINT = OUT / "tonality_v1.pt"


def _real_paths() -> list[Path]:
    paths = [Path(p) for p in list_audio_files(LIBRARY)
             if len(Path(p).relative_to(LIBRARY).parts) >= 2]
    return paths[::2][:N_REAL]


def _render_clip(index: int) -> np.ndarray | None:
    """Deterministic clip per index: synthetic first, then real crops."""
    if index < N_SYNTHETIC:
        if index % 5 < 3:
            clip, _ = random_scene(SR, seconds=1.0, bpm=90.0 + (index * 7) % 60,
                                   seed=1000 + index)
        else:
            rng = np.random.default_rng(2000 + index)
            clip = render_note(int(rng.integers(36, 85)), 1.0, SR,
                               instrument=str(rng.choice(["saw", "pluck", "sine"])))
    else:
        path = _real_paths()[index - N_SYNTHETIC]
        try:
            waveform, sr = load_audio(path)
        except Exception:
            return None
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        mono = resample(mono[None, :], sr, SR)[0]
        if mono.shape[0] < SR:
            mono = np.pad(mono, (0, SR - mono.shape[0]))
        begin = (mono.shape[0] - SR) // 2
        clip = mono[begin: begin + SR]
    peak = np.abs(clip).max()
    return None if peak < 1e-6 else clip / peak


AMOUNT_LADDER = np.linspace(-1.0, 1.0, 9)


def build_rows() -> dict:
    """Per-clip response curves over the scalar remix family, resumable.

    The rungs are the deterministic knob's own amounts ``a`` with gains
    ``(1+a, 1−a)`` — including complete layer removal at ``a=±1``, which
    log-gain families cannot reach (their ±31 dB cap left the median
    response span at 0.285 and the oracle out of reach; measured).
    Rendering the *same family the model drives* makes each clip's
    a↔measured-harmonicity curve a function to invert.
    """
    cache = OUT / "rows_amount.npz"
    n_clips = N_SYNTHETIC + len(_real_paths())
    state = {"envelopes": [], "curve_labels": [], "n_done": 0}
    if cache.exists():
        archive = np.load(cache)
        if int(archive["n_done"]) >= n_clips:
            return {key: archive[key] for key in archive.files}
        state = {"envelopes": list(archive["envelopes"]),
                 "curve_labels": list(archive["curve_labels"]),
                 "n_done": int(archive["n_done"])}
        print(f"resuming label build at clip {state['n_done']}/{n_clips}")

    start = time.perf_counter()
    for index in range(state["n_done"], n_clips):
        clip = _render_clip(index)
        if clip is not None:
            harmonic, percussive = hpss(clip, SR, n_bands=64)
            envelopes, _centers, coeff_pair = layer_envelopes(
                harmonic, percussive, SR)
            labels = []
            for a in AMOUNT_LADDER:
                gains = np.stack([np.full(MODEL_BANDS, 1.0 + a),
                                  np.full(MODEL_BANDS, 1.0 - a)])
                remix = remix_layers(coeff_pair, gains, clip.shape[0])
                labels.append(autocorrelation_pitch(remix, SR)[0])
            state["envelopes"].append(envelopes)
            state["curve_labels"].append(np.asarray(labels, dtype=np.float32))
        state["n_done"] = index + 1
        if state["n_done"] % 25 == 0 or state["n_done"] == n_clips:
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, envelopes=np.stack(state["envelopes"]),
                curve_labels=np.stack(state["curve_labels"]),
                n_done=state["n_done"])
            elapsed = time.perf_counter() - start
            print(f"  clip {state['n_done']}/{n_clips} ({elapsed:.0f}s)")
    archive = np.load(cache)
    return {key: archive[key] for key in archive.files}


def train(rows: dict) -> TonalityNet:
    rng = np.random.default_rng(4)
    n_clips = rows["envelopes"].shape[0]
    holdout_clip = rng.random(n_clips) < 0.15
    envelopes = torch.from_numpy(rows["envelopes"]).to(DEVICE)
    # Monotone δ↔label curve per clip (cumulative max stabilizes the
    # occasional non-monotone measurement) for on-the-fly inversion.
    curves = np.maximum.accumulate(rows["curve_labels"], axis=1)
    train_clips = np.flatnonzero(~holdout_clip)
    val_clips = np.flatnonzero(holdout_clip)

    model = TonalityNet(width=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TonalityNet {n_params/1e3:.0f}k params on {DEVICE}: "
          f"{train_clips.shape[0]} train clips / {val_clips.shape[0]} val clips")

    def batch_from(pool: np.ndarray, batch_rng: np.random.Generator):
        picks = pool[batch_rng.integers(0, pool.shape[0], BATCH)]
        requested = np.empty(BATCH, dtype=np.float32)
        amounts = np.empty(BATCH, dtype=np.float32)
        for i, clip_index in enumerate(picks):
            curve = curves[clip_index]
            if batch_rng.random() < 0.3:  # teach saturation beyond the range
                target = float(batch_rng.uniform(0.0, 1.0))
            else:
                target = float(batch_rng.uniform(curve[0], curve[-1]))
            requested[i] = target
            amounts[i] = np.interp(target, curve, AMOUNT_LADDER)
        return (envelopes[torch.from_numpy(picks).long()],
                torch.from_numpy(requested[:, None]).to(DEVICE),
                torch.from_numpy(amounts[:, None]).to(DEVICE))

    best_val = float("inf")
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        model.train()
        inputs, condition, target = batch_from(train_clips, rng)
        loss = ((model.amount_from(inputs, condition) - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                v_in, v_cond, v_tgt = batch_from(
                    val_clips, np.random.default_rng(5))
                val_loss = float(
                    ((model.amount_from(v_in, v_cond) - v_tgt) ** 2).mean())
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_tonality_model(model, CHECKPOINT)
                marker = "  *saved"
            if step % 500 == 0 or step == 1:
                print(f"step {step:5d}  train {loss.item():.4f}  "
                      f"val {val_loss:.4f}  ({time.perf_counter()-start:.0f}s)"
                      f"{marker}")
    np.save(OUT / "holdout_clips.npy", holdout_clip)
    from wavelet2vec.tonality_model import load_tonality_model

    return load_tonality_model(CHECKPOINT)  # best-val weights


def evaluate(model: TonalityNet, rows: dict) -> dict:
    from wavelet2vec.tonality_model import predict_layer_gains

    holdout_clip = np.load(OUT / "holdout_clips.npy")
    real_start = N_SYNTHETIC
    held_real = [i for i in np.flatnonzero(holdout_clip) if i >= real_start][:12]
    targets = [0.15, 0.3, 0.45, 0.6, 0.75]

    model_errors, anchored_errors, oracle_errors = [], [], []
    sweeps, anchored_sweeps = [], []
    for index in held_real:
        clip = _render_clip(int(index))
        if clip is None:
            continue
        harmonic, percussive = hpss(clip, SR, n_bands=64)
        envelopes, _centers, coeff_pair = layer_envelopes(
            harmonic, percussive, SR)

        # oracle scalar: the best any deterministic amount can do, per target
        amounts = np.linspace(-1.0, 1.0, 41)
        achieved_scalar = []
        for a in amounts:
            mixed = harmonic * (1.0 + a) + percussive * (1.0 - a)
            achieved_scalar.append(autocorrelation_pitch(mixed, SR)[0])
        achieved_scalar = np.array(achieved_scalar)

        # anchored inversion (the library's model-free harmonicity= path):
        # measure five anchors, invert the monotone curve
        anchor_amounts = np.linspace(-1.0, 1.0, 5)
        anchor_curve = np.maximum.accumulate(
            [achieved_scalar[i] for i in [0, 10, 20, 30, 40]])

        measured, anchored_measured = [], []
        for target in targets:
            gains = predict_layer_gains(model, envelopes, target)
            remix = remix_layers(coeff_pair, gains, clip.shape[0])
            got = autocorrelation_pitch(remix, SR)[0]
            measured.append(got)
            model_errors.append(abs(got - target))

            a_star = float(np.interp(target, anchor_curve, anchor_amounts))
            mixed = harmonic * (1.0 + a_star) + percussive * (1.0 - a_star)
            got_anchored = autocorrelation_pitch(mixed, SR)[0]
            anchored_measured.append(got_anchored)
            anchored_errors.append(abs(got_anchored - target))

            oracle_errors.append(float(np.abs(achieved_scalar - target).min()))
        sweeps.append(measured)
        anchored_sweeps.append(anchored_measured)

    sweeps = np.array(sweeps)
    anchored_sweeps = np.array(anchored_sweeps)
    return {
        "n_eval_clips": int(sweeps.shape[0]),
        "targets": targets,
        "model_mean_abs_error": float(np.mean(model_errors)),
        "anchored_mean_abs_error": float(np.mean(anchored_errors)),
        "oracle_scalar_mean_abs_error": float(np.mean(oracle_errors)),
        "mean_measured_sweep": [float(x) for x in sweeps.mean(axis=0)],
        "anchored_mean_sweep": [float(x) for x in anchored_sweeps.mean(axis=0)],
        "monotone_fraction": float(np.mean(
            np.all(np.diff(sweeps, axis=1) >= -0.02, axis=1))),
        "anchored_monotone_fraction": float(np.mean(
            np.all(np.diff(anchored_sweeps, axis=1) >= -0.02, axis=1))),
    }


def main() -> None:
    rows = build_rows()
    curves = rows["curve_labels"]
    print(f"{curves.shape[0]} clip response curves; "
          f"label range {curves.min():.3f}–{curves.max():.3f}; "
          f"median span {np.median(curves[:, -1] - curves[:, 0]):.3f}")
    if CHECKPOINT.exists() and (OUT / "holdout_clips.npy").exists():
        from wavelet2vec.tonality_model import load_tonality_model

        print("reusing", CHECKPOINT)
        model = load_tonality_model(CHECKPOINT)
    else:
        model = train(rows)
    report = evaluate(model, rows)
    (OUT / "report.json").write_text(json.dumps(report, indent=1),
                                     encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
