"""G-2 v1: train the conditioned brightness net and A/B it against G-2a.

Supervision is free (G-1): a coefficient-domain lowpass scales band
envelopes by the same mask, so (masked envelopes → full envelopes) pairs
need no audio synthesis. The pool mixes synthetic scenes/notes with
1-second crops of the real library, all on the fixed 64-band/128-frame
grid at 22 050 Hz; 15 % of each pool is held out.

Eval (held-out real crops, audio domain, through the real API paths):
- reconstruction: log-envelope L1 above the 4 kHz split for
  `imagine_highs` deterministic vs `imagine_highs(model=...)`
  (conditioned on the true target brightness), with the band-limited
  input itself as the do-nothing floor;
- conditioning: sweep the brightness knob on dull inputs and check the
  measured output brightness is monotone.

Run:  python -m experiments.train_brightness
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wavelet2vec.audio_io import list_audio_files, load_audio, resample
from wavelet2vec.brightness_model import (
    BrightnessNet,
    cutoff_mask,
    envelope_brightness,
    log_envelopes,
    save_brightness_model,
)
from wavelet2vec.datasets import axis_labels, lowpass_pair
from wavelet2vec.imagine import imagine_highs
from wavelet2vec.scenes import random_scene, render_note

LIBRARY = Path(r"c:\Users\vikto\PycharmProjects\soundgen\example_audio")
OUT = Path(__file__).parent / "output" / "brightness_v1"
SR = 22050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_HZ = 4000.0
STEPS = 10000
BATCH = 32
CHECKPOINT = OUT / "brightness_v2.pt"
TRAIN_STATE = OUT / "train_state_v2.pt"


def build_envelope_pool() -> dict:
    cache = OUT / "envelopes.npz"
    if cache.exists():
        archive = np.load(cache)
        return {key: archive[key] for key in archive.files}
    rng = np.random.default_rng(0)
    envelopes, is_real = [], []
    centers = None

    start = time.perf_counter()
    for i in range(600):  # synthetic: scenes + single notes
        if i % 5 < 3:
            clip, _events = random_scene(SR, seconds=1.0,
                                         bpm=float(rng.uniform(90, 150)), seed=i)
        else:
            clip = render_note(int(rng.integers(36, 85)), 1.0, SR,
                               instrument=str(rng.choice(["saw", "pluck", "sine"])))
        env, centers = log_envelopes(clip, SR)
        envelopes.append(env)
        is_real.append(False)

    for path in list_audio_files(LIBRARY):  # real: 1 s crops, resampled
        if len(Path(path).relative_to(LIBRARY).parts) < 2:
            continue
        try:
            waveform, sr = load_audio(path)
        except Exception:
            continue
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        mono = resample(mono[None, :], sr, SR)[0]
        if mono.shape[0] < SR:
            mono = np.pad(mono, (0, SR - mono.shape[0]))
        begin = (mono.shape[0] - SR) // 2
        env, centers = log_envelopes(mono[begin: begin + SR], SR)
        envelopes.append(env)
        is_real.append(True)

    pool = {"envelopes": np.stack(envelopes).astype(np.float32),
            "is_real": np.array(is_real), "centers": centers}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **pool)
    print(f"envelope pool {pool['envelopes'].shape} "
          f"in {time.perf_counter() - start:.0f}s")
    return pool


def train(pool: dict) -> BrightnessNet:
    rng = np.random.default_rng(1)
    envelopes = pool["envelopes"]
    centers = pool["centers"]
    holdout = rng.random(envelopes.shape[0]) < 0.15
    train_env = torch.from_numpy(envelopes[~holdout]).to(DEVICE)
    val_env = torch.from_numpy(envelopes[holdout]).to(DEVICE)
    centers_np = np.asarray(centers, dtype=np.float64)
    high_row = torch.from_numpy(
        (centers_np >= SPLIT_HZ).astype(np.float32)).to(DEVICE)[None, :, None]

    def make_batch(source: torch.Tensor, batch_rng: np.random.Generator):
        rows = batch_rng.integers(0, source.shape[0], BATCH)
        target = source[torch.from_numpy(rows).long()]
        cutoffs = np.exp(batch_rng.uniform(np.log(800.0), np.log(8000.0), BATCH))
        masks = torch.from_numpy(np.stack([
            cutoff_mask(centers_np, c) for c in cutoffs])).to(DEVICE)
        linear = torch.expm1(target)
        masked = torch.log1p(linear * masks[:, :, None])

        # v2: brightness-tilt the *hidden* bands so conditioning labels cover
        # the whole knob range (real sounds rarely measure above ~0.1). The
        # visible bands stay identical to the input — at synthesis the low
        # end is preserved, so only the hidden bands can deliver brightness.
        boost = np.ones((BATCH, centers_np.shape[0]), dtype=np.float32)
        for i in range(BATCH):
            if batch_rng.random() < 0.3:
                continue  # keep the pure-reconstruction case in distribution
            tau = float(np.exp(batch_rng.uniform(np.log(0.25), np.log(8.0))))
            slope = float(batch_rng.uniform(-0.5, 1.0))
            boost[i] = np.clip(
                tau * (centers_np / cutoffs[i]) ** slope, 0.05, 20.0)
        boost_t = torch.from_numpy(boost).to(DEVICE)
        tilted = linear * (masks + (1.0 - masks) * boost_t)[:, :, None]
        target = torch.log1p(tilted)

        condition = torch.from_numpy(np.array([
            [envelope_brightness(row, centers_np)]
            for row in target.cpu().numpy()
        ], dtype=np.float32)).to(DEVICE)
        weight = 0.05 + 0.95 * (1.0 - masks)[:, :, None]  # focus on hidden bands
        return masked[:, None], condition, target[:, None], weight[:, None]

    model = BrightnessNet(width=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    begin_step = 0
    if TRAIN_STATE.exists():
        state = torch.load(TRAIN_STATE, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        begin_step = int(state["step"])
        rng = np.random.default_rng([1, begin_step])
        print(f"resuming from step {begin_step}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BrightnessNet {n_params/1e3:.0f}k params on {DEVICE}, "
          f"{train_env.shape[0]} train / {val_env.shape[0]} held-out clips")

    start = time.perf_counter()
    for step in range(begin_step + 1, STEPS + 1):
        model.train()
        inputs, condition, target, weight = make_batch(train_env, rng)
        prediction = model(inputs, condition)
        loss = (weight * (prediction - target).abs()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                v_in, v_cond, v_tgt, v_w = make_batch(
                    val_env, np.random.default_rng(2))
                v_pred = model(v_in, v_cond)
                val_loss = (v_w * (v_pred - v_tgt).abs()).mean()
                val_high = ((v_pred - v_tgt).abs() * high_row[None]).sum() \
                    / high_row.sum() / v_pred.shape[0] / v_pred.shape[3]
            print(f"step {step:5d}  train {loss.item():.4f}  "
                  f"val {val_loss.item():.4f}  val-high-L1 {val_high.item():.4f}"
                  f"  ({time.perf_counter() - start:.0f}s)")
        if step % 500 == 0:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step},
                       TRAIN_STATE)
    model.eval().cpu()
    save_brightness_model(model, CHECKPOINT)

    # persist the holdout so the audio eval uses genuinely unseen real clips
    np.save(OUT / "holdout_mask.npy", holdout)
    return model


def _high_band_l1(reference: np.ndarray, candidate: np.ndarray,
                  centers: np.ndarray) -> float:
    rows = centers >= SPLIT_HZ
    return float(np.abs(reference[rows] - candidate[rows]).mean())


def evaluate(model: BrightnessNet, pool: dict) -> dict:
    holdout = np.load(OUT / "holdout_mask.npy")
    centers = np.asarray(pool["centers"], dtype=np.float64)
    real_held = np.flatnonzero(holdout & pool["is_real"])
    print(f"audio-domain eval on {min(len(real_held), 30)} held-out real clips")

    # Rebuild the exact held-out crops (same deterministic crop as the pool).
    crops = []
    real_index = 0
    for path in list_audio_files(LIBRARY):
        if len(Path(path).relative_to(LIBRARY).parts) < 2:
            continue
        try:
            waveform, sr = load_audio(path)
        except Exception:
            continue
        flat_index = 600 + real_index  # synthetic pool comes first
        real_index += 1
        if flat_index not in real_held:
            continue
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        mono = resample(mono[None, :], sr, SR)[0]
        if mono.shape[0] < SR:
            mono = np.pad(mono, (0, SR - mono.shape[0]))
        begin = (mono.shape[0] - SR) // 2
        crops.append(mono[begin: begin + SR])

    losses = {"floor_lowpassed": [], "deterministic": [], "model": []}
    for crop in crops[:30]:
        peak = np.abs(crop).max()
        if peak < 1e-6:
            continue
        crop = crop / peak
        true_env, _ = log_envelopes(crop, SR)
        target_brightness = envelope_brightness(true_env, centers)
        limited, _full = lowpass_pair(crop, SR, SPLIT_HZ)
        limited_env, _ = log_envelopes(limited, SR)
        det = imagine_highs(limited, SR, 1.0)
        det_env, _ = log_envelopes(det, SR)
        learned = imagine_highs(limited, SR, 1.0, model=model,
                                brightness=target_brightness)
        learned_env, _ = log_envelopes(learned, SR)
        losses["floor_lowpassed"].append(_high_band_l1(true_env, limited_env, centers))
        losses["deterministic"].append(_high_band_l1(true_env, det_env, centers))
        losses["model"].append(_high_band_l1(true_env, learned_env, centers))

    report = {name: float(np.mean(values)) for name, values in losses.items()}
    report["n_eval_clips"] = len(losses["model"])

    # Conditioning: the knob must be monotone in its own measured label.
    sweeps = []
    targets = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6]
    for crop in crops[:8]:
        peak = np.abs(crop).max()
        if peak < 1e-6:
            continue
        limited, _ = lowpass_pair(crop / peak, SR, 2000.0)  # a dull input
        measured = []
        for target in targets:
            output = imagine_highs(limited, SR, 1.0, model=model, brightness=target)
            measured.append(axis_labels(output, SR)["brightness"])
        sweeps.append(measured)
    sweeps = np.array(sweeps)
    increasing = np.mean(np.all(np.diff(sweeps, axis=1) > 0, axis=1))
    report["monotone_fraction"] = float(increasing)
    report["mean_sweep"] = [float(x) for x in sweeps.mean(axis=0)]
    report["sweep_targets"] = targets
    return report


def main() -> None:
    pool = build_envelope_pool()
    trained = False
    if TRAIN_STATE.exists():
        state = torch.load(TRAIN_STATE, map_location="cpu", weights_only=True)
        trained = int(state["step"]) >= STEPS
    if trained and CHECKPOINT.exists() and (OUT / "holdout_mask.npy").exists():
        from wavelet2vec.brightness_model import load_brightness_model

        print("reusing", CHECKPOINT)
        model = load_brightness_model(CHECKPOINT)
    else:
        model = train(pool)
    report = evaluate(model, pool)
    (OUT / "report_v2.json").write_text(json.dumps(report, indent=1),
                                        encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
