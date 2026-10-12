"""Train MorphDiffusion stage 0 (wavelet2vec.morph_diffusion): a diffusion prior over note feature images.

The evaluation experiment for a diffusion approach to vox-synth morphing
(docs/MORPH_DIFFUSION.md). It runs on the GPU and leaves the CPU free.

- Data: the cached 5-plane feature images (critic view) of all 6,515 notes in
  MIDI 36-82: vox, synth, Exhale, Harmor vox. Held out, matching MorphNet:
  the vocal takes "tze", "koh" and "EH", 8 synth families, and Exhale presets
  14 and 15.
- Batches are class-balanced: vox 30 %, synth 40 % (per envelope class and
  family), Exhale 15 %, Harmor vox 15 %. Augmentation: pitch shift by +-2
  rows (key adjusted), time shift, random noise floor.
- Model: SwinDenoiser (default; ~245 M parameters): KeyboardNet v3's Swin-L
  backbone and head, warm-started, as the denoiser (learning rate 5e-5;
  new parts 2e-4). Or the small FeatureUNet (``--arch unet``). Both use
  v-prediction, the cosine schedule, classifier-free guidance dropout 15 %
  and EMA weights.
- Every quarter epoch: a checkpoint (progress saved). Every half epoch, also
  a quick check with the EMA weights (about 45 s on the GTX 1050 Ti). It
  samples vox, synth and "and" alpha = 0.5 notes, scored by MorphNet-L's
  calibrated judges A/B: vocality, both detectors on, and KID in judge-A
  embeddings against real held-out notes.
- Listening at every check: 3 held-out synth notes as carriers, each with a
  generated vox sample, synth sample, "and" alpha 0.5 and alpha 0 -> 1
  rendered onto it (``demos/check_<epoch>/``, dashboard Listening with
  spectrograms). It uses its own random generator, so training is unchanged.
- Exact pause/resume (experiments/training_control.py): a ``PAUSE`` file or
  the dashboard button, then ``--resume``.

    PYTHONPATH=src python experiments/train_morph_diffusion.py      # CPU, Swin-L denoiser, 20k steps x batch 16

Defaults (the user's choice): train on the CPU (system memory, 16 threads) with the ~245 M Swin-L denoiser,
about 10.2 s per step, 20k steps in about 2.4 days. The half-epoch checks sample on the GPU when there is one
(about 1.5 minutes there). The small model on the GPU instead: ``--arch unet --device cuda --chans 48,96,192,256
--batch 8 --epochs 20`` (0.68 s per step, about 7-8 hours, fits 4 GB).

Full evaluation afterwards: experiments/eval_morph_diffusion.py.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_morph as tm  # noqa: E402  (catalog, pools)
import training_control as ctl  # noqa: E402
from wavelet2vec.morph import JudgeCNN, JudgeTCN, vox_probability  # noqa: E402
from wavelet2vec.morph_diffusion import (CLASSES, FRAMES, NULL, PADDED, PLANES, ROWS, FeatureUNet,  # noqa: E402
                                         SwinDenoiser, alpha_sigma, class_index, ddim_sample, judge_slice,
                                         one_hot, pad_features, shift_rows, v_target)
from wavelet2vec.morph_diffusion import render_features  # noqa: E402
from wavelet2vec.morph_large import load_v3_backbone, vocality  # noqa: E402
from wavelet2vec.wavelet_image import WaveletImage, decode, encode  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "morph" / "diffusion_s0"
CRITIC_CACHE = HERE / "output" / "morph" / "critic_cache.pt"
JUDGES = HERE / "output" / "morph" / "v2" / "judges_v2.pt"
V3_CHECKPOINT = HERE / "output" / "keyboard" / "v3" / "checkpoints" / "best.pt"
VAL_EXHALE = ("from_C1_till_B10_kontakt_vox_14.wav", "from_C1_till_B10_kontakt_vox_15.wav")


# ----------------------------------------------------------------------------------------- data


class FeatureData:
    """Cached feature images with class, key and train/val split."""

    def __init__(self):
        self.catalog = tm.Catalog()
        self.views = torch.load(CRITIC_CACHE, weights_only=False)["views"]          # [N, 5, 122, 120] fp16
        rms = torch.load(HERE / "output" / "morph" / "judge_cache.pt", weights_only=False)["rms"]
        self.pools = tm.Pools(self.catalog, rms)
        notes = self.catalog.notes
        self.midi = np.array([n["midi"] for n in notes])
        kind_to_class = {"vox": "vox", "synth": "synth", "processed_vox": "processed_vox", "synth_vox": "synth_vox"}
        self.label = np.array([class_index(kind_to_class[n["kind"]]) for n in notes])
        ok = rms > 1e-4
        val_vox = {i for i, n in enumerate(notes) if n["kind"] == "vox" and n["file"] in tm.VAL_VOX_FILES}
        val_synth = {i for i, n in enumerate(notes) if n["kind"] == "synth" and n["family"] in self.catalog.val_families}
        val_exhale = {i for i, n in enumerate(notes) if n["file"] in VAL_EXHALE}
        self.val = sorted((val_vox | val_synth | val_exhale) & set(np.nonzero(ok)[0]))
        held = set(self.val)
        self.train_by_class = {name: [i for i in np.nonzero(ok)[0] if self.label[i] == k and i not in held]
                               for k, name in enumerate(CLASSES)}
        self.val_by_class = {name: [i for i in self.val if self.label[i] == k] for k, name in enumerate(CLASSES)}

    def batch(self, rng, size):
        shares = {"vox": 0.3, "synth": 0.4, "processed_vox": 0.15, "synth_vox": 0.15}
        counts = {k: int(round(v * size)) for k, v in shares.items()}
        counts["synth"] += size - sum(counts.values())
        idx = []
        for name, count in counts.items():
            if name == "synth":
                idx += [i for i in self.pools.synth_sample(rng, "train", count)]
            else:
                pool = self.train_by_class[name]
                idx += [pool[k] for k in rng.integers(len(pool), size=count)]
        x = self.views[idx].float()
        midi = self.midi[idx].astype(np.float32)
        labels = self.label[idx].copy()
        for b in range(size):                                           # augmentation
            d = int(rng.integers(-2, 3))
            if d:
                x[b:b + 1] = shift_rows(x[b:b + 1], d)
                midi[b] += d
            s = int(rng.integers(-2, 5))
            if s:
                x[b] = torch.roll(x[b], s, dims=-1)
                fill = torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0])[:, None, None]
                if s > 0:
                    x[b, :, :, :s] = fill
                else:
                    x[b, :, :, s:] = fill
            if rng.random() < 0.5:
                floor = float(rng.uniform(-1.0, -0.5))
                x[b, :2] = torch.clamp(x[b, :2], min=floor)
        return x, torch.from_numpy(midi), torch.from_numpy(labels)


# ----------------------------------------------------------------------------------------- judges


def load_judges(device):
    saved = torch.load(JUDGES, map_location="cpu", weights_only=False)
    models = [JudgeCNN(width=32), JudgeTCN(width=256)]
    for m, key in zip(models, ("A", "B")):
        m.load_state_dict(saved[key])
        m.to(device).eval().requires_grad_(False)
    centres = [(cv.to(device), cs.to(device)) for cv, cs in saved["centres"]]
    return models, centres


@torch.no_grad()
def judge_features(models, centres, features):
    """Per judge: P(vox), vocality, both detectors on (> 0.5), embedding; for feature images [n, 5, 122, 120]."""
    view = judge_slice(features)
    out = {}
    for name, m, (cv, cs) in zip(("A", "B"), models, centres):
        logits, emb = m(view)
        out[name] = {"p_vox": vox_probability(logits), "vocality": vocality(emb, cv, cs),
                     "both": (torch.sigmoid(logits[:, 0]) > 0.5) & (torch.sigmoid(logits[:, 1]) > 0.5),
                     "embedding": emb}
    return out


def kid(x, y):
    """Unbiased KID (cubic polynomial kernel) between two embedding sets (torch, any device)."""
    x, y = x.double(), y.double()
    d = x.shape[1]
    kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
    m, n = len(x), len(y)
    return float((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1))
                 - 2 * kxy.mean())


@torch.no_grad()
def quick_check(model, data, models, centres, device, n=12, steps=15, seed=0, max_batch=8):
    """Sample vox, synth and "and" alpha = 0.5 notes; judge them; KID against real held-out notes."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    keys = torch.from_numpy(rng.integers(48, 80, n).astype(np.float32))
    report = {}
    real = {name: judge_features(models, centres, data.views[idx].float().to(device))["A"]["embedding"]
            for name, idx in data.val_by_class.items() if len(idx) > 4}
    real_all = torch.cat(list(real.values()))
    for name, guidance in (("vox", {"mode": "label", "label": "vox", "scale": 3.0}),
                           ("synth", {"mode": "label", "label": "synth", "scale": 3.0}),
                           ("and_0.5", {"mode": "and", "alpha": 0.5, "scale": 4.0})):
        feats = ddim_sample(model, keys, guidance, steps=steps, generator=gen, device=device, max_batch=max_batch)
        j = judge_features(models, centres, feats)
        ref = real.get(name, real_all)
        report[name] = {"p_vox_A": round(float(j["A"]["p_vox"].mean()), 4),
                        "vocality_A": round(float(j["A"]["vocality"].mean()), 4),
                        "vocality_B": round(float(j["B"]["vocality"].mean()), 4),
                        "both_present_A": round(float(j["A"]["both"].float().mean()), 4),
                        "both_present_B": round(float(j["B"]["both"].float().mean()), 4),
                        "kid_A": round(kid(j["A"]["embedding"], ref), 4)}
    return report


@torch.no_grad()
def write_check_demos(model, data, pairs, carriers, device, folder: Path, steps=25, seed=11, max_batch=8):
    """Render generated feature images onto held-out synth carriers (stable pitch) for listening."""
    folder.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    samples = {"and α=0.5": {"mode": "and", "alpha": 0.5, "scale": 4.0},
               "α automated 0→1": {"mode": "and", "alpha": torch.linspace(0.0, 1.0, FRAMES), "scale": 4.0},
               "vox sample": {"mode": "label", "label": "vox", "scale": 3.0},
               "synth sample": {"mode": "label", "label": "synth", "scale": 3.0}}
    entries = []
    for k, ((v, s, _), carrier) in enumerate(zip(pairs, carriers)):
        key = torch.tensor([float(data.catalog.notes[v]["midi"])])
        renders = {"carrier (synth note)": carrier}
        for name, guidance in samples.items():
            feats = ddim_sample(model, key, guidance, steps=steps, generator=gen, device=device, max_batch=max_batch)
            renders[name] = render_features(feats[0].float().cpu(), carrier, seed=k)
        entry = {"pair": k, "midi": int(key[0]), "alpha": 0.5, "vocal": "generated by the diffusion prior (no vocal note)",
                 "synth": f"carrier: {data.catalog.notes[s]['family']}: {data.catalog.notes[s]['file']}", "files": {}}
        for name, image in renders.items():
            audio = decode(image)[0].double().numpy()
            audio *= 0.99 / max(float(np.abs(audio).max()), 0.99)
            safe = (name.replace(" ", "_").replace("α", "alpha").replace("→", "-to-").replace("(", "")
                    .replace(")", "").replace("=", ""))
            path = folder / f"pair{k:02d}_{safe}.wav"
            sf.write(path, audio.T, tm.SPEC.sample_rate, subtype="FLOAT")
            entry["files"][name] = path.name
        entries.append(entry)
    (folder / "demos.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return entries


# ----------------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--arch", choices=("swin", "unet"), default="swin")
    parser.add_argument("--v3", type=Path, default=V3_CHECKPOINT, help="warm start of the Swin-L denoiser")
    parser.add_argument("--lr", type=float, default=2e-4, help="new parts (and the whole UNet)")
    parser.add_argument("--backbone-lr", type=float, default=5e-5, help="pretrained Swin-L backbone and head")
    parser.add_argument("--ema", type=float, default=0.9995)
    parser.add_argument("--cfg-drop", type=float, default=0.15, help="share of labels replaced by null")
    parser.add_argument("--chans", default="64,128,256,384", help="UNet widths per level (45 M parameters)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="training device")
    parser.add_argument("--check-device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for the half-epoch sampling checks (inference only)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    events = tm.EventLog(args.out / "metrics.jsonl")
    ctl.record_launch(args.out)
    ctl.clear_pause(args.out)

    def log(text):
        print(text, flush=True)

    data = FeatureData()
    check_device = torch.device(args.check_device)
    models, centres = load_judges(check_device)
    chans = tuple(int(c) for c in args.chans.split(","))
    if args.arch == "swin":
        state, v2_config = load_v3_backbone(args.v3)
        model = SwinDenoiser(state, v2_config).to(device)
        del state
        meta = {"arch": "swin", "v2_config": v2_config, "warm_start": str(args.v3)}
        pre = model.pretrained_parameters()
        groups = [{"params": [p for p in pre if p.ndim >= 2], "lr": args.backbone_lr, "weight_decay": 0.05},
                  {"params": [p for p in pre if p.ndim < 2], "lr": args.backbone_lr, "weight_decay": 0.0},
                  {"params": model.new_parameters(), "lr": args.lr, "weight_decay": 0.0}]
    else:
        model = FeatureUNet(chans=chans).to(device)
        meta = {"arch": "unet", "chans": chans}
        groups = [{"params": list(model.parameters()), "lr": args.lr, "weight_decay": 0.01}]
    grid = getattr(model, "grid", PADDED)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_params, model_params = list(ema.parameters()), list(model.parameters())
    opt = torch.optim.AdamW(groups)
    total = args.epochs * args.steps_per_epoch
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    rng = np.random.default_rng(1)
    gen = torch.Generator(device=device).manual_seed(1)
    start_epoch, start_step, best, history = 1, 0, float("inf"), []
    resume_parts, resume_seconds = None, 0.0
    state_path = ctl.latest_state(ckpt) if args.resume else None
    if state_path is not None:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        ctl.restore_random(saved, rng, gen)
        if device.type == "cuda" and "cuda_global" in saved:
            torch.cuda.set_rng_state(saved["cuda_global"])
        best, history = saved["best"], saved["history"]
        done = saved["step_in_epoch"]
        start_epoch, start_step = (saved["epoch"] + 1, 0) if done >= args.steps_per_epoch else (saved["epoch"], done)
        if start_step > 0:
            resume_parts, resume_seconds = saved.get("parts"), saved.get("epoch_seconds", 0.0)
        log(f"resumed from {state_path.name} at epoch {start_epoch} step {start_step}")
        events.write("resumed", epoch=start_epoch - 1 + start_step / args.steps_per_epoch, file=state_path.name, exact=True)
        del saved
    n_params = sum(p.numel() for p in model.parameters())
    run = {"task": "diffusion", "model": f"MorphDiffusion stage 0 ({args.arch} denoiser; feature-image prior, composition)",
           "device": str(device), "check_device": str(check_device), "threads": args.threads,
           "parameters": {"total": n_params}, "arch": args.arch, "grid": grid,
           "warm_start": meta.get("warm_start"), "chans": chans if args.arch == "unet" else None, "batch": args.batch,
           "steps_per_epoch": args.steps_per_epoch, "epochs_total": args.epochs, "lr": args.lr, "ema": args.ema,
           "cfg_drop": args.cfg_drop, "classes": CLASSES, "planes": PLANES, "rows": ROWS, "frames": FRAMES,
           "train_notes": {k: len(v) for k, v in data.train_by_class.items()},
           "val_notes": {k: len(v) for k, v in data.val_by_class.items()},
           "val_families": data.catalog.val_families}
    log(json.dumps(run, indent=1, default=str))
    events.write("run_start", epoch=start_epoch - 1 + start_step / args.steps_per_epoch, resumed=bool(args.resume),
                 size=f"{'Swin-L denoiser (v3 warm start)' if args.arch == 'swin' else 'FeatureUNet'} {n_params / 1e6:.1f}M", **run)
    quarter = max(1, args.steps_per_epoch // 4)
    demo_pairs = tm.sample_pairs(np.random.default_rng(31), data.catalog, data.pools, 3, "val", alpha=0.5)
    with torch.no_grad():
        carriers = []
        for _, s_idx, _ in demo_pairs:
            image = encode(torch.from_numpy(data.catalog.normalized(s_idx)[0][None]), tm.SPEC)
            carriers.append(image)
    mask = torch.zeros(1, 1, *grid, device=device)
    mask[..., :ROWS, :FRAMES] = 1.0

    def full_state(epoch, step, parts, started):
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), **ctl.random_states(rng, gen), "parts": parts,
                 "epoch_seconds": time.time() - started, "epoch": epoch, "step_in_epoch": step, "best": best,
                 "history": history, "meta": meta}
        if device.type == "cuda":
            state["cuda_global"] = torch.cuda.get_rng_state()
        return state

    def checkpoint(epoch, step, parts, started):
        nonlocal best
        point = epoch - 1 + step / args.steps_per_epoch
        tick = time.time()
        name = f"{point:g}_epoch.pth"
        if step % (2 * quarter) != 0:              # quarter only: save progress, no check
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "meta": meta}, ckpt / name)
            ctl.save_state(full_state(epoch, step, parts, started), ckpt / "last.pt")
            events.write("checkpoint", epoch=point, file=name, best=False, val_model=None,
                         size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
            log(f"== checkpoint {name}: progress saved ({time.time() - tick:.0f} s)")
            return
        check_model = copy.deepcopy(ema).to(check_device)
        check = quick_check(check_model, data, models, centres, check_device)
        try:
            folder = f"demos/check_{point:g}"
            demos = write_check_demos(check_model, data, demo_pairs, carriers, check_device, args.out / folder)
            events.write("demos", epoch=point, folder=folder, pairs=demos)
        except Exception as error:                  # listening is optional; never stop training for it
            log(f"   (listening renders failed: {error!r})")
        del check_model
        # lower is better: class fidelity (vox -> 1, synth -> 0), realism (KID) and both-present at alpha 0.5
        score = (abs(1 - check["vox"]["vocality_A"]) + abs(check["synth"]["vocality_A"])
                 + check["vox"]["kid_A"] + check["synth"]["kid_A"] + (1 - check["and_0.5"]["both_present_A"]))
        record = {"epoch": round(point, 3), "loss": round(float(np.mean([p["loss"] for p in parts])), 5) if parts else None,
                  "score": round(score, 4), "check": check}
        history.append(record)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "meta": meta, "metrics": record}, ckpt / name)
        is_best = score < best
        if is_best:
            best = score
            shutil.copyfile(ckpt / name, ckpt / "best.pt")
        ctl.save_state(full_state(epoch, step, parts, started), ckpt / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        events.write("val", epoch=point, diffusion=check, score=round(score, 4))
        events.write("checkpoint", epoch=point, file=name, best=is_best, val_model=round(score, 4),
                     size_mb=round((ckpt / name).stat().st_size / 1e6, 1))
        log(f"== checkpoint {name}: score {score:.3f}{' (best)' if is_best else ''}  vox->vocality "
            f"{check['vox']['vocality_A']:.2f} synth->{check['synth']['vocality_A']:.2f}  and(0.5): vocality "
            f"{check['and_0.5']['vocality_A']:.2f} both-on {check['and_0.5']['both_present_A']:.2f}  KID vox "
            f"{check['vox']['kid_A']:.3f} synth {check['synth']['kid_A']:.3f}  ({time.time() - tick:.0f} s)")

    for epoch in range(start_epoch, args.epochs + 1):
        started, parts = time.time(), []
        if epoch == start_epoch and resume_parts is not None:
            started, parts = time.time() - resume_seconds, list(resume_parts)
        first = start_step + 1 if epoch == start_epoch else 1
        for step in range(first, args.steps_per_epoch + 1):
            if ctl.pause_requested(args.out):
                ctl.save_state(full_state(epoch, step - 1, parts, started), ckpt / "paused.pt")
                ctl.clear_pause(args.out)
                events.write("paused", epoch=epoch - 1 + (step - 1) / args.steps_per_epoch, file="checkpoints/paused.pt")
                log(f"paused at epoch {epoch} after step {step - 1}: state saved to checkpoints/paused.pt")
                return
            tick = time.time()
            x, midi, labels = data.batch(rng, args.batch)
            drop = torch.from_numpy(rng.random(args.batch) < args.cfg_drop)
            labels = torch.where(drop, torch.full_like(labels, NULL), labels)
            x0 = pad_features(x, grid).to(device)
            t = torch.rand(args.batch, device=device, generator=gen)
            noise = torch.randn(x0.shape, device=device, generator=gen)
            a, s = alpha_sigma(t[:, None, None, None])
            v = model(a * x0 + s * noise, t, midi.to(device), one_hot(labels, device))
            loss = (((v - v_target(x0, noise, a, s)) ** 2) * mask).sum() / (mask.sum() * PLANES * args.batch)
            opt.zero_grad()
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, model_params, 1 - args.ema)
            parts.append({"loss": float(loss.detach())})
            if step % args.log_every == 0:
                window = parts[-args.log_every:]
                mean_loss = float(np.mean([p["loss"] for p in window]))
                mem = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None
                events.write("train", epoch=epoch - 1 + step / args.steps_per_epoch,
                             global_step=(epoch - 1) * args.steps_per_epoch + step, losses={"denoise": round(mean_loss, 5)},
                             lr=sched.get_last_lr()[0], grad_norm={"model": grad}, sec_per_step=round(time.time() - tick, 3),
                             gpu_gb=mem)
                log(f"epoch {epoch:3d} step {step:5d}/{args.steps_per_epoch}  loss {mean_loss:.5f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - tick:.2f} s/step{f'  gpu {mem} GB' if mem else ''}")
            if step % quarter == 0:
                checkpoint(epoch, step, parts, started)
    log(f"done; best score {best:.4f}. Full evaluation: experiments/eval_morph_diffusion.py")


if __name__ == "__main__":
    main()
