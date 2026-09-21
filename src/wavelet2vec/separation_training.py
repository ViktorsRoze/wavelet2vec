"""Single-process supervised separation, verified data position and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from wavelet2vec.feature_cache import implementation_fingerprint, model_fingerprint
from wavelet2vec.multitrack import MultitrackDataset, collate_multitracks
from wavelet2vec.pretraining import LengthBucketSampler
from wavelet2vec.synthesis import HybridSeparator
from wavelet2vec.training import separation_loss


@dataclass(frozen=True)
class SeparationRunConfig:
    total_steps: int = 1000
    warmup_steps: int = 10
    batch_size: int = 2
    accumulation_steps: int = 1
    shuffle_seed: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = .01
    max_grad_norm: float = 1.
    fft_sizes: tuple[int, ...] = (256, 1024)
    waveform_weight: float = 1.
    spectral_weight: float = 1.
    mixture_weight: float = 0.

    def __post_init__(self):
        for name in ("total_steps", "batch_size", "accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "shuffle_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must not exceed total_steps")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        weights = (self.waveform_weight, self.spectral_weight, self.mixture_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights[:2]) == 0:
            raise ValueError("loss weights must be nonnegative with a supervised term")
        object.__setattr__(self, "fft_sizes", tuple(self.fft_sizes))
        if self.spectral_weight and (not self.fft_sizes or any(isinstance(k, bool) or not isinstance(k, int)
                                                             or k < 4 or k % 2 for k in self.fft_sizes)):
            raise ValueError("fft_sizes must contain even integers >= 4")

    def to_dict(self):
        return json.loads(json.dumps({"version": "separation-run-v1", **asdict(self)}))

    def loss_options(self):
        return {key: getattr(self, key) for key in ("fft_sizes", "waveform_weight", "spectral_weight", "mixture_weight")}


def _check(model, dataset):
    if not isinstance(model, HybridSeparator) or not isinstance(dataset, MultitrackDataset):
        raise ValueError("use HybridSeparator and a verified MultitrackDataset")
    cfg = model.encoder.frontend.config
    parameter = next(model.parameters())
    if (dataset.config.sample_rate != cfg.sample_rate or dataset.config.channels != model.config.channels
            or dataset.config.hop_length != cfg.hop_length
            or dataset.manifest.source_order != model.config.source_names):
        raise ValueError("dataset rate, channels, hop and source order must match the model")
    if parameter.dtype not in (torch.float32, torch.float64) or str(parameter.dtype).split(".")[-1] != dataset.config.dtype:
        raise ValueError("model and native audio precision must match")


def _model_spec(model):
    parameter = next(model.parameters())
    return json.loads(json.dumps({"frontend": model.encoder.frontend.config.to_dict(),
        "synthesis": model.config.to_dict(), "context": model.context.config.to_dict() if model.context else None,
        "dtype": str(parameter.dtype), "device": str(parameter.device)}))


def _rng():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _runtime():
    return {"threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _prediction(model, batch, windows):
    # Distinct crop origins and lengths cannot share one frontend call.
    pieces = [model(batch.mixture[i:i+1, :, :n], sample_origin=window.sample_origin).waveform
              for i, (n, window) in enumerate(zip(batch.lengths.tolist(), windows))]
    return torch.cat([F.pad(piece, (0, batch.mixture.shape[-1] - piece.shape[-1])) for piece in pieces])


class SeparationTrainingRun:
    """AdamW with complete-update accumulation, native crops and strict resume.

    Every step completes one optimizer update. Loss accumulation is weighted
    by supervised sample count, not by padded length or microbatch count.
    Checkpoint identity includes sources, crop recipe, model/code and device.
    AMP, multi-worker decoding and distributed training are not implemented.
    """
    def __init__(self, model, training, validation=None, config=None):
        self.model, self.training, self.validation = model, training, validation
        self.config = config or SeparationRunConfig()
        _check(model, training)
        if training.split != "train":
            raise ValueError("training data must use the train split")
        if validation is not None:
            _check(model, validation)
            if (validation.split != "validation" or validation.config.mode != "fixed"
                    or validation.manifest.identity != training.manifest.identity):
                raise ValueError("validation must use fixed held-out crops from the same manifest")
        if self.config.mixture_weight and any(not r.additive for r in training.records):
            raise ValueError("mixture loss requires every training record to have complete additive targets")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        self.sampler = LengthBucketSampler(training.lengths, self.config.batch_size, seed=self.config.shuffle_seed)
        self.update_step = self.epoch = self.cursor = 0
        self._busy = self._failed = False
        self.optimizer.zero_grad(set_to_none=True)

    def _identity(self):
        return {"training": self.training.identity, "validation": self.validation.identity if self.validation else None}

    def _batches(self, epoch):
        self.sampler.set_epoch(epoch)
        return list(self.sampler)

    def _learning_rate(self):
        cfg = self.config
        if self.update_step < cfg.warmup_steps:
            return cfg.learning_rate*(self.update_step+1)/max(cfg.warmup_steps, 1)
        progress = (self.update_step-cfg.warmup_steps)/max(cfg.total_steps-cfg.warmup_steps-1, 1)
        return cfg.learning_rate*.5*(1+math.cos(math.pi*min(progress, 1.)))

    def step(self):
        if self._busy or self._failed:
            raise ValueError("run is busy or requires checkpoint recovery after an optimizer failure")
        if self.update_step >= self.config.total_steps:
            raise ValueError("configured optimizer budget is complete")
        cfg = self.config
        epoch, cursor = self.epoch, self.cursor
        plans = []
        supervised_count = mixture_count = 0
        for _ in range(cfg.accumulation_steps):
            batches = self._batches(epoch)
            if cursor == len(batches):
                epoch, cursor = epoch+1, 0
                batches = self._batches(epoch)
            indices = batches[cursor]
            plans.append((epoch, indices))
            for index in indices:
                ri = self.training.mixtures._entries[index][0]
                count = self.sampler.lengths[index]*self.training.config.channels
                supervised_count += count*len(self.training.records[ri].sources)
                mixture_count += count
            cursor += 1
        state = _rng()
        old_epoch = self.training.epoch
        modes = [(module, module.training) for module in self.model.modules()]
        self._busy = True
        stepped = False
        totals = {"waveform_l1": 0., "complex_spectral_l1": 0., "mixture_l1": 0.}
        ids, origins = [], []
        try:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for data_epoch, indices in plans:
                self.training.set_epoch(data_epoch)
                windows = [self.training[index] for index in indices]
                batch = collate_multitracks(windows, source_order=self.model.config.source_names)
                batch = batch.to(next(self.model.parameters()).device)
                prediction = _prediction(self.model, batch, windows)
                losses = separation_loss(prediction, batch, **cfg.loss_options())
                weight = sum(window.example.mixture.numel()*len(window.example.sources) for window in windows)/supervised_count
                mix_weight = sum(window.example.mixture.numel() for window in windows)/mixture_count
                loss = (cfg.waveform_weight*losses["waveform_l1"]*weight
                        + cfg.spectral_weight*losses["complex_spectral_l1"]*weight
                        + cfg.mixture_weight*losses["mixture_l1"]*mix_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite supervised separation loss")
                loss.backward()
                for key in totals:
                    totals[key] += float(losses[key].detach())*(mix_weight if key == "mixture_l1" else weight)
                ids.extend(window.window_id for window in windows)
                origins.extend(window.sample_origin for window in windows)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            lr = self._learning_rate()
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            stepped = True
            self.optimizer.step()
            self.update_step += 1
            self.epoch, self.cursor = epoch, cursor
        except Exception:
            self._failed = stepped  # optimizer failure may have partly changed parameters
            self.training.set_epoch(old_epoch)
            _restore_rng(state)
            for module, training in modes:
                module.training = training
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._busy = False
        return {**totals, "loss": cfg.waveform_weight*totals["waveform_l1"]
                + cfg.spectral_weight*totals["complex_spectral_l1"] + cfg.mixture_weight*totals["mixture_l1"],
                "update_step": self.update_step, "epoch": self.epoch, "next_batch": self.cursor,
                "learning_rate": lr, "gradient_norm": float(norm), "window_ids": ids, "sample_origins": origins}

    def save_checkpoint(self, path):
        if self._busy or self._failed:
            raise ValueError("save only after a successful optimizer update")
        payload = {"version": "separation-checkpoint-v1", "config": self.config.to_dict(),
                   "implementation": implementation_fingerprint(), "model_spec": _model_spec(self.model),
                   "data": self._identity(), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "update_step": self.update_step, "epoch": self.epoch, "cursor": self.cursor,
                   "rng": _rng(), "runtime": _runtime()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load_checkpoint(self, path):
        if self._busy:
            raise ValueError("cannot load during an update")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("version") != "separation-checkpoint-v1" or payload["config"] != self.config.to_dict():
            raise ValueError("checkpoint training configuration differs")
        if payload["implementation"] != implementation_fingerprint():
            raise ValueError("checkpoint implementation differs; use explicit weights migration")
        if payload["data"] != self._identity():
            raise ValueError("checkpoint data provenance differs")
        if payload["model_spec"] != _model_spec(self.model):
            raise ValueError("checkpoint model configuration, precision or device differs")
        if payload["runtime"] != _runtime():
            raise ValueError("checkpoint numerical runtime settings differ")
        step, epoch, cursor = (payload[name] for name in ("update_step", "epoch", "cursor"))
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (step, epoch, cursor))
                or step > self.config.total_steps or cursor > len(self.sampler)
                or epoch*len(self.sampler)+cursor != step*self.config.accumulation_steps):
            raise ValueError("invalid checkpoint data position")
        if payload["rng"]["cuda"] and len(payload["rng"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs for exact resume")
        current = self.model.state_dict()
        if current.keys() != payload["model"].keys():
            raise ValueError("checkpoint parameter keys differ")
        for key, value in current.items():
            saved = payload["model"][key]
            if isinstance(value, torch.Tensor):
                if not isinstance(saved, torch.Tensor) or saved.shape != value.shape or saved.dtype != value.dtype:
                    raise ValueError("checkpoint parameter shapes or precision differ")
            elif saved != value:
                raise ValueError("checkpoint module metadata differs")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_step, self.epoch, self.cursor = step, epoch, cursor
        self.training.set_epoch(epoch)
        _restore_rng(payload["rng"])
        self.optimizer.zero_grad(set_to_none=True)
        self._failed = False

    def evaluate(self):
        if self.validation is None:
            raise ValueError("no validation dataset configured")
        return evaluate_separation(self.model, self.validation, fft_sizes=self.config.fft_sizes,
                                   waveform_weight=self.config.waveform_weight, spectral_weight=self.config.spectral_weight)


@torch.no_grad()
def evaluate_separation(model, dataset, *, fft_sizes=(256, 1024), waveform_weight=1., spectral_weight=1.):
    """Fixed held-out crops, source metrics and sample/group-weighted baselines.

    SNR is gain-sensitive, not SI-SDR; silent targets have no SNR. Overlapping
    evaluation crops may count a physical sample multiple times. No listening
    or full-song quality claim follows from these diagnostics.
    """
    _check(model, dataset)
    if dataset.split not in ("validation", "test") or dataset.config.mode != "fixed":
        raise ValueError("evaluation requires fixed held-out crops")
    modes = [(module, module.training) for module in model.modules()]
    rows = []
    try:
        model.eval()
        fingerprint = model_fingerprint(model)
        for index in range(len(dataset)):
            window = dataset[index]
            batch = collate_multitracks([window], source_order=model.config.source_names).to(next(model.parameters()).device)
            prediction = model(batch.mixture, sample_origin=window.sample_origin).waveform
            methods = {"model": prediction, "equal_split": batch.mixture[:, None].expand_as(prediction)/len(model.config.source_names),
                       "mixture_copy": batch.mixture[:, None].expand_as(prediction), "silence": torch.zeros_like(prediction)}
            losses, sources = {}, {}
            for name, value in methods.items():
                scored = separation_loss(value, batch, fft_sizes=fft_sizes, waveform_weight=waveform_weight,
                                         spectral_weight=spectral_weight)["loss"]
                if not torch.isfinite(scored):
                    raise FloatingPointError("nonfinite separation evaluation")
                losses[name] = float(scored)
                sources[name] = {}
                for slot, source_name in enumerate(model.config.source_names):
                    if not batch.source_valid[0, slot]:
                        continue
                    target, estimated = batch.sources[0, slot].double(), value[0, slot].double()
                    energy = float(target.square().mean())
                    error = float((target-estimated).square().mean())
                    sources[name][source_name] = {"target_mean_square": energy, "error_mean_square": error,
                        "prediction_mean_square": float(estimated.square().mean()),
                        "waveform_l1": float((target-estimated).abs().mean()),
                        "snr_db": 10*math.log10(energy/max(error, energy*1e-30)) if energy else None}
            rows.append({"track_id": window.track_id, "group_id": window.group_id, "window_id": window.window_id,
                         "sample_origin": window.sample_origin, "samples": int(batch.lengths[0]),
                         "supervised_samples": int(batch.source_valid.sum())*batch.mixture.numel(),
                         "loss": losses, "sources": sources})
    finally:
        for module, training in modes:
            module.training = training
    methods = tuple(rows[0]["loss"])
    def average(items):
        total = sum(row["supervised_samples"] for row in items)
        return {method: sum(row["loss"][method]*row["supervised_samples"] for row in items)/total for method in methods}
    groups = {group: average([row for row in rows if row["group_id"] == group]) for group in sorted({r["group_id"] for r in rows})}
    return {"scope": "Fixed held-out multitrack crops; acoustic diagnostics, not perceptual/full-song qualification.",
            "manifest": dataset.manifest.identity, "dataset": dataset.identity, "crop": dataset.config.to_dict(),
            "model": fingerprint, "sample_rate": dataset.config.sample_rate, "source_order": model.config.source_names,
            "loss_config": {"fft_sizes": tuple(fft_sizes), "waveform_weight": waveform_weight, "spectral_weight": spectral_weight},
            "micro_loss": average(rows), "macro_group_loss": {method: sum(g[method] for g in groups.values())/len(groups) for method in methods},
            "groups": groups, "windows": rows}
