"""Reusable, resumable studies of named solos and their mixture variants.

This workflow deliberately describes one performance. Its temporal development
segment is not an independent test set. General grouped corpora retain the
separate MultitrackManifest workflow.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import html
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote

import numpy as np
import soundfile as sf
import torch

from wavelet2vec.corpus import AudioCropConfig
from wavelet2vec.drum_analysis import analyze_drum_track
from wavelet2vec.feature_cache import implementation_fingerprint, source_fingerprint
from wavelet2vec.hybrid import HybridConfig
from wavelet2vec.context import MusicContextConfig
from wavelet2vec.multitrack import MultitrackRecord, MultitrackManifest, MultitrackDataset
from wavelet2vec.separation_training import SeparationRunConfig, SeparationTrainingRun
from wavelet2vec.synthesis import HybridSeparator, SynthesisConfig, separate_audio
from wavelet2vec.rendering import SourceRenderConfig, _container
from wavelet2vec.training import SourceExample, collate_sources, separation_loss


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write((_canonical(value)+"\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def _lock(root):
    path = root/".study.lock"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def create_percussion_study(folder, project, *, bpm, bars, train_bars, group_id,
                             beats_per_bar=4, master_processing="unspecified", additive=False):
    """Discover the explicit only_NAME / NAME+NAME.wav filename convention.

    Writes a reviewable JSON specification, including hashes and editable model
    variants. Group/provenance comes from the caller; names are not inferred
    from audio. Original files are never modified.
    """
    folder, project = Path(folder).resolve(), Path(project).resolve()
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("provide an explicit nonempty performance group")
    if not isinstance(additive, bool) or not isinstance(master_processing, str):
        raise ValueError("invalid production metadata")
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be positive")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (bars, train_bars, beats_per_bar)) or train_bars >= bars:
        raise ValueError("require positive bar counts with 0 < train_bars < bars")
    files = sorted(folder.glob("*.wav"))
    solos = {p.stem[5:]: p for p in files if p.stem.startswith("only_")}
    mixtures = [p for p in files if not p.stem.startswith("only_")]
    if not solos or not mixtures or any(not name or "+" in name for name in solos):
        raise ValueError("expected only_NAME.wav solos and NAME+NAME.wav mixtures")
    seen = set()
    for path in mixtures:
        names = path.stem.split("+")
        if len(set(names)) != len(names) or set(names)-set(solos):
            raise ValueError(f"unknown/repeated source labels in {path.name}")
        seen.update(names)
    if seen != set(solos):
        raise ValueError("every solo must occur in at least one declared mixture")
    def entry(path):
        before = path.stat()
        info = sf.info(path)
        digest = source_fingerprint(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed during discovery")
        return {"path": os.path.relpath(path, project.parent), "sha256": digest, "sample_rate": info.samplerate,
                "channels": info.channels, "frames": info.frames, "subtype": info.subtype}
    entries = {path: entry(path) for path in files}
    rate, channels = entries[files[0]]["sample_rate"], entries[files[0]]["channels"]
    body = round(bars*beats_per_bar*60/bpm*rate)
    boundary = round(train_bars*beats_per_bar*60/bpm*rate)
    if not 0 < boundary < body:
        raise ValueError("bar intervals must contain samples")
    if any(e["sample_rate"] != rate or e["channels"] != channels or e["frames"] < body for e in entries.values()):
        raise ValueError("files must share native rate/channels and contain the declared musical body")
    recipe = {"backend": "compact_wavelet", "bands": 32, "q": 2., "dim": 32,
              "window_seconds": .2, "crops_per_mixture": 24, "batch_size": 2, "steps": 300,
              "learning_rate": .001, "seed": 17, "data_seed": 101, "balance_sources": False}
    data = {"format": "percussion-study-v1", "group_id": group_id.strip(), "bpm": bpm,
            "bars": bars, "train_bars": train_bars, "quarter_beats_per_bar": beats_per_bar,
            "sample_rate": rate, "channels": channels, "body_samples": body, "train_end": boundary,
            "master_processing": master_processing, "additive": additive,
            "sources": [{"name": name, **entries[path]} for name, path in sorted(solos.items())],
            "mixtures": [{"name": p.stem, "sources": p.stem.split("+"), **entries[p]} for p in mixtures],
            "analysis": {"sensitivity": .92, "min_spacing_seconds": .07}, "max_analysis_bytes": 512*1024**2,
            "max_evaluation_bytes": 512*1024**2,
            "variants": {"baseline": recipe, "fine": {**recipe, "bands": 96, "q": 8.},
                         "balanced": {**recipe, "balance_sources": True}}}
    _write(project, data, exclusive=True)
    return data


class PercussionStudy:
    def __init__(self, project, output):
        self.project_path, self.output = Path(project).resolve(), Path(output).resolve()
        self.spec = json.loads(self.project_path.read_text(encoding="utf-8"))
        s = self.spec
        if s.get("format") != "percussion-study-v1":
            raise ValueError("unsupported study format")
        if (not isinstance(s.get("group_id"), str) or not s["group_id"].strip()
                or not isinstance(s.get("additive"), bool) or not math.isfinite(s["bpm"]) or s["bpm"] <= 0):
            raise ValueError("invalid study provenance/tempo")
        for key in ("sample_rate", "channels", "bars", "train_bars", "quarter_beats_per_bar", "body_samples", "train_end"):
            if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] <= 0:
                raise ValueError("study dimensions must be positive integers")
        if (s["train_bars"] >= s["bars"] or s["body_samples"] != round(s["bars"]*s["quarter_beats_per_bar"]*60/s["bpm"]*s["sample_rate"])
                or s["train_end"] != round(s["train_bars"]*s["quarter_beats_per_bar"]*60/s["bpm"]*s["sample_rate"])
                or not s["train_end"] < s["body_samples"]):
            raise ValueError("study intervals do not match musical geometry")
        self.order = tuple(row["name"] for row in s["sources"])
        SynthesisConfig(source_names=self.order, channels=s["channels"])
        if not s["mixtures"] or len({row["name"] for row in s["mixtures"]}) != len(s["mixtures"]):
            raise ValueError("provide unique mixture names")
        for row in s["mixtures"]:
            if not row["sources"] or len(set(row["sources"])) != len(row["sources"]) or set(row["sources"])-set(self.order):
                raise ValueError("invalid mixture source labels")
        self.entries = [*s["sources"], *s["mixtures"]]
        for key in ("max_analysis_bytes", "max_evaluation_bytes"):
            if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] <= 0:
                raise ValueError(f"{key} must be positive")
        self.identity = _identity(s)
        self.implementation = implementation_fingerprint()
        self.output.mkdir(parents=True, exist_ok=True)
        marker = self.output/"study_identity.json"
        expected = {"study": self.identity, "implementation": self.implementation}
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError("workspace project/implementation differs; choose a new output workspace")
        else:
            if any(self.output.iterdir()):
                raise ValueError("unrecognized nonempty study workspace")
            _write(marker, expected, exclusive=True)

    def path(self, row):
        return (self.project_path.parent/row["path"]).resolve()

    def verify(self):
        if _identity(self.spec) != self.identity or _identity(json.loads(self.project_path.read_text(encoding="utf-8"))) != self.identity:
            raise ValueError("study project changed; create a new study workspace")
        for row in self.entries:
            path = self.path(row)
            if source_fingerprint(path) != row["sha256"]:
                raise ValueError(f"original input changed: {path.name}")
            info = sf.info(path)
            if (info.samplerate, info.channels, info.frames, info.subtype) != (row["sample_rate"], row["channels"], row["frames"], row["subtype"]):
                raise ValueError("input metadata differs")
            if info.samplerate != self.spec["sample_rate"] or info.channels != self.spec["channels"] or info.frames < self.spec["body_samples"]:
                raise ValueError("input incompatible with study geometry")

    def _read(self, row, start, stop):
        with sf.SoundFile(self.path(row)) as source:
            source.seek(start)
            values = source.read(stop-start, dtype="float32", always_2d=True)
        if len(values) != stop-start or not np.isfinite(values).all():
            raise ValueError("invalid study audio interval")
        return torch.from_numpy(values.T.copy())

    def audit(self):
        """Reusable hash-checked per-source cache; interrupted audits retain completed entries."""
        with _lock(self.output):
            start = time.perf_counter()
            self.verify()
            verify_seconds = time.perf_counter()-start
            rows, hits, misses = [], 0, 0
            cache = self.output/"analysis_cache"
            for source in self.spec["sources"]:
                key = _identity({"audio": source["sha256"], "samples": self.spec["body_samples"],
                    "bpm": self.spec["bpm"], "beats": self.spec["quarter_beats_per_bar"],
                    "analysis": self.spec["analysis"], "implementation": self.implementation})
                path = cache/f"{key}.json"
                begin = time.perf_counter()
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved["key"] != key or saved["checksum"] != _identity(saved["result"]):
                        raise ValueError("analysis cache checksum differs")
                    detail = saved["result"]
                    hits += 1
                else:
                    estimated = self.spec["body_samples"]*self.spec["channels"]*8 + math.ceil(
                        self.spec["body_samples"]*min(self.spec["sample_rate"], 22050)/self.spec["sample_rate"])*48*24
                    if estimated > self.spec["max_analysis_bytes"]:
                        raise ValueError("full-body event analysis exceeds max_analysis_bytes; use shorter studies or an explicit larger budget")
                    # Decode in float64 to retain any original floating-point
                    # precision for analysis; training remains explicit float32.
                    with sf.SoundFile(self.path(source)) as handle:
                        values = handle.read(self.spec["body_samples"], dtype="float64", always_2d=True)
                    detail = analyze_drum_track(values.T, self.spec["sample_rate"], bpm=self.spec["bpm"], **self.spec["analysis"])
                    for event in detail["events"]:
                        event["bar_number"] = int(event["beat_position"]//self.spec["quarter_beats_per_bar"])+1
                    _write(path, {"key": key, "checksum": _identity(detail), "result": detail})
                    misses += 1
                rows.append({"source": source["name"], "seconds": time.perf_counter()-begin, "analysis": detail})
            self.verify()
            result = {"study": self.identity, "scope": "One performance; estimated solo events, not blind mixture recognition.",
                      "cache_hits": hits, "cache_misses": misses, "verify_seconds": verify_seconds,
                      "elapsed_seconds": time.perf_counter()-start, "sources": rows}
            _write(self.output/"audit.json", result)
            return result

    def _prepare(self):
        destination = self.output/"prepared"
        if destination.exists():
            manifest = MultitrackManifest.load(destination/"multitracks.json")
            seen = set()
            for record in manifest.records:
                for asset in record.assets:
                    if asset.path not in seen:
                        asset.verify(); seen.add(asset.path)
            return manifest
        rate, channels, frames = self.spec["sample_rate"], self.spec["channels"], self.spec["train_end"]
        format_name = _container(frames, channels, "FLOAT", "auto")
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=self.output) as temporary:
            staging = Path(temporary)
            paths = {}
            for index, row in enumerate(self.entries):
                path = staging/f"asset_{index:03d}.wav"
                with sf.SoundFile(self.path(row)) as source, sf.SoundFile(path, "w", samplerate=rate, channels=channels,
                                                                        format=format_name, subtype="FLOAT") as writer:
                    remaining = frames
                    while remaining:
                        audio = source.read(min(65536, remaining), dtype="float32", always_2d=True)
                        if not len(audio) or not np.isfinite(audio).all():
                            raise ValueError("invalid preparation input")
                        writer.write(audio); remaining -= len(audio)
                paths[row["path"]] = path
            solo = {row["name"]: paths[row["path"]] for row in self.spec["sources"]}
            records = [MultitrackRecord.inspect(row["name"], paths[row["path"]],
                {name: solo[name] if name in row["sources"] else None for name in self.order},
                group_id=self.spec["group_id"], split="train", additive=self.spec["additive"]) for row in self.spec["mixtures"]]
            MultitrackManifest(records, source_order=self.order).save(staging/"multitracks.json")
            self.verify()
            os.rename(staging, destination)
        return MultitrackManifest.load(destination/"multitracks.json")

    def _setup(self, variant, device):
        if variant not in self.spec["variants"]:
            raise ValueError("unknown study variant")
        recipe = self.spec["variants"][variant]
        if not isinstance(recipe["balance_sources"], bool):
            raise ValueError("balance_sources must be bool")
        rate = self.spec["sample_rate"]
        torch.manual_seed(recipe["seed"])
        hop = max(1, round(rate*256/44100))
        crop = AudioCropConfig.from_seconds(recipe["window_seconds"], sample_rate=rate, channels=self.spec["channels"],
            hop_length=hop, crops_per_record=recipe["crops_per_mixture"], seed=recipe["data_seed"])
        dataset = MultitrackDataset(self._prepare(), "train", crop)
        config = HybridConfig(sample_rate=rate, backend=recipe["backend"], n_bands=recipe["bands"], q=recipe["q"],
            hop_length=hop, padding_samples=round(rate*1024/44100), n_fft=max(4, 2**math.ceil(math.log2(rate*2048/44100))))
        context = MusicContextConfig(dim=recipe["dim"], heads=4, local_layers=1, global_layers=1, local_tokens=16, dropout=.05)
        model = HybridSeparator(config, SynthesisConfig(source_names=self.order, channels=self.spec["channels"], dim=recipe["dim"]), context).to(device)
        weights = None
        if recipe["balance_sources"]:
            energies = []
            for source in self.spec["sources"]:
                energy = 0.; count = 0
                with sf.SoundFile(self.path(source)) as handle:
                    remaining = self.spec["train_end"]
                    while remaining:
                        values = handle.read(min(65536, remaining), dtype="float64", always_2d=True)
                        if not len(values) or not np.isfinite(values).all():
                            raise ValueError("invalid source while fitting training weights")
                        energy += float(np.square(values).sum()); count += values.size; remaining -= len(values)
                energies.append(math.sqrt(energy/count))
            inverse = 1/np.maximum(energies, .002)
            weights = tuple((inverse/inverse.mean()).tolist())
        loss_ffts = tuple(2*math.ceil(rate*size/(44100*2)) for size in (512, 2048))
        training = SeparationRunConfig(total_steps=recipe["steps"], warmup_steps=min(10, recipe["steps"]),
            batch_size=recipe["batch_size"], learning_rate=recipe["learning_rate"], fft_sizes=loss_ffts,
            source_weights=weights, shuffle_seed=recipe["data_seed"])
        return SeparationTrainingRun(model, dataset, config=training), SourceRenderConfig(crop.window_samples*2,
                   (crop.window_samples//(4*hop))*hop, max(hop, round(rate*2048/44100/hop)*hop))

    def train(self, variant="baseline", *, device="cpu", updates=25, save_every=10):
        """Resume an existing variant automatically; atomic checkpoints remain authoritative."""
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (updates, save_every)):
            raise ValueError("update/checkpoint intervals must be positive")
        with _lock(self.output):
            self.verify()
            begin = time.perf_counter()
            run, render = self._setup(variant, device)
            # Generated directory IDs avoid treating arbitrary variant labels as paths.
            directory = self.output/"runs"/_identity({"variant": variant, "recipe": self.spec["variants"][variant]})[:16]
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory/"checkpoint.pt"
            if checkpoint.exists():
                run.load_checkpoint(checkpoint)
            else:
                run.save_checkpoint(checkpoint)
            setup_seconds = time.perf_counter()-begin
            history = directory/"history.jsonl"
            if history.exists():
                # Checkpoint is authoritative after interruption. Replayed or
                # unsaved updates must not masquerade as extra training steps.
                completed = {}
                for line in history.read_text(encoding="utf-8").splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # an interrupted diagnostic write, never model state
                    if isinstance(value, dict) and isinstance(value.get("update_step"), int) and 0 < value["update_step"] <= run.update_step:
                        completed[value["update_step"]] = value
                history.write_text("".join(_canonical(completed[i])+"\n" for i in sorted(completed)), encoding="utf-8")
            previous = run.update_step
            stop = min(run.config.total_steps, previous+updates)
            start = time.perf_counter()
            with (directory/"history.jsonl").open("a", encoding="utf-8") as log:
                while run.update_step < stop:
                    metrics = run.step()
                    log.write(_canonical(metrics)+"\n"); log.flush()
                    if run.update_step % save_every == 0:
                        run.save_checkpoint(checkpoint)
            run.save_checkpoint(checkpoint)
            self.verify()
            bundle_id = run.export_bundle(directory/"separator.w2v", render_config=render, overwrite=True)
            result = {"study": self.identity, "variant": variant, "previous_step": previous, "update_step": run.update_step,
                "total_steps": run.config.total_steps, "setup_seconds": setup_seconds, "update_seconds": time.perf_counter()-start,
                "bundle_id": bundle_id, "directory": str(directory.relative_to(self.output)), "complete": run.update_step == run.config.total_steps}
            _write(directory/"status.json", result)
            return result

    def evaluate(self, variant="baseline", *, device="cpu"):
        """Evaluate every mix's temporal development interval and render audition files."""
        from wavelet2vec.inference_bundle import load_separator_bundle
        with _lock(self.output):
            self.verify()
            if variant not in self.spec["variants"]:
                raise ValueError("unknown study variant")
            directory = self.output/"runs"/_identity({"variant": variant, "recipe": self.spec["variants"][variant]})[:16]
            bundle = load_separator_bundle(directory/"separator.w2v", device=device, require_same_implementation=True)
            start, end = self.spec["train_end"], self.spec["body_samples"]
            estimated = 4*self.spec["channels"]*(end-start)*(8*len(self.order)+12)
            if estimated > self.spec["max_evaluation_bytes"]:
                raise ValueError("development tensors exceed max_evaluation_bytes; choose a shorter study or explicit larger budget")
            key = _identity({"study": self.identity, "bundle": bundle.bundle_id, "device": str(device), "start": start, "end": end})
            cached = directory/f"evaluation-{key[:16]}.json"
            if cached.exists():
                value = json.loads(cached.read_text())
                if value["key"] != key or value["checksum"] != _identity(value["result"]):
                    raise ValueError("evaluation cache checksum differs")
                for row in value["result"]["rows"]:
                    for name, digest in row["audition_hashes"].items():
                        if source_fingerprint(self.output/row["audition"]/name) != digest:
                            raise ValueError("cached audition file changed")
                _write(directory/"latest_evaluation.json", value["result"])
                self._report()
                return {**value["result"], "cache_hit": True}
            rows = []
            timer = time.perf_counter()
            fft_sizes = tuple(2*math.ceil(self.spec["sample_rate"]*n/(44100*2)) for n in (512, 2048))
            for index, mixture in enumerate(self.spec["mixtures"]):
                audio = self._read(mixture, start, end)
                targets = {source["name"]: self._read(source, start, end) if source["name"] in mixture["sources"] else torch.zeros_like(audio)
                           for source in self.spec["sources"]}
                batch = collate_sources([SourceExample(audio, targets, self.spec["sample_rate"], additive=False)], source_order=self.order)
                config = bundle.render_config
                prediction = separate_audio(bundle.model, batch.mixture.to(device), window_samples=config.window_samples,
                    overlap_samples=config.overlap_samples, context_samples=config.context_samples, sample_origin=start).waveform.cpu()
                methods = {"model": prediction, "equal_split": batch.mixture[:, None].expand_as(prediction)/len(self.order),
                           "silence": torch.zeros_like(prediction)}
                scores = {}
                for method, estimate in methods.items():
                    scores[method] = {"loss": float(separation_loss(estimate, batch, fft_sizes=fft_sizes)["loss"]), "sources": {}}
                    for slot, name in enumerate(self.order):
                        target = batch.sources[0, slot].double(); value = estimate[0, slot].double()
                        power, error = float(target.square().mean()), float((target-value).square().mean())
                        scores[method]["sources"][name] = {"snr_db": 10*math.log10(power/max(error, power*1e-30)) if power else None,
                            "waveform_l1": float((target-value).abs().mean()), "output_rms": float(value.square().mean().sqrt())}
                audition = directory/"renders"/f"{index:03d}-{bundle.bundle_id[:12]}"
                if not audition.exists():
                    bundle.render_file(self.path(mixture), audition)
                receipt = json.loads((audition/"render.json").read_text())
                if receipt["source_sha256"] != mixture["sha256"] or receipt["provenance"]["bundle_id"] != bundle.bundle_id:
                    raise ValueError("existing audition provenance differs")
                hashes = {row["file"]: source_fingerprint(audition/row["file"]) for row in receipt["sources"]}
                hashes["render.json"] = source_fingerprint(audition/"render.json")
                rows.append({"mixture": mixture["name"], "present_sources": mixture["sources"], "scores": scores,
                             "audition": str(audition.relative_to(self.output)), "audition_hashes": hashes})
            self.verify()
            result = {"scope": "Temporal development segment of one shared performance; not an independent benchmark.",
                "study": self.identity, "variant": variant, "bundle_id": bundle.bundle_id, "source_order": self.order,
                "update_step": bundle.metadata["training"]["update_step"],
                "interval": [start, end], "sample_rate": self.spec["sample_rate"], "elapsed_seconds": time.perf_counter()-timer,
                "directory": str(directory.relative_to(self.output)), "rows": rows}
            _write(cached, {"key": key, "checksum": _identity(result), "result": result})
            _write(directory/"latest_evaluation.json", result)
            self._report()
        return {**result, "cache_hit": False}

    def profile(self, variant="baseline", *, device="cpu", iterations=5, warmup=2):
        """Profile a disposable training instance; never load/change saved variant weights."""
        for value, minimum in ((iterations, 1), (warmup, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("invalid profiling iteration counts")
        requested = torch.device(device)
        devices = [requested.index if requested.index is not None else torch.cuda.current_device()] if requested.type == "cuda" else []
        with _lock(self.output), torch.random.fork_rng(devices=devices):
            self.verify()
            run, _ = self._setup(variant, device)
            if run.config.total_steps < warmup+iterations+1:
                raise ValueError("profile needs warmup + iterations + one traced update within the recipe budget")
            def synchronize():
                if devices:
                    torch.cuda.synchronize(requested)
            for _ in range(warmup):
                run.step()
            synchronize()
            if devices:
                torch.cuda.reset_peak_memory_stats(requested)
            timings = []
            for _ in range(iterations):
                synchronize(); start = time.perf_counter(); run.step(); synchronize()
                timings.append(time.perf_counter()-start)
            peak = torch.cuda.max_memory_allocated(requested) if devices else None
            activities = [torch.profiler.ProfilerActivity.CPU]
            if devices:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            options = {"acc_events": True} if "acc_events" in inspect.signature(torch.profiler.profile).parameters else {}
            with torch.profiler.profile(activities=activities, record_shapes=True, **options) as profiler:
                with torch.profiler.record_function("wavelet2vec/study_update"):
                    run.step(); synchronize()
            directory = self.output/"profiles"/_identity({"variant": variant, "device": str(requested)})[:16]
            directory.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(directory/"trace.json"))
            operators = [{"operator": event.key, "calls": event.count, "self_cpu_us": event.self_cpu_time_total,
                          "self_device_us": getattr(event, "self_device_time_total", 0.)} for event in profiler.key_averages()]
            operators.sort(key=lambda row: row["self_cpu_us"], reverse=True)
            report = {"scope": "Disposable training instance; synchronized unprofiled timings plus a separate traced update.",
                "study": self.identity, "implementation": self.implementation, "variant": variant,
                "device": str(requested), "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                "gpu_name": torch.cuda.get_device_name(requested) if devices else None,
                "seconds_per_update": timings, "median_seconds": float(np.median(timings)),
                "peak_torch_allocated_bytes": peak, "operators": operators,
                "trace": str((directory/"trace.json").relative_to(self.output))}
            _write(directory/"profile.json", report)
            return report

    def _report(self):
        evaluations = [json.loads(path.read_text()) for path in sorted((self.output/"runs").glob("*/latest_evaluation.json"))]
        body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Percussion study</title>',
                '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}audio{width:230px}</style>',
                '<h1>Percussion study</h1><p>One performance, with a temporal development segment. Source labels come from the project. Full audition files include training audio; scores cover only the development interval. Silence has 0 dB SNR for active targets.</p>']
        for result in evaluations:
            body.append(f'<h2>{html.escape(result["variant"])} — {result["update_step"]} updates</h2>')
            for row in result["rows"]:
                body.append(f'<h3>{html.escape(row["mixture"])}</h3><table><tr><th>Source</th><th>Model SNR</th><th>Solo reference</th><th>Audition</th></tr>')
                for slot, name in enumerate(self.order):
                    score = row["scores"]["model"]["sources"][name]["snr_db"]
                    text = "silent target" if score is None else f"{score:.2f} dB"
                    path = quote((Path(row["audition"])/f"source_{slot:03d}.wav").as_posix())
                    reference = quote(Path(os.path.relpath(self.path(self.spec["sources"][slot]), self.output)).as_posix())
                    body.append(f'<tr><td>{html.escape(name)}</td><td>{text}</td><td><audio controls preload="none" src="{reference}"></audio></td><td><audio controls preload="none" src="{path}"></audio></td></tr>')
                body.append('</table>')
        body.append('</html>')
        path = self.output/"report.html"
        # Report is derived; only canonical numerical JSON/checkpoints govern resume.
        path.write_text("\n".join(body), encoding="utf-8")
