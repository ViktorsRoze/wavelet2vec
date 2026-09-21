"""Prepare verified aligned stems, then train/resume a native-rate separator."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from wavelet2vec import (
    AudioCropConfig, MultitrackRecord, MultitrackManifest, MultitrackDataset,
    HybridConfig, SynthesisConfig, HybridSeparator, MusicContextConfig,
    SeparationRunConfig, SeparationTrainingRun, __version__,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("assignments", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--updates-per-invocation", type=int, default=10)
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accumulation-steps", type=int, default=1)
    train.add_argument("--window-seconds", type=float, default=.5)
    train.add_argument("--crops", type=int, default=2)
    train.add_argument("--hop-length", type=int)
    train.add_argument("--fft-sizes", type=int, nargs="+")
    train.add_argument("--backend", choices=("wavelet", "compact_wavelet", "stft"), default="compact_wavelet")
    train.add_argument("--bands", type=int, default=32)
    train.add_argument("--dim", type=int, default=32)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--data-seed", type=int, default=101)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--mixture-weight", type=float, default=0.)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        assignments = json.loads(args.assignments.read_text(encoding="utf-8"))
        records = []
        for row in assignments["tracks"]:
            if row.get("alignment") != "sample_aligned":
                parser.error("each track must explicitly declare alignment: sample_aligned")
            root = args.assignments.parent
            records.append(MultitrackRecord.inspect(
                row["track_id"], root/row["mixture"],
                {name: None if path is None else root/path for name, path in row["sources"].items()},
                group_id=row["group_id"], split=row["split"], additive=row.get("additive", False)))
        manifest = MultitrackManifest(records, source_order=assignments["source_order"])
        manifest.save(args.output)
        print(json.dumps({"tracks": len(records), "manifest": manifest.identity}))
        return
    if min(args.steps, args.updates_per_invocation, args.save_every) <= 0:
        parser.error("step and checkpoint intervals must be positive")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error("choose an empty output directory or explicitly resume")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    manifest = MultitrackManifest.load(args.manifest)
    rate, channels = manifest.records[0].mixture.sample_rate, manifest.records[0].mixture.channels
    hop = args.hop_length if args.hop_length is not None else max(1, math.ceil(rate*128/48000))
    fft_sizes = tuple(args.fft_sizes) if args.fft_sizes is not None else tuple(2*math.ceil(rate*k/(48000*2)) for k in (256, 1024))
    crop = AudioCropConfig.from_seconds(args.window_seconds, sample_rate=rate, channels=channels,
                                        hop_length=hop, crops_per_record=args.crops, seed=args.data_seed, dtype="float32")
    training = MultitrackDataset(manifest, "train", crop)
    validation = MultitrackDataset(manifest, "validation", replace(crop, mode="fixed"))
    n_fft = max(4, 2**math.ceil(math.log2(rate*1024/48000)))
    frontend = HybridConfig(sample_rate=rate, backend=args.backend, n_bands=args.bands, q=2,
                            hop_length=hop, n_fft=n_fft, padding_samples=math.ceil(rate*.025))
    synthesis = SynthesisConfig(source_names=manifest.source_order, channels=channels, dim=args.dim)
    context = MusicContextConfig(dim=args.dim, heads=4, local_layers=1, global_layers=1, local_tokens=32)
    model = HybridSeparator(frontend, synthesis, context).to(args.device)
    config = SeparationRunConfig(total_steps=args.steps, warmup_steps=min(10, args.steps),
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, shuffle_seed=args.data_seed,
        learning_rate=args.learning_rate, fft_sizes=fft_sizes, mixture_weight=args.mixture_weight)
    run = SeparationTrainingRun(model, training, validation, config)
    if args.resume:
        run.load_checkpoint(args.resume)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {"library_version": __version__, "manifest": manifest.identity,
        "frontend": frontend.to_dict(), "synthesis": synthesis.to_dict(), "context": context.to_dict(),
        "training": config.to_dict(), "crop": crop.to_dict(),
        "initial_model_seed": args.seed if args.resume is None else None,
        "resumed_from": str(args.resume) if args.resume else None}
    configuration_path = args.output/"configuration.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text(encoding="utf-8"))
        canonical = json.loads(json.dumps(configuration))
        keys = ("library_version", "manifest", "frontend", "synthesis", "context", "training", "crop")
        if any(previous.get(key) != canonical[key] for key in keys):
            parser.error("output directory configuration differs; use a matching or new output directory")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    # A resume into a new output folder also starts with a recoverable checkpoint.
    run.save_checkpoint(args.output/"checkpoint.pt")
    stop = min(args.steps, run.update_step+args.updates_per_invocation)
    while run.update_step < stop:
        metrics = run.step()
        print(json.dumps(metrics, allow_nan=False), flush=True)
        if run.update_step % args.save_every == 0:
            run.save_checkpoint(args.output/"checkpoint.pt")
    run.save_checkpoint(args.output/"checkpoint.pt")
    report = run.evaluate()
    (args.output/f"evaluation-step-{run.update_step}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["macro_group_loss"]))


if __name__ == "__main__":
    main()
