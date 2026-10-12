"""Reusable percussion studies: discovery, cached audit, training, evaluation and profiling."""
import argparse
import json

import torch

from wavelet2vec.study import PercussionStudy, create_percussion_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("dataset")
    init.add_argument("--project", required=True)
    init.add_argument("--group", required=True)
    init.add_argument("--bpm", type=float, required=True)
    init.add_argument("--bars", type=int, required=True)
    init.add_argument("--train-bars", type=int, required=True)
    init.add_argument("--beats-per-bar", type=int, default=4)
    init.add_argument("--master-processing", default="unspecified")
    init.add_argument("--additive", action="store_true")
    for name in ("audit", "train", "evaluate", "run", "profile"):
        command = commands.add_parser(name)
        command.add_argument("project")
        command.add_argument("--output", required=True)
        command.add_argument("--threads", type=int, default=1)
        if name != "audit":
            command.add_argument("--variant", default="baseline")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name in ("train", "run"):
            command.add_argument("--updates", type=int, default=25)
            command.add_argument("--save-every", type=int, default=10)
        if name == "profile":
            command.add_argument("--iterations", type=int, default=5)
            command.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.command == "init":
        value = create_percussion_study(args.dataset, args.project, bpm=args.bpm, bars=args.bars,
            train_bars=args.train_bars, group_id=args.group, beats_per_bar=args.beats_per_bar,
            master_processing=args.master_processing, additive=args.additive)
        print(json.dumps({"sources": len(value["sources"]), "mixtures": len(value["mixtures"]), "project": args.project}))
        return
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    study = PercussionStudy(args.project, args.output)
    if args.command in ("audit", "run"):
        value = study.audit()
        print(json.dumps({key: value[key] for key in ("cache_hits", "cache_misses", "elapsed_seconds")}), flush=True)
    if args.command in ("train", "run"):
        print(json.dumps(study.train(args.variant, device=args.device, updates=args.updates, save_every=args.save_every)), flush=True)
    if args.command in ("evaluate", "run"):
        value = study.evaluate(args.variant, device=args.device)
        print(json.dumps({"variant": value["variant"], "update_step": value["update_step"], "cache_hit": value["cache_hit"],
                          "report": str(study.output/"report.html")}), flush=True)
    if args.command == "profile":
        value = study.profile(args.variant, device=args.device, iterations=args.iterations, warmup=args.warmup)
        print(json.dumps({key: value[key] for key in ("variant", "device", "median_seconds", "peak_torch_allocated_bytes", "trace")}), flush=True)


if __name__ == "__main__":
    main()
