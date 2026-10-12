"""Inspect separator bundles or render native-rate floating-point stem files."""
from __future__ import annotations

import argparse
import json

import torch

from wavelet2vec.inference_bundle import inspect_separator_bundle, load_separator_bundle
from wavelet2vec.rendering import SeparationCancelled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Verify bundle metadata and every weight array.")
    inspect.add_argument("bundle")
    inspect.add_argument("--max-weight-mib", type=int, default=256)
    render = commands.add_parser("render", help="Write stems into a new output directory.")
    render.add_argument("bundle")
    render.add_argument("input")
    render.add_argument("--output", required=True)
    render.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    render.add_argument("--container", choices=("auto", "WAV", "RF64"), default="auto")
    render.add_argument("--threads", type=int, default=1)
    render.add_argument("--max-weight-mib", type=int, default=256)
    render.add_argument("--require-same-implementation", action="store_true")
    args = parser.parse_args()
    if args.max_weight_mib <= 0:
        parser.error("max-weight-mib must be positive")
    if args.command == "inspect":
        print(json.dumps(inspect_separator_bundle(args.bundle, max_weight_bytes=args.max_weight_mib*1024**2), indent=2))
        return
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        bundle = load_separator_bundle(args.bundle, device=args.device, max_weight_bytes=args.max_weight_mib*1024**2,
                                       require_same_implementation=args.require_same_implementation)
        result = bundle.render_file(args.input, args.output, container=args.container)
    except (KeyboardInterrupt, SeparationCancelled):
        parser.exit(130, "Rendering interrupted.\n")
    print(json.dumps({"bundle_id": bundle.bundle_id, "implementation_matches": bundle.implementation_matches,
                      "files": [str(path) for path in result.files], "manifest": str(result.manifest),
                      "sample_rate": result.sample_rate, "frames": result.frames, "subtype": result.subtype}, indent=2))


if __name__ == "__main__":
    main()
