"""Generated 192 kHz file-render integration; no dataset or learned quality claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from wavelet2vec import HybridConfig, HybridSeparator, SynthesisConfig, SourceRenderConfig, separate_file
from wavelet2vec.feature_cache import implementation_fingerprint


def run(durations, rate):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = HybridSeparator(HybridConfig(sample_rate=rate, backend="compact_wavelet", n_bands=16,
                            q=1, f_min=40, hop_length=128, padding_samples=128), SynthesisConfig(dim=8))
    config = SourceRenderConfig(8192, 2048, 512)
    report = {"scope": "Generated signal, initialized equal-split model, float32 file-render correctness and bounded windows only.",
              "implementation": implementation_fingerprint(), "sample_rate": rate, "dtype": "float32",
              "render": config.to_dict(), "estimated_render_buffer_bytes": config.buffer_bytes(
                  batch_size=1, sources=2, channels=2, dtype=torch.float32), "runs": []}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, seconds in enumerate(durations):
            frames = round(rate*seconds)
            source = root/f"input-{index}.wav"
            with sf.SoundFile(source, mode="w", samplerate=rate, channels=2, subtype="FLOAT") as writer:
                for start in range(0, frames, 65536):
                    samples = np.arange(start, min(frames, start+65536), dtype=np.float64)
                    signal = .12*np.sin(2*np.pi*440*samples/rate)+.04*np.sin(2*np.pi*7000*samples/rate)
                    signal[(samples.astype(np.int64) % 7919) == 0] = 1.5
                    writer.write(np.stack((signal, -.7*signal), -1).astype(np.float32))
            geometry = {"calls": 0, "maximum_model_samples": 0, "maximum_emitted_samples": 0}
            def observe(module, args):
                geometry["calls"] += 1
                geometry["maximum_model_samples"] = max(geometry["maximum_model_samples"], args[0].shape[-1])
            previous = 0
            def progress(done, total):
                nonlocal previous
                geometry["maximum_emitted_samples"] = max(geometry["maximum_emitted_samples"], done-previous)
                previous = done
            hook = model.register_forward_pre_hook(observe)
            start_time = time.perf_counter()
            try:
                result = separate_file(model, source, root/f"stems-{index}", config, progress=progress)
            finally:
                hook.remove()
            elapsed = time.perf_counter()-start_time
            peak = error = energy = 0.
            with sf.SoundFile(source) as original, sf.SoundFile(result.files[0]) as first, sf.SoundFile(result.files[1]) as second:
                while True:
                    x = original.read(65536, dtype="float64", always_2d=True)
                    if not len(x):
                        break
                    reconstructed = first.read(len(x), dtype="float64", always_2d=True)+second.read(len(x), dtype="float64", always_2d=True)
                    difference = x-reconstructed
                    peak = max(peak, float(np.abs(difference).max()))
                    error += float(np.square(difference).sum())
                    energy += float(np.square(x).sum())
            assert previous == frames and peak < 2e-6
            assert geometry["maximum_model_samples"] <= config.window_samples+2*config.context_samples
            report["runs"].append({"seconds": seconds, "frames": frames, "elapsed_seconds": elapsed,
                **geometry, "summed_source_max_error": peak, "summed_source_snr_db": 10*np.log10(energy/max(error, energy*1e-30)),
                "source_bytes": source.stat().st_size, "output_bytes": sum(path.stat().st_size for path in result.files)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, nargs="+", default=[.25, 10.])
    parser.add_argument("--sample-rate", type=int, default=192000)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/bounded_render_smoke.json"))
    args = parser.parse_args()
    if args.sample_rate <= 0 or any(not np.isfinite(v) or v <= 0 or round(v*args.sample_rate) < 1 for v in args.seconds):
        parser.error("provide positive sample rate and finite durations containing samples")
    report = run(args.seconds, args.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
