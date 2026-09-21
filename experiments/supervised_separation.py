"""Small reproducible supervised source-synthesis experiment, not a music benchmark.

Run from the repository: python -m experiments.supervised_separation --steps 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from wavelet2vec import (
    HybridConfig, HybridSeparator, MusicContextConfig, SynthesisConfig,
    SourceExample, collate_sources, separation_loss, separate_audio,
)
from wavelet2vec.feature_cache import implementation_fingerprint


def examples(count, seed, rate=8192, samples=1024):
    generator = torch.Generator().manual_seed(seed)
    time_grid = torch.arange(samples) / rate
    result = []
    for _ in range(count):
        sources = {}
        for name, low, high, pan in (("low", 110, 260, (.7, 1.)), ("high", 750, 1400, (1., .6))):
            frequency = low + (high-low)*torch.rand((), generator=generator)
            phase = 2*torch.pi*torch.rand((), generator=generator)
            envelope = .6 + .4*torch.sin(2*torch.pi*3*time_grid + phase).square()
            sound = (.12*torch.sin(2*torch.pi*frequency*time_grid+phase)
                     + .04*torch.sin(2*torch.pi*2*frequency*time_grid+phase*2)) * envelope
            sources[name] = torch.tensor(pan)[:, None] * sound
        result.append(SourceExample(sum(sources.values()), sources, rate, additive=True))
    return result


def snr(target, estimate):
    error = (target-estimate).square().sum((-1, -2)).clamp_min(1e-20)
    energy = target.square().sum((-1, -2)).clamp_min(1e-20)
    return (10*torch.log10(energy/error)).mean(0).tolist()


def run(steps, seed, backends):
    train = examples(24, 105)
    validation = collate_sources(examples(8, 206), source_order=("low", "high"))
    report = {
        "scope": "Generated 8 kHz register-separated tones with fixed source pans; pipeline/learning smoke only, not real-music quality.",
        "implementation": implementation_fingerprint(), "steps": steps, "seed": seed,
        "train_seed": 105, "validation_seed": 206, "source_order": ["low", "high"],
        "training_examples": 24, "validation_examples": 8, "sample_rate": 8192,
        "samples": 1024, "torch": str(torch.__version__), "runs": [],
    }
    for backend in backends:
        torch.manual_seed(seed)
        front = HybridConfig(sample_rate=8192, backend=backend, n_bands=24, f_min=40, q=2,
                             hop_length=32, padding_samples=128, n_fft=256)
        synth = SynthesisConfig(source_names=("low", "high"), dim=24)
        context = MusicContextConfig(dim=24, heads=2, local_layers=1, global_layers=1,
                                     local_tokens=8, dropout=0)
        model = HybridSeparator(front, synth, context)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        generator = torch.Generator().manual_seed(seed + 307)
        baseline = validation.mixture[:, None].expand_as(validation.sources) / 2
        baseline_loss = separation_loss(baseline, validation, fft_sizes=(64, 256))["loss"].item()
        start = time.perf_counter()
        first = None
        for _ in range(steps):
            indices = torch.randint(len(train), (4,), generator=generator).tolist()
            batch = collate_sources([train[i] for i in indices], source_order=synth.source_names)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.mixture, lengths=batch.lengths).waveform
            losses = separation_loss(prediction, batch, fft_sizes=(64, 256), mixture_weight=.1)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("nonfinite training loss")
            losses["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if first is None:
                first = losses["loss"].item()
        model.eval()
        with torch.no_grad():
            predicted = model(validation.mixture).waveform
            evaluation_loss = separation_loss(predicted, validation, fft_sizes=(64, 256))["loss"].item()
            chunked = separate_audio(model, validation.mixture[:1], window_samples=512,
                                     overlap_samples=128, context_samples=128).waveform
        report["runs"].append({
            "backend": backend, "frontend": front.to_dict(), "synthesis": synth.to_dict(),
            "context": context.to_dict(), "parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.perf_counter()-start, "first_training_loss": first,
            "last_training_loss": losses["loss"].item(), "last_gradient_norm": float(norm),
            "validation_loss": evaluation_loss, "equal_split_validation_loss": baseline_loss,
            "validation_source_snr_db": snr(validation.sources, predicted),
            "equal_split_source_snr_db": snr(validation.sources, baseline),
            "chunked_finite": bool(torch.isfinite(chunked).all()),
            "chunk_vs_full_mean_absolute_difference": (chunked-predicted[:1]).abs().mean().item(),
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backends", nargs="+", choices=("wavelet", "compact_wavelet", "stft"),
                        default=["wavelet", "compact_wavelet", "stft"])
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/source_synthesis_smoke.json"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    report = run(args.steps, args.seed, args.backends)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for result in report["runs"]:
        print(result["backend"], "validation loss", result["validation_loss"],
              "equal split", result["equal_split_validation_loss"])
    print("Report:", args.output)


if __name__ == "__main__":
    main()
