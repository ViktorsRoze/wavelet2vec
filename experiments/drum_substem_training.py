"""Within-performance drum separation experiment; never an independent benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from wavelet2vec import (
    AudioCropConfig, MultitrackRecord, MultitrackManifest, MultitrackDataset,
    HybridConfig, HybridAudioFrontend, SynthesisConfig, MusicContextConfig, HybridSeparator,
    SeparationRunConfig, SeparationTrainingRun, SourceExample, collate_sources, separation_loss,
    SourceRenderConfig, separate_audio, separate_file,
    load_separator_bundle,
)
from wavelet2vec.feature_cache import source_fingerprint, implementation_fingerprint


def measures(target, estimate):
    target, estimate = target.double(), estimate.double()
    energy = float(target.square().sum())
    error = float((target-estimate).square().sum())
    return {"snr_db": 10*np.log10(energy/max(error, energy*1e-30)) if energy else None,
            "waveform_l1": float((target-estimate).abs().mean()),
            "output_rms": float(estimate.square().mean().sqrt()),
            "target_rms": float(target.square().mean().sqrt())}


@torch.no_grad()
def oracle_mask(frontend, mixture, sources, origin):
    """Privileged reference-informed ratio masks; not a learned model or upper bound."""
    encoded = frontend(mixture, sample_origin=origin)
    b, s, c, n = sources.shape
    targets = frontend(sources.reshape(b*s, c, n), sample_origin=origin)
    if isinstance(encoded.coefficients, tuple):
        predicted = []
        for mix, target in zip(encoded.coefficients, targets.coefficients):
            power = target.reshape(b, s, c, -1).abs().square()
            weights = power/power.sum(1, keepdim=True).clamp_min(1e-20)
            predicted.append(mix[:, None]*weights)
        return encoded.synthesize(tuple(predicted))
    power = targets.coefficients.reshape(b, s, c, *targets.coefficients.shape[-2:]).abs().square()
    return encoded.synthesize(encoded.coefficients[:, None]*power/power.sum(1, keepdim=True).clamp_min(1e-20))


def run(root, output, steps=300, device="cpu", seed=17, evaluate_existing=False, balance_sources=False, bands=32, q=2.):
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not evaluate_existing:
        raise ValueError("choose a new experiment output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    solo_paths = {path.stem[5:]: path for path in root.glob("only_*.wav")}
    mixes = sorted((path for path in root.glob("*.wav") if not path.stem.startswith("only_")), key=lambda path:len(path.stem.split("+")))
    order = tuple(mixes[-1].stem.split("+"))
    if set(order) != set(solo_paths):
        raise ValueError("solo names and largest mixture must agree")
    rates = {sf.info(path).samplerate for path in [*solo_paths.values(), *mixes]}
    if len(rates) != 1:
        raise ValueError("native rates differ")
    rate = rates.pop()
    end = round(7.5*rate)
    boundary = round(5.625*rate)
    originals = {path.name: source_fingerprint(path) for path in [*solo_paths.values(), *mixes]}
    aligned = output/"prepared_train"
    aligned.mkdir(exist_ok=evaluate_existing)
    for path in [*solo_paths.values(), *mixes]:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[0] < end or audio.shape[1] != 2:
            raise ValueError("expected stereo recordings containing the full four bars")
        if not evaluate_existing:
            sf.write(aligned/path.name, audio[:boundary], sr, subtype="FLOAT")
    records = []
    for path in mixes:
        present = set(path.stem.split("+"))
        records.append(MultitrackRecord.inspect(path.stem, aligned/path.name,
            {name: aligned/solo_paths[name].name if name in present else None for name in order},
            group_id="single-drum-performance-bpm128", split="train", additive=False))
    manifest = MultitrackManifest(records, source_order=order)
    manifest.save(output/"training_manifest.json")
    crop = AudioCropConfig(rate, 2, 8192, hop_length=256, crops_per_record=24, seed=101)
    dataset = MultitrackDataset(manifest, "train", crop)
    front = HybridConfig(sample_rate=rate, backend="compact_wavelet", n_bands=bands, q=q,
                          hop_length=256, padding_samples=1024)
    synthesis = SynthesisConfig(source_names=order, channels=2, dim=32)
    context = MusicContextConfig(dim=32, heads=4, local_layers=1, global_layers=1, local_tokens=16, dropout=.05)
    model = HybridSeparator(front, synthesis, context).to(device)
    source_rms = np.array([np.sqrt(np.mean(sf.read(aligned/solo_paths[name].name, dtype="float64", always_2d=True)[0]**2))
                           for name in order])
    source_weights = 1/np.maximum(source_rms, .002)
    source_weights /= source_weights.mean()
    config = SeparationRunConfig(total_steps=steps, warmup_steps=min(10, steps), batch_size=2,
        learning_rate=.001, fft_sizes=(512, 2048), mixture_weight=0, shuffle_seed=101,
        source_weights=tuple(source_weights.tolist()) if balance_sources else None)
    training = SeparationTrainingRun(model, dataset, config=config)
    render = SourceRenderConfig(16384, 4096, 2048)
    history = []
    if evaluate_existing:
        bundle = load_separator_bundle(output/"separator.w2v", device=device)
        model, render, identity = bundle.model, bundle.render_config, bundle.bundle_id
        stored_config = dict(bundle.metadata["training"]["config"])
        stored_config.pop("version")
        config = SeparationRunConfig(**stored_config)
        steps = bundle.metadata["training"]["update_step"]
        front = model.encoder.frontend.config
        elapsed = None
        if (output/"history.jsonl").exists():
            history = [json.loads(line) for line in (output/"history.jsonl").read_text().splitlines()]
    else:
        training.save_checkpoint(output/"checkpoint.pt")
        start = time.perf_counter()
        with (output/"history.jsonl").open("w", encoding="utf-8") as log:
            for step in range(steps):
                metrics = training.step()
                if step == 0 or (step+1) % 25 == 0:
                    row = {key: value for key, value in metrics.items() if key not in ("window_ids", "sample_origins")}
                    history.append(row)
                    log.write(json.dumps(row)+"\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                    training.save_checkpoint(output/"checkpoint.pt")
            training.save_checkpoint(output/"checkpoint.pt")
        identity = training.export_bundle(output/"separator.w2v", render_config=render)
        elapsed = time.perf_counter()-start
    model.eval()
    report = {"scope": "Train on bars 1-3; diagnose bar 4 of the same performance. Repeated timbres/patterns and shared stems prevent independent generalization claims.",
        "dataset": str(root.resolve()), "original_hashes": originals, "native_sample_rate": rate,
        "original_subtype": "PCM_16", "prepared_subtype": "FLOAT", "source_order": order,
        "train_sample_interval": [0, boundary], "evaluation_sample_interval": [boundary, end],
        "omitted_tail_after_sample": end, "additive": False, "master_limiter_reported": True,
        "seed": seed, "updates": steps, "elapsed_training_seconds": elapsed, "history": history,
        "training_config": config.to_dict(), "source_rms_training_only": dict(zip(order, source_rms.tolist())),
        "frontend": front.to_dict(), "parameters": sum(p.numel() for p in model.parameters()),
        "implementation": implementation_fingerprint(), "bundle_id": identity, "results": []}
    solos = {name: torch.from_numpy(sf.read(path, dtype="float32", always_2d=True)[0][boundary:end].T.copy()) for name, path in solo_paths.items()}
    oracle_front = HybridAudioFrontend(front)
    for path in mixes:
        present = set(path.stem.split("+"))
        audio = torch.from_numpy(sf.read(path, dtype="float32", always_2d=True)[0][boundary:end].T.copy())
        target = {name: solos[name] if name in present else torch.zeros_like(audio) for name in order}
        batch = collate_sources([SourceExample(audio, target, rate, additive=False)], source_order=order)
        with torch.no_grad():
            estimated = separate_audio(model, batch.mixture.to(device), window_samples=render.window_samples,
                overlap_samples=render.overlap_samples, context_samples=render.context_samples, sample_origin=boundary).waveform.cpu()
            oracle = oracle_mask(oracle_front, batch.mixture, batch.sources, boundary)
        alternatives = {"model": estimated, "equal_split": batch.mixture[:, None].expand_as(estimated)/len(order),
                        "silence": torch.zeros_like(estimated), "reference_ratio_mask": oracle}
        row = {"mixture": path.name, "present_sources": sorted(present), "methods": {}}
        for method, value in alternatives.items():
            row["methods"][method] = {"loss": float(separation_loss(value, batch, fft_sizes=(512, 2048))["loss"]),
                "sources": {name: measures(batch.sources[0, i], value[0, i]) for i, name in enumerate(order)}}
        report["results"].append(row)
        print("Evaluated", path.name, row["methods"]["model"]["loss"], flush=True)
    full_mix = mixes[-1]
    if not (output/"full_mix_stems").exists():
        separate_file(model, full_mix, output/"full_mix_stems", render, provenance={"bundle_id": identity,
            "scope": "same-performance trained example, master-processed mix, solo-reference targets"})
    receipt = json.loads((output/"full_mix_stems"/"render.json").read_text())
    if receipt["provenance"]["bundle_id"] != identity or receipt["source_sha256"] != originals[full_mix.name]:
        raise ValueError("existing render provenance differs")
    report["rendered_files"] = [str(Path("full_mix_stems")/row["file"]) for row in receipt["sources"]]
    for path in [*solo_paths.values(), *mixes]:
        if source_fingerprint(path) != originals[path.name]:
            raise RuntimeError("original source changed during experiment")
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("drum_substem_dataset_BPM128_4bars"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/drum_dataset_2026-09-22/separation"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--balance-sources", action="store_true")
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--q", type=float, default=2.)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    run(args.dataset, args.output, args.steps, args.device, evaluate_existing=args.evaluate_existing,
        balance_sources=args.balance_sources, bands=args.bands, q=args.q)
