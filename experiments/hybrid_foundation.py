"""Small supervised carrier/fusion experiment; no model downloads or audio I/O.

Run ``python -m experiments.hybrid_foundation --steps 100 --seeds 0 1``.
This synthetic 8 kHz test is a plumbing/learning experiment, not a music
separation benchmark, compute-matched architecture comparison or SOTA claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from wavelet2vec import HybridAudioFrontend, HybridConfig, HybridEncoding, HybridTokenEncoder


def scenes(count: int, seed: int, n: int = 1024, sr: int = 8000):
    """Overlapping harmonic sources with different frequency distributions.

    Source 1 has a lower fundamental. Source 2 overlaps its upper harmonics.
    Random phase, amplitude and smooth envelopes vary independently. Labels
    are waveform sources, not image boxes. The task intentionally has a
    strong register cue and must not be mistaken for instrument recognition.
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / sr
    output = []
    for low, high in ((90., 240.), (300., 650.)):
        fundamental = low + (high-low)*torch.rand(count,1,generator=generator)
        phase = 2*torch.pi*torch.rand(count,3,1,generator=generator)
        harmonic = torch.arange(1,4)[None,:,None]
        signal = (torch.sin(2*torch.pi*fundamental[:,None]*harmonic*t+phase)/harmonic).sum(1)
        rate = 1 + 5*torch.rand(count,1,generator=generator)
        envelope = .55 + .45*torch.cos(2*torch.pi*rate*t)
        gain = .1 + .25*torch.rand(count,1,generator=generator)
        output.append((signal*envelope*gain)[:,None])
    return output[0]+output[1], output[0]


class _Separator(nn.Module):
    """Experimental unbounded complex-mask head on aligned hybrid tokens."""
    def __init__(self, cfg, branches, dim=24):
        super().__init__()
        self.encoder = HybridTokenEncoder(cfg, channels=1, dim=dim)
        self.branches = branches
        self.head = nn.Linear(dim, cfg.bands*2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[::2] = .5

    def forward(self, encoding):
        tokens = self.encoder(encoding, branches=self.branches).tokens
        cfg = encoding.config
        raw = self.head(tokens).transpose(1,2)
        if cfg.backend == "compact_wavelet":
            raw = raw.reshape(raw.shape[0],cfg.bands,2,raw.shape[-1])
            predicted = []
            for k, coefficient in enumerate(encoding.coefficients):
                # Same piecewise-constant raw-patch mask rule as the dense
                # wavelet head, sampled at each compact band's physical times.
                index = ((encoding.band_times_s(k)*cfg.sample_rate-encoding.sample_origin)/cfg.hop_length).floor().long()
                index = index.clamp(0,raw.shape[-1]-1)
                mask = torch.complex(raw[:,k,0,index],raw[:,k,1,index]).unsqueeze(1)
                predicted.append(coefficient*mask)
            return encoding.synthesize(tuple(predicted))
        if cfg.backend == "wavelet":
            raw = raw.repeat_interleave(cfg.hop_length, dim=-1)[..., :encoding.waveform.shape[-1]]
            pad = cfg.padding
            raw = torch.cat((raw[..., :1].expand(-1,-1,pad),raw,
                             raw[..., -1:].expand(-1,-1,pad)),dim=-1)
        else:
            # Experimental head interpolation, not a calibrated edit mask API.
            raw = F.interpolate(raw, size=encoding.coefficients.shape[-1], mode="linear", align_corners=False)
        raw = raw.reshape(raw.shape[0],1,cfg.bands,2,raw.shape[-1])
        mask = torch.complex(raw[:,:,:,0],raw[:,:,:,1])
        return encoding.synthesize(encoding.coefficients*mask)


def select(encoding: HybridEncoding, indices) -> HybridEncoding:
    coefficients = (tuple(b[indices] for b in encoding.coefficients)
                    if isinstance(encoding.coefficients,tuple) else encoding.coefficients[indices])
    return replace(encoding, waveform=encoding.waveform[indices], coefficients=coefficients)


def snr(target, estimate):
    return (10*torch.log10(target.square().sum(-1).clamp_min(1e-20)
            / (target-estimate).square().sum(-1).clamp_min(1e-20))).mean().item()


def run(steps, seeds, backends=("wavelet","stft")):
    torch.set_num_threads(1)
    mixture, target = scenes(32, 10001)
    validation_mix, validation_target = scenes(16, 20002)
    results = {
        "scope": "Synthetic 8 kHz lower-register-source extraction; CPU; no real-music/generalization claim.",
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {"train_seed":10001,"validation_seed":20002,"train_examples":32,
                 "validation_examples":16,"samples":1024,"sample_rate":8000},
        "steps":steps,"seeds":seeds,"batch_size":4,
        "half_mix_validation_snr_db":snr(validation_target,validation_mix*.5),
        "runs":[],
    }
    for backend in backends:
        cfg=HybridConfig(sample_rate=8000,backend=backend,n_bands=24,f_min=50,
                         q=4,padding_samples=128,hop_length=64,n_fft=256)
        frontend=HybridAudioFrontend(cfg)
        with torch.no_grad():
            train=frontend(mixture)
            valid=frontend(validation_mix)
            roundtrip_error=(valid.synthesize()-validation_mix).abs().max().item()
        for seed in seeds:
            for branch, context in (("waveform","complex"),("spectral","complex"),
                                    ("both","complex"),("spectral","magnitude"),("both","magnitude")):
                torch.manual_seed(seed)
                run_cfg = replace(cfg, context_mode=context)
                run_train, run_valid = replace(train, config=run_cfg), replace(valid, config=run_cfg)
                model=_Separator(run_cfg,branch)
                optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                indices=torch.Generator().manual_seed(seed+30003)
                start=time.perf_counter()
                first_loss=None
                for step in range(steps):
                    batch=torch.randint(len(mixture),(4,),generator=indices)
                    estimate=model(select(run_train,batch))
                    loss=F.mse_loss(estimate,target[batch])
                    if first_loss is None: first_loss=loss.item()
                    optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():
                    estimate=model(run_valid)
                    score=snr(validation_target,estimate)
                    # Exact-sum residual alone cannot establish source accuracy.
                    residual=validation_mix-estimate
                    consistency=(estimate+residual-validation_mix).abs().max().item()
                result={"backend":backend,"branches":branch,"context_mode":context,"seed":seed,
                        "config":run_cfg.to_dict(),"parameters_total":sum(p.numel() for p in model.parameters()),
                        "train_coefficients_bytes":train.coefficient_nbytes,
                        "roundtrip_max_error":roundtrip_error,"first_batch_mse":first_loss,
                        "last_batch_mse":loss.item(),"validation_snr_db":score,
                        "mixture_consistency_max_error":consistency,
                        "seconds":time.perf_counter()-start}
                results["runs"].append(result)
                print(f"{backend:7} {branch:8} {context:9} seed={seed}: validation SNR {score:.2f} dB",flush=True)
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    parser.add_argument("--backends",nargs="+",choices=["wavelet","compact_wavelet","stft"],default=["wavelet","stft"])
    parser.add_argument("--output",type=Path,default=Path("docs/benchmarks/hybrid_foundation.json"))
    args=parser.parse_args()
    if args.steps < 1: parser.error("steps must be positive")
    results=run(args.steps,args.seeds,args.backends)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
