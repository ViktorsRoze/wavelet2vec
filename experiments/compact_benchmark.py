"""Reproducible dense/compact carrier footprint and autograd benchmark.

Measures actual coefficient/window storage and CUDA allocator peaks, not
total process RAM, all driver allocations or large-model training memory.
"""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import wavelet2vec
from wavelet2vec import HybridConfig, HybridAudioFrontend


def synchronize(device):
    if device == "cuda": torch.cuda.synchronize()


def benchmark():
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(20260907)
    x=torch.randn(1,2,48000,generator=generator)*.1
    cfg=HybridConfig(sample_rate=48000,n_bands=64,q=8,padding_samples=12000)
    output={"scope":"One-second stereo synthetic noise; full frontend, synthesis, context and input backward.",
            "python":platform.python_version(),"torch":torch.__version__,"wavelet2vec":wavelet2vec.__version__,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed":20260907,"shape":list(x.shape),"runs":[]}
    for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
        for backend,factor,mode in [("wavelet",1,"fast"),("stft",1,"fast"),("compact_wavelet",1,"fast"),
                                    ("compact_wavelet",1,"power2"),("compact_wavelet",2,"power2")]:
            config=replace(cfg,backend=backend,compact_oversampling=factor,compact_fft_mode=mode)
            frontend=HybridAudioFrontend(config)
            for trial in range(3):
                gc.collect()
                if device=="cuda":torch.cuda.empty_cache()
                audio=x.to(device).clone().requires_grad_()
                synchronize(device)
                if device=="cuda":torch.cuda.reset_peak_memory_stats()
                baseline=torch.cuda.memory_allocated() if device=="cuda" else None
                start=time.perf_counter()
                encoding=frontend(audio)
                reconstructed=encoding.synthesize()
                features,_,_=encoding.context_features()
                (features.square().mean()+reconstructed.square().mean()).backward()
                synchronize(device)
                elapsed=(time.perf_counter()-start)*1000
                window_tuple=encoding.windows if isinstance(encoding.windows,tuple) else (encoding.windows,)
                item={"backend":backend,"oversampling":factor,"fft_mode":mode,"device":device,"trial":trial,
                      "milliseconds":elapsed,"config":config.to_dict(),
                      "coefficient_bytes":encoding.coefficient_nbytes,
                      "window_bytes":sum(w.numel()*w.element_size() for w in window_tuple),
                      "group_table_bytes":sum(g.bin_indices.numel()*g.bin_indices.element_size()
                                               +g.windows.numel()*g.windows.element_size()
                                               for g in encoding.compact_groups or ()),
                      "max_roundtrip_error":float((reconstructed-audio).abs().max().detach()),
                      "snr_db":float((10*torch.log10(audio.square().mean()/(reconstructed-audio).square().mean())).detach()),
                      "finite_gradient":bool(torch.isfinite(audio.grad).all()),
                      "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None,
                      "cuda_baseline_allocated_bytes":baseline}
                output['runs'].append(item)
                print(f"{device} {backend} {mode} x{factor} trial {trial}: {elapsed:.1f} ms; "
                      f"coefficients {encoding.coefficient_nbytes/2**20:.2f} MiB",flush=True)
                del audio,encoding,reconstructed,features,window_tuple
    # Also verify an existing short real recording without writing any audio.
    example=Path(__file__).resolve().parents[2]/'soundgen/example_audio/kicks/AA viktor rose A1 2.wav'
    if example.exists():
        import soundfile as sf
        audio,sr=sf.read(example,always_2d=True,dtype='float64')
        audio=torch.from_numpy(audio[:sr].T.copy())[None]
        real_cfg=replace(cfg,sample_rate=sr,backend='compact_wavelet',compact_fft_mode='fast')
        encoding=HybridAudioFrontend(real_cfg)(audio)
        restored=encoding.synthesize()
        output['real_audio']={"file":example.name,"sha256":hashlib.sha256(example.read_bytes()).hexdigest(),
                              "sample_rate":sr,"shape":list(audio.shape),"config":real_cfg.to_dict(),
                              "coefficient_bytes":encoding.coefficient_nbytes,
                              "snr_db":float(10*torch.log10(audio.square().mean()/(restored-audio).square().mean())),
                              "max_roundtrip_error":float((restored-audio).abs().max())}
    return output


if __name__=='__main__':
    output=benchmark()
    path=Path('docs/benchmarks/compact_memory.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
