"""Small reproducible masked-acoustic learning check; no downloads/pretrained weights.

The data are synthetic harmonic textures, not a semantic music benchmark.
Evaluate against a constant target learned ONLY from training features.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from wavelet2vec import __version__
from wavelet2vec.feature_cache import implementation_fingerprint

from wavelet2vec import (HybridConfig,HybridTokenEncoder,MusicContextConfig,MaskedPredictionConfig,
    MaskedMusicModel,PretrainingConfig,MusicPretrainer,collate_audio,sample_span_mask)


def clips(count,seed):
    generator=torch.Generator().manual_seed(seed)
    audio=[]
    for _ in range(count):
        n=int(torch.randint(384,769,(1,),generator=generator))
        t=torch.arange(n)/8192
        f=float(torch.randint(90,700,(1,),generator=generator))
        phase=2*torch.pi*torch.rand(3,generator=generator)
        gain=.04+.3*float(torch.rand(1,generator=generator))
        harmonics=sum(torch.cos(2*torch.pi*f*k*t+phase[k-1])/k for k in range(1,4))
        envelope=.6+.4*torch.cos(2*torch.pi*(1+5*float(torch.rand(1,generator=generator)))*t)
        left=gain*harmonics*envelope
        right=left*(.4+.6*float(torch.rand(1,generator=generator)))
        audio.append(torch.stack((left,right)))
    return audio


def objective(predictions,targets,mask,samples,cfg):
    error=F.smooth_l1_loss(predictions,targets,reduction='none').reshape(*mask.shape,-1,3)
    weight=mask*samples
    value=cfg.magnitude_weight*error[...,2].mean(-1)+cfg.phase_weight*error[...,:2].mean((-1,-2))
    return float((value*weight).sum()/weight.sum())


def run(steps,standardize=False,model_seed=17):
    torch.set_num_threads(1);torch.manual_seed(model_seed)
    analysis=HybridConfig(sample_rate=8192,backend='compact_wavelet',n_bands=12,f_min=64,
                          f_max=3072,q=2,hop_length=16,padding_samples=64)
    context=MusicContextConfig(dim=16,heads=2,local_layers=1,global_layers=1,local_tokens=8,dropout=0.)
    prediction=MaskedPredictionConfig(mask_fraction=.5,span_tokens=4,standardize_targets=standardize)
    model=MaskedMusicModel(HybridTokenEncoder(analysis,channels=2,dim=16),context_config=context,
                           prediction_config=prediction)
    trainer=MusicPretrainer(model,PretrainingConfig(learning_rate=.003,warmup_steps=min(5,steps),
                                                   total_steps=steps,mask_seed=31))
    train=clips(24,101);validation=clips(8,202)
    calibration=None
    if standardize:
        calibration=model.fit_target_statistics(
            (collate_audio([clip],sample_rate=8192) for clip in train),
            calibration_id='synthetic-training-clips-seed101-v1')
    validation_batch=collate_audio(validation,sample_rate=8192)
    valid,_,_=model._layout(validation_batch)
    validation_mask=sample_span_mask(valid,prediction,generator=torch.Generator().manual_seed(303))
    with torch.no_grad():
        mean=torch.zeros(model.target_dim)
        count=0
        for clip in train:
            values,_,lengths=model.teacher(clip[None]).context_features()
            mean+=(values[0]*lengths[:,None]).sum(0)
            count+=int(lengths.sum())
        mean/=count
        model.eval()
        before=model(validation_batch,mask=validation_mask)
        baseline=objective(model.to_target_space(mean).expand_as(before.targets),before.targets,
                           before.mask,before.student.valid_samples,prediction)
        initial=float(before.loss)
    generator=torch.Generator().manual_seed(404)
    history=[];start=time.perf_counter()
    for step in range(steps):
        indices=torch.randint(len(train),(4,),generator=generator).tolist()
        metrics=trainer.step(collate_audio([train[i] for i in indices],sample_rate=8192))
        if step%max(steps//5,1)==0 or step==steps-1:
            history.append(metrics)
            print(f"step {step+1}: loss={metrics['loss']:.6f}, prediction_std={metrics['prediction_std']:.4f}",flush=True)
    elapsed=time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        after=model(validation_batch,mask=validation_mask)
        raw_target=model.from_target_space(after.targets)
        raw_loss=objective(model.from_target_space(after.predictions),raw_target,after.mask,after.student.valid_samples,prediction)
        raw_baseline=objective(mean.expand_as(raw_target),raw_target,after.mask,after.student.valid_samples,prediction)
        zero_student=model.student_forward(replace(validation_batch,waveform=torch.zeros_like(validation_batch.waveform)),validation_mask)
        zero_loss=objective(zero_student.predictions,after.targets,after.mask,after.student.valid_samples,prediction)
        zero_raw_loss=objective(model.from_target_space(zero_student.predictions),raw_target,
                                after.mask,after.student.valid_samples,prediction)
    return {"scope":"Synthetic harmonic textures; masked fixed acoustic targets; no real-music semantic claim.",
            "python":platform.python_version(),"torch":str(torch.__version__),"wavelet2vec":__version__,
            "implementation":implementation_fingerprint(),"pretraining":trainer.config.to_dict(),"steps":steps,
            "train_clips":24,"validation_clips":8,"sample_rate":8192,
            "seeds":{"model":model_seed,"train_data":101,"validation_data":202,"validation_mask":303,"batches":404,"train_masks":31},
            "analysis":analysis.to_dict(),"context":context.to_dict(),"prediction":prediction.to_dict(),
            "initial_validation_loss":initial,"final_validation_loss":float(after.loss),
            "training_mean_target_baseline_loss":baseline,
            "raw_validation_loss":raw_loss,"raw_training_mean_baseline_loss":raw_baseline,
            "zero_visible_input_validation_loss":zero_loss,"zero_visible_input_raw_loss":zero_raw_loss,
            "target_calibration":calibration,
            "final_validation_metrics":{k:float(v) for k,v in after.metrics.items()},
            "seconds":elapsed,"training_history":history}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=80)
    parser.add_argument('--standardize',action='store_true')
    parser.add_argument('--model-seed',type=int,default=17)
    parser.add_argument('--output',type=Path,default=Path('docs/benchmarks/masked_music_demo.json'))
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    result=run(args.steps,args.standardize,args.model_seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('initial_validation_loss','final_validation_loss','training_mean_target_baseline_loss','seconds')},indent=2))
