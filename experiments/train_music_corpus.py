"""Prepare an explicit grouped manifest, or train/evaluate native-rate audio.

No automatic file split, resampling, downmixing or downloads are performed.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from wavelet2vec import (AudioRecord,AudioManifest,AudioCropConfig,AudioWindowDataset,HybridConfig,
    HybridTokenEncoder,MusicContextConfig,MaskedPredictionConfig,MaskedMusicModel,
    MusicPretrainer,PretrainingConfig,CorpusPretrainingRun,CorpusRunConfig,__version__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('assignments',type=Path)
    prepare.add_argument('--output',type=Path,required=True)
    train=sub.add_parser('train')
    train.add_argument('manifest',type=Path)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--steps',type=int,default=100)
    train.add_argument('--updates-per-invocation',type=int)
    train.add_argument('--resume',type=Path)
    train.add_argument('--save-every',type=int,default=10)
    train.add_argument('--batch-size',type=int,default=2)
    train.add_argument('--window-seconds',type=float,default=2.)
    train.add_argument('--mask-seconds',type=float,default=.25)
    train.add_argument('--crops',type=int,default=2)
    train.add_argument('--bands',type=int,default=32)
    train.add_argument('--dim',type=int,default=32)
    train.add_argument('--seed',type=int,default=17)
    train.add_argument('--data-seed',type=int,default=101)
    train.add_argument('--mask-seed',type=int,default=31)
    train.add_argument('--evaluation-seed',type=int,default=303)
    train.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    train.add_argument('--calibrate',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        assignments=json.loads(args.assignments.read_text(encoding='utf-8'))
        records=[AudioRecord.inspect(args.assignments.parent/row['path'],group_id=row['group_id'],split=row['split']) for row in assignments]
        manifest=AudioManifest(records);manifest.save(args.output)
        print(f"Verified {len(records)} records; manifest {manifest.identity}")
        return
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error('choose an empty output directory or explicitly resume')
    if args.steps<1 or (args.updates_per_invocation is not None and args.updates_per_invocation<1):parser.error('step limits must be positive')
    if args.save_every<1:parser.error('save-every must be positive')
    if not math.isfinite(args.mask_seconds) or args.mask_seconds<=0:parser.error('mask duration must be positive')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    manifest=AudioManifest.load(args.manifest)
    rate,channels=manifest.records[0].sample_rate,manifest.records[0].channels
    crop=AudioCropConfig.from_seconds(args.window_seconds,sample_rate=rate,channels=channels,
                                     crops_per_record=args.crops,seed=args.data_seed)
    training=AudioWindowDataset(manifest,'train',crop)
    validation=AudioWindowDataset(manifest,'validation',replace(crop,mode='fixed'))
    analysis=HybridConfig(sample_rate=rate,backend='compact_wavelet',n_bands=args.bands,q=4,hop_length=crop.hop_length)
    context=MusicContextConfig(dim=args.dim,heads=4,local_layers=1,global_layers=1,local_tokens=32)
    prediction=MaskedPredictionConfig(span_tokens=math.ceil(args.mask_seconds*rate/crop.hop_length),standardize_targets=args.calibrate)
    model=MaskedMusicModel(HybridTokenEncoder(analysis,channels=channels,dim=args.dim),context_config=context,
                           prediction_config=prediction).to(args.device)
    trainer=MusicPretrainer(model,PretrainingConfig(total_steps=args.steps,warmup_steps=min(10,args.steps),mask_seed=args.mask_seed))
    run=CorpusPretrainingRun(trainer,training,validation,CorpusRunConfig(args.batch_size,args.data_seed,args.evaluation_seed))
    if args.resume:run.load_checkpoint(args.resume)
    args.output.mkdir(parents=True,exist_ok=True)
    configuration={'library_version':__version__,'manifest_identity':manifest.identity,
                   'analysis':analysis.to_dict(),'context':context.to_dict(),'prediction':prediction.to_dict(),
                   'train_crop':crop.to_dict(),'trainer':trainer.config.to_dict(),'model_seed':args.seed}
    (args.output/'configuration.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    run.prepare()
    if not args.resume:run.save_checkpoint(args.output/'checkpoint.pt')
    stop=args.steps if args.updates_per_invocation is None else min(args.steps,trainer.update_step+args.updates_per_invocation)
    while trainer.update_step<stop:
        metrics=run.step()
        print(json.dumps(metrics),flush=True)
        if metrics['updated'] and trainer.update_step%args.save_every==0:
            run.save_checkpoint(args.output/'checkpoint.pt')
    run.save_checkpoint(args.output/'checkpoint.pt')
    report=run.evaluate()
    (args.output/f'evaluation-step-{trainer.update_step}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['macro_group_loss'],indent=2))


if __name__=='__main__':main()
