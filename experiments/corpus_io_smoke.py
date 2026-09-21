"""Read-only real-recording ingestion/forward smoke check; no held-out claim."""
import argparse
import json
from pathlib import Path

import torch

from wavelet2vec import (AudioRecord,AudioManifest,AudioCropConfig,AudioWindowDataset,collate_windows,
    HybridConfig,HybridTokenEncoder,MusicContextConfig,MaskedPredictionConfig,MaskedMusicModel,__version__)
from wavelet2vec.feature_cache import implementation_fingerprint


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('docs/benchmarks/corpus_io_smoke.json'))
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(17)
    records=[AudioRecord.inspect(p,group_id='unknown-library-provenance',split='train') for p in args.paths]
    manifest=AudioManifest(records)
    crop=AudioCropConfig.from_seconds(1.,sample_rate=records[0].sample_rate,channels=records[0].channels,
                                     crops_per_record=1,seed=7,mode='fixed')
    dataset=AudioWindowDataset(manifest,'train',crop)
    analysis=HybridConfig(sample_rate=crop.sample_rate,backend='compact_wavelet',n_bands=16,q=2,
                          hop_length=crop.hop_length,padding_samples=crop.sample_rate//8)
    model=MaskedMusicModel(HybridTokenEncoder(analysis,channels=crop.channels,dim=16),
        context_config=MusicContextConfig(dim=16,heads=2,local_layers=1,global_layers=1,local_tokens=16,dropout=0.),
        prediction_config=MaskedPredictionConfig(span_tokens=12)).eval()
    rows=[]
    with torch.no_grad():
        for i in range(len(dataset)):
            window=dataset[i];batch=collate_windows([window])
            result=model(batch,generator=torch.Generator().manual_seed(31))
            row={'file':Path(dataset.records[i].path).name,'source_frames':dataset.records[i].frames,
                 'sample_rate':window.sample_rate,'channels':window.waveform.shape[0],
                 'crop_start':window.sample_origin,'crop_samples':window.waveform.shape[-1],
                 'source_pcm_hash':window.source_hash,'finite_predictions':bool(torch.isfinite(result.predictions).all()),
                 'finite_loss':bool(torch.isfinite(result.loss)),'context_tokens':result.student.context.tokens.shape[1]}
            rows.append(row);print(json.dumps(row),flush=True)
    report={'scope':'Real-recording decoder/forward integration only; unknown provenance kept in one training group; no updates or held-out quality evaluation.',
            'version':__version__,'implementation':implementation_fingerprint(),'manifest_identity':manifest.identity,'windows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
