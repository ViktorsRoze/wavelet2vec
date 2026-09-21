"""Matched raw/calibrated objectives on the same small synthetic split."""
import json
from pathlib import Path
import statistics

from experiments.masked_music_demo import run


if __name__=='__main__':
    runs=[]
    for seed in (17,18,19):
        for calibrated in (False,True):
            result=run(80,calibrated,seed)
            runs.append(result)
            print(f"seed={seed}, calibrated={calibrated}: raw loss {result['raw_validation_loss']:.6f}",flush=True)
    summary={}
    for calibrated in (False,True):
        selected=[r for r in runs if r['prediction']['standardize_targets']==calibrated]
        summary['calibrated' if calibrated else 'raw']={
            'mean_raw_validation_loss':statistics.mean(r['raw_validation_loss'] for r in selected),
            'mean_raw_constant_baseline_loss':statistics.mean(r['raw_training_mean_baseline_loss'] for r in selected),
            'mean_zero_visible_input_raw_loss':statistics.mean(r['zero_visible_input_raw_loss'] for r in selected),
            'runs_better_than_mean_baseline':sum(r['raw_validation_loss']<r['raw_training_mean_baseline_loss'] for r in selected),
        }
    path=Path('docs/benchmarks/target_calibration_ablation.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'scope':'Three initialization seeds; fixed small synthetic split; no semantic music claim.',
                                'summary':summary,'runs':runs},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
