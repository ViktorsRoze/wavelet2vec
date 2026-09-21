"""Single-process masked-music training with accumulation and resumable state."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from wavelet2vec.self_supervised import AudioBatch,MaskedMusicModel
from wavelet2vec.feature_cache import implementation_fingerprint


@dataclass(frozen=True)
class PretrainingConfig:
    learning_rate:float=3e-4
    weight_decay:float=.01
    warmup_steps:int=100
    total_steps:int=10000
    accumulation_steps:int=1
    max_grad_norm:float=1.
    mask_seed:int=0

    def __post_init__(self):
        for name in ("total_steps","accumulation_steps"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps","mask_seed"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps>self.total_steps:raise ValueError("warmup_steps cannot exceed total_steps")
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0:raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay<0:raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm<=0:raise ValueError("max_grad_norm must be finite and positive")

    def to_dict(self):return {"version":"music-pretraining-v1",**asdict(self)}


class LengthBucketSampler:
    """Seeded batch indices grouping similar clip lengths; no audio is loaded.

    Reproducibility uses seed+epoch. Dataset position/cursor is caller-owned
    and can be stored as checkpoint data_state; this is not distributed sampling.
    """
    def __init__(self,lengths,batch_size,*,bucket_multiplier=20,seed=0,drop_last=False):
        if not lengths or any(isinstance(n,bool) or not isinstance(n,int) or n<=0 for n in lengths):
            raise ValueError("lengths must be nonempty positive integers")
        for value in (batch_size,bucket_multiplier):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError("batch and bucket sizes must be positive integers")
        if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError("seed must be a nonnegative integer")
        if not isinstance(drop_last,bool):raise ValueError("drop_last must be a bool")
        self.lengths=tuple(lengths);self.batch_size=batch_size
        self.bucket_multiplier=bucket_multiplier;self.seed=seed;self.drop_last=drop_last;self.epoch=0

    def set_epoch(self,epoch):
        if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<0:raise ValueError("epoch must be nonnegative integer")
        self.epoch=epoch

    def __len__(self):
        return len(self.lengths)//self.batch_size if self.drop_last else math.ceil(len(self.lengths)/self.batch_size)

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        order=torch.randperm(len(self.lengths),generator=generator).tolist()
        bucket=self.batch_size*self.bucket_multiplier
        batches=[]
        for start in range(0,len(order),bucket):
            items=sorted(order[start:start+bucket],key=lambda i:self.lengths[i])
            for k in range(0,len(items),self.batch_size):
                batch=items[k:k+self.batch_size]
                if not self.drop_last or len(batch)==self.batch_size:batches.append(batch)
        for i in torch.randperm(len(batches),generator=generator).tolist():yield batches[i]


class MusicPretrainer:
    """AdamW reference runner, with warmup/cosine LR and finite-gradient checks.

    Checkpoints are supported only at optimizer-update boundaries. Caller must
    supply the same next batch after resuming (restore its data_state/cursor).
    AMP, distributed training and automatic long-track decoding are not included.
    """
    def __init__(self,model:MaskedMusicModel,config:PretrainingConfig|None=None):
        self.model=model;self.config=config or PretrainingConfig()
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate,weight_decay=self.config.weight_decay)
        self.mask_generator=torch.Generator().manual_seed(self.config.mask_seed)
        self.update_step=0;self.batches_seen=0;self.pending=0
        self.optimizer.zero_grad(set_to_none=True)

    def _learning_rate(self):
        cfg=self.config
        if self.update_step<cfg.warmup_steps:
            return cfg.learning_rate*(self.update_step+1)/max(cfg.warmup_steps,1)
        progress=(self.update_step-cfg.warmup_steps)/max(cfg.total_steps-cfg.warmup_steps-1,1)
        return cfg.learning_rate*.5*(1+math.cos(math.pi*min(progress,1.)))

    def step(self,batch:AudioBatch):
        if self.update_step>=self.config.total_steps:raise ValueError("configured optimizer step budget is complete")
        self.model.train()
        try:
            result=self.model(batch,generator=self.mask_generator)
            if not torch.isfinite(result.loss):raise FloatingPointError("nonfinite masked loss")
            (result.loss/self.config.accumulation_steps).backward()
            self.pending+=1;self.batches_seen+=1
            updated=self.pending==self.config.accumulation_steps
            norm=None
            if updated:
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
                lr=self._learning_rate()
                for group in self.optimizer.param_groups:group['lr']=lr
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.pending=0;self.update_step+=1
        except Exception:
            # Never carry a partly failed accumulation into the next update.
            self.optimizer.zero_grad(set_to_none=True);self.pending=0
            raise
        metrics={name:float(value.detach().cpu()) for name,value in result.metrics.items()}
        return {"loss":float(result.loss.detach().cpu()),"magnitude_loss":float(result.magnitude_loss.detach().cpu()),
                "phase_loss":float(result.phase_loss.detach().cpu()),**metrics,"updated":updated,
                "update_step":self.update_step,"batches_seen":self.batches_seen,
                "learning_rate":self.optimizer.param_groups[0]['lr'],
                "gradient_norm":float(norm) if norm is not None else None}

    def _metadata(self):
        return {key:value for key,value in self.model.state_dict().items() if not isinstance(value,torch.Tensor)}

    def save_checkpoint(self,path:str|Path,*,data_state:dict|None=None):
        if self.pending:raise ValueError("save only at an optimizer-update boundary; accumulated gradients are not serialized")
        data_state=json.loads(json.dumps(data_state or {},allow_nan=False))
        if not isinstance(data_state,dict):raise ValueError("data_state must be a JSON object")
        payload={"version":"music-pretraining-checkpoint-v1","config":self.config.to_dict(),
                 "implementation":implementation_fingerprint(),
                 "metadata":self._metadata(),"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),
                 "update_step":self.update_step,"batches_seen":self.batches_seen,
                 "mask_rng":self.mask_generator.get_state(),"torch_rng":torch.get_rng_state(),
                 "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],"data_state":data_state}
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as handle:
                temporary=Path(handle.name)
                torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()

    def load_checkpoint(self,path:str|Path,*,expected_data_state:dict|None=None):
        if self.pending:raise ValueError("finish or discard pending accumulation before loading")
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('version')!='music-pretraining-checkpoint-v1' or payload['config']!=self.config.to_dict():
            raise ValueError("checkpoint trainer configuration differs")
        if payload.get('implementation')!=implementation_fingerprint():
            raise ValueError("checkpoint implementation differs; load weights separately for an intentional migration")
        if expected_data_state is not None and any(payload.get('data_state',{}).get(k)!=v for k,v in expected_data_state.items()):
            raise ValueError("checkpoint data provenance differs")
        if payload['metadata']!=self._metadata():raise ValueError("checkpoint model configuration differs")
        if payload['cuda_rng'] and len(payload['cuda_rng'])!=torch.cuda.device_count():
            raise ValueError("exact RNG resume requires the original CUDA device count; load model weights separately for migration")
        self.model.load_state_dict(payload['model']);self.optimizer.load_state_dict(payload['optimizer'])
        self.update_step=int(payload['update_step']);self.batches_seen=int(payload['batches_seen'])
        self.mask_generator.set_state(payload['mask_rng']);torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:torch.cuda.set_rng_state_all(payload['cuda_rng'])
        self.optimizer.zero_grad(set_to_none=True)
        return payload['data_state']
