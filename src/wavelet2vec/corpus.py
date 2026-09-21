"""Verified audio manifests and reproducible, native-rate cropped datasets."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from wavelet2vec.feature_cache import source_fingerprint
from wavelet2vec.self_supervised import collate_audio


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _stat(path):
    value=Path(path).stat()
    return value.st_size,value.st_mtime_ns


@dataclass(frozen=True)
class AudioRecord:
    path:str
    group_id:str
    split:str
    sample_rate:int
    channels:int
    frames:int
    file_hash:str
    pcm_hash:str

    def __post_init__(self):
        if not isinstance(self.path,str) or not self.path or not isinstance(self.group_id,str) or not self.group_id.strip():
            raise ValueError("records need a path and explicit nonempty source group")
        object.__setattr__(self,'path',str(Path(self.path).resolve()))
        object.__setattr__(self,'group_id',self.group_id.strip())
        if self.split not in ('train','validation','test'):raise ValueError("split must be train, validation or test")
        for value in (self.sample_rate,self.channels,self.frames):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError("audio metadata must be positive integers")
        for value in (self.file_hash,self.pcm_hash):
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError("audio identities must be SHA-256 hex")

    @classmethod
    def inspect(cls,path,*,group_id,split):
        path=Path(path).resolve()
        before=_stat(path)
        file_hash=source_fingerprint(path)
        with sf.SoundFile(path) as source:
            rate,channels,frames=source.samplerate,source.channels,len(source)
            digest=hashlib.sha256(json.dumps([rate,channels,frames]).encode())
            while True:
                values=source.read(65536,dtype='float64',always_2d=True)
                if len(values)==0:break
                if not np.isfinite(values).all():raise ValueError(f"nonfinite audio: {path}")
                values[values==0]=0.  # canonicalize signed zero across containers
                digest.update(values.astype('<f8',copy=False).tobytes())
        if _stat(path)!=before:raise ValueError("source changed while being inspected")
        return cls(str(path),group_id,split,rate,channels,frames,file_hash,digest.hexdigest())

    def verify(self):
        before=_stat(self.path)
        actual=AudioRecord.inspect(self.path,group_id=self.group_id,split=self.split)
        if actual!=self:raise ValueError(f"source no longer matches manifest: {self.path}")
        if _stat(self.path)!=before:raise ValueError("source changed during verification")
        return before


class AudioManifest:
    """Split/group integrity plus exact decoded-PCM duplicate detection.

    Groups are declared by the caller, not inferred from filenames. This cannot
    detect alternate renders, edits, resampling or other near duplicates.
    """
    def __init__(self,records):
        self.records=tuple(records)
        if not self.records or not all(isinstance(r,AudioRecord) for r in self.records):raise ValueError("provide AudioRecord entries")
        groups={};pcm=set();paths=set()
        for record in self.records:
            if record.group_id in groups and groups[record.group_id]!=record.split:
                raise ValueError("source group crosses data splits")
            groups[record.group_id]=record.split
            if record.pcm_hash in pcm:raise ValueError("duplicate decoded audio in manifest")
            if record.path in paths:raise ValueError("duplicate source path")
            pcm.add(record.pcm_hash);paths.add(record.path)

    @property
    def identity(self):
        # Paths are intentionally excluded so verified relocation is portable.
        return _digest({'schema':'audio-manifest-v1','records':[
            {k:v for k,v in asdict(r).items() if k!='path'} for r in self.records]})

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for record in self.records:
            row=asdict(record)
            row['path']=os.path.relpath(record.path,path.parent.resolve())
            rows.append(row)
        path.write_text(json.dumps({'schema':'audio-manifest-v1','records':rows},indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema')!='audio-manifest-v1':raise ValueError("unsupported manifest version")
        rows=[]
        for raw in data['records']:
            raw=dict(raw);raw['path']=str((path.parent/raw['path']).resolve())
            rows.append(AudioRecord(**raw))
        return cls(rows)


@dataclass(frozen=True)
class AudioCropConfig:
    sample_rate:int
    channels:int
    window_samples:int
    hop_length:int=256
    crops_per_record:int=2
    seed:int=0
    mode:str='random'
    dtype:str='float32'

    def __post_init__(self):
        for value in (self.sample_rate,self.channels,self.window_samples,self.hop_length,self.crops_per_record):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError("crop dimensions must be positive integers")
        if self.window_samples%self.hop_length:raise ValueError("window_samples must align to hop_length")
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:raise ValueError("seed must be a nonnegative integer")
        if self.mode not in ('random','fixed') or self.dtype not in ('float32','float64'):raise ValueError("invalid crop mode or dtype")

    @classmethod
    def from_seconds(cls,seconds,*,sample_rate,channels,hop_length=256,**kwargs):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError("duration must be finite and positive")
        return cls(sample_rate,channels,math.ceil(seconds*sample_rate/hop_length)*hop_length,hop_length,**kwargs)

    def to_dict(self):return {'schema':'audio-crop-v1',**asdict(self)}


@dataclass(frozen=True)
class AudioWindow:
    waveform:torch.Tensor  # C,N, native-rate channels preserved
    sample_rate:int
    sample_origin:int
    group_id:str
    split:str
    window_id:str
    source_hash:str


def collate_windows(windows):
    if not windows or any(w.sample_rate!=windows[0].sample_rate for w in windows):raise ValueError("windows must share sample rate")
    return collate_audio([w.waveform for w in windows],sample_rate=windows[0].sample_rate,
                         sample_origins=[w.sample_origin for w in windows])


class AudioWindowDataset:
    """Bounded seek/read crops, stateless per-record RNG and explicit epoch.

    Hashes are verified once at construction by default; size/mtime and header
    checks detect ordinary source changes during a run. This is not protection
    against adversarial filesystem changes. Fixed crops deduplicate positions.
    """
    def __init__(self,manifest:AudioManifest,split:str,config:AudioCropConfig,*,verify=True):
        if not isinstance(verify,bool):raise ValueError("verify must be a bool")
        self.verified=verify
        self.manifest=manifest;self.split=split;self.config=config;self.epoch=0
        self.records=tuple(r for r in manifest.records if r.split==split)
        if not self.records:raise ValueError(f"manifest contains no {split} records")
        self._snapshots=[];self._entries=[]
        for index,record in enumerate(self.records):
            if record.sample_rate!=config.sample_rate or record.channels!=config.channels:
                raise ValueError("corpus rate/channels differ from crop config; no implicit resampling or downmixing")
            self._snapshots.append(record.verify() if verify else _stat(record.path))
            n=min(config.window_samples,record.frames)
            maximum=(record.frames-n)//config.hop_length
            if config.mode=='fixed':
                steps=sorted(set(round(i*maximum/max(config.crops_per_record-1,1)) for i in range(config.crops_per_record)))
                self._entries.extend((index,slot,step*config.hop_length) for slot,step in enumerate(steps))
            else:self._entries.extend((index,slot,None) for slot in range(config.crops_per_record))

    @property
    def identity(self):return _digest({'manifest':self.manifest.identity,'split':self.split,'crop':self.config.to_dict()})

    @property
    def lengths(self):return [min(self.config.window_samples,self.records[index].frames) for index,_,_ in self._entries]

    def __len__(self):return len(self._entries)

    def set_epoch(self,epoch):
        if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<0:raise ValueError("epoch must be nonnegative integer")
        self.epoch=epoch

    def plan(self,index):
        record_index,slot,fixed=self._entries[index]
        record=self.records[record_index];cfg=self.config
        n=min(cfg.window_samples,record.frames)
        if fixed is None:
            seed=int(_digest([record.pcm_hash,cfg.seed,self.epoch,slot])[:16],16)
            generator=torch.Generator().manual_seed(seed)
            maximum=(record.frames-n)//cfg.hop_length
            start=int(torch.randint(maximum+1,(1,),generator=generator))*cfg.hop_length
        else:start=fixed
        # Stable across manifest reorderings and changes to the training split.
        identity=_digest([record.pcm_hash,start,n,cfg.dtype,cfg.hop_length])
        return record_index,start,n,identity

    def __getitem__(self,index):
        ri,start,n,identity=self.plan(index)
        record=self.records[ri];cfg=self.config
        if _stat(record.path)!=self._snapshots[ri]:raise ValueError("source changed after dataset construction")
        with sf.SoundFile(record.path) as source:
            if (source.samplerate,source.channels,len(source))!=(record.sample_rate,record.channels,record.frames):
                raise ValueError("source header no longer matches manifest")
            source.seek(start);audio=source.read(n,dtype=cfg.dtype,always_2d=True)
        if len(audio)!=n or not np.isfinite(audio).all() or _stat(record.path)!=self._snapshots[ri]:
            raise ValueError("invalid crop or source changed while reading")
        return AudioWindow(torch.from_numpy(audio.T.copy()),cfg.sample_rate,start,record.group_id,
                           record.split,identity,record.pcm_hash)
