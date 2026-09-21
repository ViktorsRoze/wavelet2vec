"""Content-addressed inference-token cache, independent of editor embedding caches."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import scipy
import torch

from wavelet2vec.sliding import HybridSequence


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)


def _library_version():
    from wavelet2vec import __version__
    return __version__


def source_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1024*1024), b""):digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> str:
    """Identity of available package Python source and numerical dependencies."""
    digest = hashlib.sha256()
    digest.update(_json([str(torch.__version__),np.__version__,scipy.__version__]).encode())
    package = Path(__file__).parent
    for source in sorted(package.rglob("*.py")):
        digest.update(_json(source.relative_to(package).as_posix()).encode())
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def model_fingerprint(model: torch.nn.Module) -> str:
    """Hash implementation, weights, buffers, JSON metadata and training modes.

    Recompute after state/code changes. External custom-module code revisions
    still belong in caller processing metadata.
    """
    digest = hashlib.sha256(implementation_fingerprint().encode())
    digest.update(_json([(name, type(m).__module__+"."+type(m).__qualname__, m.training) for name,m in model.named_modules()]).encode())
    for name, value in sorted(model.state_dict().items()):
        digest.update(_json(name).encode())
        if isinstance(value,torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(_json([str(tensor.dtype),list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:digest.update(_json(value).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureCacheKey:
    source_hash: str
    model_hash: str
    sample_rate: int
    channels: int
    sample_origin: int = 0
    processing_json: str = "{}"  # runtime choices/range not captured in state_dict
    library_version: str = field(default_factory=_library_version)

    def __post_init__(self):
        for h in (self.source_hash,self.model_hash):
            if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("source/model hashes must be lowercase SHA-256 hex")
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (self.sample_rate,self.channels)):
            raise ValueError("sample_rate and channels must be positive integers")
        if isinstance(self.sample_origin,bool) or not isinstance(self.sample_origin,int):raise ValueError("sample_origin must be an integer")
        parameters = json.loads(self.processing_json)
        if not isinstance(parameters,dict):raise ValueError("processing parameters must be an object")
        object.__setattr__(self,"processing_json",_json(parameters))

    @classmethod
    def for_model(cls, source_hash, model, *, sample_rate, channels, sample_origin=0, processing=None):
        from wavelet2vec import __version__
        if any(m.training for m in model.modules()):raise ValueError("use an eval-mode model for inference caching")
        return cls(source_hash,model_fingerprint(model),sample_rate,channels,sample_origin,_json(processing or {}),__version__)

    def to_dict(self):
        return {"schema":"hybrid-feature-cache-v1","source_hash":self.source_hash,"model_hash":self.model_hash,
                "sample_rate":self.sample_rate,"channels":self.channels,"sample_origin":self.sample_origin,
                "processing":json.loads(self.processing_json),"library_version":self.library_version}

    @property
    def digest(self):return hashlib.sha256(_json(self.to_dict()).encode()).hexdigest()


class FeatureCache:
    """Safe-array cache for detached HybridSequence features; no pickle loading.

    Keys must describe all runtime choices (branches, stereo, range, pooling,
    etc.). A cache cannot infer parameters omitted by its caller. Atomic
    same-directory replacement prevents partially written final entries.
    """
    def __init__(self,directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)

    def _validate(self,key,arrays):
        tokens,starts,lengths,times,support = (arrays[n] for n in ("tokens","starts","lengths","times","support"))
        if tokens.ndim!=3 or any(v==0 for v in tokens.shape) or tokens.dtype not in (np.float32,np.float64):
            raise ValueError("tokens must be nonempty [B,T,D] float32/float64")
        count = tokens.shape[1]
        if any(a.shape!=(count,) for a in (starts,lengths,times,support)):
            raise ValueError("sequence metadata must match token count")
        if starts.dtype!=np.int64 or lengths.dtype!=np.int64 or times.dtype!=np.float64 or support.dtype!=np.bool_:
            raise ValueError("invalid sequence metadata dtypes")
        if not np.isfinite(tokens).all() or not np.isfinite(times).all() or (lengths<=0).any():
            raise ValueError("cached features and intervals must be finite and valid")
        if starts[0]!=key.sample_origin or not np.array_equal(starts[1:],starts[:-1]+lengths[:-1]):
            raise ValueError("cached sequence must be contiguous and start at the keyed origin")
        if not np.allclose(times,(starts.astype(np.float64)+lengths/2)/key.sample_rate,rtol=0,atol=1e-9):
            raise ValueError("cached time coordinates disagree with sample metadata")

    def save(self,key: FeatureCacheKey,sequence: HybridSequence):
        if sequence.tokens.requires_grad:raise ValueError("cache detached inference features, not training graphs")
        if sequence.sample_rate!=key.sample_rate:raise ValueError("cache key sample rate differs")
        arrays = {"tokens":sequence.tokens.detach().cpu().numpy(),"starts":sequence.sample_starts.cpu().numpy(),
                  "lengths":sequence.valid_samples.cpu().numpy(),"times":sequence.times_s.cpu().numpy(),
                  "support":sequence.has_requested_context.cpu().numpy()}
        self._validate(key,arrays)
        target = self.directory/(key.digest+".npz")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory,prefix="."+key.digest+".",suffix=".tmp",delete=False) as output:
                temporary=Path(output.name)
                np.savez(output,metadata=np.array(_json(key.to_dict())),**arrays)
                output.flush();os.fsync(output.fileno())
            os.replace(temporary,target)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
        return target

    def load(self,key: FeatureCacheKey,*,device="cpu") -> HybridSequence | None:
        try:
            with np.load(self.directory/(key.digest+".npz"),allow_pickle=False) as saved:
                if str(saved["metadata"].item())!=_json(key.to_dict()):return None
                arrays={name:saved[name] for name in ("tokens","starts","lengths","times","support")}
            self._validate(key,arrays)
        except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile):
            return None
        tensors={name:torch.from_numpy(value).to(device) for name,value in arrays.items()}
        return HybridSequence(tensors["tokens"],tensors["starts"],tensors["lengths"],tensors["times"],tensors["support"],key.sample_rate)
