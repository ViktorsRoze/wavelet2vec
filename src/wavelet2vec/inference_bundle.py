"""Portable built-in separator bundles: JSON plus checked non-pickle arrays."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from wavelet2vec.context import MusicContextConfig
from wavelet2vec.feature_cache import implementation_fingerprint
from wavelet2vec.hybrid import HybridConfig
from wavelet2vec.rendering import SourceRenderConfig, separate_file
from wavelet2vec.synthesis import HybridSeparator, SynthesisConfig

DEFAULT_WEIGHT_LIMIT = 256 * 1024**2
_MANIFEST_LIMIT = 1024**2
_ARRAY_HEADER_LIMIT = 16384


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_weight_bytes must be a positive integer")


def _spec(model):
    return {"frontend": model.encoder.frontend.config.to_dict(), "synthesis": model.config.to_dict(),
            "context": model.context.config.to_dict() if model.context is not None else None,
            "dtype": str(next(model.parameters()).dtype).split(".")[-1]}


def _prototype(spec, render, max_weight_bytes):
    _limit(max_weight_bytes)
    if spec.get("dtype") not in ("float32", "float64"):
        raise ValueError("bundle precision must be float32 or float64")
    front = HybridConfig.from_dict(spec["frontend"])
    synth = SynthesisConfig.from_dict(spec["synthesis"])
    context = MusicContextConfig.from_dict(spec["context"]) if spec["context"] is not None else None
    # Bound Python object construction as well as tensor payload. Meta tensors
    # below do not allocate parameter storage or initialize random weights.
    if (synth.dim > 16384 or synth.channels > 256 or len(synth.source_names) > 256
            or front.bands > 1048576 or front.hop_length > 1048576
            or (context is not None and context.local_layers+context.global_layers > 256)):
        raise ValueError("architecture exceeds bundle structural limits")
    if any(n % front.hop_length for n in (render.window_samples, render.overlap_samples, render.context_samples)):
        raise ValueError("bundle render geometry must align to token hop")
    with torch.device("meta"):
        prototype = HybridSeparator(front, synth, context).to(dtype=getattr(torch, spec["dtype"]))
    size = sum(value.numel()*value.element_size() for value in prototype.state_dict().values() if isinstance(value, torch.Tensor))
    if size > max_weight_bytes:
        raise ValueError(f"bundle weight payload {size} exceeds max_weight_bytes")
    return prototype, size


def _layout(model):
    return [(name, type(module)) for name, module in model.named_modules()]


def save_separator_bundle(path, model: HybridSeparator, *, render_config: SourceRenderConfig,
                          branches="both", metadata=None, overwrite=False,
                          max_weight_bytes=DEFAULT_WEIGHT_LIMIT):
    """Atomically export a configured built-in model without optimizer/RNG state.

    Export preserves module modes and RNG. Custom module types, instance-level
    forward overrides and hooks are refused because their behavior is not part
    of this architecture format. Do not mutate/train the model during export.
    """
    if type(model) is not HybridSeparator or branches not in ("both", "waveform", "spectral"):
        raise ValueError("bundle requires a built-in HybridSeparator and valid branches")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be bool")
    metadata = json.loads(_json({} if metadata is None else metadata))
    if not isinstance(metadata, dict):
        raise ValueError("bundle metadata must be a JSON object")
    spec = _spec(model)
    prototype, size = _prototype(spec, render_config, max_weight_bytes)
    if _layout(model) != _layout(prototype):
        raise ValueError("custom module layouts cannot be exported as a built-in separator")
    for module in model.modules():
        if ("forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks):
            raise ValueError("custom forward overrides/hooks cannot be represented in the bundle")
    state, expected = model.state_dict(), prototype.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("model state does not match architecture")
    for name, reference in expected.items():
        value = state[name]
        if isinstance(reference, torch.Tensor):
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError("model weight shape/precision differs from architecture")
        elif _json(value) != _json(reference):
            raise ValueError("model module metadata differs from architecture")
    from wavelet2vec import __version__
    manifest = {"format": "separator-bundle-v1", "architecture": "hybrid-separator-v1", "spec": spec,
                "render": render_config.to_dict(), "branches": branches, "metadata": metadata,
                "producer": {"library_version": __version__, "implementation": implementation_fingerprint(),
                             "torch": str(torch.__version__)}, "weight_bytes": size, "tensors": []}
    destination = Path(path)
    if not overwrite and (destination.exists() or destination.is_symlink()):
        raise FileExistsError("bundle destination exists; choose a new path or explicit overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="."+destination.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in sorted(state):
                    value = state[name]
                    if not isinstance(value, torch.Tensor):
                        continue
                    value = value.detach().cpu().contiguous()
                    if not torch.isfinite(value).all():
                        raise ValueError("cannot export nonfinite model weights")
                    buffer = io.BytesIO()
                    np.lib.format.write_array(buffer, value.numpy(), version=(1, 0), allow_pickle=False)
                    data = buffer.getvalue()
                    member = f"weights/{len(manifest['tensors']):04d}.npy"
                    archive.writestr(member, data)
                    manifest["tensors"].append({"name": name, "member": member, "shape": list(value.shape),
                                                "dtype": spec["dtype"], "sha256": hashlib.sha256(data).hexdigest()})
                manifest["bundle_id"] = _digest(manifest)
                encoded = _json(manifest).encode()
                if len(encoded) > _MANIFEST_LIMIT:
                    raise ValueError("bundle manifest exceeds size limit")
                archive.writestr("manifest.json", encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Exclusive publication without an exists/replace race. Temp and
            # destination share a filesystem; hard-link creation never clobbers.
            os.link(temporary, destination)
        return manifest["bundle_id"]
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _open(archive, max_weight_bytes):
    infos = archive.infolist()
    names = [entry.filename for entry in infos]
    if len(names) != len(set(names)) or len(names) > 4096 or "manifest.json" not in names:
        raise ValueError("bundle has duplicate, excessive or missing members")
    if any(entry.compress_type != zipfile.ZIP_STORED for entry in infos):
        raise ValueError("bundle entries must use the stored format")
    info = archive.getinfo("manifest.json")
    if info.file_size > _MANIFEST_LIMIT:
        raise ValueError("bundle manifest exceeds size limit")
    manifest = json.loads(archive.read(info))
    if manifest.get("format") != "separator-bundle-v1" or manifest.get("architecture") != "hybrid-separator-v1":
        raise ValueError("unsupported inference bundle format/architecture")
    identity = dict(manifest)
    claimed = identity.pop("bundle_id", None)
    if _digest(identity) != claimed:
        raise ValueError("bundle manifest checksum differs")
    if manifest["branches"] not in ("both", "waveform", "spectral") or not isinstance(manifest["metadata"], dict):
        raise ValueError("invalid bundle branches/metadata")
    render = SourceRenderConfig.from_dict(manifest["render"])
    prototype, size = _prototype(manifest["spec"], render, max_weight_bytes)
    tensors = {key: value for key, value in sorted(prototype.state_dict().items()) if isinstance(value, torch.Tensor)}
    if manifest["weight_bytes"] != size or len(manifest["tensors"]) != len(tensors):
        raise ValueError("bundle weight inventory differs from architecture")
    members = {"manifest.json"}
    for index, ((key, value), row) in enumerate(zip(tensors.items(), manifest["tensors"])):
        member = f"weights/{index:04d}.npy"
        if (row["name"] != key or row["member"] != member or row["shape"] != list(value.shape)
                or row["dtype"] != manifest["spec"]["dtype"]):
            raise ValueError("bundle tensor declaration differs from architecture")
        if member not in names:
            raise ValueError("bundle weight member missing")
        nbytes = value.numel()*value.element_size()
        if not nbytes <= archive.getinfo(member).file_size <= nbytes+_ARRAY_HEADER_LIMIT:
            raise ValueError("bundle weight entry has invalid size")
        members.add(member)
    if set(names) != members:
        raise ValueError("unexpected bundle members")
    return manifest, prototype, render


def _array(archive, row):
    data = archive.read(row["member"])
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("bundle weight checksum differs")
    stream = io.BytesIO(data)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("unsupported bundle array encoding")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=_ARRAY_HEADER_LIMIT)
    if list(shape) != row["shape"] or fortran or dtype != np.dtype(row["dtype"]):
        raise ValueError("array header differs from declared shape/precision")
    expected = int(np.prod(shape, dtype=object))*dtype.itemsize
    if len(data)-stream.tell() != expected:
        raise ValueError("array payload length differs")
    array = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("bundle contains nonfinite weights")
    return array


def inspect_separator_bundle(path, *, max_weight_bytes=DEFAULT_WEIGHT_LIMIT):
    """Verify all arrays and metadata without allocating model parameter storage."""
    _limit(max_weight_bytes)
    with zipfile.ZipFile(path) as archive:
        manifest, _, _ = _open(archive, max_weight_bytes)
        for row in manifest["tensors"]:
            _array(archive, row)
    return {**manifest, "weights_verified": True,
            "implementation_matches": manifest["producer"]["implementation"] == implementation_fingerprint()}


@dataclass(frozen=True)
class SeparatorBundle:
    model: HybridSeparator
    render_config: SourceRenderConfig
    branches: str
    bundle_id: str
    metadata: dict
    producer: dict
    implementation_matches: bool

    def render_file(self, source, destination, *, render_config=None, container="auto", cancel=None, progress=None):
        subtype = "FLOAT" if next(self.model.parameters()).dtype == torch.float32 else "DOUBLE"
        return separate_file(self.model, source, destination, render_config or self.render_config,
                             subtype=subtype, container=container, branches=self.branches, cancel=cancel, progress=progress,
                             provenance={"bundle_id": self.bundle_id, "bundle_producer": self.producer,
                                         "bundle_implementation_matches": self.implementation_matches})


def load_separator_bundle(path, *, device="cpu", trainable=False, require_same_implementation=False,
                          max_weight_bytes=DEFAULT_WEIGHT_LIMIT):
    """Reconstruct on an explicit device, preserving stored float32/float64 precision.

    Supported versioned architectures can load across implementation revisions;
    the mismatch is returned explicitly, or rejected with strict checking.
    Checksums detect corruption, not publisher identity. No arbitrary Python
    objects or pickle are deserialized from these bundles.
    """
    _limit(max_weight_bytes)
    if not isinstance(trainable, bool) or not isinstance(require_same_implementation, bool):
        raise ValueError("trainable and require_same_implementation must be bool")
    with zipfile.ZipFile(path) as archive:
        manifest, model, render = _open(archive, max_weight_bytes)
        matches = manifest["producer"]["implementation"] == implementation_fingerprint()
        if require_same_implementation and not matches:
            raise ValueError("bundle implementation differs")
        model.to_empty(device=device)
        state = model.state_dict()
        with torch.no_grad():
            for row in manifest["tensors"]:
                value = torch.from_numpy(_array(archive, row).copy())
                state[row["name"]].copy_(value)
    model.eval().requires_grad_(trainable)
    return SeparatorBundle(model, render, manifest["branches"], manifest["bundle_id"], manifest["metadata"],
                           manifest["producer"], matches)
