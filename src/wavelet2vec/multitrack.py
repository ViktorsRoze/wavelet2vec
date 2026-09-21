"""Verified, sample-aligned multitrack manifests and native-rate crops."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import soundfile as sf
import torch

from wavelet2vec.corpus import AudioCropConfig, AudioManifest, AudioRecord, AudioWindowDataset, _digest, _stat
from wavelet2vec.training import SourceExample, collate_sources


def _names(names):
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError("source_order must be a sequence of names")
    names = tuple(names)
    if (not names or any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("source_order requires unique nonempty names without surrounding whitespace")
    return names


@dataclass(frozen=True)
class MultitrackRecord:
    track_id: str
    mixture: AudioRecord
    sources: Mapping[str, AudioRecord | None]  # null = known silence; omitted = unknown
    additive: bool = False
    alignment: str = "sample_aligned"

    def __post_init__(self):
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be nonempty")
        object.__setattr__(self, "track_id", self.track_id.strip())
        if not isinstance(self.mixture, AudioRecord) or not isinstance(self.sources, Mapping) or not self.sources:
            raise ValueError("provide mixture metadata and at least one source annotation")
        _names(tuple(self.sources))
        if not isinstance(self.additive, bool) or self.alignment != "sample_aligned":
            raise ValueError("additive must be bool and alignment must be sample_aligned")
        mix = self.mixture
        for source in self.sources.values():
            if source is None:
                continue
            if not isinstance(source, AudioRecord):
                raise ValueError("source annotations must be AudioRecord or None for known silence")
            if (source.sample_rate, source.channels, source.frames) != (mix.sample_rate, mix.channels, mix.frames):
                raise ValueError("source rate, channels and length must match mixture; no implicit alignment")
            if (source.group_id, source.split) != (mix.group_id, mix.split):
                raise ValueError("all assets must share their mixture's group and split")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    @classmethod
    def inspect(cls, track_id, mixture, sources, *, group_id, split, additive=False):
        if not isinstance(sources, Mapping):
            raise ValueError("sources must map names to paths or None")
        record = cls(track_id, AudioRecord.inspect(mixture, group_id=group_id, split=split), {
            name: None if path is None else AudioRecord.inspect(path, group_id=group_id, split=split)
            for name, path in sources.items()
        }, additive)
        if additive:
            _verify_additive(record)
        return record

    @property
    def assets(self):
        return (self.mixture,) + tuple(value for value in self.sources.values() if value is not None)

    def to_dict(self, *, paths=True):
        def asset(value):
            if value is None:
                return None
            row = asdict(value)
            if not paths:
                row.pop("path")
            return row
        return {"track_id": self.track_id, "mixture": asset(self.mixture),
                "sources": {name: asset(value) for name, value in self.sources.items()},
                "additive": self.additive, "alignment": self.alignment}


def _verify_additive(record):
    """Verify all samples in bounded float64 blocks; no peak normalization."""
    snapshots = {a.path: _stat(a.path) for a in record.assets}
    with ExitStack() as stack:
        mix = stack.enter_context(sf.SoundFile(record.mixture.path))
        sources = [stack.enter_context(sf.SoundFile(value.path))
                   for value in record.sources.values() if value is not None]
        while True:
            audio = mix.read(65536, dtype="float64", always_2d=True)
            if not len(audio):
                break
            summed = np.zeros_like(audio)
            for source in sources:
                value = source.read(len(audio), dtype="float64", always_2d=True)
                if value.shape != audio.shape:
                    raise ValueError("source length changed during additive verification")
                summed += value
            if not np.allclose(summed, audio, rtol=1e-5, atol=1e-6):
                raise ValueError("additive source targets do not sum to the mixture")
    if any(_stat(path) != snapshot for path, snapshot in snapshots.items()):
        raise ValueError("asset changed during additive verification")


class MultitrackManifest:
    """Source reuse must stay within one declared group, including across splits.

    Hashes identify exact PCM reuse, not alternate edits or musical provenance.
    Sample-aligned is a caller assertion; equal lengths alone do not prove it.
    """
    def __init__(self, records, *, source_order):
        self.records = tuple(records)
        self.source_order = _names(source_order)
        if not self.records or not all(isinstance(r, MultitrackRecord) for r in self.records):
            raise ValueError("provide MultitrackRecord entries")
        ids, groups, assets, paths = set(), {}, {}, {}
        # Includes duplicate mixtures and declared groups crossing splits.
        AudioManifest([record.mixture for record in self.records])
        for record in self.records:
            if record.track_id in ids:
                raise ValueError("duplicate track_id")
            ids.add(record.track_id)
            if set(record.sources) - set(self.source_order):
                raise ValueError("annotation outside source_order")
            if record.additive and set(record.sources) != set(self.source_order):
                raise ValueError("additive tracks require all source slots, including explicit silence")
            group, split = record.mixture.group_id, record.mixture.split
            if group in groups and groups[group] != split:
                raise ValueError("source group crosses data splits")
            groups[group] = split
            for asset in record.assets:
                if asset.path in paths and paths[asset.path] != asset:
                    raise ValueError("conflicting metadata for the same asset path")
                paths[asset.path] = asset
                if asset.pcm_hash in assets and assets[asset.pcm_hash] != group:
                    raise ValueError("exact audio asset reused across source groups")
                assets[asset.pcm_hash] = group

    @property
    def identity(self):
        return _digest({"schema": "multitrack-v1", "source_order": self.source_order,
                        "records": [record.to_dict(paths=False) for record in self.records]})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in self.records:
            row = record.to_dict()
            for value in (row["mixture"], *row["sources"].values()):
                if value is not None:
                    value["path"] = os.path.relpath(value["path"], path.parent.resolve())
            rows.append(row)
        path.write_text(json.dumps({"schema": "multitrack-v1", "source_order": self.source_order,
                                    "records": rows}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "multitrack-v1":
            raise ValueError("unsupported multitrack manifest")
        def asset(row):
            if row is None:
                return None
            row = dict(row)
            row["path"] = str((path.parent / row["path"]).resolve())
            return AudioRecord(**row)
        records = [MultitrackRecord(row["track_id"], asset(row["mixture"]),
                                   {name: asset(value) for name, value in row["sources"].items()},
                                   row["additive"], row["alignment"]) for row in data["records"]]
        return cls(records, source_order=data["source_order"])


@dataclass(frozen=True)
class MultitrackWindow:
    example: SourceExample
    track_id: str
    group_id: str
    split: str
    window_id: str
    sample_origin: int


def collate_multitracks(windows, *, source_order):
    return collate_sources([window.example for window in windows], source_order=source_order)


class MultitrackDataset:
    """Read the same bounded native interval from mixture and every target.

    Verification is mandatory. All selected assets are hashed at construction;
    additive tracks are checked in full. Source mutation checks then run on
    every crop. None denotes known silence; missing annotations remain absent.
    """
    def __init__(self, manifest: MultitrackManifest, split: str, config: AudioCropConfig):
        self.manifest, self.split, self.config = manifest, split, config
        self.records = tuple(r for r in manifest.records if r.mixture.split == split)
        self.mixtures = AudioWindowDataset(AudioManifest([r.mixture for r in manifest.records]), split, config)
        self._snapshots = {record.path: snapshot for record, snapshot
                           in zip(self.mixtures.records, self.mixtures._snapshots)}
        for record in self.records:
            for asset in record.assets:
                if asset.path not in self._snapshots:
                    self._snapshots[asset.path] = asset.verify()
            if record.additive:
                _verify_additive(record)

    @property
    def identity(self):
        return _digest({"manifest": self.manifest.identity, "split": self.split, "crop": self.config.to_dict()})

    @property
    def lengths(self):
        return self.mixtures.lengths

    @property
    def epoch(self):
        return self.mixtures.epoch

    def set_epoch(self, epoch):
        self.mixtures.set_epoch(epoch)

    def __len__(self):
        return len(self.mixtures)

    def plan(self, index):
        ri, start, n, identity = self.mixtures.plan(index)
        identity = _digest([identity, self.records[ri].to_dict(paths=False)])
        return ri, start, n, identity

    def __getitem__(self, index):
        ri, start, n, identity = self.plan(index)
        record = self.records[ri]
        mixture = self.mixtures[index].waveform
        sources = {}
        for name, asset in record.sources.items():
            if asset is None:
                sources[name] = torch.zeros_like(mixture)
                continue
            if _stat(asset.path) != self._snapshots[asset.path]:
                raise ValueError("source asset changed after dataset construction")
            with sf.SoundFile(asset.path) as source:
                if (source.samplerate, source.channels, len(source)) != (asset.sample_rate, asset.channels, asset.frames):
                    raise ValueError("source asset header changed")
                source.seek(start)
                value = source.read(n, dtype=self.config.dtype, always_2d=True)
            if len(value) != n or not np.isfinite(value).all() or _stat(asset.path) != self._snapshots[asset.path]:
                raise ValueError("invalid source crop or asset changed while reading")
            sources[name] = torch.from_numpy(value.T.copy())
        # Catch changes to earlier assets while later sources were being read.
        if any(_stat(a.path) != self._snapshots[a.path] for a in record.assets):
            raise ValueError("asset changed while loading aligned crop")
        example = SourceExample(mixture, sources, self.config.sample_rate, record.additive)
        return MultitrackWindow(example, record.track_id, record.mixture.group_id,
                                record.mixture.split, identity, start)
