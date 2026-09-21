"""PyTorch dataset wrappers (roadmap D-6): training data in three lines.

Everything is generated on the fly and **deterministic per index** — an
"epoch" over N indices is reproducible, datasets need no disk, and two
workers never collide. Three wrappers cover the three training stories:

- :class:`SceneDetectionDataset` — synthetic note/drum scenes (V-2) as
  ``(image [3,H,W], target)`` with COCO-style boxes/labels/pitches: food
  for DETR-family event detection (V-4).
- :class:`LadderAxisDataset` — supervision ladders (G-1) as
  ``(image, measured_axis_value)``: food for the conditioned imagination
  axes (G-2/G-8/G-9) — the label is *measured* by the same feature the
  model will condition on.
- :class:`ContrastiveViewsDataset` — two :func:`~wavelet2vec.augment.
  random_augment` views of the same clip: food for SimCLR/BYOL-style
  self-supervision over your own sample library.

Use :func:`detection_collate` for the detection loader (targets are
variable-length); the others batch with the default collate.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    HAVE_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object
    HAVE_TORCH = False

from wavelet2vec.scenes import random_scene
from wavelet2vec.vision import coeff_image, event_to_bbox

SCENE_CLASSES = ("sine", "saw", "pluck", "kick", "snare", "hat")


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise RuntimeError("wavelet2vec.data needs PyTorch (pip install torch).")


class SceneDetectionDataset(Dataset):
    """Unlimited exactly-labeled detection data from synthetic scenes.

    ``__getitem__(i)`` → ``(image [3, bands, frames] float32,
    {"boxes" [K,4] xywh, "labels" [K], "pitches" [K], "velocities" [K]})``.
    Class ids index :data:`SCENE_CLASSES` (+1; 0 is background).
    """

    def __init__(self, size: int = 1024, *, sample_rate: int = 22050,
                 seconds: float = 2.0, seed: int = 0,
                 n_bands: int = 128, n_frames: int = 128) -> None:
        _require_torch()
        self.size = int(size)
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.seed = seed
        self.n_bands = n_bands
        self.n_frames = n_frames
        self._class_id = {name: i + 1 for i, name in enumerate(SCENE_CLASSES)}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        audio, events = random_scene(
            self.sample_rate, seconds=self.seconds,
            seed=int(np.random.SeedSequence([self.seed, index]).generate_state(1)[0]),
        )
        image, meta = coeff_image(audio, self.sample_rate,
                                  n_bands=self.n_bands, n_frames=self.n_frames)
        boxes = [event_to_bbox(event, meta) for event in events]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(
                [self._class_id.get(event.label, 0) for event in events],
                dtype=torch.int64),
            "pitches": torch.tensor(
                [-1 if event.pitch is None else event.pitch for event in events],
                dtype=torch.int64),
            "velocities": torch.tensor(
                [0 if event.velocity is None else event.velocity for event in events],
                dtype=torch.int64),
        }
        return torch.from_numpy(image), target


def detection_collate(batch):
    """Stacks images; keeps the variable-length targets as a list."""
    _require_torch()
    images = torch.stack([image for image, _target in batch])
    targets = [target for _image, target in batch]
    return images, targets


class LadderAxisDataset(Dataset):
    """Conditioned-axis supervision from the G-1 ladders.

    ``axis``: ``"brightness"`` (filter-cutoff ladder), ``"harmonicity"``
    (noise-mix ladder) or ``"attack_ms"`` (attack ladder). Each item renders
    one ladder step of a random pluck and returns ``(image, label)`` where
    the label is **measured** by :func:`wavelet2vec.datasets.axis_labels` —
    the same feature the model conditions on, so labels are free and honest.
    """

    _AXES = ("brightness", "harmonicity", "attack_ms")

    def __init__(self, axis: str = "brightness", size: int = 1024, *,
                 sample_rate: int = 22050, seed: int = 0,
                 n_bands: int = 128, n_frames: int = 128) -> None:
        _require_torch()
        if axis not in self._AXES:
            raise ValueError(f"axis must be one of {self._AXES}")
        self.axis = axis
        self.size = int(size)
        self.sample_rate = sample_rate
        self.seed = seed
        self.n_bands = n_bands
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.size

    def _render(self, rng: np.random.Generator) -> np.ndarray:
        from wavelet2vec.datasets import (
            attack_ladder, filter_ladder, noise_mix_ladder,
        )

        f0 = float(rng.uniform(80.0, 500.0))
        # Render only the sampled ladder step (each generator returns
        # [(audio, param_value), ...] for the requested parameter list).
        if self.axis == "brightness":
            cutoff = float(np.exp(rng.uniform(
                np.log(400.0), np.log(0.28 * self.sample_rate))))
            return filter_ladder(f0, self.sample_rate, [cutoff],
                                 seconds=1.0)[0][0]
        if self.axis == "harmonicity":
            mix = float(rng.uniform(0.0, 0.9))
            return noise_mix_ladder(f0, self.sample_rate, [mix], seconds=1.0,
                                    seed=int(rng.integers(2**31)))[0][0]
        attack_ms = float(rng.uniform(1.0, 200.0))
        return attack_ladder(f0, self.sample_rate, [attack_ms],
                             seconds=1.0)[0][0]

    def __getitem__(self, index: int):
        from wavelet2vec.datasets import axis_labels

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        audio = self._render(rng)
        image, _meta = coeff_image(audio, self.sample_rate,
                                   n_bands=self.n_bands, n_frames=self.n_frames)
        label = axis_labels(audio, self.sample_rate)[self.axis]
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)


class ContrastiveViewsDataset(Dataset):
    """Two augmented views per clip for SimCLR/BYOL-style training.

    ``clips`` is a list of mono/stereo arrays (e.g. bar-crops of a sample
    library) at one sample rate. Views come from :func:`wavelet2vec.
    augment.random_augment`, deterministic per ``(seed, index, view)``;
    ``transform`` (e.g. a ``WaveletFrontend`` or ``sequence_tokens``
    wrapper) maps each view before batching, else raw float32 audio is
    returned.
    """

    def __init__(self, clips: list, sample_rate: int, *, seed: int = 0,
                 transform=None, max_semitones: int = 4,
                 effect_probability: float = 0.5) -> None:
        _require_torch()
        if not clips:
            raise ValueError("clips must be a non-empty list of arrays.")
        self.clips = [np.asarray(clip, dtype=np.float32) for clip in clips]
        self.sample_rate = sample_rate
        self.seed = seed
        self.transform = transform
        self.max_semitones = max_semitones
        self.effect_probability = effect_probability

    def __len__(self) -> int:
        return len(self.clips)

    def _view(self, clip: np.ndarray, index: int, view: int):
        from wavelet2vec.augment import random_augment

        augmented = random_augment(
            clip, self.sample_rate,
            seed=self.seed * 2 + view, index=index,
            max_semitones=self.max_semitones,
            effect_probability=self.effect_probability,
        )
        if self.transform is not None:
            return self.transform(augmented)
        return torch.from_numpy(augmented)

    def __getitem__(self, index: int):
        clip = self.clips[index]
        return self._view(clip, index, 0), self._view(clip, index, 1)
