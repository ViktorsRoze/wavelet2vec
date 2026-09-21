"""V-4 (small-scale): event detection on coefficient images.

Faster R-CNN with a MobileNetV3-FPN backbone (fits the 4 GB card;
torchvision's battle-tested pipeline) trained on D-6's
:class:`~wavelet2vec.data.SceneDetectionDataset` — unlimited exactly-
labeled scenes, generated on the fly. The DETR-family upgrade the roadmap
names is a scale-time swap once a bigger GPU / the user's datasets land;
this proves the pipeline and measures gate 4 ("small-scale F1 > 0.9 on
synthetic scenes") honestly.

Eval: greedy class-aware matching at IoU ≥ 0.5 on held-out scenes
(disjoint seed space) → precision / recall / F1 overall and per class,
plus median onset-timing error in milliseconds.

Run:  python -m experiments.train_detector   (resumable per epoch)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wavelet2vec.data import SCENE_CLASSES, SceneDetectionDataset, detection_collate

OUT = Path(__file__).parent / "output" / "detector_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SECONDS = 2.0
IMAGE = 128
TRAIN_SIZE = 2400
EVAL_SIZE = 150
EPOCHS = 12
BATCH = 8
CHECKPOINT = OUT / "detector_v1.pt"
STATE = OUT / "train_state.pt"


def _build_cache(size: int, seed: int, path: Path) -> dict:
    """Pre-renders D-6 scenes to disk (resumable): rendering costs ~0.4 s
    per scene on CPU, which would starve the GPU if done per epoch."""
    base = SceneDetectionDataset(size, seconds=SECONDS, seed=seed,
                                 n_bands=IMAGE, n_frames=IMAGE)
    state = {"images": [], "boxes": [], "labels": [], "n_done": 0}
    if path.exists():
        try:
            archive = np.load(path, allow_pickle=True)
            if int(archive["n_done"]) >= size:
                return {key: archive[key] for key in archive.files}
            state = {"images": list(archive["images"]),
                     "boxes": list(archive["boxes"]),
                     "labels": list(archive["labels"]),
                     "n_done": int(archive["n_done"])}
            print(f"resuming scene cache at {state['n_done']}/{size}")
        except Exception:  # interrupted mid-write: rebuild from scratch
            print(f"cache {path.name} unreadable (interrupted write) - rebuilding")
    start = time.perf_counter()
    for index in range(state["n_done"], size):
        image, target = base[index]
        boxes = target["boxes"].clone()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # xywh -> xyxy
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = boxes.clamp(0.0, float(IMAGE))
        keep = (boxes[:, 2] - boxes[:, 0] > 0.5) & (boxes[:, 3] - boxes[:, 1] > 0.5)
        state["images"].append(image.numpy().astype(np.float16))
        state["boxes"].append(boxes[keep].numpy().astype(np.float32))
        state["labels"].append(target["labels"][keep].numpy().astype(np.int64))
        state["n_done"] = index + 1
        if state["n_done"] % 200 == 0 or state["n_done"] == size:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, images=np.stack(state["images"]),
                boxes=np.array(state["boxes"], dtype=object),
                labels=np.array(state["labels"], dtype=object),
                n_done=state["n_done"])
            print(f"  scene {state['n_done']}/{size} "
                  f"({time.perf_counter()-start:.0f}s)", flush=True)
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}


PITCHED_IDS = {i + 1 for i, name in enumerate(SCENE_CLASSES)
               if name in ("sine", "saw", "pluck")}
NOTE_HALF_HEIGHT = 5.0  # px apron for pitched boxes (see class docstring)


class _XyxyScenes(Dataset):
    """Cached scenes with targets in torchvision's dialect (xyxy).

    Pitched boxes get a symmetric vertical apron (±5 px ≈ ±4 semitones):
    the raw ±1.5-semitone box is ~4 px tall on the 128-row grid, which no
    anchor assignment or IoU-0.5 metric can credit (measured: note F1 0.03
    then 0.01 across two anchor configurations while drums scored 0.87–1.0).
    Pitch decodes from the box *center*, which symmetric padding preserves.
    """

    def __init__(self, size: int, seed: int, cache_name: str) -> None:
        cache = _build_cache(size, seed, OUT / cache_name)
        self.images = cache["images"]
        self.boxes = cache["boxes"]
        self.labels = cache["labels"]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        boxes = torch.from_numpy(self.boxes[index]).clone()
        labels = torch.from_numpy(self.labels[index])
        pitched = torch.tensor([int(label) in PITCHED_IDS for label in labels],
                               dtype=torch.bool)
        if pitched.any():
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            boxes[pitched, 1] = (cy[pitched] - NOTE_HALF_HEIGHT).clamp(
                0.0, float(IMAGE))
            boxes[pitched, 3] = (cy[pitched] + NOTE_HALF_HEIGHT).clamp(
                0.0, float(IMAGE))
        return (torch.from_numpy(self.images[index].astype(np.float32)),
                {"boxes": boxes, "labels": labels})


def make_model():
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    # Anchor experiments, measured: shrinking anchors to the note scale
    # (4..64 or 8..128) lifted notes only marginally and *degraded* the
    # drum classes (kick 1.0 → 0.74–0.78, hat 0.87 → 0.57–0.61). The stock
    # anchors + the ±5 px note apron + transcription-style matching keep
    # every regime's best behavior; small-anchor configs are the recorded
    # dead end, not the fix.
    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=len(SCENE_CLASSES) + 1,
        min_size=IMAGE, max_size=IMAGE,
    )


def train() -> torch.nn.Module:
    model = make_model().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1)
    begin_epoch = 0
    if STATE.exists():
        state = torch.load(STATE, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        begin_epoch = int(state["epoch"])
        print(f"resuming at epoch {begin_epoch}")

    loader = DataLoader(_XyxyScenes(TRAIN_SIZE, seed=0, cache_name="train.npz"),
                        batch_size=BATCH, shuffle=True,
                        collate_fn=detection_collate, num_workers=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Faster R-CNN mobilenet-FPN {n_params/1e6:.1f}M params on {DEVICE}")

    model.train()
    start = time.perf_counter()
    for epoch in range(begin_epoch, EPOCHS):
        running = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"epoch {epoch}  step {step}/{len(loader)}  "
                      f"loss {running/step:.3f}  "
                      f"({time.perf_counter()-start:.0f}s)", flush=True)
        scheduler.step()
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1},
                   STATE)
        print(f"epoch {epoch} done  mean loss {running/len(loader):.3f}")
    torch.save({"model": model.state_dict()}, CHECKPOINT)
    return model


def evaluate(model: torch.nn.Module, score_threshold: float = 0.5) -> dict:
    """Transcription-style scoring: pitched events match on onset ±50 ms +
    pitch-center within ±1.5 semitones (mir_eval's spirit; the 128-row grid
    quantizes pitch to ~0.8 st/px, so 50-cent tolerance is sub-pixel here);
    drums match on IoU ≥ 0.5 (their boxes span octaves — IoU is fair)."""
    dataset = _XyxyScenes(EVAL_SIZE, seed=99000,  # disjoint seed space
                          cache_name="eval.npz")
    model.eval().to(DEVICE)
    ms_per_pixel = SECONDS * 1000.0 / IMAGE
    onset_tolerance_px = 50.0 / ms_per_pixel
    rows_per_semitone = IMAGE / (np.log2(11025.0 / 30.0) * 12.0)
    pitch_tolerance_px = 1.5 * rows_per_semitone

    def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import box_iou

        return box_iou(a, b)

    tp = {name: 0 for name in SCENE_CLASSES}
    fp = {name: 0 for name in SCENE_CLASSES}
    fn = {name: 0 for name in SCENE_CLASSES}
    onset_errors = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, truth = dataset[index]
            prediction = model([image.to(DEVICE)])[0]
            keep = prediction["scores"] > score_threshold
            boxes = prediction["boxes"][keep].cpu()
            labels = prediction["labels"][keep].cpu()
            matched_truth = set()
            order = torch.argsort(prediction["scores"][keep], descending=True)
            for i in order.tolist():
                class_id = int(labels[i])
                name = SCENE_CLASSES[class_id - 1]
                candidates = [
                    j for j in range(truth["labels"].shape[0])
                    if j not in matched_truth
                    and int(truth["labels"][j]) == class_id
                ]
                if not candidates:
                    fp[name] += 1
                    continue
                if class_id in PITCHED_IDS:
                    cy = float(boxes[i][1] + boxes[i][3]) / 2.0
                    x1 = float(boxes[i][0])
                    scored = []
                    for j in candidates:
                        onset_px = abs(x1 - float(truth["boxes"][j][0]))
                        pitch_px = abs(cy - float(
                            truth["boxes"][j][1] + truth["boxes"][j][3]) / 2.0)
                        if (onset_px <= onset_tolerance_px
                                and pitch_px <= pitch_tolerance_px):
                            scored.append((onset_px, j))
                    if scored:
                        onset_px, j = min(scored)
                        matched_truth.add(j)
                        tp[name] += 1
                        onset_errors.append(onset_px * ms_per_pixel)
                    else:
                        fp[name] += 1
                    continue
                ious = iou_matrix(boxes[i][None], truth["boxes"][candidates])[0]
                best = int(torch.argmax(ious))
                if float(ious[best]) >= 0.5:
                    j = candidates[best]
                    matched_truth.add(j)
                    tp[name] += 1
                    onset_errors.append(
                        abs(float(boxes[i][0]) - float(truth["boxes"][j][0]))
                        * ms_per_pixel)
                else:
                    fp[name] += 1
            for j in range(truth["labels"].shape[0]):
                if j not in matched_truth:
                    fn[SCENE_CLASSES[int(truth["labels"][j]) - 1]] += 1

    def f1(tp_n, fp_n, fn_n):
        precision = tp_n / (tp_n + fp_n) if tp_n + fp_n else 0.0
        recall = tp_n / (tp_n + fn_n) if tp_n + fn_n else 0.0
        score = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)
        return precision, recall, score

    total = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_class = {name: f1(tp[name], fp[name], fn[name])
                 for name in SCENE_CLASSES}
    report = {
        "n_scenes": len(dataset), "score_threshold": score_threshold,
        "precision": total[0], "recall": total[1], "f1": total[2],
        "median_onset_error_ms": float(np.median(onset_errors)),
        "per_class": {name: {"precision": values[0], "recall": values[1],
                             "f1": values[2],
                             "support": tp[name] + fn[name]}
                      for name, values in per_class.items()},
    }
    return report


def main() -> None:
    trained = STATE.exists() and int(
        torch.load(STATE, map_location="cpu",
                   weights_only=True)["epoch"]) >= EPOCHS
    if trained and CHECKPOINT.exists():
        print("reusing", CHECKPOINT)
        model = make_model()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu",
                                         weights_only=True)["model"])
    else:
        model = train()
    report = evaluate(model)
    (OUT / "report.json").write_text(json.dumps(report, indent=1),
                                     encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
