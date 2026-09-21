"""V-3: frozen vision backbones on coefficient tiles — the go/no-go probe.

Question: do generic pretrained vision features (DINOv2 ViT-S/14, Swin-T)
read our 3-channel coefficient images well enough to justify fine-tuning
(V-4+)? Measured on the real library (folders as labels) with the D-7
harness (leave-one-out kNN purity / recall) plus a 5-fold linear probe,
against the deterministic 570-dim embedding v1 as the baseline.

Run:  python -m experiments.frozen_probe
Features and tiles are cached under experiments/output/frozen_probe/ so
re-runs only redo what is missing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from wavelet2vec import Wavelet2Vec, coeff_image, retrieval_report
from wavelet2vec.audio_io import list_audio_files, load_audio

LIBRARY = Path(r"c:\Users\vikto\PycharmProjects\soundgen\example_audio")
OUT = Path(__file__).parent / "output" / "frozen_probe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_tiles() -> tuple[np.ndarray, np.ndarray]:
    """[N, 3, 128, 128] coefficient images + folder labels, cached."""
    cache = OUT / "tiles.npz"
    if cache.exists():
        archive = np.load(cache, allow_pickle=False)
        return archive["tiles"], archive["labels"]
    tiles, labels = [], []
    start = time.perf_counter()
    for path in list_audio_files(LIBRARY):
        rel = Path(path).relative_to(LIBRARY)
        if len(rel.parts) < 2:
            continue
        try:
            waveform, sr = load_audio(path)
        except Exception as error:
            print("skip", rel, error)
            continue
        mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
        image, _meta = coeff_image(mono, sr)
        tiles.append(image)
        labels.append(rel.parts[0])
    tiles = np.stack(tiles).astype(np.float32)
    labels = np.array(labels)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tiles=tiles, labels=labels)
    print(f"tiles: {tiles.shape} in {time.perf_counter() - start:.0f}s")
    return tiles, labels


def embedding_baseline() -> np.ndarray:
    """570-dim embedding v1 per file, cached (order matches build_tiles)."""
    cache = OUT / "embedding_v1.npz"
    if cache.exists():
        return np.load(cache)["vectors"]
    embedder = Wavelet2Vec()
    vectors = []
    start = time.perf_counter()
    for path in list_audio_files(LIBRARY):
        rel = Path(path).relative_to(LIBRARY)
        if len(rel.parts) < 2:
            continue
        try:
            waveform, sr = load_audio(path)
        except Exception:
            continue
        vectors.append(embedder.embed(waveform, sr))
    vectors = np.stack(vectors)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vectors=vectors)
    print(f"embedding v1: {vectors.shape} in {time.perf_counter() - start:.0f}s")
    return vectors


def _prepare_batch(tiles: np.ndarray, magnitude_only: bool) -> torch.Tensor:
    batch = torch.from_numpy(tiles)
    if magnitude_only:
        batch = batch[:, :1].repeat(1, 3, 1, 1)  # log-magnitude on all channels
    batch = torch.nn.functional.interpolate(
        batch, size=(224, 224), mode="bilinear", align_corners=False)
    return (batch - _IMAGENET_MEAN) / _IMAGENET_STD


def extract_features(name: str, model, tiles: np.ndarray, *,
                     magnitude_only: bool = False, chunk: int = 16) -> np.ndarray:
    cache = OUT / f"features_{name}.npz"
    if cache.exists():
        return np.load(cache)["features"]
    model.eval().to(DEVICE)
    features = []
    start = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, tiles.shape[0], chunk):
            batch = _prepare_batch(tiles[begin: begin + chunk], magnitude_only)
            features.append(model(batch.to(DEVICE)).cpu().numpy())
    model.cpu()
    torch.cuda.empty_cache()
    features = np.concatenate(features).astype(np.float32)
    np.savez_compressed(cache, features=features)
    print(f"{name}: {features.shape} in {time.perf_counter() - start:.0f}s")
    return features


def linear_probe(features: np.ndarray, labels: np.ndarray) -> float:
    """5-fold stratified logistic-regression accuracy on frozen features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(cross_val_score(pipeline, features, labels, cv=folds).mean())


def main() -> None:
    tiles, labels = build_tiles()
    print(f"{tiles.shape[0]} tiles, {len(set(labels.tolist()))} labels, "
          f"device {DEVICE}")

    candidates: dict[str, np.ndarray] = {"embedding_v1": embedding_baseline()}

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                            verbose=False)
    candidates["dinov2_vits14"] = extract_features("dinov2_vits14", dinov2, tiles)
    candidates["dinov2_vits14_magonly"] = extract_features(
        "dinov2_vits14_magonly", dinov2, tiles, magnitude_only=True)

    from torchvision.models import Swin_T_Weights, swin_t

    swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    swin.head = torch.nn.Identity()
    candidates["swin_t"] = extract_features("swin_t", swin, tiles)

    report = {}
    for name, features in candidates.items():
        result = retrieval_report(features, labels, ks=(1, 5, 10))
        result["linear_probe_acc"] = linear_probe(features, labels)
        report[name] = result
        print(f"\n== {name} ({features.shape[1]}d): "
              f"purity@1 {result['purity@1']:.3f}  "
              f"recall@5 {result['recall'][5]:.3f}  "
              f"linear probe {result['linear_probe_acc']:.3f}")
        for label, purity in sorted(result["per_label_purity@1"].items()):
            print(f"   {label:28s} {purity:.3f}")

    (OUT / "report.json").write_text(json.dumps(report, indent=1),
                                     encoding="utf-8")
    print("\nreport ->", OUT / "report.json")


if __name__ == "__main__":
    main()
