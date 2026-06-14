"""Quantitative metrics: pitch recovery, cluster quality, invariance, PCA.

These turn the mel-vs-wavelet comparison into numbers and double as a
regression harness (see ``docs/EXPERIMENTS_PLAN.md`` §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pitch_class_correct(predicted: int, label: int, tolerant: bool = False) -> bool:
    """Octave-folded pitch-class match; ``tolerant`` allows ±1 semitone."""
    diff = (predicted - label) % 12
    diff = min(diff, 12 - diff)
    return diff <= (1 if tolerant else 0)


@dataclass
class PitchResult:
    n: int
    mel_accuracy: float
    wavelet_accuracy: float
    mel_correct: int
    wavelet_correct: int


def score_pitch_recovery(
    labels: list[int],
    mel_predictions: list[int],
    wavelet_predictions: list[int],
    tolerant: bool = False,
) -> PitchResult:
    n = len(labels)
    mel_ok = sum(pitch_class_correct(p, y, tolerant) for p, y in zip(mel_predictions, labels))
    wav_ok = sum(pitch_class_correct(p, y, tolerant) for p, y in zip(wavelet_predictions, labels))
    return PitchResult(
        n=n,
        mel_accuracy=mel_ok / max(n, 1),
        wavelet_accuracy=wav_ok / max(n, 1),
        mel_correct=mel_ok,
        wavelet_correct=wav_ok,
    )


def confusion_matrix(labels: list[int], predictions: list[int], n_classes: int = 12) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for label, pred in zip(labels, predictions):
        matrix[label % n_classes, pred % n_classes] += 1
    return matrix


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project rows to 2D via PCA (numpy SVD; no sklearn dependency)."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T


def knn_label_purity(vectors: np.ndarray, labels: list[str], k: int = 5) -> float:
    """Fraction of k nearest neighbors (cosine) sharing a point's own label."""
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -np.inf)
    label_array = np.asarray(labels)
    purities = []
    for index in range(len(labels)):
        neighbors = np.argsort(similarity[index])[::-1][:k]
        purities.append(np.mean(label_array[neighbors] == label_array[index]))
    return float(np.mean(purities)) if purities else 0.0


def silhouette_score(vectors: np.ndarray, labels: list[str]) -> float:
    """Mean silhouette over points (cosine distance). Falls back to 0 if degenerate."""
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    distance = 1.0 - normed @ normed.T
    np.clip(distance, 0.0, 2.0, out=distance)
    label_array = np.asarray(labels)
    scores = []
    for index in range(len(labels)):
        same = label_array == label_array[index]
        same[index] = False
        if not same.any():
            continue
        a = distance[index, same].mean()
        b = min(
            distance[index, label_array == other].mean()
            for other in unique
            if other != label_array[index]
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0
