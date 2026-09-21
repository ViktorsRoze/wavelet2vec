"""Embedding rigor: standardization/whitening + retrieval eval (roadmap D-7).

Two halves that keep the embedding honest:

- :class:`EmbeddingStandardizer` — fit per-dimension standardization or PCA
  whitening on a reference set (your library), then transform queries the
  same way. The 570-dim embedding concatenates sections with very different
  natural scales; standardizing stops the loudest section from silently
  owning the cosine. Save/load round-trips exactly and records the
  embedding version it was fitted on.
- :func:`retrieval_report` / :func:`knn_purity` / :func:`recall_at_k` —
  leave-one-out retrieval metrics over labeled vectors, so any change to
  the embedding (or a standardizer) gets a number, not an opinion.
  Queries whose label has no other member are excluded from recall —
  they have no correct answer to find.

Both operate on plain ``[N, dim]`` arrays: nothing here needs audio, so the
harness also works on vectors loaded from a saved
:class:`~wavelet2vec.index.SimilarityIndex`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wavelet2vec.embedder import EMBEDDING_VERSION


def _rows_normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class EmbeddingStandardizer:
    """Per-dimension standardization or PCA whitening for embeddings.

    ``mode="standardize"``: z-score each dimension by the reference set's
    mean/std. ``mode="whiten"``: rotate onto principal axes and equalize
    their variances (eigenvalues floored at ``shrinkage`` × their mean, so
    near-null directions don't explode noise). ``transform`` re-normalizes
    rows by default so cosine remains the metric downstream.
    """

    def __init__(self, mode: str = "standardize", *, shrinkage: float = 1e-3) -> None:
        if mode not in ("standardize", "whiten"):
            raise ValueError("mode must be 'standardize' or 'whiten'")
        self.mode = mode
        self.shrinkage = float(shrinkage)
        self.embedding_version = EMBEDDING_VERSION
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None  # standardize: per-dim std
        self.components_: np.ndarray | None = None  # whiten: [dim, dim]

    def fit(self, vectors: np.ndarray) -> "EmbeddingStandardizer":
        X = np.asarray(vectors, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("fit needs an [N, dim] array with N >= 2.")
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        if self.mode == "standardize":
            std = centered.std(axis=0)
            floor = max(std.mean(), 1e-12) * self.shrinkage
            self.scale_ = np.maximum(std, floor)
        else:
            cov = centered.T @ centered / (X.shape[0] - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            floor = max(eigenvalues.mean(), 1e-24) * self.shrinkage
            self.components_ = eigenvectors / np.sqrt(
                np.maximum(eigenvalues, floor))
        return self

    def transform(self, vectors: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        if self.mean_ is None:
            raise ValueError("Call fit() (or load()) before transform().")
        single = np.asarray(vectors).ndim == 1
        X = np.asarray(vectors, dtype=np.float64)
        if single:
            X = X[None, :]
        centered = X - self.mean_
        if self.mode == "standardize":
            out = centered / self.scale_
        else:
            out = centered @ self.components_
        if normalize:
            out = _rows_normalized(out)
        out = out.astype(np.float32)
        return out[0] if single else out

    def fit_transform(self, vectors: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        return self.fit(vectors).transform(vectors, normalize=normalize)

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if self.mean_ is None:
            raise ValueError("Nothing to save: fit() first.")
        path = Path(path)
        payload = {
            "mode": np.array(self.mode), "shrinkage": np.array(self.shrinkage),
            "embedding_version": np.array(self.embedding_version),
            "mean": self.mean_,
        }
        if self.scale_ is not None:
            payload["scale"] = self.scale_
        if self.components_ is not None:
            payload["components"] = self.components_
        np.savez_compressed(path, **payload)
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStandardizer":
        archive = np.load(Path(path))
        out = cls(str(archive["mode"]), shrinkage=float(archive["shrinkage"]))
        out.embedding_version = str(archive["embedding_version"])
        out.mean_ = archive["mean"]
        out.scale_ = archive["scale"] if "scale" in archive else None
        out.components_ = archive["components"] if "components" in archive else None
        return out


# ---------------------------------------------------------------------------
# Retrieval eval harness — leave-one-out, exact cosine
# ---------------------------------------------------------------------------
def _neighbor_order(vectors: np.ndarray) -> np.ndarray:
    """[N, N-1] neighbor indices per query, nearest first, self excluded."""
    normalized = _rows_normalized(vectors)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argsort(similarities, axis=1)[:, ::-1][:, : similarities.shape[0] - 1]


def knn_purity(vectors: np.ndarray, labels, k: int = 1) -> float:
    """Leave-one-out k-NN accuracy: fraction of items whose majority label
    among the ``k`` nearest others matches their own (ties → nearest wins)."""
    labels = np.asarray(labels)
    if labels.shape[0] != np.asarray(vectors).shape[0]:
        raise ValueError("vectors and labels must have equal length.")
    if labels.shape[0] < 2:
        raise ValueError("Need at least 2 items.")
    order = _neighbor_order(vectors)[:, : max(k, 1)]
    correct = 0
    for i, neighbors in enumerate(order):
        neighbor_labels = labels[neighbors]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        winners = values[counts == counts.max()]
        if len(winners) > 1:  # break ties by the single nearest neighbor
            vote = neighbor_labels[0] if neighbor_labels[0] in winners else winners[0]
        else:
            vote = winners[0]
        correct += int(vote == labels[i])
    return correct / labels.shape[0]


def recall_at_k(vectors: np.ndarray, labels, ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    """For each k: the fraction of queries with at least one same-label item
    in their top ``k`` neighbors. Labels with a single member are excluded
    (there is nothing correct to retrieve for them)."""
    labels = np.asarray(labels)
    order = _neighbor_order(vectors)
    values, counts = np.unique(labels, return_counts=True)
    eligible = np.isin(labels, values[counts > 1])
    if not eligible.any():
        raise ValueError("Every label is a singleton: recall is undefined.")
    result: dict[int, float] = {}
    for k in ks:
        hits = [
            bool((labels[order[i, : max(k, 1)]] == labels[i]).any())
            for i in np.flatnonzero(eligible)
        ]
        result[int(k)] = float(np.mean(hits))
    return result


def retrieval_report(vectors: np.ndarray, labels,
                     ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """One honest summary: kNN purity (k=1 and 5), recall@k, and per-label
    1-NN purity so a strong global number can't hide a broken class."""
    labels = np.asarray(labels)
    order = _neighbor_order(vectors)
    nearest = labels[order[:, 0]]
    per_label = {
        str(value): float(np.mean(nearest[labels == value] == value))
        for value in np.unique(labels)
    }
    return {
        "n": int(labels.shape[0]),
        "purity@1": knn_purity(vectors, labels, k=1),
        "purity@5": knn_purity(vectors, labels, k=5),
        "recall": recall_at_k(vectors, labels, ks=ks),
        "per_label_purity@1": per_label,
        "embedding_version": EMBEDDING_VERSION,
    }
