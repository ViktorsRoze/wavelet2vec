"""D-7: embedding standardization/whitening, retrieval eval, version tag."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec.eval import (
    EMBEDDING_VERSION,
    EmbeddingStandardizer,
    knn_purity,
    recall_at_k,
    retrieval_report,
)


def _clusters(n_per=20, dim=32, spread=0.05, seed=0):
    """Three well-separated Gaussian clusters + labels."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, dim)) * 2.0
    vectors, labels = [], []
    for c, center in enumerate(centers):
        vectors.append(center + spread * rng.standard_normal((n_per, dim)))
        labels += [f"class{c}"] * n_per
    return np.concatenate(vectors), np.array(labels)


# --- retrieval metrics ---------------------------------------------------------
def test_purity_and_recall_on_separated_clusters():
    vectors, labels = _clusters()
    assert knn_purity(vectors, labels, k=1) == 1.0
    assert knn_purity(vectors, labels, k=5) == 1.0
    recall = recall_at_k(vectors, labels, ks=(1, 5, 10))
    assert recall[1] == recall[5] == recall[10] == 1.0

    shuffled = np.array(labels)
    np.random.default_rng(1).shuffle(shuffled)
    assert knn_purity(vectors, shuffled, k=1) < 0.6  # broken labels score low


def test_recall_monotonic_and_singletons_excluded():
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((30, 16))  # unstructured: recall grows with k
    labels = np.array(["a", "b", "c"] * 10)
    recall = recall_at_k(vectors, labels, ks=(1, 5, 20))
    assert recall[1] <= recall[5] <= recall[20]

    # A singleton label must not count against recall (nothing to retrieve).
    vectors2, labels2 = _clusters(n_per=5)
    lone = np.concatenate([vectors2, [vectors2.mean(axis=0)]])
    lone_labels = np.append(labels2, "unique")
    assert recall_at_k(lone, lone_labels, ks=(1,))[1] == 1.0
    with pytest.raises(ValueError):
        recall_at_k(vectors2[:3], np.array(["x", "y", "z"]), ks=(1,))


def test_retrieval_report_structure():
    vectors, labels = _clusters(n_per=8)
    report = retrieval_report(vectors, labels, ks=(1, 5))
    assert report["n"] == 24
    assert report["purity@1"] == 1.0
    assert set(report["per_label_purity@1"]) == {"class0", "class1", "class2"}
    assert report["embedding_version"] == EMBEDDING_VERSION


# --- standardizer ----------------------------------------------------------------
def test_standardize_rescues_scale_dominated_cosine():
    # Clusters live in dims 0..9 at small scale; dims 10..14 are huge
    # label-free noise that owns the raw cosine (a single noise dim is NOT
    # enough — ranking there is second-order and clusters survive; measured).
    # Standardizing restores retrieval. Whitening does not rescue this case:
    # it flattens between-cluster variance too (measured 0.30 vs 0.97).
    rng = np.random.default_rng(30)
    vectors, labels = _clusters(dim=10, spread=0.05, seed=3)
    noise = rng.standard_normal((vectors.shape[0], 5)) * 100.0
    vectors = np.concatenate([vectors * 0.01, noise], axis=1)

    raw = knn_purity(vectors, labels, k=1)
    fixed = knn_purity(EmbeddingStandardizer("standardize").fit_transform(vectors),
                       labels, k=1)
    assert raw < 0.5 and fixed > 0.9


def test_whiten_decorrelates():
    rng = np.random.default_rng(4)
    latent = rng.standard_normal((400, 4))
    mixing = rng.standard_normal((4, 12))
    data = latent @ mixing + rng.standard_normal((400, 12)) * 0.1
    white = EmbeddingStandardizer("whiten", shrinkage=1e-6).fit_transform(
        data, normalize=False)
    cov = np.cov(white.astype(np.float64).T)
    off_diagonal = cov - np.diag(np.diag(cov))
    strong = np.diag(cov) > 0.5  # directions above the shrinkage floor
    assert np.abs(np.diag(cov)[strong] - 1.0).max() < 0.05  # unit variance
    assert np.abs(off_diagonal).max() < 0.05  # decorrelated


def test_standardizer_save_load_roundtrip(tmp_path):
    vectors, _labels = _clusters(n_per=10)
    for mode in ("standardize", "whiten"):
        original = EmbeddingStandardizer(mode).fit(vectors)
        saved = original.save(tmp_path / f"{mode}.npz")
        restored = EmbeddingStandardizer.load(saved)
        assert restored.mode == mode
        assert restored.embedding_version == EMBEDDING_VERSION
        assert np.array_equal(original.transform(vectors),
                              restored.transform(vectors))
    with pytest.raises(ValueError):
        EmbeddingStandardizer("median")
    with pytest.raises(ValueError):
        EmbeddingStandardizer().transform(vectors)  # unfitted


# --- version tag ------------------------------------------------------------------
def test_version_tag_travels_with_index(tmp_path, caplog):
    import logging

    from wavelet2vec import SimilarityIndex, Wavelet2Vec

    assert Wavelet2Vec().version == EMBEDDING_VERSION

    index = SimilarityIndex(version=EMBEDDING_VERSION)
    rng = np.random.default_rng(5)
    index.add_many(["a", "b"], rng.standard_normal((2, 8)))
    path = index.save(tmp_path / "lib.npz")
    assert SimilarityIndex.load(path).version == EMBEDDING_VERSION

    stale = SimilarityIndex(version="w2v570-v0")
    stale.add_many(["a"], rng.standard_normal((1, 8)))
    stale_path = stale.save(tmp_path / "stale.npz")
    with caplog.at_level(logging.WARNING, logger="wavelet2vec.index"):
        SimilarityIndex.load(stale_path)
    assert any("rebuild" in message for message in caplog.messages)

    untagged = SimilarityIndex()
    untagged.add_many(["a"], rng.standard_normal((1, 8)))
    untagged_path = untagged.save(tmp_path / "untagged.npz")
    assert SimilarityIndex.load(untagged_path).version is None  # backward compat
