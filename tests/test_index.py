"""I-4: similarity index + duplicate detection."""

from __future__ import annotations

import numpy as np
import pytest

from wavelet2vec import SimilarityIndex

RNG = np.random.default_rng(0)


def _cluster_vectors(n_clusters=4, per_cluster=8, dim=64, noise=0.05):
    ids, vectors = [], []
    for c in range(n_clusters):
        center = RNG.standard_normal(dim)
        for i in range(per_cluster):
            ids.append(f"c{c}_{i}")
            vectors.append(center + noise * RNG.standard_normal(dim))
    return ids, np.stack(vectors)


def test_search_orders_by_similarity():
    ids, vectors = _cluster_vectors()
    index = SimilarityIndex()
    index.add_many(ids, vectors)
    results = index.search("c2_0", k=7)
    assert len(results) == 7
    assert results[0][0].startswith("c2_")  # nearest neighbour shares the cluster
    names = [name for name, _s in results]
    assert "c2_0" not in names  # id queries exclude themselves
    similarities = [s for _n, s in results]
    assert similarities == sorted(similarities, reverse=True)
    # All same-cluster mates rank above any foreign vector.
    same = [n for n in names if n.startswith("c2_")]
    assert len(same) == 7


def test_search_by_vector_and_k_clip():
    ids, vectors = _cluster_vectors()
    index = SimilarityIndex()
    index.add_many(ids, vectors)
    hits = index.search(vectors[5], k=3)
    assert hits[0][0] == ids[5]  # the vector itself is in the index
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert index.search(vectors[0], k=0) == []


def test_duplicates_groups_near_copies():
    ids, vectors = _cluster_vectors(noise=0.3)
    index = SimilarityIndex()
    index.add_many(ids, vectors)
    # Inject two exact copies and one near-copy of c0_0.
    base = vectors[0]
    index.add("copy_a", base)
    index.add("copy_b", base * 2.0)  # scale-invariant: normalized
    index.add("near", base + 1e-3 * RNG.standard_normal(base.shape))
    groups = index.duplicates(threshold=0.999)
    assert len(groups) == 1
    assert set(groups[0]) == {"c0_0", "copy_a", "copy_b", "near"}


def test_save_load_round_trip(tmp_path):
    ids, vectors = _cluster_vectors()
    index = SimilarityIndex()
    index.add_many(ids, vectors)
    path = tmp_path / "library.npz"
    index.save(path)
    loaded = SimilarityIndex.load(path)
    assert loaded.ids == index.ids
    before, after = index.search("c1_2", k=3), loaded.search("c1_2", k=3)
    assert [n for n, _s in before] == [n for n, _s in after]
    for (_na, sa), (_nb, sb) in zip(before, after):
        assert sa == pytest.approx(sb, abs=1e-5)  # float32 renormalization jitter


def test_rejects_mismatched_and_duplicate_ids():
    index = SimilarityIndex()
    index.add_many(["a"], RNG.standard_normal((1, 8)))
    with pytest.raises(ValueError):
        index.add_many(["b"], RNG.standard_normal((1, 9)))  # dim mismatch
    with pytest.raises(ValueError):
        index.add("a", RNG.standard_normal(8))  # duplicate id
    with pytest.raises(KeyError):
        index.search("missing")


@pytest.mark.skipif(not __import__("wavelet2vec.index", fromlist=["HAVE_FAISS"]).HAVE_FAISS,
                    reason="faiss not installed")
def test_faiss_backend_matches_numpy():
    from wavelet2vec import index as index_module

    ids = [f"v{i}" for i in range(3000)]
    vectors = RNG.standard_normal((3000, 32)).astype(np.float32)
    index = SimilarityIndex()
    index.add_many(ids, vectors)
    assert index.backend == "faiss"
    faiss_hits = index.search(vectors[7], k=5)
    # Force the numpy path and compare.
    original = index_module._FAISS_MIN_SIZE
    index_module._FAISS_MIN_SIZE = 10**9
    try:
        numpy_hits = index.search(vectors[7], k=5)
    finally:
        index_module._FAISS_MIN_SIZE = original
    assert [n for n, _s in faiss_hits] == [n for n, _s in numpy_hits]
    for (_na, sa), (_nb, sb) in zip(faiss_hits, numpy_hits):
        assert sa == pytest.approx(sb, abs=1e-5)


def test_from_folder_finds_similar_tone(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.arange(sr) / sr
    for name, freq in (("a440.wav", 440.0), ("a442.wav", 442.0), ("high3k.wav", 3000.0)):
        sf.write(tmp_path / name, np.sin(2 * np.pi * freq * t) * 0.5, sr)
    index = SimilarityIndex.from_folder(tmp_path)
    assert len(index) == 3
    results = index.search("a440.wav", k=2)
    assert results[0][0] == "a442.wav"  # 2 Hz away beats 3 kHz away
