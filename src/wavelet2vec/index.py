"""Similarity index + duplicate detection for sound libraries (roadmap I-4).

The editor's library backend: build once over a folder of samples, then
query nearest sounds and near-duplicate groups instantly. Vectors are
L2-normalized 570-dim Wavelet2Vec embeddings, so similarity = inner product
= cosine.

Backends: exact numpy (a single GEMM — entirely adequate up to ~100 k
sounds) and FAISS ``IndexFlatIP`` when installed and the library is large
(also exact; same results, faster queries). No approximate search until a
library actually needs it — exactness keeps dedup honest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:  # optional accelerator
    import faiss

    HAVE_FAISS = True
except Exception:  # pragma: no cover - exercised only without faiss
    faiss = None
    HAVE_FAISS = False

_FAISS_MIN_SIZE = 2048  # below this, numpy's GEMM wins on overhead alone


class SimilarityIndex:
    """Exact cosine-similarity index over named embedding vectors."""

    def __init__(self, dim: int | None = None, *, version: str | None = None) -> None:
        self.dim = dim
        self.version = version  # embedding version tag (D-7); None = untagged
        self.ids: list[str] = []
        self._vectors: np.ndarray | None = None  # [N, dim], L2-normalized
        self._faiss_index = None
        self._faiss_dirty = True

    # --- building -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    @property
    def backend(self) -> str:
        if HAVE_FAISS and len(self) >= _FAISS_MIN_SIZE:
            return "faiss"
        return "numpy"

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def add_many(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = self._normalize(vectors)
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have equal length.")
        if self.dim is None:
            self.dim = vectors.shape[1]
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}.")
        duplicates = set(ids) & set(self.ids)
        if duplicates:
            raise ValueError(f"Duplicate ids: {sorted(duplicates)[:3]}…")
        self.ids.extend(ids)
        self._vectors = (vectors if self._vectors is None
                         else np.concatenate([self._vectors, vectors]))
        self._faiss_dirty = True

    def add(self, identifier: str, vector: np.ndarray) -> None:
        self.add_many([identifier], np.asarray(vector)[None, :])

    @classmethod
    def from_folder(cls, folder: str | Path, *, embedder=None) -> "SimilarityIndex":
        """Embed every audio file under ``folder`` (recursive) and index it.
        Ids are paths relative to the folder."""
        from wavelet2vec.audio_io import list_audio_files, load_audio

        if embedder is None:
            from wavelet2vec.embedder import Wavelet2Vec

            embedder = Wavelet2Vec()
        folder = Path(folder)
        index = cls(version=getattr(embedder, "version", None))
        ids, vectors = [], []
        for path in list_audio_files(folder):
            try:
                waveform, sample_rate = load_audio(path)
                vectors.append(embedder.embed(waveform, sample_rate))
                ids.append(str(Path(path).relative_to(folder)))
            except Exception:
                log.exception("Skipping unreadable file %s", path)
        if ids:
            index.add_many(ids, np.stack(vectors))
        return index

    # --- queries ----------------------------------------------------------------
    def _query_matrix(self, queries: np.ndarray) -> np.ndarray:
        """Similarities [Q, N] for normalized query rows."""
        if self._vectors is None:
            raise ValueError("The index is empty.")
        if self.backend == "faiss":
            if self._faiss_dirty or self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(self._vectors)
                self._faiss_dirty = False
            similarities, order = self._faiss_index.search(queries, len(self))
            full = np.empty((queries.shape[0], len(self)), dtype=np.float32)
            for row in range(queries.shape[0]):
                full[row, order[row]] = similarities[row]
            return full
        return queries @ self._vectors.T

    def search(self, query, k: int = 10) -> list[tuple[str, float]]:
        """Nearest sounds to a vector or an indexed id. Searching by id
        excludes the id itself from its results."""
        exclude = None
        if isinstance(query, str):
            if query not in self.ids:
                raise KeyError(query)
            exclude = self.ids.index(query)
            vector = self._vectors[exclude][None, :]
        else:
            vector = self._normalize(np.asarray(query))
        similarities = self._query_matrix(vector)[0]
        if exclude is not None:
            similarities = similarities.copy()
            similarities[exclude] = -np.inf
        order = np.argsort(similarities)[::-1][: max(k, 0)]
        return [(self.ids[i], float(similarities[i])) for i in order]

    def duplicates(self, threshold: float = 0.995) -> list[list[str]]:
        """Groups of near-identical sounds (cosine ≥ ``threshold``),
        each group sorted, largest groups first. Union-find over the exact
        pairwise matrix, blockwise to stay memory-flat."""
        n = len(self)
        if n < 2:
            return []
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        block = 1024
        for start in range(0, n, block):
            rows = self._vectors[start: start + block] @ self._vectors.T
            for local, row in enumerate(rows):
                i = start + local
                for j in np.flatnonzero(row >= threshold):
                    if j > i:
                        ra, rb = find(i), int(find(int(j)))
                        if ra != rb:
                            parent[rb] = ra
        groups: dict[int, list[str]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(self.ids[i])
        result = [sorted(members) for members in groups.values() if len(members) > 1]
        return sorted(result, key=len, reverse=True)

    # --- persistence ---------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if self._vectors is None:
            raise ValueError("Nothing to save: the index is empty.")
        payload = {
            "vectors": self._vectors,
            "ids": np.frombuffer(json.dumps(self.ids).encode(), dtype=np.uint8),
        }
        if self.version is not None:
            payload["version"] = np.array(self.version)
        np.savez_compressed(path, **payload)
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "SimilarityIndex":
        archive = np.load(Path(path))
        version = str(archive["version"]) if "version" in archive else None
        if version is not None:
            from wavelet2vec.embedder import EMBEDDING_VERSION

            if version != EMBEDDING_VERSION:
                log.warning(
                    "Index %s was built with embedding %s but the library is "
                    "now %s — rebuild before mixing with fresh embeddings.",
                    path, version, EMBEDDING_VERSION)
        index = cls(version=version)
        vectors = archive["vectors"]
        ids = json.loads(bytes(archive["ids"]).decode())
        index.add_many(ids, vectors)
        return index
