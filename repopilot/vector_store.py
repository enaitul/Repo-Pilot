"""
Stores CodeChunk embeddings and finds the closest ones to a query vector.

This is the Phase 6 counterpart to how earlier phases wrap an external
tool behind a narrow interface: cloner.py wraps `git`, the parsers wrap
`ast`/Tree-sitter, sentence_transformer_provider.py wraps
sentence-transformers. This module wraps FAISS — nothing outside this
file needs to know FAISS is involved at all.

WHY FAISS: it's a free, local, no-API-key library built specifically to
answer "which of these vectors is closest to this one?" quickly, even
for large collections. It does NOT handle metadata or persistence on its
own — that's exactly the gap this module fills, by keeping a plain
Python mapping (FAISS integer ID -> CodeChunk) alongside the FAISS index
itself, and saving/loading both together.

WHY NORMALIZE + INNER PRODUCT INSTEAD OF L2 DISTANCE: our embedding
model (all-MiniLM-L6-v2, from Phase 5) is designed to be compared with
COSINE similarity (the angle between two vectors, ignoring their
length) — not straight-line (L2/Euclidean) distance, which IS affected
by vector length. Cosine similarity is mathematically just "inner
product, divided by both vectors' lengths, to cancel length out." If
every vector is rescaled to length 1 first (normalized), that division
no longer changes anything — so plain inner product on normalized
vectors already gives the same ranking as cosine similarity. This is
the standard, well-documented way to get FAISS to behave like cosine
similarity, since FAISS has no cosine-specific index type of its own.

WHY IndexIDMap: a plain FAISS index assigns vectors sequential
positions (0, 1, 2, ...) automatically, with no way to tie a position
back to something meaningful, and no way to remove a specific one later.
Wrapping the index in `IndexIDMap` lets us assign OUR OWN integer ID to
each vector (tied to its chunk), which is what makes future per-chunk
update/delete support possible without redesigning this module later —
even though THIS phase only implements a full `rebuild()`, not
per-chunk deletion, deliberately deferred as a follow-up (see
exceptions.py's VectorStoreError docstring for the same "system failure
vs. expected outcome" philosophy applied here).
"""

from __future__ import annotations

import json
from pathlib import Path
try:
    import faiss
except ImportError:
    faiss = None

import numpy as np

from repopilot.exceptions import VectorStoreError
from repopilot.models import CodeChunk, EmbeddedChunk, SearchResult


class VectorStore:
    """Stores embedded chunks in a local FAISS index and searches them."""

    def __init__(self):
        # Dimensions and the underlying FAISS index are both unknown
        # until the first batch of vectors arrives — we infer the
        # dimension from that first batch rather than requiring the
        # caller to specify it up front, matching the flexibility
        # EmbeddingProvider already has via its own `.dimensions`.
        self._dimensions: int | None = None
        self._index: faiss.IndexIDMap | None = None
        self._id_to_chunk: dict[int, CodeChunk] = {}
        self._next_id: int = 0

    @property
    def size(self) -> int:
        """How many vectors are currently stored."""
        return len(self._id_to_chunk)

    # -- building the index --------------------------------------------

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Add embedded chunks to the store. Safe to call multiple times to
        grow the index incrementally (e.g. as more files get embedded).

        Every chunk added in the store's lifetime must share the same
        vector dimension — mixing vectors from two different embedding
        models would silently produce meaningless similarity scores, so
        a mismatch raises VectorStoreError immediately rather than
        letting it corrupt search results later.
        """
        if not embedded_chunks:
            return  # nothing to add — not an error, just a no-op

        dimension = embedded_chunks[0].dimensions
        if self._dimensions is None:
            self._dimensions = dimension
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        elif dimension != self._dimensions:
            raise VectorStoreError(
                f"Cannot add vectors with {dimension} dimensions to a store "
                f"already using {self._dimensions} dimensions."
            )

        vectors = np.array([ec.vector for ec in embedded_chunks], dtype="float32")
        if vectors.shape[1] != self._dimensions:
            raise VectorStoreError(
                f"Expected vectors of {self._dimensions} dimensions, "
                f"got shape {vectors.shape}."
            )

        # Normalizing here (rather than trusting the caller to have done
        # it already) is what makes inner product behave like cosine
        # similarity — see the module docstring for why.
        faiss.normalize_L2(vectors)

        ids = np.arange(self._next_id, self._next_id + len(embedded_chunks), dtype="int64")
        self._index.add_with_ids(vectors, ids)

        for faiss_id, embedded_chunk in zip(ids, embedded_chunks):
            self._id_to_chunk[int(faiss_id)] = embedded_chunk.chunk

        self._next_id += len(embedded_chunks)

    def rebuild(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Wipe the store and rebuild it from scratch with `embedded_chunks`.

        This is the supported way to handle a repository whose files
        changed: rather than trying to surgically remove and re-add
        individual stale chunks (which FAISS's simplest index types
        don't support well, and which adds real bookkeeping complexity),
        we just start over with the latest full set of chunks. Simple,
        always correct, and — for a single-repo, prototype-scale tool —
        fast enough not to need anything cleverer yet. `IndexIDMap` is
        still used underneath so a smarter per-chunk update strategy can
        be added later without changing this class's shape again.
        """
        self.clear()
        self.add(embedded_chunks)

    def clear(self) -> None:
        """Remove everything from the store, resetting it to empty."""
        self._dimensions = None
        self._index = None
        self._id_to_chunk = {}
        self._next_id = 0

    # -- searching --------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """
        Return the `top_k` stored chunks whose vectors are most similar
        to `query_vector`, ranked by cosine similarity (highest first).

        Returns an empty list if the store has nothing in it yet —
        that's an expected state (e.g. before anything's been indexed),
        not an error.
        """
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k}.")

        if self._index is None or self.size == 0:
            return []

        if len(query_vector) != self._dimensions:
            raise VectorStoreError(
                f"Query vector has {len(query_vector)} dimensions, but this "
                f"store holds {self._dimensions}-dimension vectors."
            )

        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)

        # FAISS returns fewer than top_k matches if the store holds
        # fewer vectors than that; it pads missing slots with score -1
        # and id -1 rather than raising, so we filter those out below.
        k = min(top_k, self.size)
        scores, ids = self._index.search(query, k)

        results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:
                continue
            chunk = self._id_to_chunk.get(int(faiss_id))
            if chunk is None:
                # Defensive: should never happen if add()/load() kept the
                # index and the mapping in sync, but a missing mapping
                # entry is exactly the kind of silent corruption we don't
                # want to paper over with a made-up result.
                raise VectorStoreError(
                    f"No chunk metadata found for vector ID {faiss_id}; "
                    "the index and metadata may be out of sync."
                )
            results.append(SearchResult(chunk=chunk, score=float(score)))

        return results

    # -- persistence --------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Persist the store to `directory` as two files: `index.faiss` (the
        raw FAISS index) and `metadata.json` (the ID -> CodeChunk
        mapping). Saving only the FAISS index would not be enough — it
        has no idea a given vector ID represents `auth.py`'s `login`
        function; that relationship lives entirely in `metadata.json`.
        """
        if self._index is None:
            raise VectorStoreError("Cannot save an empty vector store (nothing has been added yet).")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self._index, str(path / "index.faiss"))
        except Exception as exc:
            raise VectorStoreError(f"Failed to write FAISS index to '{directory}': {exc}") from exc

        metadata = {
            "dimensions": self._dimensions,
            "next_id": self._next_id,
            "chunks": {
                str(faiss_id): chunk.to_dict()
                for faiss_id, chunk in self._id_to_chunk.items()
            },
        }
        try:
            (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        except OSError as exc:
            raise VectorStoreError(f"Failed to write metadata to '{directory}': {exc}") from exc

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a store previously saved with `.save(directory)`. Raises
        VectorStoreError if either file is missing or unreadable — a
        half-saved or corrupted store is a system-level failure, not
        something to silently paper over with an empty store.
        """
        path = Path(directory)
        index_path = path / "index.faiss"
        metadata_path = path / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(
                f"No saved vector store found at '{directory}' "
                "(expected index.faiss and metadata.json)."
            )

        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"Could not read FAISS index at '{index_path}': {exc}") from exc

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Could not read metadata at '{metadata_path}': {exc}") from exc

        store = cls()
        store._index = index
        store._dimensions = metadata["dimensions"]
        store._next_id = metadata["next_id"]
        store._id_to_chunk = {
            int(faiss_id): CodeChunk.from_dict(chunk_dict)
            for faiss_id, chunk_dict in metadata["chunks"].items()
        }
        return store
