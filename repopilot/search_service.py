"""
Orchestrates Phase 6's semantic search: turns a plain-English question
into a query vector, then asks a VectorStore for the closest matches.

Mirrors the shape of every previous phase's *_service.py: this is the
only module that knows the end-to-end SEARCH pipeline order (embed the
query -> search the vector store -> return ranked results). It doesn't
know how FAISS works internally (that's VectorStore's job) and doesn't
know how the embedding model works internally (that's EmbeddingProvider's
job, unchanged from Phase 5) — it just wires the two together.

Deliberately reuses Phase 5's EmbeddingProvider AS-IS for embedding the
query text, rather than writing new embedding logic: the query and the
stored code chunks must be embedded by the exact same model, or their
vectors are not comparable to each other at all (see vector_store.py's
docstring for why cosine similarity specifically requires this).
"""

from __future__ import annotations

from repopilot.config import DEFAULT_TOP_K
from repopilot.embedding_provider import EmbeddingProvider
from repopilot.exceptions import VectorStoreError
from repopilot.models import SearchResult
from repopilot.sentence_transformer_provider import SentenceTransformerProvider
from repopilot.vector_store import VectorStore


class SearchService:
    """High-level entry point for Phase 6: semantic search over a VectorStore."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        # The vector store is REQUIRED (there's nothing meaningful to
        # search without one), while the embedding provider defaults to
        # the same local, free model used in Phase 5 — constructor
        # injection here again, so tests can substitute a fake provider
        # and a fake store without touching a real model or real FAISS.
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider or SentenceTransformerProvider()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """
        Return the `top_k` CodeChunks most semantically similar to
        `query`, ranked by similarity (most similar first).

        Returns an empty list if the underlying store has nothing
        indexed yet — an expected state, not an error. Raises
        VectorStoreError for a blank query or an invalid `top_k`, since
        those are caller mistakes rather than "no relevant code found."
        """
        if not query or not query.strip():
            raise VectorStoreError("Search query must not be empty.")
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k}.")

        query_vectors = self._embedding_provider.embed_texts([query])
        query_vector = query_vectors[0]

        return self._vector_store.search(query_vector, top_k)
