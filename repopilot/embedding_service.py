"""
Orchestrates Phase 5 (embeddings) over an already-chunked repository.

Mirrors the shape of parsing_service.py and chunking_service.py: this is
the only module that knows the end-to-end EMBEDDING pipeline order
(build the text to embed for each chunk -> group into batches -> ask the
provider for vectors -> pair each vector back with its original chunk).
It does not know or care which model/API is actually behind the vectors
— that's EmbeddingProvider's job.

INPUT:  list[CodeChunk]        (Phase 4's output)
OUTPUT: list[EmbeddedChunk]    (one per input chunk, same order)
"""

from __future__ import annotations

from repopilot.config import EMBEDDING_BATCH_SIZE
from repopilot.embedding_provider import EmbeddingProvider
from repopilot.exceptions import EmbeddingError
from repopilot.models import CodeChunk, EmbeddedChunk
from repopilot.sentence_transformer_provider import SentenceTransformerProvider


class EmbeddingService:
    """High-level entry point for Phase 5: embedding a repository's code chunks."""

    def __init__(self, provider: EmbeddingProvider | None = None):
        # Constructor injection, same pattern as every earlier service —
        # lets tests substitute a fake provider so no real model is ever
        # loaded or called during automated tests.
        self._provider = provider or SentenceTransformerProvider()

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
        """
        Embed every chunk in `chunks`, returning one EmbeddedChunk per
        input chunk, in the same order.

        Chunks are grouped into batches of EMBEDDING_BATCH_SIZE before
        being sent to the provider — one embedding call per batch rather
        than one per chunk, which matters a lot once a repo has hundreds
        or thousands of chunks. If one batch's provider call fails, that
        raises EmbeddingError immediately: an embedding call is a single
        all-or-nothing operation, so there's no meaningful partial
        result to keep going with for that specific batch. Earlier
        successful batches' results are simply not returned in that
        case — callers who want "best effort across the whole repo" can
        catch EmbeddingError per smaller sub-list of chunks themselves.
        """
        if not chunks:
            return []

        embedded: list[EmbeddedChunk] = []
        for batch in self._batches(chunks, EMBEDDING_BATCH_SIZE):
            embedded.extend(self._embed_batch(batch))
        return embedded

    # -- internal helpers ---------------------------------------------------

    def _embed_batch(self, batch: list[CodeChunk]) -> list[EmbeddedChunk]:
        texts = [self._build_embedding_text(chunk) for chunk in batch]
        vectors = self._provider.embed_texts(texts)

        if len(vectors) != len(batch):
            # Defensive, mirrors the same check inside the provider — if
            # a future provider implementation forgets this invariant,
            # we still refuse to silently pair chunks with the wrong
            # vectors rather than trusting the count blindly.
            raise EmbeddingError(
                f"Expected {len(batch)} vectors for this batch, got {len(vectors)}."
            )

        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
                model_name=self._provider.model_name,
                dimensions=self._provider.dimensions,
            )
            for chunk, vector in zip(batch, vectors)
        ]

    @staticmethod
    def _build_embedding_text(chunk: CodeChunk) -> str:
        """
        Build the text actually sent to the embedding model for one
        chunk: a small metadata header, then the code itself.

        We do NOT embed the raw code alone — a bare `def login(...)` with
        no context embeds almost identically to an unrelated `login`
        function in a completely different file. Prepending the file
        path and symbol name gives the model useful disambiguating
        context. This never modifies chunk.content itself — the header
        is only built here, temporarily, for the embedding call.
        """
        symbol_line = chunk.symbol_name or "(module-level code)"
        if chunk.parent:
            symbol_line = f"{chunk.parent}.{symbol_line}"

        header = (
            f"File: {chunk.file_path}\n"
            f"Symbol: {symbol_line}\n"
            f"Language: {chunk.language}\n"
        )
        return f"{header}\n{chunk.content}"

    @staticmethod
    def _batches(items: list[CodeChunk], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start:start + batch_size]
