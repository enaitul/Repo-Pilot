"""
Common interface every embedding provider implements.

Same role in Phase 5 as parser_base.py plays in Phase 3: EmbeddingService
depends on this one small contract, not on any specific model or API.
Today the only implementation is SentenceTransformerProvider (a local,
free, offline model). Swapping to a hosted API later (OpenAI, Gemini,
etc.) or adding a second option means writing one new class that
satisfies this interface — EmbeddingService itself would not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Turns a batch of text strings into a batch of embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A short identifier for the underlying model, e.g. 'all-MiniLM-L6-v2'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        How many numbers are in each vector this provider produces.

        Callers (and eventually Phase 6, when configuring a vector
        database) need this up front — every vector from one provider
        must have the same length, and that length is part of what
        determines how the vectors can be stored/compared later.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts, returning one vector per input text, in
        the SAME ORDER as `texts`. The returned list must have exactly
        `len(texts)` vectors, and every vector must have `self.dimensions`
        numbers in it.

        Implementations MUST raise EmbeddingError (not let a raw
        provider-specific exception escape) if the underlying model/API
        call fails for any reason — network issues, invalid input, a
        missing model file, etc. A malformed individual text is expected
        input here, not a bug, but a provider that can't produce a
        result at all for the batch has genuinely failed and callers
        need to know that clearly.
        """
        raise NotImplementedError
