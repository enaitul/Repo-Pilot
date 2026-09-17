"""
Common interface every LLM provider implements.

Same role in Phase 7 as embedding_provider.py plays in Phase 5:
RAGService depends on this one small contract, not on any specific LLM
or API. Today the only implementation is GroqProvider (free-tier,
serves fast open-weight models). Swapping to a different provider later
(Gemini, OpenAI, a local Ollama server, etc.) means writing one new
class that satisfies this interface — RAGService itself never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Generates a text answer given a system prompt and a user prompt."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """A short identifier for the underlying model, e.g. 'llama-3.3-70b-versatile'."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response given `system_prompt` (behavioral
        instructions for the whole request) and `user_prompt` (the
        actual question, plus any retrieved context it references).

        Implementations MUST raise RAGError (not let a raw
        provider-specific exception escape) if the request fails for
        any reason — network issues, an invalid/missing API key, a rate
        limit, or a response that doesn't contain usable text.
        """
        raise NotImplementedError
