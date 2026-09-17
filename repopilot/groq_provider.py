"""
Generates answers using Groq's free-tier API, which serves fast
open-weight models (Llama, Mixtral, etc.) on custom low-latency
hardware.

Chosen for Phase 7 because it requires no credit card for its free
tier, and its inference speed makes a live RAG demo feel close to
instant — a real, measurable trade-off, not just a brand preference.
Unlike Phase 5's local embedding model, this DOES require an internet
connection and an API key: there is no equivalent free, high-quality,
fully local LLM option that matches Groq's ease of setup and speed for
this project's purposes, so this is a deliberate, informed departure
from the "fully offline" approach used for embeddings.

SECURITY NOTE: the API key is read from an environment variable, never
hardcoded, logged, or printed — matching the project's existing secrets
handling. Because this provider sends prompt text (which includes
retrieved source code) to a third-party API, RAGService's context stays
limited to only the retrieved Top-K chunks (not the whole repository) —
both for relevance and to avoid sending more of a person's codebase to
an external service than necessary for the question being asked.
"""

from __future__ import annotations

import os

from repopilot.config import GROQ_MODEL_NAME
from repopilot.exceptions import RAGError
from repopilot.llm_provider import LLMProvider

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore[assignment,misc]
    _GROQ_AVAILABLE = False


class GroqProvider(LLMProvider):
    """LLM provider backed by Groq's free-tier chat completion API."""

    def __init__(self, model_name: str = GROQ_MODEL_NAME, api_key: str | None = None):
        self._model_name = model_name
        # Request-scoped BYOK support. Never serialized, logged, or persisted.
        self._api_key = api_key
        self._client = None  # created lazily — see _get_client()

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            # Covers network failures, invalid/expired API keys, rate
            # limits, and any other request-level failure. This is a
            # SYSTEM-level failure — there's no partial answer to
            # salvage from a failed request.
            raise RAGError(f"Groq request failed: {exc}") from exc

        try:
            return response.choices[0].message.content
        except (IndexError, AttributeError, TypeError) as exc:
            # Defensive: a response that doesn't have the expected shape
            # (e.g. an empty choices list) is an invalid response we
            # should surface clearly rather than crash on obscurely.
            raise RAGError(f"Unexpected response shape from Groq: {exc}") from exc

    def _get_client(self):
        if self._client is None:
            if not _GROQ_AVAILABLE:
                raise RAGError(
                    "The 'groq' package is not installed. Install it with "
                    "'pip install groq' to use GroqProvider."
                )
            api_key = self._api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RAGError(
                    "GROQ_API_KEY environment variable is not set. Get a free "
                    "API key at https://console.groq.com/keys and set it "
                    "(e.g. `export GROQ_API_KEY=...`) before using GroqProvider."
                )
            try:
                self._client = Groq(api_key=api_key)
            except Exception as exc:
                raise RAGError(f"Could not initialize the Groq client: {exc}") from exc
        return self._client
