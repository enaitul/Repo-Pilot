"""
Embeds text locally using the `sentence-transformers` library — no API
key, no account, no per-call cost, and no code ever leaves the machine.

Chosen for Phase 5 specifically because it keeps the same "free, easy to
run, no external account needed" spirit as the rest of this project:
`git` is the only other external tool RepoPilot depends on, and this
provider similarly only needs one `pip install` plus a one-time model
download that's then cached on disk.

The model itself is loaded lazily (only on first use, not at import
time) so simply importing this module — or importing the package as a
whole — never triggers a slow model load or a network call. This
matters most for tests: test files can import this module freely
without accidentally downloading anything, since tests mock this
provider entirely rather than exercising the real model.
"""

from __future__ import annotations

from repopilot.config import EMBEDDING_MODEL_NAME
from repopilot.embedding_provider import EmbeddingProvider
from repopilot.exceptions import EmbeddingError

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider backed by a sentence-transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._model = None  # loaded lazily — see _get_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        # get_sentence_embedding_dimension() is a real method on a loaded
        # sentence-transformers model — asking the model directly (rather
        # than hardcoding a number here) means this stays correct even
        # if EMBEDDING_MODEL_NAME is changed to a model with a different
        # vector size.
        return self._get_model().get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        try:
            vectors = model.encode(texts, show_progress_bar=False)
        except Exception as exc:
            # Any failure here (out-of-memory, a corrupted cached model
            # file, an unexpected input type slipping through, etc.) is a
            # SYSTEM-level failure for this batch — there's no partial
            # result to salvage from one encode() call, so we surface it
            # clearly rather than returning something misleading.
            raise EmbeddingError(f"Embedding failed for a batch of {len(texts)} text(s): {exc}") from exc

        # .encode() returns a numpy array; convert to plain Python lists
        # so the rest of the pipeline (and JSON serialization later) never
        # needs to know or care that numpy was involved here.
        result = [vector.tolist() for vector in vectors]

        if len(result) != len(texts):
            # Defensive: should never happen with a well-behaved model,
            # but a provider returning a different count than it was
            # given is exactly the kind of "invalid response" that would
            # silently corrupt the chunk<->vector pairing downstream if
            # we didn't catch it here.
            raise EmbeddingError(
                f"Provider returned {len(result)} vectors for {len(texts)} input texts."
            )

        return result

    def _get_model(self):
        if self._model is None:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install it with "
                    "'pip install sentence-transformers' to use local embeddings."
                )
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                # Covers: no internet on first-ever run (the model hasn't
                # been downloaded and cached yet), a bad/unknown model
                # name, or a corrupted local cache.
                raise EmbeddingError(
                    f"Could not load embedding model '{self._model_name}': {exc}"
                ) from exc
        return self._model
