from unittest.mock import MagicMock

import pytest

from repopilot.embedding_provider import EmbeddingProvider
from repopilot.exceptions import VectorStoreError
from repopilot.models import CodeChunk, SearchResult
from repopilot.search_service import SearchService
from repopilot.vector_store import VectorStore


def _fake_provider(query_vector):
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_texts.return_value = [query_vector]
    return provider


def _fake_result():
    chunk = CodeChunk(
        chunk_id="auth.py:1-2", file_path="auth.py", language="Python",
        symbol_name="login", symbol_type="function", parent=None,
        start_line=1, end_line=2, content="def login(): pass",
    )
    return SearchResult(chunk=chunk, score=0.97)


def test_search_embeds_query_with_the_same_provider_used_for_chunks():
    provider = _fake_provider([1.0, 0.0, 0.0])
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []

    SearchService(vector_store=store, embedding_provider=provider).search("where is login")

    provider.embed_texts.assert_called_once_with(["where is login"])


def test_search_passes_query_vector_and_top_k_to_store():
    provider = _fake_provider([1.0, 0.0, 0.0])
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []

    SearchService(vector_store=store, embedding_provider=provider).search("q", top_k=3)

    store.search.assert_called_once_with([1.0, 0.0, 0.0], 3)


def test_search_returns_results_from_store():
    provider = _fake_provider([1.0, 0.0, 0.0])
    store = MagicMock(spec=VectorStore)
    expected = [_fake_result()]
    store.search.return_value = expected

    results = SearchService(vector_store=store, embedding_provider=provider).search("where is login")

    assert results == expected
    assert results[0].chunk.symbol_name == "login"


def test_empty_query_raises():
    store = MagicMock(spec=VectorStore)
    provider = _fake_provider([1.0, 0.0, 0.0])
    service = SearchService(vector_store=store, embedding_provider=provider)

    with pytest.raises(VectorStoreError):
        service.search("   ")


def test_invalid_top_k_raises():
    store = MagicMock(spec=VectorStore)
    provider = _fake_provider([1.0, 0.0, 0.0])
    service = SearchService(vector_store=store, embedding_provider=provider)

    with pytest.raises(VectorStoreError):
        service.search("valid query", top_k=0)


def test_empty_store_returns_empty_results():
    # An empty store is a normal, expected state — not an error.
    provider = _fake_provider([1.0, 0.0, 0.0])
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []

    results = SearchService(vector_store=store, embedding_provider=provider).search("anything")

    assert results == []


def test_default_top_k_is_used_when_not_specified():
    from repopilot.config import DEFAULT_TOP_K

    provider = _fake_provider([1.0, 0.0, 0.0])
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []

    SearchService(vector_store=store, embedding_provider=provider).search("q")

    store.search.assert_called_once_with([1.0, 0.0, 0.0], DEFAULT_TOP_K)
