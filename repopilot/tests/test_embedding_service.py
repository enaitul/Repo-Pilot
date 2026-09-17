from unittest.mock import MagicMock

import pytest

from repopilot.embedding_provider import EmbeddingProvider
from repopilot.embedding_service import EmbeddingService
from repopilot.exceptions import EmbeddingError
from repopilot.models import CodeChunk


def _chunk(chunk_id="a.py:1-2", symbol_name="greet", parent=None, content="def greet():\n    pass"):
    return CodeChunk(
        chunk_id=chunk_id,
        file_path="a.py",
        language="Python",
        symbol_name=symbol_name,
        symbol_type="function",
        parent=parent,
        start_line=1,
        end_line=2,
        content=content,
    )


def _fake_provider(vectors=None, dimensions=3, model_name="fake-model"):
    """A MagicMock standing in for a real EmbeddingProvider — no real
    model is ever loaded or called in these tests."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.model_name = model_name
    provider.dimensions = dimensions
    if vectors is not None:
        provider.embed_texts.return_value = vectors
    return provider


# --- basic behavior ----------------------------------------------------

def test_empty_input_returns_empty_list():
    service = EmbeddingService(provider=_fake_provider())
    assert service.embed_chunks([]) == []


def test_one_chunk_produces_one_embedding():
    provider = _fake_provider(vectors=[[0.1, 0.2, 0.3]])
    service = EmbeddingService(provider=provider)

    result = service.embed_chunks([_chunk()])

    assert len(result) == 1
    assert result[0].vector == [0.1, 0.2, 0.3]
    assert result[0].model_name == "fake-model"
    assert result[0].dimensions == 3


def test_multiple_chunks_produce_multiple_embeddings():
    chunks = [_chunk(chunk_id=f"a.py:{i}-{i+1}") for i in range(5)]
    vectors = [[float(i)] for i in range(5)]
    provider = _fake_provider(vectors=vectors)
    service = EmbeddingService(provider=provider)

    result = service.embed_chunks(chunks)

    assert len(result) == 5
    assert [ec.vector for ec in result] == vectors


def test_vectors_are_paired_with_the_correct_chunk_in_order():
    chunk_a = _chunk(chunk_id="a.py:1-2", symbol_name="a")
    chunk_b = _chunk(chunk_id="a.py:3-4", symbol_name="b")
    provider = _fake_provider(vectors=[[1.0], [2.0]])
    service = EmbeddingService(provider=provider)

    result = service.embed_chunks([chunk_a, chunk_b])

    assert result[0].chunk.symbol_name == "a"
    assert result[0].vector == [1.0]
    assert result[1].chunk.symbol_name == "b"
    assert result[1].vector == [2.0]


def test_original_source_text_is_preserved():
    chunk = _chunk(content="def greet():\n    print('hi')\n")
    provider = _fake_provider(vectors=[[0.1]])
    service = EmbeddingService(provider=provider)

    result = service.embed_chunks([chunk])

    assert result[0].chunk.content == "def greet():\n    print('hi')\n"


def test_chunk_metadata_is_preserved():
    chunk = _chunk(symbol_name="login", parent="AuthService")
    provider = _fake_provider(vectors=[[0.1]])
    service = EmbeddingService(provider=provider)

    result = service.embed_chunks([chunk])

    assert result[0].chunk.symbol_name == "login"
    assert result[0].chunk.parent == "AuthService"
    assert result[0].chunk.file_path == "a.py"


# --- text built for embedding -------------------------------------------

def test_embedding_text_includes_file_and_symbol_context():
    chunk = _chunk(symbol_name="login", parent="AuthService", content="def login():\n    pass")
    provider = _fake_provider(vectors=[[0.1]])
    service = EmbeddingService(provider=provider)

    service.embed_chunks([chunk])

    sent_texts = provider.embed_texts.call_args[0][0]
    assert len(sent_texts) == 1
    assert "AuthService.login" in sent_texts[0]
    assert "a.py" in sent_texts[0]
    assert "def login():" in sent_texts[0]


# --- batching -------------------------------------------------------------

def test_batching_splits_chunks_into_configured_batch_size(monkeypatch):
    import repopilot.embedding_service as embedding_service_module
    monkeypatch.setattr(embedding_service_module, "EMBEDDING_BATCH_SIZE", 2)

    chunks = [_chunk(chunk_id=f"a.py:{i}-{i+1}") for i in range(5)]
    provider = _fake_provider()
    # Each call returns as many vectors as texts it was given.
    provider.embed_texts.side_effect = lambda texts: [[0.0] for _ in texts]

    service = EmbeddingService(provider=provider)
    result = service.embed_chunks(chunks)

    assert len(result) == 5
    # 5 chunks at batch size 2 -> batches of 2, 2, 1 -> 3 calls
    assert provider.embed_texts.call_count == 3
    call_sizes = [len(call.args[0]) for call in provider.embed_texts.call_args_list]
    assert call_sizes == [2, 2, 1]


# --- error handling ---------------------------------------------------------

def test_provider_error_propagates_as_embedding_error():
    provider = _fake_provider()
    provider.embed_texts.side_effect = EmbeddingError("model exploded")

    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingError):
        service.embed_chunks([_chunk()])


def test_mismatched_vector_count_raises_embedding_error():
    # Provider claims success but returns the wrong number of vectors —
    # must be caught rather than silently mis-pairing chunks and vectors.
    provider = _fake_provider(vectors=[[0.1], [0.2]])  # 2 vectors...
    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingError):
        service.embed_chunks([_chunk()])  # ...for 1 chunk


# --- provider abstraction ---------------------------------------------------

def test_service_works_with_any_provider_satisfying_the_interface():
    # Demonstrates the abstraction: a totally different fake provider
    # (not sentence-transformers at all) works with zero changes to
    # EmbeddingService.
    class TinyFakeProvider(EmbeddingProvider):
        model_name = "tiny-fake"
        dimensions = 2

        def embed_texts(self, texts):
            return [[len(t), 0.0] for t in texts]

    service = EmbeddingService(provider=TinyFakeProvider())
    result = service.embed_chunks([_chunk(content="x")])

    assert result[0].model_name == "tiny-fake"
    assert result[0].dimensions == 2
