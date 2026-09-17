from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from repopilot.exceptions import EmbeddingError
from repopilot.sentence_transformer_provider import SentenceTransformerProvider


def _mock_model(dimensions=3, encode_return=None):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimensions
    if encode_return is not None:
        model.encode.return_value = encode_return
    return model


# --- lazy loading -----------------------------------------------------------

@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_model_is_not_loaded_until_first_use(mock_st_class):
    SentenceTransformerProvider(model_name="fake-model")
    mock_st_class.assert_not_called()


@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_model_is_loaded_on_first_embed_call(mock_st_class):
    mock_st_class.return_value = _mock_model(encode_return=np.array([[0.1, 0.2, 0.3]]))
    provider = SentenceTransformerProvider(model_name="fake-model")

    provider.embed_texts(["some code"])

    mock_st_class.assert_called_once_with("fake-model")


@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_model_is_only_loaded_once_across_multiple_calls(mock_st_class):
    mock_st_class.return_value = _mock_model(encode_return=np.array([[0.1]]))
    provider = SentenceTransformerProvider(model_name="fake-model")

    provider.embed_texts(["a"])
    provider.embed_texts(["b"])

    mock_st_class.assert_called_once()


# --- basic embedding behavior ------------------------------------------------

@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_embed_texts_returns_plain_python_lists(mock_st_class):
    mock_st_class.return_value = _mock_model(encode_return=np.array([[0.1, 0.2], [0.3, 0.4]]))
    provider = SentenceTransformerProvider(model_name="fake-model")

    result = provider.embed_texts(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_empty_input_returns_empty_list_without_touching_the_model():
    provider = SentenceTransformerProvider(model_name="fake-model")
    assert provider.embed_texts([]) == []


@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_dimensions_reflects_the_real_model(mock_st_class):
    mock_st_class.return_value = _mock_model(dimensions=384)
    provider = SentenceTransformerProvider(model_name="fake-model")

    assert provider.dimensions == 384


def test_model_name_property():
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    assert provider.model_name == "all-MiniLM-L6-v2"


# --- error handling ---------------------------------------------------------

@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_model_load_failure_raises_embedding_error(mock_st_class):
    mock_st_class.side_effect = OSError("no internet, model not cached")
    provider = SentenceTransformerProvider(model_name="fake-model")

    with pytest.raises(EmbeddingError):
        provider.embed_texts(["some code"])


@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_encode_failure_raises_embedding_error(mock_st_class):
    model = _mock_model()
    model.encode.side_effect = RuntimeError("out of memory")
    mock_st_class.return_value = model
    provider = SentenceTransformerProvider(model_name="fake-model")

    with pytest.raises(EmbeddingError):
        provider.embed_texts(["some code"])


@patch("repopilot.sentence_transformer_provider.SentenceTransformer")
def test_mismatched_vector_count_raises_embedding_error(mock_st_class):
    # Simulate a misbehaving model returning 1 vector for 2 input texts.
    mock_st_class.return_value = _mock_model(encode_return=np.array([[0.1, 0.2]]))
    provider = SentenceTransformerProvider(model_name="fake-model")

    with pytest.raises(EmbeddingError):
        provider.embed_texts(["a", "b"])


@patch("repopilot.sentence_transformer_provider.SentenceTransformer", None)
@patch("repopilot.sentence_transformer_provider._SENTENCE_TRANSFORMERS_AVAILABLE", False)
def test_missing_dependency_raises_clear_embedding_error():
    provider = SentenceTransformerProvider(model_name="fake-model")

    with pytest.raises(EmbeddingError, match="not installed"):
        provider.embed_texts(["some code"])
