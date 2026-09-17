from unittest.mock import MagicMock, patch

import pytest

from repopilot.exceptions import RAGError
from repopilot.groq_provider import GroqProvider


def _mock_groq_response(text="a generated answer"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


# --- lazy client creation -------------------------------------------------

@patch("repopilot.groq_provider.Groq")
def test_client_is_not_created_until_first_use(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    GroqProvider(model_name="fake-model")
    mock_groq_class.assert_not_called()


@patch("repopilot.groq_provider.Groq")
def test_client_created_with_api_key_on_first_use(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-123")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response()
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="fake-model")
    provider.generate("system", "user")

    mock_groq_class.assert_called_once_with(api_key="fake-key-123")


@patch("repopilot.groq_provider.Groq")
def test_client_only_created_once_across_multiple_calls(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response()
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="fake-model")
    provider.generate("s", "u")
    provider.generate("s", "u")

    mock_groq_class.assert_called_once()


# --- missing API key --------------------------------------------------------

def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqProvider(model_name="fake-model")

    with pytest.raises(RAGError, match="GROQ_API_KEY"):
        provider.generate("system", "user")


# --- basic generation behavior ----------------------------------------------

@patch("repopilot.groq_provider.Groq")
def test_generate_returns_message_content(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response("Auth happens in login().")
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="fake-model")
    result = provider.generate("system prompt", "user prompt")

    assert result == "Auth happens in login()."


@patch("repopilot.groq_provider.Groq")
def test_generate_sends_correct_messages(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response()
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="my-model")
    provider.generate("SYSTEM TEXT", "USER TEXT")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "my-model"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]


def test_model_name_property():
    provider = GroqProvider(model_name="llama-3.3-70b-versatile")
    assert provider.model_name == "llama-3.3-70b-versatile"


# --- error handling ---------------------------------------------------------

@patch("repopilot.groq_provider.Groq")
def test_request_failure_raises_rag_error(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("network down")
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="fake-model")

    with pytest.raises(RAGError):
        provider.generate("system", "user")


@patch("repopilot.groq_provider.Groq")
def test_malformed_response_raises_rag_error(mock_groq_class, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    mock_client = MagicMock()
    bad_response = MagicMock()
    bad_response.choices = []  # empty — no message to read
    mock_client.chat.completions.create.return_value = bad_response
    mock_groq_class.return_value = mock_client

    provider = GroqProvider(model_name="fake-model")

    with pytest.raises(RAGError):
        provider.generate("system", "user")


@patch("repopilot.groq_provider.Groq", None)
@patch("repopilot.groq_provider._GROQ_AVAILABLE", False)
def test_missing_dependency_raises_clear_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    provider = GroqProvider(model_name="fake-model")

    with pytest.raises(RAGError, match="not installed"):
        provider.generate("system", "user")
