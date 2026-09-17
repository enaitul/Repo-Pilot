from unittest.mock import MagicMock

import pytest

from repopilot.exceptions import RAGError
from repopilot.llm_provider import LLMProvider
from repopilot.models import CodeChunk, SearchResult
from repopilot.rag_service import RAGService
from repopilot.search_service import SearchService


def _search_result(file_path="auth.py", symbol_name="login", content="def login(): pass", score=0.9):
    chunk = CodeChunk(
        chunk_id=f"{file_path}:1-2", file_path=file_path, language="Python",
        symbol_name=symbol_name, symbol_type="function", parent=None,
        start_line=1, end_line=2, content=content,
    )
    return SearchResult(chunk=chunk, score=score)


def _fake_search_service(results):
    service = MagicMock(spec=SearchService)
    service.search.return_value = results
    return service


def _fake_llm(return_value="a generated answer"):
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = return_value
    provider.model_name = "fake-llm"
    return provider


# --- basic flow -------------------------------------------------------

def test_answer_calls_search_with_question_and_top_k():
    search_service = _fake_search_service([_search_result()])
    llm = _fake_llm()

    RAGService(search_service, llm_provider=llm).answer("how does login work?", top_k=3)

    search_service.search.assert_called_once_with("how does login work?", top_k=3)


def test_answer_calls_llm_with_system_and_user_prompt():
    search_service = _fake_search_service([_search_result()])
    llm = _fake_llm()

    RAGService(search_service, llm_provider=llm).answer("how does login work?")

    llm.generate.assert_called_once()
    system_prompt, user_prompt = llm.generate.call_args[0]
    assert isinstance(system_prompt, str) and len(system_prompt) > 0
    assert "how does login work?" in user_prompt


def test_prompt_contains_retrieved_code():
    result = _search_result(content="def login(username, password):\n    return True")
    search_service = _fake_search_service([result])
    llm = _fake_llm()

    RAGService(search_service, llm_provider=llm).answer("how does login work?")

    _, user_prompt = llm.generate.call_args[0]
    assert "def login(username, password):" in user_prompt


def test_prompt_contains_file_path_and_symbol():
    result = _search_result(file_path="backend/auth.py", symbol_name="login")
    search_service = _fake_search_service([result])
    llm = _fake_llm()

    RAGService(search_service, llm_provider=llm).answer("how does login work?")

    _, user_prompt = llm.generate.call_args[0]
    assert "backend/auth.py" in user_prompt
    assert "login" in user_prompt


def test_response_contains_the_llm_answer():
    search_service = _fake_search_service([_search_result()])
    llm = _fake_llm(return_value="Authentication happens in login().")

    response = RAGService(search_service, llm_provider=llm).answer("how does login work?")

    assert response.answer == "Authentication happens in login()."


def test_sources_correspond_to_retrieved_chunks():
    result = _search_result(file_path="backend/auth.py", symbol_name="login", score=0.93)
    search_service = _fake_search_service([result])
    llm = _fake_llm()

    response = RAGService(search_service, llm_provider=llm).answer("how does login work?")

    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.file_path == "backend/auth.py"
    assert source.symbol_name == "login"
    assert source.score == 0.93


def test_response_retains_full_retrieval_results():
    result = _search_result()
    search_service = _fake_search_service([result])
    llm = _fake_llm()

    response = RAGService(search_service, llm_provider=llm).answer("how does login work?")

    assert response.retrieval_results == [result]


# --- no retrieval results — handled gracefully, not an error ------------

def test_no_retrieval_results_still_calls_llm_with_placeholder_context():
    search_service = _fake_search_service([])
    llm = _fake_llm()

    response = RAGService(search_service, llm_provider=llm).answer("does this use Redis?")

    _, user_prompt = llm.generate.call_args[0]
    assert "no relevant code" in user_prompt.lower()
    assert response.sources == []


# --- error handling ---------------------------------------------------------

def test_empty_question_raises_without_calling_search_or_llm():
    search_service = _fake_search_service([])
    llm = _fake_llm()

    with pytest.raises(RAGError):
        RAGService(search_service, llm_provider=llm).answer("   ")

    search_service.search.assert_not_called()
    llm.generate.assert_not_called()


def test_llm_failure_propagates_as_rag_error():
    search_service = _fake_search_service([_search_result()])
    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = RAGError("Groq request failed: network error")

    with pytest.raises(RAGError):
        RAGService(search_service, llm_provider=llm).answer("how does login work?")


def test_search_failure_propagates():
    search_service = MagicMock(spec=SearchService)
    search_service.search.side_effect = RAGError("vector store not found")
    llm = _fake_llm()

    with pytest.raises(RAGError):
        RAGService(search_service, llm_provider=llm).answer("how does login work?")


# --- provider abstraction ---------------------------------------------------

def test_rag_service_works_with_any_llm_provider_satisfying_the_interface():
    class TinyFakeLLM(LLMProvider):
        model_name = "tiny-fake"

        def generate(self, system_prompt, user_prompt):
            return f"echo: {user_prompt[:20]}"

    search_service = _fake_search_service([_search_result()])
    response = RAGService(search_service, llm_provider=TinyFakeLLM()).answer("how does login work?")

    assert response.answer.startswith("echo:")


def test_default_llm_provider_is_groq_when_none_given():
    from repopilot.groq_provider import GroqProvider

    search_service = _fake_search_service([])
    service = RAGService(search_service)

    assert isinstance(service._llm_provider, GroqProvider)
