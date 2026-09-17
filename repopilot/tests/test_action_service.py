import json
from unittest.mock import MagicMock

import pytest

from repopilot.action_context_builder import ActionContextBuilder
from repopilot.action_service import ActionService
from repopilot.exceptions import ActionError
from repopilot.llm_provider import LLMProvider
from repopilot.models import ActionRequest, ActionType, CodeChunk, SearchResult


def _search_result(file_path="auth.py", symbol_name="login", score=0.9):
    chunk = CodeChunk(
        chunk_id=f"{file_path}:1-2", file_path=file_path, language="Python",
        symbol_name=symbol_name, symbol_type="function", parent=None,
        start_line=1, end_line=2, content="def login(): pass",
    )
    return SearchResult(chunk=chunk, score=score)


def _fake_context_builder(context_text="some context", search_results=None):
    builder = MagicMock(spec=ActionContextBuilder)
    builder.build.return_value = (context_text, search_results or [])
    return builder


def _fake_llm(json_payload: dict):
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = json.dumps(json_payload)
    provider.model_name = "fake-llm"
    return provider


_VALID_PAYLOAD = {
    "summary": "Login verifies credentials against the user store.",
    "generated_content": "def test_login():\n    assert login('a', 'b') is True",
    "findings": [
        {"category": "fact", "description": "login() calls find_user()", "affected_file": "auth.py", "affected_symbol": "login"}
    ],
    "assumptions": ["Assumed pytest since no test framework was visible."],
    "warnings": [],
}


# --- basic flow -------------------------------------------------------

def test_execute_returns_parsed_action_result():
    request = ActionRequest(action_type=ActionType.TEST_GENERATION, request="write tests for login", target_symbol="login")
    context_builder = _fake_context_builder(search_results=[_search_result()])
    llm = _fake_llm(_VALID_PAYLOAD)

    result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert result.action_type == "test_generation"
    assert result.summary == _VALID_PAYLOAD["summary"]
    assert result.generated_content == _VALID_PAYLOAD["generated_content"]
    assert len(result.findings) == 1
    assert result.findings[0].category == "fact"


def test_routes_to_the_correct_prompt_builder_per_action_type():
    for action_type in ActionType:
        request = ActionRequest(action_type=action_type, request="do the thing")
        context_builder = _fake_context_builder()
        llm = _fake_llm(_VALID_PAYLOAD)

        result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

        assert result.action_type == action_type.value
        llm.generate.assert_called_once()


def test_sources_are_built_from_search_results_not_llm_output():
    # The LLM's JSON says nothing about sources at all — sources must
    # still be populated correctly from real retrieval metadata.
    request = ActionRequest(action_type=ActionType.REFACTORING, request="improve this class")
    result_1 = _search_result(file_path="auth.py", symbol_name="login", score=0.95)
    context_builder = _fake_context_builder(search_results=[result_1])
    llm = _fake_llm(_VALID_PAYLOAD)

    result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert len(result.sources) == 1
    assert result.sources[0].file_path == "auth.py"
    assert result.sources[0].symbol_name == "login"
    assert result.sources[0].score == 0.95


def test_change_plan_has_no_sources_when_context_builder_returns_none():
    request = ActionRequest(action_type=ActionType.CHANGE_PLAN, request="add JWT auth")
    context_builder = _fake_context_builder(search_results=[])  # CHANGE_PLAN never has per-chunk results
    llm = _fake_llm(_VALID_PAYLOAD)

    result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert result.sources == []


# --- JSON parsing, including malformed output -----------------------------

def test_tolerates_markdown_fenced_json():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="document this")
    context_builder = _fake_context_builder()
    llm = MagicMock(spec=LLMProvider)
    llm.generate.return_value = "```json\n" + json.dumps(_VALID_PAYLOAD) + "\n```"

    result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert result.summary == _VALID_PAYLOAD["summary"]


def test_malformed_json_raises_action_error():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="document this")
    context_builder = _fake_context_builder()
    llm = MagicMock(spec=LLMProvider)
    llm.generate.return_value = "this is not JSON at all {{{"

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(request)


def test_json_array_instead_of_object_raises_action_error():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="document this")
    context_builder = _fake_context_builder()
    llm = MagicMock(spec=LLMProvider)
    llm.generate.return_value = json.dumps(["not", "an", "object"])

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(request)


def test_missing_optional_keys_default_safely():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="document this")
    context_builder = _fake_context_builder()
    llm = _fake_llm({"summary": "A minimal response."})  # no generated_content/findings/assumptions/warnings

    result = ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert result.summary == "A minimal response."
    assert result.generated_content is None
    assert result.findings == []
    assert result.assumptions == []
    assert result.warnings == []


# --- error handling ---------------------------------------------------------

def test_none_request_raises():
    context_builder = _fake_context_builder()
    llm = _fake_llm(_VALID_PAYLOAD)

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(None)


def test_empty_request_text_raises():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="   ")
    context_builder = _fake_context_builder()
    llm = _fake_llm(_VALID_PAYLOAD)

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(request)


def test_missing_target_propagates_as_action_error():
    # Context builder itself raises when required context isn't available
    # (e.g. CHANGE_PLAN with no RepositoryModel) — ActionService must not
    # swallow that.
    request = ActionRequest(action_type=ActionType.CHANGE_PLAN, request="add JWT auth")
    context_builder = MagicMock(spec=ActionContextBuilder)
    context_builder.build.side_effect = ActionError("CHANGE_PLAN requires a RepositoryModel.")
    llm = _fake_llm(_VALID_PAYLOAD)

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(request)


def test_llm_failure_becomes_action_error():
    request = ActionRequest(action_type=ActionType.BUG_ANALYSIS, request="null pointer error")
    context_builder = _fake_context_builder()
    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = RuntimeError("network exploded")

    with pytest.raises(ActionError):
        ActionService(context_builder=context_builder, llm_provider=llm).execute(request)


# --- read-only guarantee (no filesystem/shell access anywhere) ------------

def test_action_service_never_touches_the_filesystem(tmp_path, monkeypatch):
    # A structural guarantee: executing any action must not create,
    # modify, or delete any file on disk.
    marker_file = tmp_path / "untouched.txt"
    marker_file.write_text("original")

    request = ActionRequest(action_type=ActionType.TEST_GENERATION, request="write tests for login")
    context_builder = _fake_context_builder(search_results=[_search_result()])
    llm = _fake_llm(_VALID_PAYLOAD)

    ActionService(context_builder=context_builder, llm_provider=llm).execute(request)

    assert marker_file.read_text() == "original"
