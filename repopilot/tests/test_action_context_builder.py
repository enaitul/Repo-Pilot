from unittest.mock import MagicMock

import pytest

from repopilot.action_context_builder import ActionContextBuilder
from repopilot.exceptions import ActionError
from repopilot.models import ActionRequest, ActionType, CodeChunk, FileMetadata, RepositoryModel, SearchResult
from repopilot.search_service import SearchService


def _search_result(file_path="auth.py", symbol_name="login"):
    chunk = CodeChunk(
        chunk_id=f"{file_path}:1-2", file_path=file_path, language="Python",
        symbol_name=symbol_name, symbol_type="function", parent=None,
        start_line=1, end_line=2, content="def login(): pass",
    )
    return SearchResult(chunk=chunk, score=0.9)


def _repo_model():
    return RepositoryModel(
        repo_url="https://github.com/fake/repo", local_path="/tmp/fake",
        files=[FileMetadata(path="main.py", language="Python", size_bytes=10, imports=[])],
    )


# --- retrieval-based actions --------------------------------------------

def test_test_generation_uses_search_service():
    search_service = MagicMock(spec=SearchService)
    search_service.search.return_value = [_search_result()]
    request = ActionRequest(action_type=ActionType.TEST_GENERATION, request="write tests for login", target_symbol="login")

    context_text, results = ActionContextBuilder(search_service=search_service).build(request, None, top_k=5)

    assert "login" in context_text
    assert len(results) == 1


def test_retrieval_query_includes_target_symbol_and_error_message():
    search_service = MagicMock(spec=SearchService)
    search_service.search.return_value = []
    request = ActionRequest(
        action_type=ActionType.BUG_ANALYSIS, request="null pointer error",
        target_symbol="login", error_message="NoneType has no attribute",
    )

    ActionContextBuilder(search_service=search_service).build(request, None, top_k=5)

    query = search_service.search.call_args[0][0]
    assert "login" in query
    assert "NoneType has no attribute" in query


def test_missing_search_service_raises_for_retrieval_based_actions():
    request = ActionRequest(action_type=ActionType.DOCUMENTATION, request="document this module")

    with pytest.raises(ActionError):
        ActionContextBuilder(search_service=None).build(request, None, top_k=5)


def test_empty_search_results_still_returns_valid_context():
    search_service = MagicMock(spec=SearchService)
    search_service.search.return_value = []
    request = ActionRequest(action_type=ActionType.REFACTORING, request="improve this class")

    context_text, results = ActionContextBuilder(search_service=search_service).build(request, None, top_k=5)

    assert "no relevant code" in context_text.lower()
    assert results == []


# --- CHANGE_PLAN uses RepositoryOverview instead of retrieval ------------

def test_change_plan_uses_repository_overview_not_search():
    search_service = MagicMock(spec=SearchService)
    request = ActionRequest(action_type=ActionType.CHANGE_PLAN, request="add JWT authentication")

    context_text, results = ActionContextBuilder(search_service=search_service).build(request, _repo_model(), top_k=5)

    assert "main.py" in context_text
    assert results == []
    search_service.search.assert_not_called()


def test_change_plan_without_repo_model_raises():
    request = ActionRequest(action_type=ActionType.CHANGE_PLAN, request="add JWT authentication")

    with pytest.raises(ActionError):
        ActionContextBuilder().build(request, None, top_k=5)
