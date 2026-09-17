from unittest.mock import MagicMock

import pytest

from repopilot.exceptions import RepositoryIntelligenceError
from repopilot.llm_provider import LLMProvider
from repopilot.models import FileMetadata, RepositoryModel
from repopilot.repository_intelligence_service import RepositoryIntelligenceService


def _repo(files, repo_url="https://github.com/fake/repo"):
    return RepositoryModel(repo_url=repo_url, local_path="/tmp/fake", files=files)


def _file(path, imports=None, line_count=10):
    return FileMetadata(path=path, language="Python", size_bytes=100, imports=imports or [], line_count=line_count)


def _fake_llm(return_value="This repo has a simple structure."):
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = return_value
    provider.model_name = "fake-llm"
    return provider


# --- basic flow -------------------------------------------------------

def test_analyze_calls_llm_and_returns_answer():
    repo = _repo([_file("main.py")])
    llm = _fake_llm("A simple single-file Python script.")

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert analysis.answer == "A simple single-file Python script."


def test_prompt_uses_architecture_specific_system_prompt():
    repo = _repo([_file("main.py", imports=["auth"]), _file("auth.py")])
    llm = _fake_llm()

    RepositoryIntelligenceService(llm_provider=llm).analyze(repo, question="Explain the architecture.")

    system_prompt, user_prompt = llm.generate.call_args[0]
    assert "not invent" in system_prompt.lower()
    assert "call graph" in system_prompt.lower()
    assert "Explain the architecture." in user_prompt


def test_prompt_contains_deterministic_facts_not_raw_source():
    repo = _repo([_file("main.py", imports=["auth"]), _file("auth.py")])
    llm = _fake_llm()

    RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    _, user_prompt = llm.generate.call_args[0]
    assert "main.py -> auth.py" in user_prompt


def test_default_question_used_when_none_given():
    repo = _repo([_file("main.py")])
    llm = _fake_llm()

    RepositoryIntelligenceService(llm_provider=llm).analyze(repo, question=None)

    _, user_prompt = llm.generate.call_args[0]
    assert "architecture" in user_prompt.lower()


# --- deterministic fields are correct, not LLM-controlled ------------------

def test_technologies_come_from_deterministic_detection_not_llm():
    repo = _repo([_file("requirements.txt")])
    llm = _fake_llm("I have no idea what technologies this uses.")  # LLM says nothing useful

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert any(t.name == "Python (pip)" for t in analysis.technologies)


def test_important_files_and_entry_points_are_deterministic():
    repo = _repo([_file("main.py"), _file("utils.py")])
    llm = _fake_llm()

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert "main.py" in analysis.important_files
    assert "main.py" in analysis.entry_points
    assert "utils.py" not in analysis.important_files


def test_dependency_graph_is_included_in_the_response():
    repo = _repo([_file("main.py", imports=["auth"]), _file("auth.py")])
    llm = _fake_llm()

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert any(e.source == "main.py" and e.target == "auth.py" for e in analysis.dependency_graph.edges)


def test_sources_reference_real_important_files_with_line_counts():
    repo = _repo([_file("main.py", line_count=42)])
    llm = _fake_llm()

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert len(analysis.sources) == 1
    source = analysis.sources[0]
    assert source.file_path == "main.py"
    assert source.end_line == 42


# --- empty / edge cases -------------------------------------------------

def test_empty_repository_is_handled_gracefully_not_an_error():
    repo = _repo([])
    llm = _fake_llm("There isn't enough information to describe an architecture.")

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert analysis.technologies == []
    assert analysis.important_files == []
    assert analysis.dependency_graph.nodes == []
    llm.generate.assert_called_once()  # still asked, with honest "nothing detected" facts


def test_single_file_repository_works():
    repo = _repo([_file("script.py")])
    llm = _fake_llm()

    analysis = RepositoryIntelligenceService(llm_provider=llm).analyze(repo)

    assert "script.py" in analysis.dependency_graph.nodes


def test_none_repo_model_raises():
    llm = _fake_llm()
    with pytest.raises(RepositoryIntelligenceError):
        RepositoryIntelligenceService(llm_provider=llm).analyze(None)


def test_llm_failure_becomes_repository_intelligence_error():
    repo = _repo([_file("main.py")])
    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = RuntimeError("network exploded")

    with pytest.raises(RepositoryIntelligenceError):
        RepositoryIntelligenceService(llm_provider=llm).analyze(repo)


def test_existing_repopilot_error_from_llm_propagates_unchanged():
    from repopilot.exceptions import RAGError  # any RepoPilotError subtype

    repo = _repo([_file("main.py")])
    llm = MagicMock(spec=LLMProvider)
    llm.generate.side_effect = RAGError("some upstream RAG-layer error")

    with pytest.raises(RepositoryIntelligenceError):
        RepositoryIntelligenceService(llm_provider=llm).analyze(repo)
