import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repopilot.exceptions import ReviewError
from repopilot.llm_provider import LLMProvider
from repopilot.models import (
    AgentResult,
    FileDiff,
    FileMetadata,
    FindingKind,
    RepositoryModel,
    ReviewCategory,
    ReviewFinding,
    ReviewMode,
    ReviewReport,
    ReviewRequest,
    ReviewSeverity,
    TestResult as RepoTestResult,
    ValidationResult,
)
from repopilot.prompt_builder import PromptBuilder
from repopilot.review_context_builder import ReviewContextBuilder
from repopilot.review_service import ReviewService


def _fake_llm(payload: dict) -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = json.dumps(payload)
    provider.model_name = "mock-groq"
    return provider


def _sample_repo(tmp_path: Path) -> RepositoryModel:
    auth_file = tmp_path / "auth.py"
    auth_file.write_text("def login(user, pw):\n    return True\n", encoding="utf-8")

    server_file = tmp_path / "server.py"
    server_file.write_text("import auth\ndef run():\n    auth.login('admin', '123')\n", encoding="utf-8")

    files = [
        FileMetadata(path="auth.py", language="Python", size_bytes=len(auth_file.read_bytes()), imports=[]),
        FileMetadata(path="server.py", language="Python", size_bytes=len(server_file.read_bytes()), imports=["auth"]),
    ]
    return RepositoryModel(repo_url="https://github.com/example/repo", local_path=str(tmp_path), files=files)



# ---------------------------------------------------------------------------
# Models & Serialization
# ---------------------------------------------------------------------------


def test_review_models_serialization():
    finding = ReviewFinding(
        category=ReviewCategory.SECURITY.value,
        severity=ReviewSeverity.HIGH.value,
        title="SQL Injection",
        description="Raw query execution without parameterization.",
        file_path="auth.py",
        line_number=42,
        snippet="cursor.execute(f'SELECT {user}')",
        evidence="f-string used in DB execute",
        finding_kind=FindingKind.INFERENCE.value,
        recommendation="Use parameterized query.",
        confidence="HIGH",
    )
    d = finding.to_dict()
    assert d["category"] == "SECURITY"
    assert d["severity"] == "HIGH"
    assert d["line_number"] == 42
    assert d["finding_kind"] == "INFERENCE"

    req = ReviewRequest(
        mode=ReviewMode.DIFF,
        target_files=["auth.py"],
        focus_categories=["SECURITY"],
        custom_instructions="Check passwords",
    )
    rd = req.to_dict()
    assert rd["mode"] == "diff"
    assert rd["target_files"] == ["auth.py"]

    rep = ReviewReport(
        mode="diff",
        summary="Review complete",
        findings=[finding],
        severity_counts={"HIGH": 1},
        category_counts={"SECURITY": 1},
        reviewed_files=["auth.py"],
        passed_checks=["Linting clean"],
        warnings=[],
    )
    rpd = rep.to_dict()
    assert rpd["summary"] == "Review complete"
    assert len(rpd["findings"]) == 1


# ---------------------------------------------------------------------------
# ReviewContextBuilder
# ---------------------------------------------------------------------------


def test_review_context_builder_diff_mode(tmp_path):
    repo = _sample_repo(tmp_path)
    builder = ReviewContextBuilder()

    diff = FileDiff(
        file_path="auth.py",
        diff="--- a/auth.py\n+++ b/auth.py\n@@ -1 +1,2 @@\n+import hashlib\n def login(user, pw):",
        lines_added=1,
        lines_removed=0,
    )
    req = ReviewRequest(
        mode=ReviewMode.DIFF,
        diffs=[diff],
        focus_categories=["SECURITY", "ARCHITECTURE"],
    )

    ctx = builder.build(repo, req)
    assert "=== REVIEW MODE: CHANGED FILES / DIFF REVIEW ===" in ctx
    assert "Focus categories: SECURITY, ARCHITECTURE" in ctx
    assert "--- Diff for: auth.py (+1 / -0) ---" in ctx
    # Reverse dependency check: server.py imports auth
    assert "server.py" in ctx
    assert "File 'auth.py' is imported by:" in ctx
    assert "=== CURRENT FILE CONTENTS ===" in ctx
    assert "def login(user, pw):" in ctx


def test_review_context_builder_with_phase10_agent_result(tmp_path):
    repo = _sample_repo(tmp_path)
    builder = ReviewContextBuilder()

    diff = FileDiff(file_path="auth.py", diff="+added line", lines_added=1, lines_removed=0)
    validation = ValidationResult(valid=True, messages=["Syntax check passed"])
    test_res = RepoTestResult(
        status="passed",
        command=["pytest"],
        exit_code=0,
        stdout="1 passed in 0.05s",
        stderr="",
    )
    agent_result = AgentResult(
        status="tested",
        change_plan=None,
        proposed_changes=[],
        applied_changes=[],
        diffs=[diff],
        validation_result=validation,
        test_result=test_res,
        warnings=["Non-critical warning"],
        errors=[],
    )

    req = ReviewRequest(mode=ReviewMode.DIFF, agent_result=agent_result)
    ctx = builder.build(repo, req)

    assert "=== PHASE 10 EXECUTION STATUS ===" in ctx
    assert "Workflow status: tested" in ctx
    assert "Test Status: passed" in ctx
    assert "1 passed in 0.05s" in ctx
    assert "Validation valid: True" in ctx


def test_review_context_builder_full_repo_mode(tmp_path):
    repo = _sample_repo(tmp_path)
    builder = ReviewContextBuilder()

    req = ReviewRequest(mode=ReviewMode.FULL_REPO, focus_categories=["ARCHITECTURE"])
    ctx = builder.build(repo, req)

    assert "=== REVIEW MODE: FULL REPOSITORY AUDIT ===" in ctx
    assert "=== ARCHITECTURAL CONTEXT & DEPENDENCIES ===" in ctx
    assert "Total scanned files: 2" in ctx


# ---------------------------------------------------------------------------
# PromptBuilder for Phase 11
# ---------------------------------------------------------------------------


def test_prompt_builder_review_methods():
    req = ReviewRequest(
        mode=ReviewMode.DIFF,
        focus_categories=["SECURITY"],
        custom_instructions="Focus on cryptographic security",
    )
    sys_prompt, user_prompt = PromptBuilder.build_review_diff("context here", req)
    assert "principal software engineer" in sys_prompt
    assert "findings" in sys_prompt
    assert "context here" in user_prompt
    assert "Focus on cryptographic security" in user_prompt

    sys_full, user_full = PromptBuilder.build_review_full_repo("context here", req)
    assert "principal software architect" in sys_full
    assert "context here" in user_full


# ---------------------------------------------------------------------------
# ReviewService
# ---------------------------------------------------------------------------


def test_review_service_diff_review_and_severity_sorting(tmp_path):
    repo = _sample_repo(tmp_path)
    diff = FileDiff(file_path="auth.py", diff="+some change", lines_added=1, lines_removed=0)

    llm_payload = {
        "summary": "Diff introduces a hardcoded credential and a minor formatting issue.",
        "findings": [
            {
                "category": "CODE_QUALITY",
                "severity": "LOW",
                "title": "Trailing Whitespace",
                "description": "Trailing whitespace detected on line 5.",
                "file_path": "auth.py",
                "line_number": 5,
                "evidence": "Extra space after return",
                "finding_kind": "FACT",
                "recommendation": "Remove whitespace",
                "confidence": "HIGH",
            },
            {
                "category": "SECURITY",
                "severity": "CRITICAL",
                "title": "Hardcoded Secret",
                "description": "Production key hardcoded in source.",
                "file_path": "auth.py",
                "line_number": 2,
                "evidence": "API_KEY = 'secret'",
                "finding_kind": "FACT",
                "recommendation": "Use environment variable.",
                "confidence": "HIGH",
            },
            {
                "category": "BUG",
                "severity": "HIGH",
                "title": "Unhandled None",
                "description": "Function may return None unexpectedly.",
                "file_path": "auth.py",
                "line_number": 3,
                "evidence": "Missing else clause",
                "finding_kind": "INFERENCE",
                "recommendation": "Add fallback return.",
                "confidence": "MEDIUM",
            },
        ],
        "passed_checks": ["Unit tests executed cleanly", "No circular imports"],
        "warnings": ["Ensure API keys are revoked if exposed."],
    }

    service = ReviewService(llm_provider=_fake_llm(llm_payload))
    report = service.review_diffs([diff], repo)

    assert report.mode == "diff"
    assert report.summary == llm_payload["summary"]
    assert len(report.findings) == 3

    # Verify severity sorting: CRITICAL -> HIGH -> LOW
    assert report.findings[0].severity == "CRITICAL"
    assert report.findings[1].severity == "HIGH"
    assert report.findings[2].severity == "LOW"

    # Verify counts
    assert report.severity_counts["CRITICAL"] == 1
    assert report.severity_counts["HIGH"] == 1
    assert report.severity_counts["LOW"] == 1
    assert report.severity_counts["MEDIUM"] == 0
    assert report.category_counts["SECURITY"] == 1
    assert report.category_counts["BUG"] == 1
    assert report.category_counts["CODE_QUALITY"] == 1

    assert "auth.py" in report.reviewed_files
    assert len(report.passed_checks) == 2


def test_review_service_handles_markdown_wrapped_json(tmp_path):
    repo = _sample_repo(tmp_path)
    diff = FileDiff(file_path="auth.py", diff="+code", lines_added=1, lines_removed=0)

    raw_json = json.dumps({
        "summary": "Looks good",
        "findings": [],
        "passed_checks": ["All clear"],
        "warnings": [],
    })
    markdown_wrapped = f"```json\n{raw_json}\n```"

    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = markdown_wrapped
    service = ReviewService(llm_provider=provider)

    report = service.review_diffs([diff], repo)
    assert report.summary == "Looks good"
    assert len(report.findings) == 0


def test_review_service_detects_hallucinated_file_paths(tmp_path):
    repo = _sample_repo(tmp_path)
    diff = FileDiff(file_path="auth.py", diff="+code", lines_added=1, lines_removed=0)

    llm_payload = {
        "summary": "Audit with hallucinated file reference",
        "findings": [
            {
                "category": "SECURITY",
                "severity": "HIGH",
                "title": "Bug in non-existent file",
                "description": "Issue in file that doesn't exist",
                "file_path": "phantom/does_not_exist.py",
                "evidence": "Fake evidence",
                "finding_kind": "INFERENCE",
                "recommendation": "Fix it",
                "confidence": "LOW",
            }
        ],
        "passed_checks": [],
        "warnings": [],
    }

    service = ReviewService(llm_provider=_fake_llm(llm_payload))
    report = service.review_diffs([diff], repo)

    assert len(report.findings) == 1
    # Check that a system warning was appended for the non-existent file
    assert any("non-existent repository path" in w for w in report.warnings)


def test_review_service_agent_result_integration(tmp_path):
    repo = _sample_repo(tmp_path)
    diff = FileDiff(file_path="auth.py", diff="+code", lines_added=1, lines_removed=0)
    agent_result = AgentResult(
        status="tested",
        change_plan=None,
        proposed_changes=[],
        applied_changes=[],
        diffs=[diff],
        validation_result=ValidationResult(valid=True, messages=[]),
        test_result=RepoTestResult(status="passed", command=["pytest"], exit_code=0, stdout="OK", stderr=""),
        warnings=[],
        errors=[],
    )

    llm_payload = {
        "summary": "Phase 10 change reviewed.",
        "findings": [],
        "passed_checks": ["Tested cleanly"],
        "warnings": [],
    }
    service = ReviewService(llm_provider=_fake_llm(llm_payload))
    report = service.review_agent_result(agent_result, repo)

    assert report.mode == "diff"
    assert "auth.py" in report.reviewed_files


def test_review_service_full_repo_review(tmp_path):
    repo = _sample_repo(tmp_path)
    llm_payload = {
        "summary": "Repository architecture is clean and modular.",
        "findings": [],
        "passed_checks": ["Layering consistent"],
        "warnings": [],
    }
    service = ReviewService(llm_provider=_fake_llm(llm_payload))
    report = service.review_repository(repo)

    assert report.mode == "full_repo"
    assert report.summary == "Repository architecture is clean and modular."
    assert "auth.py" in report.reviewed_files
    assert "server.py" in report.reviewed_files


def test_review_service_invalid_json_raises_review_error(tmp_path):
    repo = _sample_repo(tmp_path)
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = "Sorry, I cannot process this as JSON."
    service = ReviewService(llm_provider=provider)

    with pytest.raises(ReviewError, match="Failed to parse LLM review output as JSON"):
        service.review_repository(repo)


def test_review_service_missing_arguments_raise_error(tmp_path):
    service = ReviewService()
    with pytest.raises(ReviewError, match="A non-empty ReviewRequest is required"):
        service.review(None, None)

    with pytest.raises(ReviewError, match="A RepositoryModel is required"):
        service.review(ReviewRequest(), None)
