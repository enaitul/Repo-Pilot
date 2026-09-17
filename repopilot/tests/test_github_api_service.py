"""Tests for Phase 12 GitHubAPIService."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest
from repopilot.exceptions import AuthenticationError, PullRequestError
from repopilot.github_api_service import GitHubAPIService
from repopilot.models import ReviewFinding, ReviewReport


def test_verify_token_empty():
    service = GitHubAPIService()
    with pytest.raises(AuthenticationError, match="non-empty GitHub token is required"):
        service.verify_token("")


@patch("urllib.request.urlopen")
def test_verify_token_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"login": "octocat"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    service = GitHubAPIService()
    res = service.verify_token("valid_token")
    assert res["login"] == "octocat"


@patch("urllib.request.urlopen")
def test_create_pull_request_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "html_url": "https://github.com/owner/repo/pull/1",
        "number": 1,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    service = GitHubAPIService()
    details = service.create_pull_request(
        owner="owner",
        repo="repo",
        title="Test PR",
        body="PR description",
        head_branch="feature/test",
        base_branch="main",
        token="valid_token",
    )
    assert details.title == "Test PR"
    assert details.number == 1
    assert details.html_url == "https://github.com/owner/repo/pull/1"


def test_format_pr_description():
    service = GitHubAPIService()
    review = ReviewReport(
        mode="diff",
        summary="Code review passed.",
        findings=[
            ReviewFinding(
                category="SECURITY",
                severity="HIGH",
                title="Unsafe eval use",
                description="eval() called on untrusted input",
                file_path="main.py",
            )
        ],
        severity_counts={"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0, "INFO": 0},
        category_counts={"SECURITY": 1},
        reviewed_files=["main.py"],
        passed_checks=["Syntax check"],
        warnings=[],
    )

    desc = service.format_pr_description(
        change_plan_summary="Refactored auth module",
        test_status="passed",
        test_details="12 passed in 0.5s",
        review_report=review,
    )

    assert "Refactored auth module" in desc
    assert "PASSED" in desc
    assert "Unsafe eval use" in desc
    assert "RepoPilot Phase 12" in desc
