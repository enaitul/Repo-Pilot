"""Tests for Phase 12 GitHubWorkflowService."""

from unittest.mock import MagicMock

import pytest
from repopilot.exceptions import GitHubWorkflowError
from repopilot.github_workflow_service import GitHubWorkflowService
from repopilot.models import (
    AgentResult,
    GitBranchDetails,
    GitHubWorkflowRequest,
    PullRequestDetails,
    RepositoryModel,
)


def test_github_workflow_missing_inputs():
    service = GitHubWorkflowService()
    with pytest.raises(GitHubWorkflowError, match="A non-empty GitHubWorkflowRequest is required"):
        service.execute(request=None, repo_model=None)  # type: ignore


def test_github_workflow_unapproved_human_gate(tmp_path):
    repo_model = RepositoryModel(
        repo_url="https://github.com/owner/repo",
        local_path=str(tmp_path),
        files=[],
    )

    mock_git = MagicMock()
    mock_git.create_feature_branch.return_value = "feature/repopilot-update"
    mock_git.stage_and_commit.return_value = GitBranchDetails(
        branch_name="feature/repopilot-update",
        base_branch="main",
        commit_hash="abc1234",
        staged_files=[],
    )

    service = GitHubWorkflowService(git_service=mock_git)
    req = GitHubWorkflowRequest(
        repo_owner="owner",
        repo_name="repo",
        approved=False,  # Unapproved
    )

    result = service.execute(req, repo_model)
    assert result.status == "missing_approval"
    assert not result.approved
    assert result.pr_details is None
    assert len(result.warnings) > 0
    assert "Human approval was not granted" in result.warnings[0]


def test_github_workflow_approved_with_token(tmp_path):
    repo_model = RepositoryModel(
        repo_url="https://github.com/owner/repo",
        local_path=str(tmp_path),
        files=[],
    )


    mock_git = MagicMock()
    mock_git.create_feature_branch.return_value = "feature/repopilot-update"
    mock_git.stage_and_commit.return_value = GitBranchDetails(
        branch_name="feature/repopilot-update",
        base_branch="main",
        commit_hash="abc1234",
        staged_files=["app.py"],
    )

    mock_api = MagicMock()
    mock_api.format_pr_description.return_value = "PR Description Body"
    mock_api.create_pull_request.return_value = PullRequestDetails(
        title="RepoPilot Edits for repo",
        body="PR Description Body",
        head_branch="feature/repopilot-update",
        base_branch="main",
        html_url="https://github.com/owner/repo/pull/1",
        number=1,
    )

    service = GitHubWorkflowService(git_service=mock_git, api_service=mock_api)
    req = GitHubWorkflowRequest(
        repo_owner="owner",
        repo_name="repo",
        approved=True,  # Approved
        token="valid_token",
    )

    result = service.execute(req, repo_model)
    assert result.status == "approved_and_created"
    assert result.approved
    assert result.pr_details is not None
    assert result.pr_details.number == 1
    assert result.pr_details.html_url == "https://github.com/owner/repo/pull/1"
