"""Tests for Phase 12 GitService."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from repopilot.exceptions import BranchError
from repopilot.git_service import GitService


def test_git_service_missing_repo_path():
    service = GitService()
    with pytest.raises(BranchError, match="Repository path does not exist"):
        service.get_current_branch("/nonexistent/path/for/repopilot/test")


def test_create_feature_branch_and_commit(tmp_path):
    # Initialize real git repo in temp folder
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)

    test_file = tmp_path / "hello.py"
    test_file.write_text("print('hello world')\n")

    service = GitService()
    branch = service.create_feature_branch(str(tmp_path), branch_name="feature/test-branch")
    assert branch == "feature/test-branch"

    details = service.stage_and_commit(
        repo_path=str(tmp_path),
        file_paths=["hello.py"],
        commit_message="feat: add hello.py",
        branch_name=branch,
    )
    assert details.branch_name == "feature/test-branch"
    assert details.commit_hash is not None
    assert details.staged_files == ["hello.py"]


@patch("subprocess.run")
def test_push_branch_calls_git_push(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create dummy .git dir
    (tmp_path / ".git").mkdir()

    service = GitService()
    service.push_branch(str(tmp_path), branch_name="feature/test-branch", remote_name="origin", token="fake_token")

    assert mock_run.called
    called_args = mock_run.call_args[0][0]
    assert called_args == ["git", "push", "-u", "origin", "feature/test-branch"]
