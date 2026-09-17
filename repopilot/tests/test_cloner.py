import subprocess
from unittest.mock import patch, MagicMock

import pytest

from repopilot.cloner import GitCloner, validate_repo_url
from repopilot.exceptions import InvalidRepoURLError, CloneError


# --- URL validation -------------------------------------------------------

def test_accepts_valid_github_url():
    assert validate_repo_url("https://github.com/psf/requests") == "https://github.com/psf/requests.git"


def test_accepts_url_already_ending_in_git():
    assert validate_repo_url("https://github.com/psf/requests.git") == "https://github.com/psf/requests.git"


def test_rejects_empty_url():
    with pytest.raises(InvalidRepoURLError):
        validate_repo_url("")


def test_rejects_non_github_url():
    with pytest.raises(InvalidRepoURLError):
        validate_repo_url("https://gitlab.com/psf/requests")


def test_rejects_shell_injection_attempt():
    with pytest.raises(InvalidRepoURLError):
        validate_repo_url("https://github.com/foo/bar; rm -rf /")


def test_rejects_local_file_url():
    with pytest.raises(InvalidRepoURLError):
        validate_repo_url("file:///etc/passwd")


def test_rejects_non_string_input():
    with pytest.raises(InvalidRepoURLError):
        validate_repo_url(None)  # type: ignore[arg-type]


# --- Cloning (subprocess mocked out — no real network calls in unit tests) --

@patch("repopilot.cloner.subprocess.run")
@patch("repopilot.cloner.tempfile.mkdtemp")
def test_clone_success_returns_path(mock_mkdtemp, mock_run, tmp_path):
    fake_dir = tmp_path / "clone"
    fake_dir.mkdir()
    (fake_dir / "README.md").write_text("hi")  # so dir isn't "empty"
    mock_mkdtemp.return_value = str(fake_dir)
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    path = GitCloner().clone("https://github.com/psf/requests")

    assert path == str(fake_dir)
    mock_run.assert_called_once()
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "git"
    assert "--depth" in called_args


@patch("repopilot.cloner.subprocess.run")
@patch("repopilot.cloner.tempfile.mkdtemp")
def test_clone_failure_raises_clone_error(mock_mkdtemp, mock_run, tmp_path):
    fake_dir = tmp_path / "clone"
    fake_dir.mkdir()
    mock_mkdtemp.return_value = str(fake_dir)
    mock_run.return_value = MagicMock(returncode=128, stderr="fatal: repository not found")

    with pytest.raises(CloneError):
        GitCloner().clone("https://github.com/psf/doesnotexist12345")


@patch("repopilot.cloner.subprocess.run")
@patch("repopilot.cloner.tempfile.mkdtemp")
def test_clone_timeout_raises_clone_error(mock_mkdtemp, mock_run, tmp_path):
    fake_dir = tmp_path / "clone"
    fake_dir.mkdir()
    mock_mkdtemp.return_value = str(fake_dir)
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=120)

    with pytest.raises(CloneError):
        GitCloner().clone("https://github.com/psf/requests")


def test_cleanup_removes_directory(tmp_path):
    target = tmp_path / "to_remove"
    target.mkdir()
    (target / "file.txt").write_text("x")

    GitCloner.cleanup(str(target))

    assert not target.exists()


def test_cleanup_is_safe_on_nonexistent_path():
    # Should not raise even if the path is already gone
    GitCloner.cleanup("/tmp/definitely_does_not_exist_repopilot_xyz")
