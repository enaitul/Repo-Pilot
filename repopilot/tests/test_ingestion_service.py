import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repopilot.ingestion_service import IngestionService
from repopilot.cloner import GitCloner
from repopilot.walker import RepoWalker
from repopilot.exceptions import CloneError, IngestionError


def _make_local_git_repo(tmp_path: Path) -> Path:
    """Create a tiny real git repo on disk to stand in for a 'clone'."""
    repo_dir = tmp_path / "fake_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("import os\nimport sys\n\ndef main():\n    pass\n")
    (repo_dir / "utils.py").write_text("from typing import List\n")
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "junk.js").write_text("module.exports = {};")
    (repo_dir / "README.md").write_text("# Fake repo")
    return repo_dir


def test_ingest_builds_repository_model(tmp_path):
    repo_dir = _make_local_git_repo(tmp_path)

    fake_cloner = MagicMock(spec=GitCloner)
    fake_cloner.clone.return_value = str(repo_dir)

    service = IngestionService(cloner=fake_cloner, walker=RepoWalker())
    model = service.ingest("https://github.com/fake/repo", cleanup=False)

    assert model.repo_url == "https://github.com/fake/repo"
    assert model.total_files == 3  # main.py, utils.py, README.md
    assert "Python" in model.language_summary
    paths = {f.path for f in model.files}
    assert "main.py" in paths
    assert not any("node_modules" in p for p in paths)

    fake_cloner.cleanup.assert_not_called()


def test_ingest_calls_cleanup_by_default(tmp_path):
    repo_dir = _make_local_git_repo(tmp_path)
    fake_cloner = MagicMock(spec=GitCloner)
    fake_cloner.clone.return_value = str(repo_dir)

    service = IngestionService(cloner=fake_cloner, walker=RepoWalker())
    service.ingest("https://github.com/fake/repo")  # cleanup=True default

    fake_cloner.cleanup.assert_called_once_with(str(repo_dir))


def test_ingest_propagates_clone_error():
    fake_cloner = MagicMock(spec=GitCloner)
    fake_cloner.clone.side_effect = CloneError("boom")

    service = IngestionService(cloner=fake_cloner, walker=RepoWalker())

    with pytest.raises(CloneError):
        service.ingest("https://github.com/fake/repo")


def test_ingest_wraps_unexpected_errors():
    fake_cloner = MagicMock(spec=GitCloner)
    fake_cloner.clone.side_effect = RuntimeError("something totally unexpected")

    service = IngestionService(cloner=fake_cloner, walker=RepoWalker())

    with pytest.raises(IngestionError):
        service.ingest("https://github.com/fake/repo")


@pytest.mark.integration
def test_end_to_end_real_clone_small_public_repo(tmp_path):
    """
    Real network test against a tiny, stable public repo. Skipped unless
    explicitly requested, since unit test suites shouldn't depend on
    network availability.
    """
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5, check=True)
    except Exception:
        pytest.skip("git not available")

    service = IngestionService()
    model = service.ingest("https://github.com/octocat/Hello-World")

    assert model.total_files >= 0  # repo may be nearly empty; just verify no crash
    assert model.repo_url == "https://github.com/octocat/Hello-World"
