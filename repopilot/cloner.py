"""
Handles all interaction with git and the network. This is the highest-risk
module in the system (external process, untrusted input, network I/O), so
its job is narrow and defensive on purpose: validate the URL, clone
shallowly into an isolated temp directory, and hand back a path. Nothing
else in the system should shell out to git.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from repopilot.config import CLONE_TIMEOUT_SECONDS
from repopilot.exceptions import CloneError, InvalidRepoURLError

# Deliberately narrow: Phase 1 only supports public GitHub HTTPS URLs.
# Rejecting everything else (file://, ssh with arbitrary hosts, other
# forges) keeps the attack surface small and matches the stated scope.
_GITHUB_HTTPS_PATTERN = re.compile(
    r"^https://github\.com/[\w.\-]+/[\w.\-]+(?:\.git)?/?$"
)


def validate_repo_url(url: str) -> str:
    """
    Validate that `url` is an acceptable public GitHub repository URL.

    Returns the normalized URL (with a trailing .git) on success.
    Raises InvalidRepoURLError otherwise.

    This validation happens BEFORE the URL ever reaches subprocess, which
    is the real defense here — not the subprocess call style. Never build
    the git command with shell=True + string interpolation regardless.
    """
    if not url or not isinstance(url, str):
        raise InvalidRepoURLError("Repository URL must be a non-empty string.")

    url = url.strip()

    if not _GITHUB_HTTPS_PATTERN.match(url):
        raise InvalidRepoURLError(
            f"'{url}' is not a valid public GitHub URL. "
            "Expected format: https://github.com/<owner>/<repo>"
        )

    normalized = url.rstrip("/")
    if not normalized.endswith(".git"):
        normalized += ".git"
    return normalized


class GitCloner:
    """Clones a validated GitHub URL into an isolated temporary directory."""

    def clone(self, repo_url: str) -> str:
        """
        Shallow-clone `repo_url` into a fresh temp directory and return
        the local path. Raises CloneError on any failure.
        """
        normalized_url = validate_repo_url(repo_url)
        dest_dir = tempfile.mkdtemp(prefix="repopilot_")

        # List-args subprocess call — NEVER shell=True with interpolated
        # input. This is what actually prevents command injection; the
        # URL validation above is a second, independent layer of defense.
        command = [
            "git", "clone",
            "--depth", "1",           # shallow: we only need the current tree
            "--single-branch",
            normalized_url,
            dest_dir,
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._cleanup(dest_dir)
            raise CloneError(
                f"Clone of '{repo_url}' timed out after {CLONE_TIMEOUT_SECONDS}s."
            ) from exc
        except FileNotFoundError as exc:
            self._cleanup(dest_dir)
            raise CloneError(
                "git executable not found. Is git installed and on PATH?"
            ) from exc

        if result.returncode != 0:
            self._cleanup(dest_dir)
            raise CloneError(
                f"git clone failed for '{repo_url}': {result.stderr.strip()}"
            )

        if not any(Path(dest_dir).iterdir()):
            self._cleanup(dest_dir)
            raise CloneError(f"Clone of '{repo_url}' produced an empty directory.")

        return dest_dir

    @staticmethod
    def cleanup(local_path: str) -> None:
        """Public cleanup entry point — remove a cloned repo from disk."""
        GitCloner._cleanup(local_path)

    @staticmethod
    def _cleanup(local_path: str) -> None:
        shutil.rmtree(local_path, ignore_errors=True)
