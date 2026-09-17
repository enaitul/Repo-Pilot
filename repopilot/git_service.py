"""
Phase 12: Git Service.

Handles local Git actions (branch creation, file staging, commit, push)
using safe list-arg subprocess calls.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from repopilot.config import DEFAULT_BASE_BRANCH, DEFAULT_BRANCH_PREFIX, DEFAULT_REMOTE_NAME
from repopilot.exceptions import BranchError
from repopilot.models import GitBranchDetails


class GitService:
    """Safely manages local Git workspace operations via subprocess."""

    def __init__(self, timeout_seconds: int = 30):
        self._timeout_seconds = timeout_seconds

    def _run_git(self, repo_path: str, args: list[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
        """Executes a git command defensively using list arguments."""
        if not repo_path or not Path(repo_path).exists():
            raise BranchError(f"Repository path does not exist: {repo_path}")

        command = ["git"] + args
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)

        try:
            result = subprocess.run(
                command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
                env=cmd_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise BranchError(f"Git command '{' '.join(command)}' timed out after {self._timeout_seconds}s.") from exc
        except FileNotFoundError as exc:
            raise BranchError("git executable not found. Is git installed and on PATH?") from exc

        return result

    def ensure_git_repo(self, repo_path: str) -> None:
        """Ensures that repo_path is an initialized Git repository."""
        git_dir = Path(repo_path) / ".git"
        if not git_dir.exists():
            res = self._run_git(repo_path, ["init"])
            if res.returncode != 0:
                raise BranchError(f"Failed to initialize git repository: {res.stderr.strip()}")

    def get_current_branch(self, repo_path: str) -> str:
        """Gets current active Git branch name."""
        self.ensure_git_repo(repo_path)
        res = self._run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
        return DEFAULT_BASE_BRANCH

    def create_feature_branch(
        self, repo_path: str, branch_name: Optional[str] = None, base_branch: str = DEFAULT_BASE_BRANCH
    ) -> str:
        """Creates and checks out a new feature branch in the workspace."""
        self.ensure_git_repo(repo_path)
        if not branch_name:
            import time
            branch_name = f"{DEFAULT_BRANCH_PREFIX}{int(time.time())}"

        # Create branch
        res = self._run_git(repo_path, ["checkout", "-b", branch_name])
        if res.returncode != 0:
            # If branch already exists, attempt checking it out
            res_checkout = self._run_git(repo_path, ["checkout", branch_name])
            if res_checkout.returncode != 0:
                raise BranchError(f"Failed to create or checkout branch '{branch_name}': {res.stderr.strip()}")

        return branch_name

    def stage_and_commit(
        self, repo_path: str, file_paths: list[str], commit_message: str, branch_name: str
    ) -> GitBranchDetails:
        """Stages specified files and creates a Git commit."""
        self.ensure_git_repo(repo_path)
        if not file_paths:
            # Stage all changes if no specific files are provided
            res_add = self._run_git(repo_path, ["add", "-A"])
        else:
            res_add = self._run_git(repo_path, ["add"] + file_paths)

        if res_add.returncode != 0:
            raise BranchError(f"Failed to stage files: {res_add.stderr.strip()}")

        # Commit
        res_commit = self._run_git(repo_path, ["commit", "-m", commit_message])
        if res_commit.returncode != 0 and "nothing to commit" not in res_commit.stdout.lower():
            raise BranchError(f"Failed to commit changes: {res_commit.stderr.strip()}")

        # Get commit hash
        res_hash = self._run_git(repo_path, ["rev-parse", "HEAD"])
        commit_hash = res_hash.stdout.strip() if res_hash.returncode == 0 else "HEAD"

        return GitBranchDetails(
            branch_name=branch_name,
            base_branch=DEFAULT_BASE_BRANCH,
            commit_hash=commit_hash,
            staged_files=file_paths,
        )

    def push_branch(
        self,
        repo_path: str,
        branch_name: str,
        remote_name: str = DEFAULT_REMOTE_NAME,
        token: Optional[str] = None,
    ) -> None:
        """Pushes local feature branch to remote GitHub repository."""
        self.ensure_git_repo(repo_path)
        cmd_args = ["push", "-u", remote_name, branch_name]
        env = {}
        if token:
            # Mask token in git askpass / env if needed
            env["GITHUB_TOKEN"] = token

        res = self._run_git(repo_path, cmd_args, env=env)
        if res.returncode != 0:
            raise BranchError(f"Failed to push branch '{branch_name}' to '{remote_name}': {res.stderr.strip()}")
