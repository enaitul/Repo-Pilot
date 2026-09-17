"""Deterministic, allowlisted test execution for a controlled workspace."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from repopilot.config import MAX_TEST_OUTPUT_CHARS, TEST_TIMEOUT_SECONDS
from repopilot.models import TestResult


class TestRunner:
    """Selects a command from repository configuration; no LLM command is accepted."""

    def select_command(self, workspace_path: str) -> list[str] | None:
        root = Path(workspace_path)
        if any((root / name).exists() for name in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")) or (root / "tests").is_dir():
            return [sys.executable, "-m", "pytest"]
        return None

    def run(self, workspace_path: str) -> TestResult:
        command = self.select_command(workspace_path)
        if command is None:
            return TestResult(status="not_available", command=None, exit_code=None,
                              stderr="No supported deterministic test configuration was found.")
        try:
            completed = subprocess.run(
                command, cwd=workspace_path, shell=False, text=True, capture_output=True,
                timeout=TEST_TIMEOUT_SECONDS, check=False,
                env={"PATH": os.defpath},
            )
        except subprocess.TimeoutExpired as exc:
            return TestResult(status="timed_out", command=command, exit_code=None,
                              stdout=_limit(exc.stdout), stderr=_limit(exc.stderr))
        except OSError as exc:
            return TestResult(status="failed", command=command, exit_code=None, stderr=str(exc))
        return TestResult(status="passed" if completed.returncode == 0 else "failed", command=command,
                          exit_code=completed.returncode, stdout=_limit(completed.stdout), stderr=_limit(completed.stderr))


def _limit(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output or ""
    return output[:MAX_TEST_OUTPUT_CHARS]
