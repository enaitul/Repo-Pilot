"""
Flags files as "probably important" or "probably an entry point" using
simple filename-pattern matching against config.py's small, maintainable
pattern lists.

HONESTY NOTE, worth restating here since it matters for how RepoPilot's
output should be read: "important" is a HEURISTIC based on common naming
conventions, not a guarantee. A repository is free to name its real
entry point something this list doesn't recognize (e.g. `run.py`
instead of `main.py`) — in that case, this detector will simply miss it,
rather than guessing incorrectly.
"""

from __future__ import annotations

from repopilot.config import ENTRY_POINT_FILENAMES, IMPORTANT_FILE_PATTERNS
from repopilot.models import RepositoryModel


class ImportantFileDetector:
    """Detects likely-important files and entry points by filename convention."""

    def detect_important_files(self, repo_model: RepositoryModel) -> list[str]:
        return [f.path for f in repo_model.files if self._basename(f.path) in IMPORTANT_FILE_PATTERNS]

    def detect_entry_points(self, repo_model: RepositoryModel) -> list[str]:
        return [f.path for f in repo_model.files if self._basename(f.path) in ENTRY_POINT_FILENAMES]

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rsplit("/", 1)[-1]
