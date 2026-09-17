"""Bounded file operations for Phase 10's disposable clone workspace."""

from __future__ import annotations

import difflib
from pathlib import Path

from repopilot.config import MAX_MODIFIED_FILE_SIZE_BYTES, MODIFICATION_ALLOWED_EXTENSIONS
from repopilot.exceptions import ModificationError
from repopilot.models import ChangeOperation, FileDiff, ProposedChange

_SENSITIVE_PATH_PARTS = (".env", "secret", "credential", "token", "api_key")


class WorkspaceTools:
    """Applies validated text changes without granting general filesystem access."""

    def __init__(self, workspace_path: str):
        self._root = Path(workspace_path).resolve()
        if not self._root.is_dir():
            raise ModificationError("Controlled workspace path must be an existing directory.")
        self._before: dict[str, str | None] = {}

    def validate_change(self, change: ProposedChange) -> Path:
        target = self._resolve(change.file_path)
        if target.suffix.lower() not in MODIFICATION_ALLOWED_EXTENSIONS:
            raise ModificationError(f"File type is not allowed for modification: {change.file_path}")
        if self._is_sensitive(change.file_path):
            raise ModificationError(f"Sensitive file is not allowed for modification: {change.file_path}")
        if change.operation == ChangeOperation.CREATE:
            if target.exists():
                raise ModificationError(f"Cannot create existing file: {change.file_path}")
            if not isinstance(change.new_content, str):
                raise ModificationError("Create changes require text new_content.")
        elif change.operation == ChangeOperation.REPLACE:
            if not target.is_file():
                raise ModificationError(f"Replace target does not exist: {change.file_path}")
            if not isinstance(change.new_content, str):
                raise ModificationError("Replace changes require text new_content.")
            if not isinstance(change.start_line, int) or not isinstance(change.end_line, int):
                raise ModificationError("Replace changes require integer start_line and end_line.")
            line_count = self._read(target).count("\n") + 1
            if change.start_line < 1 or change.end_line < change.start_line or change.end_line > line_count:
                raise ModificationError(f"Replace line range is invalid for {change.file_path}.")
        elif change.operation == ChangeOperation.DELETE:
            if not target.is_file():
                raise ModificationError(f"Delete target does not exist: {change.file_path}")
        else:
            raise ModificationError(f"Unsupported change operation: {change.operation!r}")
        return target

    def apply(self, changes: list[ProposedChange]) -> list[ProposedChange]:
        # Validate the complete batch before the first write, avoiding partial batches.
        targets = [(change, self.validate_change(change)) for change in changes]
        if len({str(path) for _, path in targets}) != len(targets):
            raise ModificationError("Only one structured change per file is allowed in a workflow run.")

        for change, target in targets:
            key = change.file_path
            self._before[key] = self._read(target) if target.exists() else None
            if change.operation == ChangeOperation.CREATE:
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write(target, change.new_content or "")
            elif change.operation == ChangeOperation.REPLACE:
                lines = self._read(target).splitlines(keepends=True)
                replacement = change.new_content or ""
                replaced_lines = lines[change.start_line - 1:change.end_line]
                if replacement and not replacement.endswith("\n") and replaced_lines[-1].endswith("\n"):
                    replacement += "\n"
                lines[change.start_line - 1:change.end_line] = [replacement]
                self._write(target, "".join(lines))
            else:
                target.unlink()
        return changes

    def diffs(self) -> list[FileDiff]:
        results: list[FileDiff] = []
        for file_path, before in self._before.items():
            target = self._resolve(file_path)
            after = self._read(target) if target.exists() else None
            diff_lines = list(difflib.unified_diff(
                (before or "").splitlines(keepends=True),
                (after or "").splitlines(keepends=True),
                fromfile=f"a/{file_path}" if before is not None else "/dev/null",
                tofile=f"b/{file_path}" if after is not None else "/dev/null",
            ))
            results.append(FileDiff(
                file_path=file_path,
                diff="".join(diff_lines),
                lines_added=sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")),
                lines_removed=sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---")),
            ))
        return results

    def rollback(self) -> None:
        for file_path, before in self._before.items():
            target = self._resolve(file_path)
            if before is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write(target, before)

    def path_for(self, file_path: str) -> Path:
        """Return a guarded workspace path for deterministic post-apply checks."""
        return self._resolve(file_path)

    def read_context(self, file_path: str) -> str:
        """Read one explicitly requested, non-sensitive source file for LLM context."""
        target = self._resolve(file_path)
        if target.suffix.lower() not in MODIFICATION_ALLOWED_EXTENSIONS or self._is_sensitive(file_path):
            raise ModificationError("Requested context file is not permitted.")
        if not target.is_file():
            raise ModificationError("Requested context file does not exist.")
        return self._read(target)

    def _resolve(self, file_path: str) -> Path:
        if not isinstance(file_path, str) or not file_path or Path(file_path).is_absolute():
            raise ModificationError("File paths must be non-empty, relative workspace paths.")
        candidate = self._root / file_path
        # Resolving both existing targets and their existing parents catches symlink escapes.
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ModificationError(f"Unsafe path outside controlled workspace: {file_path}") from exc
        return resolved

    @staticmethod
    def _is_sensitive(file_path: str) -> bool:
        lowered = Path(file_path).name.lower()
        return any(part in lowered for part in _SENSITIVE_PATH_PARTS)

    @staticmethod
    def _read(path: Path) -> str:
        try:
            if path.stat().st_size > MAX_MODIFIED_FILE_SIZE_BYTES:
                raise ModificationError(f"File exceeds modification size limit: {path.name}")
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ModificationError(f"File is not valid UTF-8 text: {path.name}") from exc
        except OSError as exc:
            raise ModificationError(f"Could not read workspace file: {path.name}") from exc

    @staticmethod
    def _write(path: Path, content: str) -> None:
        if len(content.encode("utf-8")) > MAX_MODIFIED_FILE_SIZE_BYTES:
            raise ModificationError(f"Modified file exceeds size limit: {path.name}")
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ModificationError(f"Could not write workspace file: {path.name}") from exc
