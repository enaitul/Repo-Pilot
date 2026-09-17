"""Deterministic validation for changes already confined to a workspace."""

from __future__ import annotations

from repopilot.exceptions import ValidationError
from repopilot.models import ProposedChange, ValidationResult
from repopilot.workspace_tools import WorkspaceTools


class ValidationService:
    def validate(self, workspace: WorkspaceTools, changes: list[ProposedChange]) -> ValidationResult:
        messages: list[str] = []
        for change in changes:
            path = workspace.path_for(change.file_path) if change.operation.value != "delete" else None
            if path is not None and path.suffix == ".py":
                try:
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")
                except (SyntaxError, UnicodeDecodeError) as exc:
                    raise ValidationError(f"Python syntax validation failed for {change.file_path}: {exc}") from exc
            messages.append(f"Validated {change.file_path}.")
        return ValidationResult(valid=True, messages=messages)
