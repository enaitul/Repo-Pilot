"""Phase 10 orchestration: plan, structured generation, apply, validate, test."""

from __future__ import annotations

import json
from pathlib import Path

from repopilot.action_service import ActionService
from repopilot.exceptions import ActionError, ModificationError, RepoPilotError, ValidationError
from repopilot.groq_provider import GroqProvider
from repopilot.llm_provider import LLMProvider
from repopilot.models import (
    ActionRequest, ActionType, AgentResult, AgentState, ChangeOperation,
    ProposedChange, RepositoryModel,
)
from repopilot.controlled_test_runner import TestRunner
from repopilot.validation_service import ValidationService
from repopilot.workspace_tools import WorkspaceTools

_GENERATION_SYSTEM_PROMPT = """You produce structured, reviewable repository edits.
Return ONLY JSON with exactly these keys: "changes" (a list) and "warnings" (a list of strings).
Each change must contain file_path, operation (create, replace, or delete), start_line,
end_line, and new_content. For create, start_line and end_line must be null. For delete,
start_line, end_line, and new_content must be null. For replace, line numbers are inclusive
and new_content replaces exactly that range. Never provide shell commands. Do not include
secrets or unrelated files. Only propose edits supported by the supplied context."""


class ModificationService:
    """Extends Phase 9's CHANGE_PLAN using a clone as the sole write boundary."""

    def __init__(
        self,
        action_service: ActionService | None = None,
        llm_provider: LLMProvider | None = None,
        validator: ValidationService | None = None,
        test_runner: TestRunner | None = None,
    ):
        self._llm_provider = llm_provider or GroqProvider()
        self._action_service = action_service or ActionService(llm_provider=self._llm_provider)
        self._validator = validator or ValidationService()
        self._test_runner = test_runner or TestRunner()

    def execute(self, request: ActionRequest, repo_model: RepositoryModel) -> AgentResult:
        if request is None or not request.request or not request.request.strip():
            raise ModificationError("A non-empty ActionRequest is required.")
        if repo_model is None:
            raise ModificationError("A RepositoryModel with a controlled workspace is required.")
        workspace = WorkspaceTools(repo_model.local_path)
        state = AgentState(user_request=request, workspace_path=repo_model.local_path)
        applied = False
        try:
            # Reuse the Phase 9 architecture-grounded planning path verbatim.
            plan_request = ActionRequest(
                action_type=ActionType.CHANGE_PLAN, request=request.request,
                target_file=request.target_file, target_symbol=request.target_symbol,
                additional_context=request.additional_context,
            )
            state.change_plan = self._action_service.execute(plan_request, repo_model=repo_model)
            state.status = "planned"

            raw = self._llm_provider.generate(
                _GENERATION_SYSTEM_PROMPT,
                self._generation_prompt(request, repo_model, state.change_plan.summary),
            )
            state.proposed_changes, warnings = self._parse_changes(raw)
            state.warnings.extend(warnings)
            state.status = "proposed"

            state.applied_changes = workspace.apply(state.proposed_changes)
            applied = True
            state.status = "applied"
            state.diffs = workspace.diffs()
            state.validation_result = self._validator.validate(workspace, state.applied_changes)
            state.status = "validated"
            state.test_result = self._test_runner.run(repo_model.local_path)
            if state.test_result.status in ("failed", "timed_out"):
                workspace.rollback()
                applied = False
                state.status = "failed"
                state.warnings.append("Applied workspace changes were rolled back after unsuccessful tests.")
                state.errors.append(f"Tests were not successful: {state.test_result.status}.")
            else:
                state.status = "tested" if state.test_result.status == "passed" else "validated"
                if state.test_result.status == "not_available":
                    state.warnings.append("No supported deterministic test command was available.")
            return self._result(state)
        except (RepoPilotError, ValueError, TypeError) as exc:
            if applied:
                workspace.rollback()
                state.warnings.append("Applied workspace changes were rolled back after workflow failure.")
            state.status = "failed"
            state.errors.append(str(exc))
            return self._result(state)
        except Exception as exc:
            if applied:
                workspace.rollback()
                state.warnings.append("Applied workspace changes were rolled back after workflow failure.")
            state.status = "failed"
            state.errors.append(f"Unexpected modification workflow failure: {exc}")
            return self._result(state)

    @staticmethod
    def _generation_prompt(request: ActionRequest, repo_model: RepositoryModel, plan: str) -> str:
        files = "\n".join(f.path for f in repo_model.files[:200])
        target_content = ""
        if request.target_file:
            target_content = WorkspaceTools(repo_model.local_path).read_context(request.target_file)[:20_000]
        return (
            f"Developer request: {request.request}\n\nPhase 9 change plan:\n{plan}\n\n"
            f"Repository file list:\n{files}\n\nTarget file content (only if requested):\n{target_content}"
        )

    @staticmethod
    def _parse_changes(raw_response: str) -> tuple[list[ProposedChange], list[str]]:
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModificationError(f"LLM modification response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("changes"), list):
            raise ModificationError("LLM modification response must contain a changes list.")
        changes: list[ProposedChange] = []
        for item in payload["changes"]:
            if not isinstance(item, dict):
                raise ModificationError("Every proposed change must be an object.")
            try:
                operation = ChangeOperation(item.get("operation"))
            except ValueError as exc:
                raise ModificationError("Every proposed change needs a supported operation.") from exc
            change = ProposedChange(
                file_path=item.get("file_path"), operation=operation,
                start_line=item.get("start_line"), end_line=item.get("end_line"),
                new_content=item.get("new_content"),
            )
            changes.append(change)
        if not changes:
            raise ModificationError("LLM proposed no modifications.")
        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
            raise ModificationError("LLM warnings must be a list of strings.")
        return changes, warnings

    @staticmethod
    def _result(state: AgentState) -> AgentResult:
        return AgentResult(
            status=state.status, change_plan=state.change_plan,
            proposed_changes=state.proposed_changes, applied_changes=state.applied_changes,
            diffs=state.diffs, validation_result=state.validation_result,
            test_result=state.test_result, warnings=state.warnings, errors=state.errors,
        )
