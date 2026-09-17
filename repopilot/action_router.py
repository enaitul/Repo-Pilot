"""
Deterministic mapping from ActionType to the right PromptBuilder method.

No LLM is involved in routing. ActionRequest already carries an explicit
`action_type`, so routing is a plain, predictable dictionary lookup —
not a natural-language classification step. Automatically classifying
vague free text into an action type would be a genuinely separate
problem; this phase deliberately doesn't build it, since the caller
already states the action type directly.
"""

from __future__ import annotations

from typing import Callable

from repopilot.exceptions import ActionError
from repopilot.models import ActionType
from repopilot.prompt_builder import PromptBuilder

_PROMPT_BUILDERS: dict[ActionType, Callable] = {
    ActionType.TEST_GENERATION: PromptBuilder.build_test_generation,
    ActionType.DOCUMENTATION: PromptBuilder.build_documentation,
    ActionType.REFACTORING: PromptBuilder.build_refactoring,
    ActionType.BUG_ANALYSIS: PromptBuilder.build_bug_analysis,
    ActionType.CHANGE_PLAN: PromptBuilder.build_change_plan,
}


class ActionRouter:
    """Routes an ActionType to its corresponding PromptBuilder method."""

    @staticmethod
    def get_prompt_builder(action_type: ActionType) -> Callable:
        try:
            return _PROMPT_BUILDERS[action_type]
        except KeyError:
            raise ActionError(f"Unknown action type: {action_type!r}")
