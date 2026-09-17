import pytest

from repopilot.action_router import ActionRouter
from repopilot.exceptions import ActionError
from repopilot.models import ActionType
from repopilot.prompt_builder import PromptBuilder


@pytest.mark.parametrize(
    "action_type,expected_method",
    [
        (ActionType.TEST_GENERATION, PromptBuilder.build_test_generation),
        (ActionType.DOCUMENTATION, PromptBuilder.build_documentation),
        (ActionType.REFACTORING, PromptBuilder.build_refactoring),
        (ActionType.BUG_ANALYSIS, PromptBuilder.build_bug_analysis),
        (ActionType.CHANGE_PLAN, PromptBuilder.build_change_plan),
    ],
)
def test_routes_each_action_type_to_the_correct_prompt_builder(action_type, expected_method):
    assert ActionRouter.get_prompt_builder(action_type) is expected_method


def test_all_five_action_types_are_routable():
    for action_type in ActionType:
        # Should not raise for any real ActionType.
        ActionRouter.get_prompt_builder(action_type)
