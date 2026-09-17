import json
import subprocess
from unittest.mock import MagicMock

from repopilot.action_service import ActionService
from repopilot.llm_provider import LLMProvider
from repopilot.models import ActionRequest, ActionResult, ActionType, RepositoryModel
from repopilot.modification_service import ModificationService
from repopilot.controlled_test_runner import TestRunner


def _plan_service():
    service = MagicMock(spec=ActionService)
    service.execute.return_value = ActionResult(
        action_type="change_plan", summary="Replace the selected line.", generated_content=None,
        findings=[], assumptions=[], warnings=[], sources=[],
    )
    return service


def _provider(payload):
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = json.dumps(payload)
    provider.model_name = "fake"
    return provider


def _repo(tmp_path):
    return RepositoryModel("https://github.com/example/project", str(tmp_path), [])


def test_modification_workflow_applies_validated_change_and_reports_diff(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n")
    provider = _provider({"changes": [{"file_path": "module.py", "operation": "replace", "start_line": 1, "end_line": 1, "new_content": "value = 2"}], "warnings": []})
    runner = MagicMock(spec=TestRunner)
    runner.run.return_value = MagicMock(status="passed")

    result = ModificationService(_plan_service(), provider, test_runner=runner).execute(
        ActionRequest(ActionType.REFACTORING, "change value", target_file="module.py"), _repo(tmp_path)
    )

    assert result.status == "tested"
    assert (tmp_path / "module.py").read_text() == "value = 2\n"
    assert result.validation_result.valid is True
    assert "-value = 1" in result.diffs[0].diff
    runner.run.assert_called_once_with(str(tmp_path))


def test_malformed_llm_changes_fail_without_write(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n")
    provider = _provider({"changes": "not-a-list", "warnings": []})

    result = ModificationService(_plan_service(), provider).execute(
        ActionRequest(ActionType.REFACTORING, "change value"), _repo(tmp_path)
    )

    assert result.status == "failed"
    assert (tmp_path / "module.py").read_text() == "value = 1\n"


def test_validation_failure_rolls_back_workspace(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n")
    provider = _provider({"changes": [{"file_path": "module.py", "operation": "replace", "start_line": 1, "end_line": 1, "new_content": "def broken(:"}], "warnings": []})

    result = ModificationService(_plan_service(), provider).execute(
        ActionRequest(ActionType.REFACTORING, "break code"), _repo(tmp_path)
    )

    assert result.status == "failed"
    assert "rolled back" in result.warnings[0]
    assert (tmp_path / "module.py").read_text() == "value = 1\n"


def test_failed_tests_roll_back_workspace_but_keep_diff_for_review(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n")
    provider = _provider({"changes": [{"file_path": "module.py", "operation": "replace", "start_line": 1, "end_line": 1, "new_content": "value = 2"}], "warnings": []})
    runner = MagicMock(spec=TestRunner)
    runner.run.return_value = MagicMock(status="failed")

    result = ModificationService(_plan_service(), provider, test_runner=runner).execute(
        ActionRequest(ActionType.REFACTORING, "change value"), _repo(tmp_path)
    )

    assert result.status == "failed"
    assert "-value = 1" in result.diffs[0].diff
    assert (tmp_path / "module.py").read_text() == "value = 1\n"


def test_test_runner_uses_fixed_pytest_command_and_handles_timeout(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    runner = TestRunner()
    assert runner.select_command(str(tmp_path))[1:] == ["-m", "pytest"]

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output="partial", stderr="slow")

    monkeypatch.setattr("repopilot.controlled_test_runner.subprocess.run", timeout)
    result = runner.run(str(tmp_path))
    assert result.status == "timed_out"
    assert result.command[1:] == ["-m", "pytest"]
