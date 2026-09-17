import pytest

from repopilot.exceptions import ModificationError
from repopilot.models import ChangeOperation, ProposedChange
from repopilot.workspace_tools import WorkspaceTools


def test_create_replace_delete_and_actual_diffs(tmp_path):
    (tmp_path / "module.py").write_text("one\ntwo\n")
    tools = WorkspaceTools(str(tmp_path))
    changes = [
        ProposedChange("module.py", ChangeOperation.REPLACE, 2, 2, "changed"),
        ProposedChange("new.py", ChangeOperation.CREATE, new_content="value = 1\n"),
    ]

    tools.apply(changes)

    assert (tmp_path / "module.py").read_text() == "one\nchanged\n"
    assert (tmp_path / "new.py").read_text() == "value = 1\n"
    assert "-two" in tools.diffs()[0].diff
    assert "+changed" in tools.diffs()[0].diff


def test_rejects_path_traversal_and_symlink_escape(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ModificationError):
        tools.apply([ProposedChange("../../outside.py", ChangeOperation.CREATE, new_content="x")])

    outside = tmp_path.parent / "outside.py"
    outside.write_text("x")
    (tmp_path / "escape.py").symlink_to(outside)
    with pytest.raises(ModificationError):
        tools.apply([ProposedChange("escape.py", ChangeOperation.REPLACE, 1, 1, "y")])


def test_rejects_sensitive_file_as_llm_context(tmp_path):
    (tmp_path / "api_key.txt").write_text("do-not-send")
    with pytest.raises(ModificationError):
        WorkspaceTools(str(tmp_path)).read_context("api_key.txt")


def test_rejects_malformed_changes_without_partial_write(tmp_path):
    (tmp_path / "existing.py").write_text("x = 1\n")
    tools = WorkspaceTools(str(tmp_path))
    with pytest.raises(ModificationError):
        tools.apply([
            ProposedChange("new.py", ChangeOperation.CREATE, new_content="x = 2\n"),
            ProposedChange("existing.py", ChangeOperation.REPLACE, 9, 9, "x = 3"),
        ])
    assert not (tmp_path / "new.py").exists()
    assert (tmp_path / "existing.py").read_text() == "x = 1\n"


def test_rollback_restores_create_replace_and_delete(tmp_path):
    (tmp_path / "module.py").write_text("old\n")
    (tmp_path / "remove.py").write_text("remove\n")
    tools = WorkspaceTools(str(tmp_path))
    tools.apply([
        ProposedChange("module.py", ChangeOperation.REPLACE, 1, 1, "new"),
        ProposedChange("new.py", ChangeOperation.CREATE, new_content="new\n"),
        ProposedChange("remove.py", ChangeOperation.DELETE),
    ])
    tools.rollback()

    assert (tmp_path / "module.py").read_text() == "old\n"
    assert not (tmp_path / "new.py").exists()
    assert (tmp_path / "remove.py").read_text() == "remove\n"
