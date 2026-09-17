import os
from pathlib import Path

from repopilot.walker import RepoWalker
from repopilot.exceptions import WalkError

import pytest


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_walks_and_finds_source_files(tmp_path):
    _write(tmp_path / "main.py", "import os\nprint('hi')\n")
    _write(tmp_path / "app.js", "const x = 1;\n")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert "main.py" in paths
    assert "app.js" in paths


def test_ignores_configured_directories(tmp_path):
    _write(tmp_path / "node_modules" / "lib" / "index.js", "module.exports = {};")
    _write(tmp_path / "__pycache__" / "main.cpython-312.pyc", "junk")
    _write(tmp_path / "src" / "main.py", "import sys")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert not any("node_modules" in p for p in paths)
    assert not any("__pycache__" in p for p in paths)
    # Use os.path.join instead of a hardcoded "/" so this works on both
    # Windows (backslash paths) and Mac/Linux (forward-slash paths).
    assert os.path.join("src", "main.py") in paths


def test_ignores_git_directory(tmp_path):
    _write(tmp_path / ".git" / "config", "[core]")
    _write(tmp_path / "main.py", "x = 1")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert not any(".git" in p for p in paths)
    assert "main.py" in paths


def test_ignores_binary_files_by_extension(tmp_path):
    _write(tmp_path / "logo.png", "fake binary content")
    _write(tmp_path / "main.py", "x = 1")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert "logo.png" not in paths
    assert "main.py" in paths


def test_ignores_binary_content_by_null_byte_sniff(tmp_path):
    # A file with no recognizable "binary" extension but null-byte content
    # would only be caught if it also happened to have a source extension.
    binary_path = tmp_path / "weird.py"
    binary_path.write_bytes(b"\x00\x01\x02binarydata")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert "weird.py" not in paths


def test_ignores_unrecognized_extensions(tmp_path):
    _write(tmp_path / "notes.txt", "just some notes")
    _write(tmp_path / "main.py", "x = 1")

    files = RepoWalker().walk(str(tmp_path))
    paths = {f.path for f in files}

    assert "notes.txt" not in paths
    assert "main.py" in paths


def test_extracts_imports_for_python_file(tmp_path):
    _write(tmp_path / "main.py", "import os\nimport sys\n")

    files = RepoWalker().walk(str(tmp_path))
    main_file = next(f for f in files if f.path == "main.py")

    assert "os" in main_file.imports
    assert "sys" in main_file.imports
    assert main_file.language == "Python"


def test_raises_on_nonexistent_directory():
    with pytest.raises(WalkError):
        RepoWalker().walk("/this/path/does/not/exist")


def test_large_file_marked_skipped_content(tmp_path, monkeypatch):
    import repopilot.walker as walker_module
    monkeypatch.setattr(walker_module, "MAX_FILE_SIZE_BYTES", 10)

    _write(tmp_path / "big.py", "x = 'this is definitely more than ten bytes'")

    files = RepoWalker().walk(str(tmp_path))
    big_file = next(f for f in files if f.path == "big.py")

    assert big_file.skipped_content is True
    assert big_file.imports == []
