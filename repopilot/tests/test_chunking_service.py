from pathlib import Path
from unittest.mock import MagicMock

from repopilot.chunker import Chunker
from repopilot.chunking_service import ChunkingService
from repopilot.models import (
    CodeChunk,
    CodeSymbol,
    ParsedFile,
    ParseStatus,
    RepositoryModel,
    SymbolType,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_chunk_repository_reads_files_and_delegates_to_chunker(tmp_path):
    _write(tmp_path / "greet.py", "def greet():\n    print('hi')\n")

    repo_model = RepositoryModel(
        repo_url="https://github.com/fake/repo",
        local_path=str(tmp_path),
        files=[],
    )
    parsed_files = [
        ParsedFile(
            path="greet.py",
            language="Python",
            status=ParseStatus.SUCCESS,
            symbols=[
                CodeSymbol(
                    name="greet", symbol_type=SymbolType.FUNCTION,
                    start_line=1, end_line=2, parent=None,
                )
            ],
        )
    ]

    chunks = ChunkingService(chunker=Chunker()).chunk_repository(repo_model, parsed_files)

    assert len(chunks) == 1
    assert chunks[0].symbol_name == "greet"
    assert "print('hi')" in chunks[0].content


def test_chunk_repository_skips_unsupported_and_empty_files(tmp_path):
    _write(tmp_path / "notes.txt", "just notes")

    repo_model = RepositoryModel(
        repo_url="https://github.com/fake/repo", local_path=str(tmp_path), files=[]
    )
    parsed_files = [
        ParsedFile(path="notes.txt", language="Unknown", status=ParseStatus.UNSUPPORTED_LANGUAGE),
        ParsedFile(path="blank.py", language="Python", status=ParseStatus.EMPTY_FILE),
    ]

    chunks = ChunkingService().chunk_repository(repo_model, parsed_files)

    assert chunks == []


def test_chunk_repository_skips_unreadable_file_without_crashing(tmp_path):
    repo_model = RepositoryModel(
        repo_url="https://github.com/fake/repo", local_path=str(tmp_path), files=[]
    )
    # File doesn't actually exist on disk -> read_text will raise OSError,
    # which chunk_repository must swallow rather than propagate.
    parsed_files = [
        ParsedFile(
            path="missing.py",
            language="Python",
            status=ParseStatus.SUCCESS,
            symbols=[
                CodeSymbol(
                    name="f", symbol_type=SymbolType.FUNCTION,
                    start_line=1, end_line=1, parent=None,
                )
            ],
        )
    ]

    chunks = ChunkingService().chunk_repository(repo_model, parsed_files)

    assert chunks == []


def test_chunk_repository_uses_injected_chunker(tmp_path):
    _write(tmp_path / "a.py", "def a():\n    pass\n")

    fake_chunk = CodeChunk(
        chunk_id="a.py:1-2", file_path="a.py", language="Python",
        symbol_name="a", symbol_type="function", parent=None,
        start_line=1, end_line=2, content="def a():\n    pass",
    )
    fake_chunker = MagicMock(spec=Chunker)
    fake_chunker.chunk_file.return_value = [fake_chunk]

    repo_model = RepositoryModel(
        repo_url="https://github.com/fake/repo", local_path=str(tmp_path), files=[]
    )
    parsed_files = [
        ParsedFile(
            path="a.py", language="Python", status=ParseStatus.SUCCESS,
            symbols=[CodeSymbol(name="a", symbol_type=SymbolType.FUNCTION, start_line=1, end_line=2)],
        )
    ]

    chunks = ChunkingService(chunker=fake_chunker).chunk_repository(repo_model, parsed_files)

    assert chunks == [fake_chunk]
    fake_chunker.chunk_file.assert_called_once()


def test_chunk_repository_aggregates_across_multiple_files(tmp_path):
    _write(tmp_path / "a.py", "def a():\n    pass\n")
    _write(tmp_path / "b.py", "def b():\n    pass\n")

    repo_model = RepositoryModel(
        repo_url="https://github.com/fake/repo", local_path=str(tmp_path), files=[]
    )
    parsed_files = [
        ParsedFile(
            path="a.py", language="Python", status=ParseStatus.SUCCESS,
            symbols=[CodeSymbol(name="a", symbol_type=SymbolType.FUNCTION, start_line=1, end_line=2)],
        ),
        ParsedFile(
            path="b.py", language="Python", status=ParseStatus.SUCCESS,
            symbols=[CodeSymbol(name="b", symbol_type=SymbolType.FUNCTION, start_line=1, end_line=2)],
        ),
    ]

    chunks = ChunkingService().chunk_repository(repo_model, parsed_files)

    assert {c.file_path for c in chunks} == {"a.py", "b.py"}
    assert len(chunks) == 2
