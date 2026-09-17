"""
Walks a cloned repository's directory tree, applies ignore rules, and
produces FileMetadata for every recognized source file.

Kept separate from cloning (no network/git knowledge here) and separate
from language/import logic (this module just decides WHICH files are
in scope and reads their raw content; it delegates "what language is
this" and "what does it import" to their own modules).
"""

from __future__ import annotations

import os
from pathlib import Path

from repopilot.config import (
    IGNORED_DIRECTORIES,
    IGNORED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    BINARY_SNIFF_BYTES,
)
from repopilot.exceptions import WalkError
from repopilot.language_detector import detect_language, is_recognized_source_file
from repopilot.import_extractor import extract_imports
from repopilot.models import FileMetadata


def _looks_binary(file_path: Path) -> bool:
    """
    Heuristic binary check: read the first chunk of the file and look
    for a null byte. This is the same trick `git` itself uses to decide
    whether to treat a file as binary. Cheap, and catches binaries that
    slipped past extension-based filtering (e.g. no extension, or a
    misleading one).
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(BINARY_SNIFF_BYTES)
        return b"\x00" in chunk
    except OSError:
        # If we can't even read it, treat it as unreadable/binary and skip.
        return True


def _should_ignore_dir(dirname: str) -> bool:
    return dirname in IGNORED_DIRECTORIES or dirname.endswith(".egg-info")


def _should_ignore_file(file_path: Path) -> bool:
    if file_path.suffix.lower() in IGNORED_EXTENSIONS:
        return True
    if file_path.name.startswith("."):
        # dotfiles (.env, .DS_Store, etc.) — not source code we want to index
        return True
    return False


class RepoWalker:
    """Traverses a local repository directory and extracts file metadata."""

    def walk(self, root_path: str) -> list[FileMetadata]:
        """
        Walk `root_path`, returning FileMetadata for every recognized,
        non-ignored, non-binary source file found.
        """
        root = Path(root_path)
        if not root.exists() or not root.is_dir():
            raise WalkError(f"'{root_path}' is not a valid directory to walk.")

        results: list[FileMetadata] = []

        for current_dir, dirnames, filenames in os.walk(root_path):
            # Prune ignored directories IN PLACE so os.walk never
            # descends into them — critical for skipping node_modules
            # efficiently rather than filtering after a full walk.
            dirnames[:] = [d for d in dirnames if not _should_ignore_dir(d)]

            for filename in filenames:
                file_path = Path(current_dir) / filename
                metadata = self._process_file(file_path, root)
                if metadata is not None:
                    results.append(metadata)

        return results

    def _process_file(self, file_path: Path, root: Path) -> FileMetadata | None:
        """
        Build FileMetadata for a single file, or return None if it
        should be skipped (ignored extension, dotfile, binary, or an
        extension we don't recognize as source code).

        Any per-file OSError is swallowed and logged as a skip rather
        than aborting the whole walk — one unreadable file (broken
        symlink, permissions issue) shouldn't kill ingestion of an
        otherwise-healthy repo.
        """
        if _should_ignore_file(file_path):
            return None

        relative_path = str(file_path.relative_to(root))

        if not is_recognized_source_file(str(file_path)):
            return None

        try:
            size_bytes = file_path.stat().st_size
        except OSError:
            return None

        if _looks_binary(file_path):
            return None

        language = detect_language(str(file_path))

        if size_bytes > MAX_FILE_SIZE_BYTES:
            # Too large to safely read into memory for import parsing —
            # still report it exists, just without content-derived fields.
            return FileMetadata(
                path=relative_path,
                language=language,
                size_bytes=size_bytes,
                imports=[],
                line_count=None,
                skipped_content=True,
            )

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return FileMetadata(
                path=relative_path,
                language=language,
                size_bytes=size_bytes,
                imports=[],
                line_count=None,
                skipped_content=True,
            )

        imports = extract_imports(content, language)
        line_count = content.count("\n") + 1 if content else 0

        return FileMetadata(
            path=relative_path,
            language=language,
            size_bytes=size_bytes,
            imports=imports,
            line_count=line_count,
            skipped_content=False,
        )
