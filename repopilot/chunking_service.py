"""
Orchestrates Phase 4 (chunking) over an already-parsed repository.

Mirrors the shape of ingestion_service.py and parsing_service.py on
purpose: this is the only module that knows the end-to-end CHUNKING
pipeline order (for each parsed file -> read its source text -> ask the
Chunker for CodeChunks -> collect everything). Callers should call into
THIS module rather than wiring Chunker + file reading themselves.

Takes a RepositoryModel (Phase 1/2's output, for local_path) together
with the list[ParsedFile] (Phase 3's output, for symbols) — the same
two-input shape cli.py already assembles today. Like ParsingService, it
re-reads each file's content itself rather than requiring it to be
carried through the pipeline in memory; this keeps every phase simple
and independent, at the cost of reading each file's bytes more than
once across the whole run.
"""

from __future__ import annotations

from pathlib import Path

from repopilot.chunker import Chunker
from repopilot.models import CodeChunk, ParsedFile, ParseStatus, RepositoryModel


class ChunkingService:
    """High-level entry point for Phase 4: chunking an already-parsed repository."""

    def __init__(self, chunker: Chunker | None = None):
        # Constructor injection, same pattern as IngestionService/ParsingService
        # — lets tests substitute a fake chunker without real chunking logic.
        self._chunker = chunker or Chunker()

    def chunk_repository(
        self, repo_model: RepositoryModel, parsed_files: list[ParsedFile]
    ) -> list[CodeChunk]:
        """
        Chunk every file in `parsed_files` whose source lives under
        `repo_model.local_path`.

        One unreadable or symbol-less file never aborts the rest of the
        batch — it just contributes zero chunks — matching the same
        "one bad file can't kill the run" philosophy used throughout
        ingestion and parsing.
        """
        all_chunks: list[CodeChunk] = []
        for parsed_file in parsed_files:
            all_chunks.extend(self._chunk_one_file(repo_model.local_path, parsed_file))
        return all_chunks

    def _chunk_one_file(self, local_path: str, parsed_file: ParsedFile) -> list[CodeChunk]:
        # Nothing to chunk for a file we don't parse, or one with no content.
        if parsed_file.status in (ParseStatus.UNSUPPORTED_LANGUAGE, ParseStatus.EMPTY_FILE):
            return []
        if not parsed_file.symbols:
            return []

        full_path = Path(local_path) / parsed_file.path
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            # Same defensive stance as ParsingService.parse_file: an
            # unreadable file (permissions, disappeared mid-run) just
            # contributes no chunks, rather than crashing the whole batch.
            return []

        return self._chunker.chunk_file(parsed_file, content)
