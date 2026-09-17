"""
Orchestrates Phase 3 (code parsing) over an already-ingested repository.

Mirrors the shape of ingestion_service.py: this is the only module that
knows the end-to-end PARSING pipeline order (pick a parser -> read the
file -> parse -> collect results). Callers (CLI today; later an HTTP
endpoint or agent tool) should call into THIS module rather than
wiring ParserFactory + file reading themselves.

Deliberately takes a RepositoryModel (Phase 1/2's output) rather than a
repo_url — parsing is a distinct phase that operates on files already
on disk. It re-reads file content itself (rather than requiring
FileMetadata to carry full content) to keep Phase 1/2's FileMetadata
lightweight and keep this phase self-contained.
"""

from __future__ import annotations

from pathlib import Path

from repopilot.models import FileMetadata, ParsedFile, ParseStatus, RepositoryModel
from repopilot.parser_factory import ParserFactory


class ParsingService:
    """High-level entry point for Phase 3: parsing an ingested repository."""

    def __init__(self, factory: ParserFactory | None = None):
        # Constructor injection, same pattern as IngestionService — lets
        # tests substitute a fake factory/parser without real parsing.
        self._factory = factory or ParserFactory()

    def parse_repository(self, repo_model: RepositoryModel) -> list[ParsedFile]:
        """
        Parse every file in `repo_model`.

        One bad/unsupported/unreadable file must NEVER abort parsing of
        the rest of the repository — each file's outcome (including
        failure) is captured as its own ParsedFile entry in the returned
        list, never raised as an exception. See parse_file.
        """
        return [self.parse_file(repo_model.local_path, file_meta) for file_meta in repo_model.files]

    def parse_file(self, local_path: str, file_meta: FileMetadata) -> ParsedFile:
        """
        Parse a single file described by `file_meta`, whose content lives
        at `local_path / file_meta.path` on disk.
        """
        parser = self._factory.get_parser(file_meta.language)
        if parser is None:
            return ParsedFile(
                path=file_meta.path,
                language=file_meta.language,
                status=ParseStatus.UNSUPPORTED_LANGUAGE,
            )

        if file_meta.skipped_content:
            # Walker (Phase 1/2) already decided this file was too large
            # to safely read into memory. Respect that policy here too
            # rather than re-reading a potentially huge file just to parse it.
            return ParsedFile(
                path=file_meta.path,
                language=file_meta.language,
                status=ParseStatus.PARSER_ERROR,
                error_message="File content was skipped during ingestion (exceeds size limit).",
            )

        full_path = Path(local_path) / file_meta.path
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return ParsedFile(
                path=file_meta.path,
                language=file_meta.language,
                status=ParseStatus.PARSER_ERROR,
                error_message=f"Could not read file: {exc}",
            )

        return parser.parse(content, file_meta.path, file_meta.language)
