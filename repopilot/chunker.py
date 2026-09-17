"""
Splits ONE already-parsed file's source code into CodeChunks.

This is the Phase 4 counterpart to python_parser.py/treesitter_parser.py:
those turned source TEXT into CodeSymbols (locations + names, no text).
This module turns CodeSymbols + the original TEXT back into CodeChunks
(locations + names + the actual text), sized to be useful later as input
to an embedding model.

Chunking strategy, in order of preference:
  1. If a symbol's own text already fits under MAX_CHUNK_CHARS, keep it
     as ONE chunk — the common case for most functions/methods.
  2. If a CLASS is too big, don't chunk the whole class as one blob —
     chunk each of its own methods individually instead. We already know
     which symbols belong to it via CodeSymbol.parent, so this is just
     recursion, not new parsing work.
  3. If a single FUNCTION/METHOD is too big and has no smaller structural
     unit to fall back on, fall back to a raw line-based sliding window
     with a small overlap between consecutive windows, so information
     sitting exactly at a cut boundary isn't fully lost from both sides.

A Chunker never raises on a "difficult" file — same philosophy as every
other worker in this pipeline (import_extractor, the parsers): a file
that produces zero usable symbols simply produces zero chunks.
"""

from __future__ import annotations

from repopilot.config import CHUNK_OVERLAP_LINES, MAX_CHUNK_CHARS
from repopilot.models import CodeChunk, CodeSymbol, ParsedFile, SymbolType


class Chunker:
    """Turns one file's (ParsedFile, source text) pair into CodeChunks."""

    def chunk_file(self, parsed_file: ParsedFile, content: str) -> list[CodeChunk]:
        """
        Return the CodeChunks for `parsed_file`, whose full source text is
        `content`. Returns an empty list if there are no symbols to chunk
        (e.g. an empty file, an unsupported language, or a file that
        failed to parse with zero recovered symbols).
        """
        if not parsed_file.symbols:
            return []

        lines = content.splitlines()

        # Only start from TOP-LEVEL symbols (parent is None). Symbols with
        # a parent (e.g. a method inside a class) are only visited if we
        # recurse into them from their parent class below — this avoids
        # ever emitting both "the whole class" AND "each of its methods"
        # as separate, overlapping chunks when the class was small enough
        # to keep whole in the first place.
        top_level_symbols = [s for s in parsed_file.symbols if s.parent is None]

        chunks: list[CodeChunk] = []
        for symbol in top_level_symbols:
            chunks.extend(self._chunk_symbol(symbol, parsed_file, lines))
        return chunks

    # -- internal helpers ---------------------------------------------------

    def _chunk_symbol(
        self, symbol: CodeSymbol, parsed_file: ParsedFile, lines: list[str]
    ) -> list[CodeChunk]:
        """Decide HOW to chunk one symbol, and return one or more CodeChunks."""
        text = self._slice_lines(lines, symbol.start_line, symbol.end_line)

        if len(text) <= MAX_CHUNK_CHARS:
            return [
                self._make_chunk(
                    parsed_file, text, symbol.start_line, symbol.end_line,
                    symbol.name, symbol.symbol_type.value, symbol.parent,
                )
            ]

        # Too big for one chunk. Prefer a structural split over a blind one.
        if symbol.symbol_type == SymbolType.CLASS:
            methods = [s for s in parsed_file.symbols if s.parent == symbol.name]
            if methods:
                result: list[CodeChunk] = []
                for method in methods:
                    result.extend(self._chunk_symbol(method, parsed_file, lines))
                return result
            # A class with no tracked methods (e.g. an empty class, or one
            # whose body isn't methods) has nothing smaller to fall back
            # on — fall through to the raw split below.

        return self._split_oversized_symbol(
            parsed_file, lines, symbol.start_line, symbol.end_line,
            symbol.name, symbol.symbol_type.value, symbol.parent,
        )

    def _split_oversized_symbol(
        self,
        parsed_file: ParsedFile,
        lines: list[str],
        start_line: int,
        end_line: int,
        symbol_name: str,
        symbol_type: str,
        parent: str | None,
    ) -> list[CodeChunk]:
        """
        Last resort: cut a single large symbol into several smaller chunks
        using a sliding window over its lines, with a small overlap
        between consecutive windows (CHUNK_OVERLAP_LINES).
        """
        full_text = self._slice_lines(lines, start_line, end_line)
        total_lines = end_line - start_line + 1

        # Estimate how many lines fit under MAX_CHUNK_CHARS using this
        # symbol's OWN average characters-per-line, rather than a fixed
        # guess — dense code (long lines) gets fewer lines per chunk,
        # sparse code (short lines) gets more, and both stay under budget.
        avg_line_len = max(len(full_text) // max(total_lines, 1), 1)
        lines_per_window = max(MAX_CHUNK_CHARS // avg_line_len, 1)

        windows: list[tuple[int, int]] = []
        current_start = start_line
        while current_start <= end_line:
            current_end = min(current_start + lines_per_window - 1, end_line)
            windows.append((current_start, current_end))
            if current_end >= end_line:
                break
            # Step forward, but re-include the last few lines of the
            # previous window so context isn't lost right at the cut.
            current_start = current_end - CHUNK_OVERLAP_LINES + 1

        chunk_count = len(windows)
        chunks = []
        for index, (window_start, window_end) in enumerate(windows):
            text = self._slice_lines(lines, window_start, window_end)
            chunks.append(
                self._make_chunk(
                    parsed_file, text, window_start, window_end,
                    symbol_name, symbol_type, parent,
                    chunk_index=index, chunk_count=chunk_count,
                )
            )
        return chunks

    @staticmethod
    def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
        # CodeSymbol line numbers are 1-based (line 1 = first line of the
        # file). Python lists are 0-based, so we subtract 1 from
        # start_line to convert to a list index. We do NOT subtract 1
        # from end_line: Python's slice lines[a:b] already excludes b,
        # which lines up exactly with "give me everything up to and
        # including end_line".
        return "\n".join(lines[start_line - 1:end_line])

    @staticmethod
    def _make_chunk(
        parsed_file: ParsedFile,
        text: str,
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        symbol_type: str | None,
        parent: str | None,
        chunk_index: int = 0,
        chunk_count: int = 1,
    ) -> CodeChunk:
        chunk_id = f"{parsed_file.path}:{start_line}-{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=parsed_file.path,
            language=parsed_file.language,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            parent=parent,
            start_line=start_line,
            end_line=end_line,
            content=text,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )
