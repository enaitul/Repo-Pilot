"""
Common interface every language-specific parser implements.

Keeping this as a tiny ABC (rather than duck-typing) means ParserFactory
and ParsingService can depend on a single contract regardless of what's
underneath — Python's built-in `ast`, Tree-sitter, or (later) something
else entirely for a new language. Adding a language means writing one
new class that satisfies this interface; nothing upstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from repopilot.models import ParsedFile


class Parser(ABC):
    """Parses source code for one or more languages into a ParsedFile."""

    @abstractmethod
    def parse(self, content: str, file_path: str, language: str) -> ParsedFile:
        """
        Parse `content` (the full text of one source file) and return a
        ParsedFile describing what was found.

        `file_path` is used for the returned ParsedFile.path (and, for
        parsers that cover more than one language/grammar, may be used
        to disambiguate — e.g. .ts vs .tsx).

        `language` is the language name as already determined by
        language_detector (e.g. "Python", "TypeScript") — implementations
        should not need to re-detect it.

        MUST NOT raise for malformed/unparseable source. Every failure
        mode short of a genuine bug in the parser implementation should
        be caught and reported via ParsedFile.status /
        ParsedFile.error_message instead — a broken file in the target
        repository is expected input, not an application error.
        """
        raise NotImplementedError
