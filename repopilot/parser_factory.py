"""
Decides which Parser implementation handles a given language.

This keeps language-selection logic in exactly one place. ParsingService
doesn't know or care that Python goes through `ast` and JS/TS go through
Tree-sitter — it just asks the factory for a parser and calls it. Adding
a language later means registering it here; nothing in ParsingService
changes.
"""

from __future__ import annotations

from typing import Optional

from repopilot.parser_base import Parser
from repopilot.python_parser import PythonParser
from repopilot.treesitter_parser import TreeSitterParser

# Languages currently routed to the Tree-sitter parser. Extending this
# set (plus TreeSitterParser's own grammar selection) is how a future
# language like Java or Go would be added without touching ParsingService.
_TREE_SITTER_LANGUAGES = frozenset({"JavaScript", "TypeScript"})


class ParserFactory:
    """Maps a language name to a Parser instance capable of handling it."""

    def __init__(self):
        # Parsers are stateless w.r.t. any single file, so one instance
        # per parser type is reused across every file of that language
        # rather than constructed per-call.
        self._python_parser = PythonParser()
        self._treesitter_parser = TreeSitterParser()

    def get_parser(self, language: str) -> Optional[Parser]:
        """
        Return the Parser for `language`, or None if RepoPilot doesn't
        parse that language yet. None (not an exception) is the signal
        for "unsupported" — ParsingService turns that into a
        ParseStatus.UNSUPPORTED_LANGUAGE result for the affected file.
        """
        if language == "Python":
            return self._python_parser
        if language in _TREE_SITTER_LANGUAGES:
            return self._treesitter_parser
        return None
