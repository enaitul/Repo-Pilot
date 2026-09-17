"""
Maps a file path to a human-readable language name.

Deliberately a pure function with no I/O and no state — this makes it
trivially unit-testable and safely reusable by other modules (the
ImportExtractor needs to know the language too, and depends on this
module rather than duplicating the mapping).
"""

from __future__ import annotations

from pathlib import Path

from repopilot.config import LANGUAGE_EXTENSION_MAP


def detect_language(file_path: str) -> str:
    """
    Return the detected language name for `file_path`, or "Unknown" if
    the extension isn't in our map (e.g. a file with no recognized
    source-code extension).
    """
    extension = Path(file_path).suffix.lower()
    return LANGUAGE_EXTENSION_MAP.get(extension, "Unknown")


def is_recognized_source_file(file_path: str) -> bool:
    """True if we have a language mapping for this file's extension."""
    return detect_language(file_path) != "Unknown"
