"""
Best-effort import/dependency extraction from source file content.

IMPORTANT DESIGN NOTE (say this out loud in an interview): this is a
regex-based heuristic, not a real parser. It will miss dynamic imports,
can be fooled by strings/comments that look like import statements, and
doesn't resolve relative imports to real file paths. It's intentionally
scoped this way for Phase 1: zero extra dependencies, fast, and covers
the common case for the languages we care about first.

The clean upgrade path — swapping this module's internals for
`ast` (Python) or `tree-sitter` (multi-language) — is exactly why this
lives behind one function signature (`extract_imports`) rather than
being inlined into the walker. Callers never need to know it changed.
"""

from __future__ import annotations

import re

# Each pattern captures the imported module/path as group(1).
_PYTHON_PATTERNS = [
    re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),
    re.compile(r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE),
]

_JS_TS_PATTERNS = [
    re.compile(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]"""),
    re.compile(r"""import\s+['"]([^'"]+)['"]"""),               # side-effect import
    re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
]

_JAVA_PATTERNS = [
    re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE),
]

_GO_PATTERNS = [
    re.compile(r'^\s*import\s+"([^"]+)"', re.MULTILINE),
    re.compile(r'^\s*"([^"]+)"', re.MULTILINE),  # lines inside import ( ... ) blocks
]

_RUBY_PATTERNS = [
    re.compile(r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]""", re.MULTILINE),
]

_LANGUAGE_PATTERNS: dict[str, list[re.Pattern]] = {
    "Python": _PYTHON_PATTERNS,
    "JavaScript": _JS_TS_PATTERNS,
    "TypeScript": _JS_TS_PATTERNS,
    "Java": _JAVA_PATTERNS,
    "Go": _GO_PATTERNS,
    "Ruby": _RUBY_PATTERNS,
}


def extract_imports(content: str, language: str) -> list[str]:
    """
    Return a de-duplicated, order-preserving list of import targets found
    in `content` for the given `language`. Returns an empty list for
    languages we don't have patterns for, or on any parsing hiccup —
    this function must never raise, since a malformed source file
    shouldn't be able to crash the whole ingestion pipeline.
    """
    patterns = _LANGUAGE_PATTERNS.get(language)
    if not patterns:
        return []

    try:
        found: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(content):
                module = match.group(1).strip()
                if module and module not in seen:
                    seen.add(module)
                    found.append(module)
        return found
    except re.error:
        # Defensive: should never happen with our fixed patterns, but
        # extraction failures must degrade gracefully, not crash ingestion.
        return []
