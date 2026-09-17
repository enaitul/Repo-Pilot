"""
Parses Python source using the standard-library `ast` module.

Python already ships a solid, grammar-aware parser, so there's no
reason to route Python through Tree-sitter or any other dependency —
this keeps the common case (Python repos) fast and dependency-free.

Only classes, functions, and methods are extracted (Phase 3 scope).
A function is classified as a METHOD when its nearest enclosing scope
is a class body; otherwise it's a FUNCTION (this also correctly
classifies nested/closure functions defined inside another function
as FUNCTION, not METHOD).
"""

from __future__ import annotations

import ast
from typing import Optional

from repopilot.models import CodeSymbol, ParsedFile, ParseStatus, SymbolType
from repopilot.parser_base import Parser


def _end_line(node: ast.AST) -> int:
    # end_lineno is available on Python 3.8+ for all statement nodes we
    # care about here. Fall back to lineno defensively in case a node
    # type somehow lacks it.
    return getattr(node, "end_lineno", None) or node.lineno


def _walk_for_symbols(
    node: ast.AST,
    parent_name: Optional[str] = None,
    parent_is_class: bool = False,
) -> list[CodeSymbol]:
    """
    Recursively walk direct-and-nested child nodes of `node`, collecting
    CodeSymbols for every ClassDef/FunctionDef/AsyncFunctionDef found —
    including ones nested inside if/try/with blocks, not just ones
    directly in a class or module body.

    `parent_name`/`parent_is_class` describe the nearest enclosing
    class or function, threaded through unchanged for node types that
    aren't themselves a class/function (If, Try, With, ...).
    """
    symbols: list[CodeSymbol] = []

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            symbols.append(
                CodeSymbol(
                    name=child.name,
                    symbol_type=SymbolType.CLASS,
                    start_line=child.lineno,
                    end_line=_end_line(child),
                    parent=parent_name,
                )
            )
            symbols.extend(
                _walk_for_symbols(child, parent_name=child.name, parent_is_class=True)
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_type = SymbolType.METHOD if parent_is_class else SymbolType.FUNCTION
            symbols.append(
                CodeSymbol(
                    name=child.name,
                    symbol_type=symbol_type,
                    start_line=child.lineno,
                    end_line=_end_line(child),
                    parent=parent_name,
                )
            )
            symbols.extend(
                _walk_for_symbols(child, parent_name=child.name, parent_is_class=False)
            )
        else:
            symbols.extend(
                _walk_for_symbols(child, parent_name=parent_name, parent_is_class=parent_is_class)
            )

    return symbols


class PythonParser(Parser):
    """Extracts classes/functions/methods from Python source via `ast`."""

    def parse(self, content: str, file_path: str, language: str = "Python") -> ParsedFile:
        if not content or not content.strip():
            return ParsedFile(path=file_path, language="Python", status=ParseStatus.EMPTY_FILE)

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as exc:
            return ParsedFile(
                path=file_path,
                language="Python",
                status=ParseStatus.SYNTAX_ERROR,
                error_message=f"line {exc.lineno}: {exc.msg}",
            )
        except (ValueError, RecursionError) as exc:
            # ValueError: e.g. null bytes in source. RecursionError: e.g.
            # pathologically deep nesting. Neither is "our" bug.
            return ParsedFile(
                path=file_path,
                language="Python",
                status=ParseStatus.PARSER_ERROR,
                error_message=str(exc),
            )

        try:
            symbols = _walk_for_symbols(tree)
        except Exception as exc:  # defensive: symbol extraction must never crash the pipeline
            return ParsedFile(
                path=file_path,
                language="Python",
                status=ParseStatus.PARSER_ERROR,
                error_message=f"Symbol extraction failed: {exc}",
            )

        return ParsedFile(
            path=file_path,
            language="Python",
            status=ParseStatus.SUCCESS,
            symbols=symbols,
        )
