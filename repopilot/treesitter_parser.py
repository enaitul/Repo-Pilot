"""
Parses JavaScript/TypeScript source using Tree-sitter.

Python gets its own dependency-free parser (see python_parser.py)
because the standard library already provides one. Every other
language RepoPilot supports goes through Tree-sitter instead of a
bespoke parser per language, so adding language #4, #5, #6 later is
"add a grammar + a node-type mapping", not "write a new parser".

Node types handled (Phase 3 scope — classes/functions/methods only):
  class_declaration        -> CLASS
  function_declaration     -> FUNCTION  (or METHOD if lexically inside a class)
  generator_function_declaration -> same as above
  method_definition         -> METHOD   (always inside a class_body)

Arrow functions and function expressions assigned to variables are
intentionally out of scope for Phase 3 — they're common but structurally
ambiguous (is `const x = () => {}` a "function symbol"? sometimes),
and the notes for this phase scope symbols to classes/functions/methods.
"""

from __future__ import annotations

from typing import Optional

from repopilot.models import CodeSymbol, ParsedFile, ParseStatus, SymbolType
from repopilot.parser_base import Parser

try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_typescript as ts_typescript

    _JS_LANGUAGE = Language(ts_javascript.language())
    _TS_LANGUAGE = Language(ts_typescript.language_typescript())
    _TSX_LANGUAGE = Language(ts_typescript.language_tsx())
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _JS_LANGUAGE = _TS_LANGUAGE = _TSX_LANGUAGE = None
    _TREE_SITTER_AVAILABLE = False


_CLASS_NODE_TYPES = {"class_declaration"}
_FUNCTION_NODE_TYPES = {"function_declaration", "generator_function_declaration"}
_METHOD_NODE_TYPES = {"method_definition"}


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def _symbol_name(node, source_bytes: bytes) -> Optional[str]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return _node_text(name_node, source_bytes)


def _walk_for_symbols(
    node,
    source_bytes: bytes,
    parent_name: Optional[str] = None,
    parent_is_class: bool = False,
) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []

    for child in node.children:
        node_type = child.type
        start_line = child.start_point[0] + 1
        end_line = child.end_point[0] + 1

        if node_type in _CLASS_NODE_TYPES:
            name = _symbol_name(child, source_bytes)
            if name:
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=SymbolType.CLASS,
                        start_line=start_line,
                        end_line=end_line,
                        parent=parent_name,
                    )
                )
            symbols.extend(
                _walk_for_symbols(child, source_bytes, parent_name=name, parent_is_class=True)
            )
        elif node_type in _METHOD_NODE_TYPES:
            name = _symbol_name(child, source_bytes)
            if name:
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=SymbolType.METHOD,
                        start_line=start_line,
                        end_line=end_line,
                        parent=parent_name,
                    )
                )
            symbols.extend(
                _walk_for_symbols(child, source_bytes, parent_name=name, parent_is_class=False)
            )
        elif node_type in _FUNCTION_NODE_TYPES:
            name = _symbol_name(child, source_bytes)
            symbol_type = SymbolType.METHOD if parent_is_class else SymbolType.FUNCTION
            if name:
                symbols.append(
                    CodeSymbol(
                        name=name,
                        symbol_type=symbol_type,
                        start_line=start_line,
                        end_line=end_line,
                        parent=parent_name,
                    )
                )
            symbols.extend(
                _walk_for_symbols(child, source_bytes, parent_name=name, parent_is_class=False)
            )
        else:
            symbols.extend(
                _walk_for_symbols(
                    child, source_bytes, parent_name=parent_name, parent_is_class=parent_is_class
                )
            )

    return symbols


class TreeSitterParser(Parser):
    """Extracts classes/functions/methods from JS/TS source via Tree-sitter."""

    def _select_grammar(self, file_path: str, language: str):
        if language == "TypeScript" and file_path.endswith(".tsx"):
            return _TSX_LANGUAGE
        if language == "TypeScript":
            return _TS_LANGUAGE
        return _JS_LANGUAGE

    def parse(self, content: str, file_path: str, language: str) -> ParsedFile:
        if not content or not content.strip():
            return ParsedFile(path=file_path, language=language, status=ParseStatus.EMPTY_FILE)

        if not _TREE_SITTER_AVAILABLE:
            return ParsedFile(
                path=file_path,
                language=language,
                status=ParseStatus.PARSER_ERROR,
                error_message=(
                    "Tree-sitter is not installed. Install tree-sitter, "
                    "tree-sitter-javascript, and tree-sitter-typescript to parse "
                    f"{language} files."
                ),
            )

        try:
            ts_language = self._select_grammar(file_path, language)
            ts_parser = TSParser(ts_language)
            source_bytes = content.encode("utf8")
            tree = ts_parser.parse(source_bytes)
        except Exception as exc:  # defensive: a grammar/runtime failure is our bug, not the file's
            return ParsedFile(
                path=file_path,
                language=language,
                status=ParseStatus.PARSER_ERROR,
                error_message=str(exc),
            )

        try:
            symbols = _walk_for_symbols(tree.root_node, source_bytes)
        except Exception as exc:
            return ParsedFile(
                path=file_path,
                language=language,
                status=ParseStatus.PARSER_ERROR,
                error_message=f"Symbol extraction failed: {exc}",
            )

        # Tree-sitter is error-tolerant: it still returns a tree (and often
        # still-usable symbols) for malformed source rather than raising.
        # We surface that as SYNTAX_ERROR while still returning whatever
        # symbols were successfully recovered around the broken section.
        if tree.root_node.has_error:
            return ParsedFile(
                path=file_path,
                language=language,
                status=ParseStatus.SYNTAX_ERROR,
                symbols=symbols,
                error_message="Tree-sitter reported one or more syntax errors in this file.",
            )

        return ParsedFile(
            path=file_path,
            language=language,
            status=ParseStatus.SUCCESS,
            symbols=symbols,
        )
