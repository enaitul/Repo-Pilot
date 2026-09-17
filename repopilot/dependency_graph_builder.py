"""
Builds a repository-internal dependency graph from data Phase 1 already
extracted — FileMetadata.imports. No new parsing or scanning happens
here; this is purely NEW INTERPRETATION of existing data.

IMPORTANT HONESTY NOTE: FileMetadata.imports stores import strings
exactly as written in source code (e.g. "os", "typing", "sample.simple")
— NOT resolved file paths. Most of those imports are external libraries
that say nothing about this repository's own internal structure (e.g.
"os", "pytest"). This module's real job is figuring out which import
strings correspond to an ACTUAL file inside this repository, and only
turning THOSE into graph edges. An import that can't be resolved to a
real file in the repo is simply not represented as an edge — never
guessed at.

This is a DEPENDENCY graph (which files reference which other files),
explicitly NOT a call graph (which functions call which functions at
runtime) — see repository_overview_builder.py's docstring for why that
distinction matters and why only the former is attempted here.

Resolution is intentionally simple: it only recognizes Python-style
dotted import paths (e.g. "package.module" -> "package/module.py"),
since that's what RepoPilot's own import_extractor.py already reliably
produces for Python. JS/TS import resolution (relative paths, bundler
aliases, etc.) is a genuinely harder problem and is deliberately left
unresolved for now rather than guessed at — those imports just won't
appear as edges, which is more honest than a wrong guess.
"""

from __future__ import annotations

from repopilot.models import DependencyEdge, DependencyGraph, RepositoryModel


class DependencyGraphBuilder:
    """Builds a DependencyGraph from a RepositoryModel's existing import data."""

    def build(self, repo_model: RepositoryModel) -> DependencyGraph:
        nodes = sorted(self._normalize(f.path) for f in repo_model.files)
        file_paths = set(nodes)

        edges: list[DependencyEdge] = []
        seen_edges: set[tuple[str, str]] = set()
        for file_meta in repo_model.files:
            source_path = self._normalize(file_meta.path)
            for import_name in file_meta.imports:
                target_path = self._resolve(import_name, file_paths)
                if target_path and target_path != source_path:
                    edge_key = (source_path, target_path)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(DependencyEdge(source=source_path, target=target_path))

        return DependencyGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _normalize(path: str) -> str:
        # Windows produces backslash paths; keep the graph consistent
        # regardless of which OS RepoPilot ran on.
        return path.replace("\\", "/")

    @staticmethod
    def _resolve(import_name: str, file_paths: set[str]) -> str | None:
        """
        Try to resolve a Python-style dotted import (e.g. "pkg.module")
        to an actual file path already known to exist in this
        repository. Returns None if no match is found — a miss is not
        an error, it just means this import doesn't produce an edge.
        """
        candidate_suffixes = (
            import_name.replace(".", "/") + ".py",
            import_name.replace(".", "/") + "/__init__.py",
        )
        for path in file_paths:
            for suffix in candidate_suffixes:
                if path == suffix or path.endswith("/" + suffix):
                    return path
        return None
