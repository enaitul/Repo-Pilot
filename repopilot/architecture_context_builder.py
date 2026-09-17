"""
Formats a RepositoryOverview into one clean, labeled text block for the
LLM — the Phase 8 counterpart to context_builder.py from Phase 7.

WHY NOT SEND THE RAW RepositoryOverview OBJECT: same reasoning as Phase
7's ContextBuilder — a clearly labeled, human-readable layout is easier
for the model to read accurately than Python object syntax, and it lets
us control exactly what does and doesn't get sent (no raw source code
dumped in bulk here, only structural facts).
"""

from __future__ import annotations

from repopilot.models import RepositoryOverview


class ArchitectureContextBuilder:
    """Formats a RepositoryOverview into LLM-ready architecture context text."""

    @staticmethod
    def build(overview: RepositoryOverview) -> str:
        lines: list[str] = [f"Repository: {overview.repo_url}", ""]

        lines.append("Languages:")
        lines.extend(ArchitectureContextBuilder._language_lines(overview))
        lines.append("")

        lines.append("Technologies:")
        lines.extend(ArchitectureContextBuilder._technology_lines(overview))
        lines.append("")

        lines.append("Entry points:")
        lines.extend(ArchitectureContextBuilder._path_lines(overview.entry_points))
        lines.append("")

        lines.append("Important files:")
        lines.extend(ArchitectureContextBuilder._path_lines(overview.important_files))
        lines.append("")

        lines.append(
            "Internal dependency graph (file imports file, based on detected "
            "import statements — this is NOT a runtime call graph):"
        )
        lines.extend(ArchitectureContextBuilder._edge_lines(overview))

        isolated = overview.dependency_graph.isolated_nodes
        if isolated:
            lines.append("")
            lines.append(
                "Files with no detected internal dependency connections: "
                + ", ".join(isolated)
            )

        return "\n".join(lines)

    @staticmethod
    def _language_lines(overview: RepositoryOverview) -> list[str]:
        if not overview.languages:
            return ["  (none detected)"]
        ranked = sorted(overview.languages.items(), key=lambda kv: -kv[1])
        return [f"  - {lang}: {count} file(s)" for lang, count in ranked]

    @staticmethod
    def _technology_lines(overview: RepositoryOverview) -> list[str]:
        if not overview.technologies:
            return ["  (none detected)"]
        return [f"  - {t.name} [{t.confidence}] ({t.evidence})" for t in overview.technologies]

    @staticmethod
    def _path_lines(paths: list[str]) -> list[str]:
        if not paths:
            return ["  (none detected)"]
        return [f"  - {p}" for p in paths]

    @staticmethod
    def _edge_lines(overview: RepositoryOverview) -> list[str]:
        if not overview.dependency_graph.edges:
            return ["  (no internal dependencies resolved)"]
        return [f"  - {e.source} -> {e.target}" for e in overview.dependency_graph.edges]
