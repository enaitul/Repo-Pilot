"""
Builds deterministic context for Phase 11 Intelligent Code Review.

Reuses Phase 6 (SearchService), Phase 8 (RepositoryOverview, DependencyGraph),
and Phase 10 (FileDiffs, ValidationResult, TestResult) without duplicating
any parsing or graph building logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from repopilot.architecture_context_builder import ArchitectureContextBuilder
from repopilot.dependency_graph_builder import DependencyGraphBuilder
from repopilot.exceptions import ReviewError
from repopilot.models import (
    FileDiff,
    RepositoryModel,
    ReviewMode,
    ReviewRequest,
)
from repopilot.repository_overview_builder import RepositoryOverviewBuilder
from repopilot.search_service import SearchService


class ReviewContextBuilder:
    """Builds deterministic, evidence-based context for code review."""

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        overview_builder: Optional[RepositoryOverviewBuilder] = None,
        graph_builder: Optional[DependencyGraphBuilder] = None,
    ):
        self._search_service = search_service
        self._overview_builder = overview_builder or RepositoryOverviewBuilder()
        self._graph_builder = graph_builder or DependencyGraphBuilder()

    def build(self, repo_model: RepositoryModel, request: ReviewRequest) -> str:
        """Route to appropriate context builder based on review mode."""
        if repo_model is None:
            raise ReviewError("A RepositoryModel is required to build review context.")

        mode = request.mode if isinstance(request.mode, ReviewMode) else ReviewMode(request.mode)
        if mode == ReviewMode.DIFF:
            return self.build_diff_context(repo_model, request)
        elif mode == ReviewMode.FULL_REPO:
            return self.build_full_repo_context(repo_model, request)
        else:
            raise ReviewError(f"Unsupported review mode: {request.mode}")

    def build_diff_context(self, repo_model: RepositoryModel, request: ReviewRequest) -> str:
        """
        Build context for reviewing diffs, combining:
        1. Changed files and diff hunks.
        2. Reverse dependencies from Phase 8 DependencyGraph.
        3. Phase 10 validation and test execution results.
        4. Current contents of modified files.
        """
        diffs = list(request.diffs)
        if request.agent_result and request.agent_result.diffs:
            diffs.extend(request.agent_result.diffs)

        # Deduplicate diffs by file_path
        seen_paths = set()
        unique_diffs: list[FileDiff] = []
        for d in diffs:
            if d.file_path not in seen_paths:
                seen_paths.add(d.file_path)
                unique_diffs.append(d)

        changed_files = [d.file_path for d in unique_diffs]
        if not changed_files and request.target_files:
            changed_files = list(request.target_files)

        sections: list[str] = [
            "=== REVIEW MODE: CHANGED FILES / DIFF REVIEW ===",
            f"Repository: {getattr(repo_model, 'repo_url', getattr(repo_model, 'url', ''))}",
        ]

        if request.focus_categories:
            sections.append(f"Focus categories: {', '.join(request.focus_categories)}")
        if request.custom_instructions:
            sections.append(f"Custom instructions: {request.custom_instructions}")

        sections.append(f"\nChanged files count: {len(changed_files)}")
        for path in changed_files:
            sections.append(f" - {path}")

        # Diffs
        sections.append("\n=== GIT DIFFS ===")
        if unique_diffs:
            for d in unique_diffs:
                sections.append(
                    f"--- Diff for: {d.file_path} (+{d.lines_added} / -{d.lines_removed}) ---\n"
                    f"{d.diff}\n"
                )
        else:
            sections.append("No explicit git diffs provided; reviewing target files directly.")

        # Phase 8 Reverse Dependency Analysis
        sections.append("\n=== REVERSE DEPENDENCY ANALYSIS (IMPORTERS) ===")
        graph = self._graph_builder.build(repo_model)
        importers_map: dict[str, list[str]] = {}
        for changed in changed_files:
            importers_map[changed] = []
            changed_norm = changed.replace("\\", "/").lower()
            for edge in graph.edges:
                target_norm = edge.target.replace("\\", "/").lower()
                if target_norm.endswith(changed_norm) or changed_norm.endswith(target_norm):
                    importers_map[changed].append(edge.source)

        has_importers = False
        for changed, importers in importers_map.items():
            if importers:
                has_importers = True
                sections.append(
                    f"File '{changed}' is imported by:\n" +
                    "\n".join(f"  * {imp}" for imp in sorted(set(importers)))
                )
        if not has_importers:
            sections.append("No internal reverse dependencies found importing the changed files.")

        # Phase 10 Validation and Test Results
        if request.agent_result:
            ar = request.agent_result
            sections.append("\n=== PHASE 10 EXECUTION STATUS ===")
            sections.append(f"Workflow status: {ar.status}")
            if ar.validation_result:
                sections.append(
                    f"Validation valid: {ar.validation_result.valid}\n"
                    f"Validation messages: {', '.join(ar.validation_result.messages) if ar.validation_result.messages else 'None'}"
                )
            if ar.test_result:
                tr = ar.test_result
                cmd_str = " ".join(tr.command) if tr.command else "None"
                sections.append(
                    f"Test Status: {tr.status}\n"
                    f"Command: {cmd_str}\n"
                    f"Exit Code: {tr.exit_code}"
                )
                if tr.stdout:
                    sections.append(f"Stdout (first 1000 chars):\n{tr.stdout[:1000]}")
                if tr.stderr:
                    sections.append(f"Stderr (first 1000 chars):\n{tr.stderr[:1000]}")

        # Current File Contents (bounded)
        sections.append("\n=== CURRENT FILE CONTENTS ===")
        repo_root = Path(repo_model.local_path)
        for rel_path in changed_files:
            file_disk_path = repo_root / rel_path
            if file_disk_path.exists() and file_disk_path.is_file():
                try:
                    content = file_disk_path.read_text(encoding="utf-8", errors="replace")
                    truncated = content[:15_000]
                    suffix = "\n... [truncated]" if len(content) > 15_000 else ""
                    sections.append(f"--- File: {rel_path} ---\n{truncated}{suffix}\n")
                except Exception as exc:
                    sections.append(f"--- File: {rel_path} (could not read: {exc}) ---")

        return "\n".join(sections)

    def build_full_repo_context(self, repo_model: RepositoryModel, request: ReviewRequest) -> str:
        """
        Build context for a full repository health/architecture review:
        1. Overview (languages, technologies, important files, entry points).
        2. Architecture facts and dependency graph metrics.
        3. Key entry point file snippets.
        """
        overview = self._overview_builder.build(repo_model)
        arch_facts = ArchitectureContextBuilder.build(overview)

        sections: list[str] = [
            "=== REVIEW MODE: FULL REPOSITORY AUDIT ===",
            f"Repository: {getattr(repo_model, 'repo_url', getattr(repo_model, 'url', ''))}",
            f"Total scanned files: {len(repo_model.files)}",
        ]

        if request.focus_categories:
            sections.append(f"Focus categories: {', '.join(request.focus_categories)}")
        if request.custom_instructions:
            sections.append(f"Custom instructions: {request.custom_instructions}")

        sections.append("\n=== ARCHITECTURAL CONTEXT & DEPENDENCIES ===")
        sections.append(arch_facts)

        # Sample entry points or important files
        sections.append("\n=== SAMPLE ENTRY POINTS & KEY MODULES ===")
        repo_root = Path(repo_model.local_path)
        sampled_files = list(overview.entry_points) + list(overview.important_files)
        # Unique preserving order
        unique_samples = []
        for f in sampled_files:
            if f not in unique_samples:
                unique_samples.append(f)

        for rel_path in unique_samples[:5]:
            file_disk_path = repo_root / rel_path
            if file_disk_path.exists() and file_disk_path.is_file():
                try:
                    content = file_disk_path.read_text(encoding="utf-8", errors="replace")
                    truncated = content[:4000]
                    suffix = "\n... [truncated]" if len(content) > 4000 else ""
                    sections.append(f"--- Entry / Key File: {rel_path} ---\n{truncated}{suffix}\n")
                except Exception:
                    pass

        return "\n".join(sections)
