"""
Combines the three purely deterministic Phase 8 detectors
(TechnologyDetector, ImportantFileDetector, DependencyGraphBuilder) into
one RepositoryOverview. Nothing in this module involves an LLM — every
field it produces is a plain, explainable fact derived from Phases 1-3's
existing data.

This is the "deterministic analysis" half of Phase 8's hybrid design.
The LLM only ever sees the OUTPUT of this module (via
ArchitectureContextBuilder) — it never gets to discover these facts
itself, which is exactly what prevents it from inventing files,
technologies, or relationships that don't actually exist.
"""

from __future__ import annotations

from repopilot.dependency_graph_builder import DependencyGraphBuilder
from repopilot.important_file_detector import ImportantFileDetector
from repopilot.models import RepositoryModel, RepositoryOverview
from repopilot.technology_detector import TechnologyDetector


class RepositoryOverviewBuilder:
    """Builds a RepositoryOverview from a RepositoryModel — no LLM involved."""

    def __init__(
        self,
        technology_detector: TechnologyDetector | None = None,
        important_file_detector: ImportantFileDetector | None = None,
        dependency_graph_builder: DependencyGraphBuilder | None = None,
    ):
        # Constructor injection, same pattern as every earlier service —
        # lets tests substitute fakes if a detector's own behavior isn't
        # what's under test.
        self._technology_detector = technology_detector or TechnologyDetector()
        self._important_file_detector = important_file_detector or ImportantFileDetector()
        self._dependency_graph_builder = dependency_graph_builder or DependencyGraphBuilder()

    def build(self, repo_model: RepositoryModel) -> RepositoryOverview:
        return RepositoryOverview(
            repo_url=repo_model.repo_url,
            languages=repo_model.language_summary,
            technologies=self._technology_detector.detect(repo_model),
            important_files=self._important_file_detector.detect_important_files(repo_model),
            entry_points=self._important_file_detector.detect_entry_points(repo_model),
            dependency_graph=self._dependency_graph_builder.build(repo_model),
        )
