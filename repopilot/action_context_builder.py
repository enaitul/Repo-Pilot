"""
Builds the right kind of context for each ActionType, reusing Phase 6/7/8
machinery directly — no new retrieval, embedding, or architecture logic.

Most actions (test generation, documentation, refactoring, bug analysis)
need SPECIFIC CODE, so they go through the exact same
SearchService.search() -> ContextBuilder.build() pipeline Phase 7's
RAGService already uses. CHANGE_PLAN is different — it needs
REPOSITORY-LEVEL architecture facts, not retrieved code snippets, so it
goes through Phase 8's RepositoryOverviewBuilder ->
ArchitectureContextBuilder instead. This mirrors the actual reasoning
requirement of each action, rather than forcing every action through
one identical pipeline.
"""

from __future__ import annotations

from repopilot.architecture_context_builder import ArchitectureContextBuilder
from repopilot.context_builder import ContextBuilder
from repopilot.exceptions import ActionError
from repopilot.models import ActionRequest, ActionType, RepositoryModel, SearchResult
from repopilot.repository_overview_builder import RepositoryOverviewBuilder
from repopilot.search_service import SearchService


class ActionContextBuilder:
    """Builds action-appropriate context text, reusing existing Phase 6/7/8 services."""

    def __init__(
        self,
        search_service: SearchService | None = None,
        overview_builder: RepositoryOverviewBuilder | None = None,
    ):
        # search_service is optional here (not every caller has a saved
        # vector store handy) but REQUIRED for any action except
        # CHANGE_PLAN — enforced in build() below, not at construction
        # time, so a caller doing only change-planning never needs one.
        self._search_service = search_service
        self._overview_builder = overview_builder or RepositoryOverviewBuilder()

    def build(
        self, request: ActionRequest, repo_model: RepositoryModel | None, top_k: int
    ) -> tuple[str, list[SearchResult]]:
        """
        Return (context_text, search_results). `search_results` is empty
        for CHANGE_PLAN (its context comes from RepositoryOverview facts,
        not retrieval) — ActionService uses whatever's returned here to
        build real, traceable `sources` for the final ActionResult.
        """
        if request.action_type == ActionType.CHANGE_PLAN:
            if repo_model is None:
                raise ActionError("CHANGE_PLAN requires a RepositoryModel for architecture context.")
            overview = self._overview_builder.build(repo_model)
            return ArchitectureContextBuilder.build(overview), []

        if self._search_service is None:
            raise ActionError(
                f"{request.action_type.value} requires a SearchService for code retrieval "
                "(build one from a saved vector store, e.g. VectorStore.load(...))."
            )

        query = self._build_retrieval_query(request)
        search_results = self._search_service.search(query, top_k=top_k)
        context_text = ContextBuilder.build(search_results)
        return context_text, search_results

    @staticmethod
    def _build_retrieval_query(request: ActionRequest) -> str:
        """
        Combine whatever the caller provided into one retrieval query —
        the target symbol/file (if known) and any error message give
        semantic search stronger signal than the free-text request alone.
        """
        parts = [request.request]
        if request.target_symbol:
            parts.append(request.target_symbol)
        if request.target_file:
            parts.append(request.target_file)
        if request.error_message:
            parts.append(request.error_message)
        return " ".join(parts)
