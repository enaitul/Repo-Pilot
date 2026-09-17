"""
Orchestrates Phase 8's repository-level intelligence: turns a
RepositoryModel into a grounded, sourced answer to a repository-level
question (architecture, components, onboarding, etc.).

    RepositoryModel  (Phase 1/2, UNCHANGED)
        |
        v
    RepositoryOverviewBuilder.build()   <- Phase 8, 100% deterministic
        |
        v
    RepositoryOverview
        |
        v
    ArchitectureContextBuilder.build()  <- Phase 8, formats facts as text
        |
        v
    PromptBuilder.build_architecture()  <- Phase 7's PromptBuilder, extended
        |
        v
    LLMProvider.generate()              <- Phase 7's GroqProvider, UNCHANGED
        |
        v
    RepositoryAnalysis

Deliberately does NOT touch chunking, embeddings, or FAISS — a
repository-level architecture question doesn't need any of that;
RepositoryOverviewBuilder works directly from Phase 1/2's FileMetadata.
"""

from __future__ import annotations

from repopilot.architecture_context_builder import ArchitectureContextBuilder
from repopilot.exceptions import RepositoryIntelligenceError
from repopilot.groq_provider import GroqProvider
from repopilot.llm_provider import LLMProvider
from repopilot.models import RepositoryAnalysis, RepositoryModel, Source
from repopilot.prompt_builder import PromptBuilder
from repopilot.repository_overview_builder import RepositoryOverviewBuilder

_DEFAULT_QUESTION = (
    "Explain the architecture of this repository: its major components, "
    "how they relate, and where a new developer should start reading."
)


class RepositoryIntelligenceService:
    """High-level entry point for Phase 8: repository-level architecture Q&A."""

    def __init__(
        self,
        overview_builder: RepositoryOverviewBuilder | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        # Same constructor-injection pattern as every earlier service —
        # lets tests substitute fakes with zero real LLM calls.
        self._overview_builder = overview_builder or RepositoryOverviewBuilder()
        self._llm_provider = llm_provider or GroqProvider()

    def analyze(self, repo_model: RepositoryModel, question: str | None = None) -> RepositoryAnalysis:
        """
        Answer a repository-level question about `repo_model`. If
        `question` is omitted, defaults to a general architecture
        overview request.

        A repository with zero files is NOT an error — it produces a
        valid, minimal RepositoryOverview (empty languages/technologies/
        files/graph), and the LLM is asked to work with that honestly
        (it will typically say there isn't enough information), matching
        the same "expected empty state, not a failure" philosophy used
        throughout every earlier phase.
        """
        if repo_model is None:
            raise RepositoryIntelligenceError("A RepositoryModel is required for analysis.")

        question = question if question and question.strip() else _DEFAULT_QUESTION

        overview = self._overview_builder.build(repo_model)
        context_text = ArchitectureContextBuilder.build(overview)
        system_prompt, user_prompt = PromptBuilder.build_architecture(context_text, question)

        try:
            answer_text = self._llm_provider.generate(system_prompt, user_prompt)
        except RepositoryIntelligenceError:
            raise
        except Exception as exc:
            # Mirrors RAGService: an LLM-layer failure here is a
            # SYSTEM-level failure, not an expected empty result.
            raise RepositoryIntelligenceError(f"Repository analysis failed: {exc}") from exc

        file_line_counts = {f.path: f.line_count for f in repo_model.files}
        sources = [
            Source(
                file_path=path,
                symbol_name=None,
                start_line=1,
                end_line=file_line_counts.get(path) or 1,
                score=1.0,  # deterministic file-level evidence — maximal confidence, not a similarity score
            )
            for path in overview.important_files
        ]

        return RepositoryAnalysis(
            answer=answer_text,
            technologies=overview.technologies,
            important_files=overview.important_files,
            entry_points=overview.entry_points,
            dependency_graph=overview.dependency_graph,
            sources=sources,
        )
