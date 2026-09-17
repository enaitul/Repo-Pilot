"""
Orchestrates Phase 7's RAG pipeline: turns a question into a grounded,
sourced answer.

Mirrors the shape of every previous phase's *_service.py: this is the
only module that knows the end-to-end RAG pipeline order (retrieve ->
build context -> build prompt -> generate -> build response). It
DELIBERATELY reuses Phase 6's SearchService unchanged rather than
touching FAISS/embeddings itself — RAG's whole job is "do something
useful with what retrieval already found," not "retrieve differently."

    User question
        |
        v
    SearchService.search()      <- Phase 6, UNCHANGED
        |
        v
    list[SearchResult]
        |
        v
    ContextBuilder.build()      <- Phase 7, NEW
        |
        v
    PromptBuilder.build()       <- Phase 7, NEW
        |
        v
    LLMProvider.generate()      <- Phase 7, NEW (abstract; Groq by default)
        |
        v
    RAGResponse (answer + sources + raw retrieval results)

Phase 7 is READ-ONLY: it only ever asks the LLM to explain retrieved
code in text. Nothing here modifies files, runs commands, or writes to
the repository or anywhere else.
"""

from __future__ import annotations

from repopilot.config import DEFAULT_TOP_K
from repopilot.context_builder import ContextBuilder
from repopilot.exceptions import RAGError
from repopilot.groq_provider import GroqProvider
from repopilot.llm_provider import LLMProvider
from repopilot.models import RAGResponse, Source
from repopilot.prompt_builder import PromptBuilder
from repopilot.search_service import SearchService


class RAGService:
    """High-level entry point for Phase 7: retrieval-augmented Q&A."""

    def __init__(self, search_service: SearchService, llm_provider: LLMProvider | None = None):
        # search_service is REQUIRED — RAG has nothing to ground answers
        # in without it. llm_provider defaults to Groq's free tier, but
        # constructor injection (same pattern as every earlier service)
        # lets tests substitute a fake provider with zero real API calls.
        self._search_service = search_service
        self._llm_provider = llm_provider or GroqProvider()

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        """
        Answer `question` using the top_k most relevant retrieved code
        chunks as grounding context.

        A question that retrieves ZERO relevant chunks is NOT an error —
        ContextBuilder produces an explicit "no relevant code was found"
        placeholder, which is passed to the LLM as-is, so the model can
        honestly say it found nothing relevant rather than the pipeline
        silently failing or fabricating an answer with no evidence.
        """
        if not question or not question.strip():
            raise RAGError("Question must not be empty.")

        search_results = self._search_service.search(question, top_k=top_k)

        context_text = ContextBuilder.build(search_results)
        system_prompt, user_prompt = PromptBuilder.build(context_text, question)

        answer_text = self._llm_provider.generate(system_prompt, user_prompt)

        sources = [
            Source(
                file_path=result.chunk.file_path,
                symbol_name=result.chunk.symbol_name,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                score=result.score,
            )
            for result in search_results
        ]

        return RAGResponse(answer=answer_text, sources=sources, retrieval_results=search_results)
