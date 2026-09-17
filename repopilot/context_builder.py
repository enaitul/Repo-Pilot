"""
Turns Phase 6's ranked SearchResults into one clean, clearly-labeled
block of text an LLM can read as context.

WHY NOT JUST DUMP RAW CodeChunk OBJECTS INTO THE PROMPT: an LLM reading
Python object syntax like `CodeChunk(chunk_id='...', file_path='...')`
has to mentally parse programming syntax just to find the information
it needs, and it's easy for such formatting to bury the actual code
under noise. A plain, consistent, human-readable layout — clearly
labeled File/Symbol/Lines/Language, then the code itself — is easier
for the model to read accurately and reference correctly in its answer.

WHY ORDER MATTERS: SearchResults arrive already ranked by similarity
(highest first, from VectorStore/FAISS) — this module preserves that
order rather than reshuffling it, since the most relevant chunk should
naturally get read first, the same way a human would want the most
relevant document on top of the pile.
"""

from __future__ import annotations

from repopilot.models import SearchResult

_NO_RESULTS_TEXT = (
    "(No relevant code was found in the repository for this question. "
    "There is no context available to answer from.)"
)


class ContextBuilder:
    """Formats retrieved SearchResults into LLM-ready context text."""

    @staticmethod
    def build(search_results: list[SearchResult]) -> str:
        """
        Build one text block from `search_results`, in the order given
        (i.e. already ranked — this does not re-sort). Returns a clear
        placeholder message, not an empty string, when there are no
        results — an empty string could be silently mistaken for "no
        context needed" rather than "no context was found."
        """
        if not search_results:
            return _NO_RESULTS_TEXT

        blocks = [
            ContextBuilder._format_one(index, result)
            for index, result in enumerate(search_results, start=1)
        ]
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _format_one(index: int, result: SearchResult) -> str:
        chunk = result.chunk
        symbol_line = chunk.symbol_name or "(module-level code)"
        if chunk.parent:
            symbol_line = f"{chunk.parent}.{symbol_line}"

        return (
            f"[Chunk {index}]\n"
            f"File: {chunk.file_path}\n"
            f"Symbol: {symbol_line}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Language: {chunk.language}\n"
            f"Relevance score: {result.score:.2f}\n\n"
            f"{chunk.content}"
        )
