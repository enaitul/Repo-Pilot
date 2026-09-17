"""
Command-line entrypoint for RepoPilot.

Deliberately thin: all real logic lives in IngestionService (Phase 1/2)
and ParsingService (Phase 3). This file only handles argument parsing,
calling the services in order, and formatting output/errors for a human
at a terminal.

Pipeline:

    repo_url --> IngestionService.ingest --> RepositoryModel
              --> ParsingService.parse_repository --> list[ParsedFile]
              --> ChunkingService.chunk_repository --> list[CodeChunk]
              --> EmbeddingService.embed_chunks (only with --embed) --> list[EmbeddedChunk]
              --> VectorStore.rebuild + save (only with --index) --> saved to disk
              --> combined JSON on stdout

Separately, --query searches a PREVIOUSLY saved vector store (built by
a prior --index run) without re-cloning or re-embedding anything.
"""

from __future__ import annotations

import argparse
import json
import sys

from repopilot.config import DEFAULT_TOP_K, VECTOR_STORE_DIR
from repopilot.ingestion_service import IngestionService
from repopilot.parsing_service import ParsingService
from repopilot.chunking_service import ChunkingService
try:
    from repopilot.vector_store import VectorStore
    from repopilot.search_service import SearchService
    from repopilot.rag_service import RAGService
except ImportError:
    VectorStore = None  # type: ignore
    SearchService = None  # type: ignore
    RAGService = None  # type: ignore

from repopilot.repository_intelligence_service import RepositoryIntelligenceService
from repopilot.review_service import ReviewService
from repopilot.github_workflow_service import GitHubWorkflowService
from repopilot.models import (
    CodeChunk,
    EmbeddedChunk,
    GitHubWorkflowRequest,
    ParsedFile,
    RepositoryModel,
    ReviewMode,
    ReviewRequest,
    SearchResult,
)
from repopilot.exceptions import RepoPilotError



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repopilot",
        description="RepoPilot: ingest a public GitHub repository, parse its "
        "source files, and print a structured representation as JSON.",
    )
    parser.add_argument(
        "repo_url",
        nargs="?",
        default=None,
        help=(
            "Public GitHub repo URL, e.g. https://github.com/owner/repo. "
            "Not needed when using --query, which searches a previously "
            "saved vector store instead."
        ),
    )
    parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="Do not delete the cloned repo from disk after processing.",
    )
    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Run ingestion only (Phase 1/2) and skip code parsing (Phase 3) and chunking (Phase 4).",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Run ingestion and parsing but skip chunking (Phase 4).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Also generate embeddings for each chunk (Phase 5). Off by "
            "default: it downloads a local model on first use (~80MB, "
            "one-time) and adds inference time per chunk. Requires "
            "'pip install sentence-transformers'. Ignored if parsing or "
            "chunking was skipped, since embedding needs chunks to embed."
        ),
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "Build a searchable vector store (Phase 6) from the embedded "
            f"chunks and save it to '{VECTOR_STORE_DIR}/' for later "
            "searching with --query. Requires --embed. Requires "
            "'pip install faiss-cpu'."
        ),
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help=(
            "Search a previously saved vector store (built with --index) "
            "for the chunks most similar to TEXT. When used, repo_url and "
            "every other pipeline flag is ignored — this only loads the "
            f"saved store from '{VECTOR_STORE_DIR}/' and searches it."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many results --query or --ask should retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help=(
            "Ask a question about a previously saved vector store (built "
            "with --index) and get a grounded, sourced answer generated "
            "by an LLM (Phase 7 — RAG). Like --query, this only loads "
            f"the saved store from '{VECTOR_STORE_DIR}/' — repo_url and "
            "other pipeline flags are ignored. Requires "
            "'pip install groq' and a GROQ_API_KEY environment variable "
            "(free at https://console.groq.com/keys)."
        ),
    )
    parser.add_argument(
        "--architecture",
        metavar="QUESTION",
        help=(
            "Ask a repository-LEVEL question (Phase 8 — architecture, "
            "components, entry points, onboarding) about repo_url, e.g. "
            "'Explain the architecture of this repository.' Unlike --ask, "
            "this only needs ingestion (Phase 1/2) — parsing, chunking, "
            "embedding, and a saved index are NOT required. Requires "
            "'pip install groq' and a GROQ_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "Run Phase 11 Intelligent Code Review on repo_url. "
            "Analyzes code quality, security, architecture, and testing. "
            "Requires 'pip install groq' and a GROQ_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--review-mode",
        choices=["diff", "full_repo"],
        default="full_repo",
        help="Review mode for --review: 'full_repo' (default) or 'diff'.",
    )
    parser.add_argument(
        "--review-categories",
        nargs="+",
        help="Specific categories to focus on during code review (e.g. SECURITY BUG ARCHITECTURE).",
    )
    parser.add_argument(
        "--github-workflow",
        action="store_true",
        help="Run Phase 12 GitHub Developer Workflow Integration.",
    )
    parser.add_argument(
        "--branch-name",
        metavar="NAME",
        help="Feature branch name for Phase 12 GitHub workflow.",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Request creation of a Pull Request on GitHub (requires --approve-push and GITHUB_TOKEN).",
    )
    parser.add_argument(
        "--approve-push",
        action="store_true",
        help="Explicitly grant human approval for remote Git branch push and PR creation.",
    )
    return parser



def _parsing_summary(parsed_files: list[ParsedFile]) -> dict:
    """
    Small aggregate view over per-file parse results — how many files
    succeeded/failed and why, plus a total symbol count. Useful at a
    glance without scrolling through every file's symbol list.
    """
    status_counts: dict[str, int] = {}
    total_symbols = 0
    for pf in parsed_files:
        status_counts[pf.status.value] = status_counts.get(pf.status.value, 0) + 1
        total_symbols += len(pf.symbols)

    return {
        "total_files_parsed": len(parsed_files),
        "total_symbols_found": total_symbols,
        "status_counts": status_counts,
    }


def _chunking_summary(chunks: list[CodeChunk]) -> dict:
    """
    Small aggregate view over chunking results — total chunk count and
    the average chunk size, useful to sanity-check chunking behavior
    (e.g. "did most files end up as one chunk, or did something get
    split apart a lot more than expected?") without reading every chunk.
    """
    total_chunks = len(chunks)
    total_chars = sum(len(c.content) for c in chunks)
    avg_chars = total_chars // total_chunks if total_chunks else 0

    return {
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "avg_chunk_chars": avg_chars,
    }


def _embedding_summary(embedded_chunks: list[EmbeddedChunk]) -> dict:
    """
    Small aggregate view over embedding results — how many chunks got
    embedded, with which model, and at what dimension. Doesn't repeat
    the (large) vectors themselves.
    """
    if not embedded_chunks:
        return {"total_embedded": 0, "model_name": None, "dimensions": None}

    return {
        "total_embedded": len(embedded_chunks),
        "model_name": embedded_chunks[0].model_name,
        "dimensions": embedded_chunks[0].dimensions,
    }


def _build_output(
    repo_model: RepositoryModel,
    parsed_files: list[ParsedFile] | None,
    chunks: list[CodeChunk] | None,
    embedded_chunks: list[EmbeddedChunk] | None,
) -> dict:
    output = repo_model.to_dict()
    if parsed_files is not None:
        output["parsing"] = {
            "summary": _parsing_summary(parsed_files),
            "files": [pf.to_dict() for pf in parsed_files],
        }
    if chunks is not None:
        output["chunking"] = {
            "summary": _chunking_summary(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }
    if embedded_chunks is not None:
        output["embedding"] = {
            "summary": _embedding_summary(embedded_chunks),
            "embedded_chunks": [ec.to_dict() for ec in embedded_chunks],
        }
    return output


def _run_query(query: str, top_k: int, pretty: bool) -> int:
    """
    Handle `--query`: load a previously saved VectorStore (Phase 6) and
    search it. Completely separate from the ingest/parse/chunk/embed
    pipeline above — this is what makes searching fast after the first
    --index run, since nothing gets re-cloned or re-embedded.
    """
    try:
        vector_store = VectorStore.load(VECTOR_STORE_DIR)
    except RepoPilotError as exc:
        print(f"Could not load vector store: {exc}", file=sys.stderr)
        print(f"(Build one first with: repopilot <repo_url> --embed --index)", file=sys.stderr)
        return 1

    search_service = SearchService(vector_store=vector_store)
    try:
        results = search_service.search(query, top_k=top_k)
    except RepoPilotError as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    output = {
        "query": query,
        "top_k": top_k,
        "results": [r.to_dict() for r in results],
    }
    indent = 2 if pretty else None
    print(json.dumps(output, indent=indent))
    return 0


def _run_ask(question: str, top_k: int, pretty: bool) -> int:
    """
    Handle `--ask`: load a previously saved VectorStore (Phase 6), run
    Phase 7's RAG pipeline over it, and print a grounded, sourced
    answer. Like _run_query, this never re-clones or re-embeds anything.
    """
    try:
        vector_store = VectorStore.load(VECTOR_STORE_DIR)
    except RepoPilotError as exc:
        print(f"Could not load vector store: {exc}", file=sys.stderr)
        print(f"(Build one first with: repopilot <repo_url> --embed --index)", file=sys.stderr)
        return 1

    search_service = SearchService(vector_store=vector_store)
    rag_service = RAGService(search_service=search_service)
    try:
        response = rag_service.answer(question, top_k=top_k)
    except RepoPilotError as exc:
        print(f"RAG query failed: {exc}", file=sys.stderr)
        return 1

    indent = 2 if pretty else None
    print(json.dumps(response.to_dict(), indent=indent))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.ask:
        return _run_ask(args.ask, args.top_k, args.pretty)

    if args.query:
        return _run_query(args.query, args.top_k, args.pretty)

    if not args.repo_url:
        parser.error("repo_url is required unless --query or --ask is used.")

    ingestion_service = IngestionService()

    # Parsing (Phase 3) needs the cloned files still on disk, so we always
    # ingest with cleanup=False here and take responsibility for cleanup
    # ourselves below, once parsing (if any) is done. This avoids the trap
    # of IngestionService deleting the clone before ParsingService ever
    # gets to read it.
    try:
        repo_model = ingestion_service.ingest(args.repo_url, cleanup=False)
    except RepoPilotError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    # Architecture analysis (Phase 8) only needs Phase 1/2's output — it
    # deliberately skips parsing/chunking/embedding entirely, since a
    # repository-level question doesn't need any of that. Handled here,
    # separately from the parsing/chunking/embedding pipeline below.
    if args.architecture:
        try:
            intelligence_service = RepositoryIntelligenceService()
            analysis = intelligence_service.analyze(repo_model, question=args.architecture)
        except RepoPilotError as exc:
            print(f"Architecture analysis failed: {exc}", file=sys.stderr)
            if not args.keep_clone:
                ingestion_service._cloner.cleanup(repo_model.local_path)
            return 1

        if not args.keep_clone:
            ingestion_service._cloner.cleanup(repo_model.local_path)

        indent = 2 if args.pretty else None
        print(json.dumps(analysis.to_dict(), indent=indent))
        return 0

    # Phase 11: Intelligent Code Review
    if args.review:
        try:
            review_service = ReviewService()
            mode = ReviewMode.DIFF if args.review_mode == "diff" else ReviewMode.FULL_REPO
            review_req = ReviewRequest(
                mode=mode,
                focus_categories=args.review_categories or [],
            )
            report = review_service.review(review_req, repo_model)
        except RepoPilotError as exc:
            print(f"Code review failed: {exc}", file=sys.stderr)
            if not args.keep_clone:
                ingestion_service._cloner.cleanup(repo_model.local_path)
            return 1

        indent = 2 if args.pretty else None
        print(json.dumps(report.to_dict(), indent=indent))
        return 0

    # Phase 12: GitHub Developer Workflow Integration
    if args.github_workflow:
        try:
            workflow_service = GitHubWorkflowService()
            parts = repo_model.repo_url.rstrip("/").removesuffix(".git").split("/")
            owner = parts[-2] if len(parts) >= 2 else "owner"
            name = parts[-1] if len(parts) >= 1 else "repo"
            wf_req = GitHubWorkflowRequest(
                repo_owner=owner,
                repo_name=name,
                branch_name=args.branch_name,
                approved=args.approve_push,
            )
            wf_result = workflow_service.execute(wf_req, repo_model)

        except RepoPilotError as exc:
            print(f"GitHub workflow failed: {exc}", file=sys.stderr)
            if not args.keep_clone:
                ingestion_service._cloner.cleanup(repo_model.local_path)
            return 1

        if not args.keep_clone:
            ingestion_service._cloner.cleanup(repo_model.local_path)

        indent = 2 if args.pretty else None
        print(json.dumps(wf_result.to_dict(), indent=indent))
        return 0


    parsed_files = None
    chunks = None
    embedded_chunks = None
    exit_code = 0
    try:
        if not args.skip_parsing:
            parsing_service = ParsingService()
            try:
                parsed_files = parsing_service.parse_repository(repo_model)
            except RepoPilotError as exc:
                print(f"Parsing failed: {exc}", file=sys.stderr)
                exit_code = 1

            # Chunking (Phase 4) needs Phase 3's output, so it only runs
            # when parsing actually happened and succeeded, and wasn't
            # explicitly skipped.
            if exit_code == 0 and not args.skip_chunking:
                chunking_service = ChunkingService()
                try:
                    chunks = chunking_service.chunk_repository(repo_model, parsed_files)
                except RepoPilotError as exc:
                    print(f"Chunking failed: {exc}", file=sys.stderr)
                    exit_code = 1

                # Embedding (Phase 5) needs Phase 4's output, and is
                # opt-in via --embed since it's the first stage that
                # downloads a model and does real inference work.
                if exit_code == 0 and args.embed:
                    embedding_service = EmbeddingService()
                    try:
                        embedded_chunks = embedding_service.embed_chunks(chunks)
                    except RepoPilotError as exc:
                        print(f"Embedding failed: {exc}", file=sys.stderr)
                        exit_code = 1

                    # Indexing (Phase 6) needs Phase 5's output, and is
                    # opt-in via --index since it requires faiss-cpu and
                    # writes files to disk.
                    if exit_code == 0 and args.index:
                        vector_store = VectorStore()
                        try:
                            vector_store.rebuild(embedded_chunks)
                            vector_store.save(VECTOR_STORE_DIR)
                        except RepoPilotError as exc:
                            print(f"Indexing failed: {exc}", file=sys.stderr)
                            exit_code = 1
    finally:
        if not args.keep_clone:
            ingestion_service._cloner.cleanup(repo_model.local_path)

    if exit_code != 0:
        return exit_code

    indent = 2 if args.pretty else None
    print(json.dumps(_build_output(repo_model, parsed_files, chunks, embedded_chunks), indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
