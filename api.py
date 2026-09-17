"""RepoPilot's production HTTP application.

This is deliberately a thin adapter: the Phase 1-12 package remains the
source of all repository, retrieval, modification, review, and GitHub logic.
Each analysis is held in an isolated clone for the lifetime of a short-lived
browser session; no original repository is modified.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repopilot.action_service import ActionService
from repopilot.chunking_service import ChunkingService
from repopilot.embedding_service import EmbeddingService
from repopilot.exceptions import RepoPilotError
from repopilot.ingestion_service import IngestionService
from repopilot.modification_service import ModificationService
from repopilot.models import ActionRequest, ActionType, GitHubWorkflowRequest, ReviewMode, ReviewRequest
from repopilot.parsing_service import ParsingService
from repopilot.rag_service import RAGService
from repopilot.repository_intelligence_service import RepositoryIntelligenceService
from repopilot.review_service import ReviewService
from repopilot.search_service import SearchService
from repopilot.vector_store import VectorStore
from repopilot.github_workflow_service import GitHubWorkflowService
from repopilot.groq_provider import GroqProvider

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SESSION_TTL_SECONDS = int(os.getenv("REPOPILOT_SESSION_TTL_SECONDS", "3600"))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("REPOPILOT_ALLOWED_ORIGINS", "").split(",") if origin.strip()]


@dataclass
class AnalysisSession:
    repository: Any
    search: SearchService | None
    created_at: float = field(default_factory=time.time)
    agent_result: Any | None = None
    review_report: Any | None = None


sessions: dict[str, AnalysisSession] = {}


class AnalyzeBody(BaseModel):
    repo_url: str = Field(min_length=1, max_length=500)


class QuestionBody(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)
    groq_api_key: str | None = Field(default=None, max_length=500)


class ActionBody(QuestionBody):
    action_type: ActionType
    target_file: str | None = Field(default=None, max_length=500)
    target_symbol: str | None = Field(default=None, max_length=300)
    additional_context: str | None = Field(default=None, max_length=4000)


class ModifyBody(QuestionBody):
    target_file: str | None = Field(default=None, max_length=500)
    target_symbol: str | None = Field(default=None, max_length=300)


class ReviewBody(BaseModel):
    session_id: str
    mode: ReviewMode = ReviewMode.FULL_REPO
    focus_categories: list[str] = Field(default_factory=list, max_length=8)
    instructions: str | None = Field(default=None, max_length=2000)
    groq_api_key: str | None = Field(default=None, max_length=500)


class GitHubBody(BaseModel):
    session_id: str
    repo_owner: str = Field(min_length=1, max_length=100)
    repo_name: str = Field(min_length=1, max_length=100)
    branch_name: str | None = Field(default=None, max_length=120)
    base_branch: str = Field(default="main", max_length=120)
    pr_title: str | None = Field(default=None, max_length=200)
    approved: bool = False


app = FastAPI(title="RepoPilot", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def prewarm_embedding_model() -> None:
    try:
        from repopilot.sentence_transformer_provider import SentenceTransformerProvider
        SentenceTransformerProvider()._get_model()
    except Exception as exc:
        print(f"Startup model pre-warm warning: {exc}")



def _cleanup_expired_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    for session_id, session in list(sessions.items()):
        if session.created_at < cutoff:
            shutil.rmtree(session.repository.local_path, ignore_errors=True)
            sessions.pop(session_id, None)


def _session(session_id: str) -> AnalysisSession:
    _cleanup_expired_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session was not found or expired. Analyze the repository again.")
    return session


def _error(exc: Exception) -> HTTPException:
    # Domain exceptions have intentionally safe messages. Unexpected errors
    # are logged by the server, not exposed to the browser.
    if isinstance(exc, RepoPilotError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="RepoPilot could not complete that request. Check server logs and configuration.")


def _groq_provider(api_key: str | None) -> GroqProvider:
    """Create an ephemeral provider; user keys never enter session state."""
    if not (api_key and api_key.strip()) and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=400, detail="Please provide your Groq API key to use AI features.")
    return GroqProvider(api_key=api_key.strip() if api_key else None)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_configured": str(bool(os.getenv("GROQ_API_KEY"))).lower()}


@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> dict[str, Any]:
    _cleanup_expired_sessions()
    try:
        # cleanup=False is required only while this controlled browser session
        # exists; it is deleted on expiry or process shutdown.
        repository = IngestionService().ingest(body.repo_url, cleanup=False)
        parsed = ParsingService().parse_repository(repository)
        chunks = ChunkingService().chunk_repository(repository, parsed)

        search: SearchService | None = None
        index_warning: str | None = None
        if chunks:
            try:
                embeddings = EmbeddingService().embed_chunks(chunks)
                store = VectorStore()
                store.rebuild(embeddings)
                search = SearchService(store)
            except Exception as exc:
                # Analysis remains useful without RAG if a deployment lacks
                # the local model or FAISS; the UI makes this explicit.
                index_warning = f"Semantic retrieval is unavailable: {exc}"
        else:
            index_warning = "No supported code chunks were found to index."

        overview = RepositoryIntelligenceService()._overview_builder.build(repository).to_dict()
        session_id = uuid.uuid4().hex
        sessions[session_id] = AnalysisSession(repository=repository, search=search)
        return {
            "session_id": session_id,
            "repository": repository.to_dict(),
            "overview": overview,
            "parsed_files": len(parsed),
            "chunks": len(chunks),
            "rag_ready": search is not None,
            "warning": index_warning,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.post("/api/question")
def question(body: QuestionBody) -> dict[str, Any]:
    session = _session(body.session_id)
    if not session.search:
        raise HTTPException(status_code=503, detail="Semantic retrieval is not ready for this repository. See the analysis warning.")
    try:
        return RAGService(session.search, llm_provider=_groq_provider(body.groq_api_key)).answer(body.question).to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.post("/api/actions")
def action(body: ActionBody) -> dict[str, Any]:
    session = _session(body.session_id)
    try:
        request = ActionRequest(body.action_type, body.question, body.target_file, body.target_symbol, additional_context=body.additional_context)
        # Change plans use deterministic architecture context; all retrieval
        # actions reuse this session's in-memory FAISS index.
        if body.action_type == ActionType.CHANGE_PLAN:
            result = ActionService(llm_provider=_groq_provider(body.groq_api_key)).execute(request, repo_model=session.repository)
        elif session.search:
            from repopilot.action_context_builder import ActionContextBuilder
            result = ActionService(context_builder=ActionContextBuilder(search_service=session.search), llm_provider=_groq_provider(body.groq_api_key)).execute(request, repo_model=session.repository)
        else:
            raise HTTPException(status_code=503, detail="This action needs semantic retrieval, which is not ready for this session.")
        return result.to_dict()
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.post("/api/modify")
def modify(body: ModifyBody) -> dict[str, Any]:
    session = _session(body.session_id)
    try:
        request = ActionRequest(ActionType.CHANGE_PLAN, body.question, body.target_file, body.target_symbol)
        provider = _groq_provider(body.groq_api_key)
        result = ModificationService(action_service=ActionService(llm_provider=provider), llm_provider=provider).execute(request, session.repository)
        session.agent_result = result
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.post("/api/review")
def review(body: ReviewBody) -> dict[str, Any]:
    session = _session(body.session_id)
    try:
        request = ReviewRequest(mode=body.mode, focus_categories=body.focus_categories, custom_instructions=body.instructions)
        if body.mode == ReviewMode.DIFF and session.agent_result:
            request = ReviewRequest(mode=ReviewMode.DIFF, agent_result=session.agent_result, diffs=session.agent_result.diffs, focus_categories=body.focus_categories, custom_instructions=body.instructions)
        report = ReviewService(llm_provider=_groq_provider(body.groq_api_key)).review(request, session.repository)
        session.review_report = report
        return report.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.post("/api/github-workflow")
def github_workflow(body: GitHubBody) -> dict[str, Any]:
    session = _session(body.session_id)
    if not session.agent_result:
        raise HTTPException(status_code=409, detail="Run a controlled modification before starting the GitHub workflow.")
    try:
        request = GitHubWorkflowRequest(
            repo_owner=body.repo_owner, repo_name=body.repo_name, branch_name=body.branch_name,
            base_branch=body.base_branch, pr_title=body.pr_title, approved=body.approved,
        )
        return GitHubWorkflowService().execute(request, session.repository, session.agent_result, session.review_report).to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(exc) from exc


@app.on_event("shutdown")
def cleanup_sessions() -> None:
    for session in sessions.values():
        shutil.rmtree(session.repository.local_path, ignore_errors=True)
    sessions.clear()
