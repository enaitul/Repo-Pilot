"""
Typed data contracts for the ingestion + parsing subsystems.

These dataclasses ARE the API of RepoPilot. Every later phase (chunking,
embeddings, RAG, agent tools) should depend on these shapes, not on
whatever internal representation a given module happens to use. Using
dataclasses (rather than raw dicts) buys us:
  - static type checking / IDE autocomplete
  - a single, explicit source of truth for the schema
  - easy serialization via `asdict()` for JSON output or API responses

Phase 1/2 contract:  FileMetadata, RepositoryModel
Phase 3 contract:    SymbolType, ParseStatus, CodeSymbol, ParsedFile
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class FileMetadata:
    """Metadata for a single source file discovered during ingestion."""

    path: str                      # path relative to the repo root
    language: str                  # e.g. "Python", "TypeScript"
    size_bytes: int
    imports: list[str] = field(default_factory=list)
    line_count: Optional[int] = None
    skipped_content: bool = False  # True if file was too large to read content

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepositoryModel:
    """Structured representation of an entire ingested repository."""

    repo_url: str
    local_path: str
    files: list[FileMetadata] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def language_summary(self) -> dict[str, int]:
        """Count of files per detected language, useful for a quick repo overview."""
        summary: dict[str, int] = {}
        for f in self.files:
            summary[f.language] = summary.get(f.language, 0) + 1
        return summary

    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "local_path": self.local_path,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "language_summary": self.language_summary,
            "files": [f.to_dict() for f in self.files],
        }


# ---------------------------------------------------------------------------
# Phase 3: Code Parsing
# ---------------------------------------------------------------------------


class SymbolType(str, Enum):
    """The kinds of named code structures RepoPilot currently extracts."""

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class ParseStatus(str, Enum):
    """
    File-level parsing outcome.

    These are NOT exceptions. A repository can legitimately contain
    empty files, files with syntax errors, or files in languages we
    don't parse yet — none of that is an application error, so it's
    represented as data here rather than raised as a RepoPilotError.
    See exceptions.ParsingError for the distinction.
    """

    SUCCESS = "success"
    EMPTY_FILE = "empty_file"
    SYNTAX_ERROR = "syntax_error"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    PARSER_ERROR = "parser_error"


@dataclass(frozen=True)
class CodeSymbol:
    """
    One meaningful named code structure found inside a file — a class,
    a function, or a method.

    `parent` is the name of the enclosing class/function, or None for
    top-level symbols. This is enough to reconstruct simple hierarchy
    (e.g. "AuthService.login") without needing a full nested tree.
    """

    name: str
    symbol_type: SymbolType
    start_line: int
    end_line: int
    parent: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol_type": self.symbol_type.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent": self.parent,
        }


@dataclass
class ParsedFile:
    """
    The parsing result for a single source file.

    `status` always reflects what actually happened for THIS file.
    `symbols` may be non-empty even when status is SYNTAX_ERROR (e.g.
    Tree-sitter's error-tolerant parsing can still recover some
    structure around a broken section of a file) — callers should not
    assume SUCCESS is the only status worth reading symbols from.
    """

    path: str
    language: str
    status: ParseStatus
    symbols: list[CodeSymbol] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "status": self.status.value,
            "symbols": [s.to_dict() for s in self.symbols],
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Phase 4: Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeChunk:
    """
    One piece of source code, sized to be meaningful on its own and small
    enough to eventually hand to an embedding model (Phase 5).

    Usually a chunk is one whole class, function, or method — the same
    unit Phase 3 already identified as a CodeSymbol. The difference from
    CodeSymbol is that a CodeChunk also carries the actual source text
    (`content`), not just its location.

    A symbol that was too large to fit in one chunk may be represented by
    several CodeChunks instead of one — see Chunker for how that split
    happens. `chunk_index`/`chunk_count` let callers reassemble or at
    least recognize "these N chunks all came from the same original
    symbol" even after a big symbol has been split.
    """

    chunk_id: str                    # unique within a repo, e.g. "app/user_service.py:5-15"
    file_path: str                   # path relative to the repo root
    language: str                    # e.g. "Python"
    symbol_name: Optional[str]       # e.g. "login"; None only if we ever chunk non-symbol text
    symbol_type: Optional[str]       # "class" / "function" / "method"; None if not applicable
    parent: Optional[str]            # enclosing class name, same meaning as CodeSymbol.parent
    start_line: int
    end_line: int
    content: str                     # the actual code text for this chunk
    chunk_index: int = 0             # 0-based position among sibling chunks from the same symbol
    chunk_count: int = 1             # how many chunks that original symbol was split into

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent": self.parent,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        """
        Reconstruct a CodeChunk from the dict produced by to_dict().

        Needed by Phase 6's persistence: when a VectorStore is saved to
        disk and loaded back later, the FAISS index only stores raw
        vectors — the actual CodeChunk data has to be saved/restored
        separately (as plain JSON), and this is the other half of that
        round trip.
        """
        return cls(**data)


# ---------------------------------------------------------------------------
# Phase 5: Embeddings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A CodeChunk together with its embedding vector.

    Deliberately wraps the ORIGINAL CodeChunk rather than copying its
    fields out flat — this guarantees the raw source text and all of
    Phase 4's metadata survive into Phase 5's output unchanged, so
    nothing downstream (Phase 6, or a human debugging the pipeline) ever
    has to wonder whether embedding the chunk altered it.

    `model_name` and `dimensions` travel WITH each embedded chunk (not
    just logged once) because Phase 6 will need to know exactly which
    model/dimension produced a given vector — mixing vectors from two
    different models in one search index silently produces garbage
    results, since the numbers from different models aren't comparable.
    """

    chunk: CodeChunk
    vector: list[float]
    model_name: str
    dimensions: int

    def to_dict(self) -> dict:
        return {
            "chunk": self.chunk.to_dict(),
            "vector": self.vector,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
        }


# ---------------------------------------------------------------------------
# Phase 6: Vector Database + Semantic Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """
    One match from a semantic search: a CodeChunk, together with how
    similar it was to the search query.

    `score` comes from cosine similarity (via normalized vectors + FAISS
    inner product — see VectorStore): closer to 1.0 means more similar
    in meaning, closer to 0 means unrelated, closer to -1 means opposite
    (rare in practice for code). Deliberately minimal — a future step
    (RAG, not part of this phase) only needs "which code matched, and
    how confident was the match."
    """

    chunk: CodeChunk
    score: float

    def to_dict(self) -> dict:
        return {"chunk": self.chunk.to_dict(), "score": self.score}


# ---------------------------------------------------------------------------
# Phase 7: RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """
    A citation pointing back to one piece of real retrieved code that
    supported a RAG answer. Deliberately a slimmer shape than a full
    CodeChunk/SearchResult — this is what a UI would actually show a
    developer under "Sources:", not the full code text again.
    """

    file_path: str
    symbol_name: str | None
    start_line: int
    end_line: int
    score: float

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": self.score,
        }


@dataclass(frozen=True)
class RAGResponse:
    """
    The complete result of a RAG query: the LLM's written answer,
    together with the sources that grounded it.

    `retrieval_results` keeps the FULL Phase 6 SearchResults (including
    complete code content) alongside the slimmer `sources` list, so
    callers who need to inspect or debug exactly what the LLM saw can,
    without RAGService needing a second retrieval call.
    """

    answer: str
    sources: list[Source]
    retrieval_results: list[SearchResult]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
        }


# ---------------------------------------------------------------------------
# Phase 8: Repository Intelligence & Architecture Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Technology:
    """
    One detected/inferred technology, with its confidence tier stated
    explicitly so a "detected" fact (a real manifest file was found)
    is never blurred together with a lower-confidence "inferred" one
    (a framework name showed up in import statements).
    """

    name: str
    confidence: str  # "detected" or "inferred" — never anything blurrier than that
    evidence: str     # e.g. "found requirements.txt" or "imported in 3 file(s)"

    def to_dict(self) -> dict:
        return {"name": self.name, "confidence": self.confidence, "evidence": self.evidence}


@dataclass(frozen=True)
class DependencyEdge:
    """
    One directed internal dependency: `source` imports `target`. Built
    only from import statements that could be resolved to an ACTUAL
    file in this repository — an import of an external library (like
    `os` or `pytest`) never becomes an edge, since it says nothing about
    this repository's own internal structure.
    """

    source: str
    target: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True)
class DependencyGraph:
    """
    A simple in-memory dependency graph: which files import which other
    files, INTERNAL to this repository only.

    This is a DEPENDENCY graph, built from import statements — it is
    explicitly NOT a call graph. It cannot tell you which functions call
    which other functions at runtime, only which files reference which
    other files structurally. See repository_intelligence's design notes
    for why that distinction matters.
    """

    nodes: list[str]           # every file path in the repository
    edges: list[DependencyEdge]  # resolved internal import relationships

    @property
    def isolated_nodes(self) -> list[str]:
        """Files that neither import, nor are imported by, any other file in this repo."""
        connected = {e.source for e in self.edges} | {e.target for e in self.edges}
        return [n for n in self.nodes if n not in connected]

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "isolated_nodes": self.isolated_nodes,
        }


@dataclass(frozen=True)
class RepositoryOverview:
    """
    Purely DETERMINISTIC repository-level facts — nothing here comes
    from an LLM. Every field is either copied directly from earlier
    phases' data (languages) or computed with a plain, explainable rule
    (technologies, important files, entry points, dependency graph).
    This is what gets handed to the LLM as grounding evidence in Phase
    8's architecture analysis — the LLM interprets these facts, it does
    not get to invent new ones.
    """

    repo_url: str
    languages: dict[str, int]
    technologies: list[Technology]
    important_files: list[str]
    entry_points: list[str]
    dependency_graph: DependencyGraph

    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "languages": self.languages,
            "technologies": [t.to_dict() for t in self.technologies],
            "important_files": self.important_files,
            "entry_points": self.entry_points,
            "dependency_graph": self.dependency_graph.to_dict(),
        }


@dataclass(frozen=True)
class RepositoryAnalysis:
    """
    The result of a Phase 8 repository-level question. Deliberately
    similar in shape to RAGResponse (Phase 7): the LLM only ever fills
    in `answer` — every other field is copied straight from the
    deterministic RepositoryOverview or built from real file evidence
    (`sources`), so the LLM has no field it could use to sneak in an
    invented fact. `sources` here cite whole files (deterministic
    evidence), not retrieved snippets — see RepositoryIntelligenceService
    for how they're built.
    """

    answer: str
    technologies: list[Technology]
    important_files: list[str]
    entry_points: list[str]
    dependency_graph: DependencyGraph
    sources: list[Source]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "technologies": [t.to_dict() for t in self.technologies],
            "important_files": self.important_files,
            "entry_points": self.entry_points,
            "dependency_graph": self.dependency_graph.to_dict(),
            "sources": [s.to_dict() for s in self.sources],
        }


# ---------------------------------------------------------------------------
# Phase 9: Intelligent Developer Actions
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """
    The 5 supported developer actions. Deliberately a fixed, closed set —
    routing (see action_router.py) is a plain dictionary lookup on this
    enum, not an LLM guessing what the caller wants.
    """

    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    BUG_ANALYSIS = "bug_analysis"
    CHANGE_PLAN = "change_plan"


@dataclass(frozen=True)
class ActionRequest:
    """
    A structured developer action request. `action_type` is REQUIRED and
    explicit — the caller states which of the 5 actions they want, so
    routing never has to guess from free text. `target_file`/
    `target_symbol` are optional hints (if known) to help context
    retrieval; `error_message` is only meaningful for BUG_ANALYSIS.
    """

    action_type: ActionType
    request: str
    target_file: Optional[str] = None
    target_symbol: Optional[str] = None
    error_message: Optional[str] = None
    additional_context: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "request": self.request,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "error_message": self.error_message,
            "additional_context": self.additional_context,
        }


@dataclass(frozen=True)
class Finding:
    """
    One labeled claim within an ActionResult — a refactor issue, a bug
    hypothesis, or a change-plan step are all, structurally, "a claim
    about the repo, tagged by confidence, optionally pointing at a
    file/symbol." One shape, reused across all 5 actions, rather than a
    separate model per action type.

    `category` is the explicit anti-hallucination label required by this
    phase's design: "fact" (directly supported by retrieved evidence),
    "inference" (a reasonable conclusion beyond the literal evidence), or
    "suggestion" (a proposed action, not a claim about current reality).
    These three categories are never blurred together.
    """

    category: str  # "fact" / "inference" / "suggestion"
    description: str
    affected_file: Optional[str] = None
    affected_symbol: Optional[str] = None
    severity_or_confidence: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "description": self.description,
            "affected_file": self.affected_file,
            "affected_symbol": self.affected_symbol,
            "severity_or_confidence": self.severity_or_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        """
        Build a Finding from the LLM's parsed JSON output. Missing keys
        default safely rather than raising — a Finding missing an
        optional field is fine; ActionService's own JSON-validity check
        is what catches genuinely malformed LLM output before this
        point ever runs.
        """
        return cls(
            category=data.get("category", "suggestion"),
            description=data.get("description", ""),
            affected_file=data.get("affected_file"),
            affected_symbol=data.get("affected_symbol"),
            severity_or_confidence=data.get("severity_or_confidence"),
        )


@dataclass(frozen=True)
class ActionResult:
    """
    The structured result of a developer action. Same grounding
    philosophy as RAGResponse (Phase 7) and RepositoryAnalysis (Phase
    8): the LLM only ever supplies the interpretive/generative fields
    (`summary`, `generated_content`, `findings`' descriptions,
    `assumptions`, `warnings`) — `sources` is built directly by
    ActionService from real retrieval metadata (SearchResult /
    RepositoryOverview evidence), NEVER parsed from whatever the LLM
    claims about file paths or line numbers. This is what makes source
    references trustworthy rather than just plausible-looking.

    `generated_content` is kept as its own field, separate from
    `summary`'s prose, so a caller can clearly tell "this is proposed
    code" apart from "this is an explanation" (e.g. a draft test's
    source code vs. the paragraph explaining the test strategy).
    """

    action_type: str
    summary: str
    generated_content: Optional[str]
    findings: list[Finding]
    assumptions: list[str]
    warnings: list[str]
    sources: list[Source]

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "summary": self.summary,
            "generated_content": self.generated_content,
            "findings": [f.to_dict() for f in self.findings],
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "sources": [s.to_dict() for s in self.sources],
        }


# ---------------------------------------------------------------------------
# Phase 10: Controlled agentic code modification
# ---------------------------------------------------------------------------


class ChangeOperation(str, Enum):
    CREATE = "create"
    REPLACE = "replace"
    DELETE = "delete"


@dataclass(frozen=True)
class ProposedChange:
    """One untrusted, structured instruction produced by the LLM."""

    file_path: str
    operation: ChangeOperation
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    new_content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "operation": self.operation.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "new_content": self.new_content,
        }


@dataclass(frozen=True)
class FileDiff:
    file_path: str
    diff: str
    lines_added: int
    lines_removed: int

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "diff": self.diff,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "messages": self.messages}


@dataclass(frozen=True)
class TestResult:
    status: str  # passed / failed / timed_out / not_available
    command: Optional[list[str]]
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class AgentState:
    """Minimal state retained through a single Phase 10 workflow run."""

    user_request: ActionRequest
    workspace_path: str
    status: str = "requested"
    change_plan: Optional[ActionResult] = None
    proposed_changes: list[ProposedChange] = field(default_factory=list)
    applied_changes: list[ProposedChange] = field(default_factory=list)
    diffs: list[FileDiff] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    test_result: Optional[TestResult] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResult:
    """Human-reviewable result. The original repository is never changed."""

    status: str
    change_plan: Optional[ActionResult]
    proposed_changes: list[ProposedChange]
    applied_changes: list[ProposedChange]
    diffs: list[FileDiff]
    validation_result: Optional[ValidationResult]
    test_result: Optional[TestResult]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "change_plan": self.change_plan.to_dict() if self.change_plan else None,
            "proposed_changes": [change.to_dict() for change in self.proposed_changes],
            "applied_changes": [change.to_dict() for change in self.applied_changes],
            "diffs": [diff.to_dict() for diff in self.diffs],
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "test_result": self.test_result.to_dict() if self.test_result else None,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Phase 11: Intelligent repository-level code review
# ---------------------------------------------------------------------------


class ReviewMode(str, Enum):
    DIFF = "diff"
    FULL_REPO = "full_repo"


class ReviewSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ReviewCategory(str, Enum):
    SECURITY = "SECURITY"
    BUG = "BUG"
    ARCHITECTURE = "ARCHITECTURE"
    TESTING = "TESTING"
    ERROR_HANDLING = "ERROR_HANDLING"
    MAINTAINABILITY = "MAINTAINABILITY"
    PERFORMANCE = "PERFORMANCE"
    CODE_QUALITY = "CODE_QUALITY"
    DOCUMENTATION = "DOCUMENTATION"


class FindingKind(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    SUGGESTION = "SUGGESTION"


@dataclass(frozen=True)
class ReviewFinding:
    """A structured, evidence-based code review finding."""

    category: str
    severity: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[str] = None
    evidence: str = ""
    finding_kind: str = FindingKind.INFERENCE.value
    recommendation: str = ""
    confidence: str = "HIGH"

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet,
            "evidence": self.evidence,
            "finding_kind": self.finding_kind,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ReviewRequest:
    """Input parameters for a code review run."""

    mode: ReviewMode = ReviewMode.DIFF
    target_files: list[str] = field(default_factory=list)
    focus_categories: list[str] = field(default_factory=list)
    custom_instructions: Optional[str] = None
    diffs: list[FileDiff] = field(default_factory=list)
    agent_result: Optional[AgentResult] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value if isinstance(self.mode, ReviewMode) else self.mode,
            "target_files": self.target_files,
            "focus_categories": self.focus_categories,
            "custom_instructions": self.custom_instructions,
            "diffs": [d.to_dict() for d in self.diffs],
            "agent_result": self.agent_result.to_dict() if self.agent_result else None,
        }


@dataclass(frozen=True)
class ReviewReport:
    """Final, comprehensive code review report."""

    mode: str
    summary: str
    findings: list[ReviewFinding]
    severity_counts: dict[str, int]
    category_counts: dict[str, int]
    reviewed_files: list[str]
    passed_checks: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "severity_counts": self.severity_counts,
            "category_counts": self.category_counts,
            "reviewed_files": self.reviewed_files,
            "passed_checks": self.passed_checks,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Phase 12: GitHub Developer Workflow Integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubCredentials:
    """Authentication settings for GitHub API & Git operations."""

    token: Optional[str] = None
    auth_type: str = "PAT"

    def to_dict(self) -> dict:
        return {
            "token_present": self.token is not None and len(self.token) > 0,
            "auth_type": self.auth_type,
        }


@dataclass(frozen=True)
class GitBranchDetails:
    """Metadata regarding local Git feature branch and commits."""

    branch_name: str
    base_branch: str = "main"
    commit_hash: Optional[str] = None
    staged_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "commit_hash": self.commit_hash,
            "staged_files": self.staged_files,
        }


@dataclass(frozen=True)
class PullRequestDetails:
    """Metadata for created GitHub Pull Request."""

    title: str
    body: str
    head_branch: str
    base_branch: str
    html_url: Optional[str] = None
    number: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "head_branch": self.head_branch,
            "base_branch": self.base_branch,
            "html_url": self.html_url,
            "number": self.number,
        }


@dataclass(frozen=True)
class GitHubWorkflowRequest:
    """Inputs required for Phase 12 GitHub developer workflow."""

    repo_owner: str
    repo_name: str
    branch_name: Optional[str] = None
    base_branch: str = "main"
    pr_title: Optional[str] = None
    approved: bool = False
    token: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "branch_name": self.branch_name,
            "base_branch": self.base_branch,
            "pr_title": self.pr_title,
            "approved": self.approved,
            "token_present": self.token is not None and len(self.token) > 0,
        }


@dataclass(frozen=True)
class GitHubWorkflowResult:
    """Final output of Phase 12 workflow."""

    status: str  # approved_and_created / missing_approval / branch_only / failed
    approved: bool
    branch_details: Optional[GitBranchDetails] = None
    pr_details: Optional[PullRequestDetails] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "approved": self.approved,
            "branch_details": self.branch_details.to_dict() if self.branch_details else None,
            "pr_details": self.pr_details.to_dict() if self.pr_details else None,
            "warnings": self.warnings,
            "errors": self.errors,
        }


