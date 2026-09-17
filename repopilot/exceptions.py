"""
Domain-specific exceptions.

Rather than letting subprocess.CalledProcessError, OSError, or
requests-level exceptions leak out of this package, every failure mode
is translated into one of these. Callers (a CLI today; an HTTP endpoint
or an agent tool later) can catch RepoPilotError generically or catch a
specific subtype to react precisely — e.g. "bad input, ask the user to
fix the URL" vs "transient failure, retry."
"""


class RepoPilotError(Exception):
    """Base class for all errors raised by RepoPilot's ingestion subsystem."""


class InvalidRepoURLError(RepoPilotError):
    """Raised when the given URL is not an acceptable public GitHub repo URL."""


class CloneError(RepoPilotError):
    """Raised when `git clone` fails, times out, or produces no usable output."""


class WalkError(RepoPilotError):
    """Raised when directory traversal encounters an unrecoverable filesystem error."""


class IngestionError(RepoPilotError):
    """Raised for failures in the overall ingestion pipeline orchestration."""


class ParsingError(RepoPilotError):
    """
    Raised for APPLICATION-level parsing failures only — e.g. the parser
    subsystem itself is misconfigured (no parser registered, a required
    parsing dependency missing at the factory level, etc).

    IMPORTANT DISTINCTION (Phase 3):
    A malformed/unparseable *source file* inside the repository being
    analyzed is NOT a RepoPilotError. That is an expected, per-file
    outcome and is represented as a `ParsedFile` with a non-SUCCESS
    `ParseStatus` (SYNTAX_ERROR, UNSUPPORTED_LANGUAGE, PARSER_ERROR).
    One broken file in the target repo must never raise an exception
    that aborts parsing of the rest of the repository — see
    ParsingService.parse_repository.
    """


class EmbeddingError(RepoPilotError):
    """
    Raised when the embedding subsystem itself fails — the model can't be
    loaded, an embedding call raises for a whole batch, or a provider
    returns a response that doesn't match what was asked of it (e.g. a
    different number of vectors than input texts).

    This is a SYSTEM-level failure, not a per-chunk one: an embedding
    call is one all-or-nothing operation over a batch of texts, so there
    is no meaningful "this one chunk in the batch failed but the rest
    succeeded" outcome the way there is for parsing individual files.
    EmbeddingService catches this per-batch so one bad batch doesn't
    necessarily have to abort every other batch in the run.
    """


class VectorStoreError(RepoPilotError):
    """
    Raised for failures in Phase 6's vector storage/search subsystem:
    a dimension mismatch between vectors, an empty or missing index
    being searched incorrectly, a corrupted or missing persisted index
    file, invalid search parameters (e.g. top_k <= 0), or a search
    query that fails to embed. Like EmbeddingError, this covers
    SYSTEM-level failures of the search mechanism itself — not "no
    results were relevant," which is simply an empty (but valid)
    list[SearchResult], not an error.
    """


class RAGError(RepoPilotError):
    """
    Raised for failures in Phase 7's RAG pipeline: an empty question, an
    LLM provider that fails to respond (network issue, missing/invalid
    API key, rate limit, malformed response shape), or a prompt that
    fails to construct. Retrieval finding zero relevant chunks is NOT
    this — that's an expected, gracefully-handled outcome (the LLM is
    told explicitly that no relevant code was found, rather than the
    pipeline raising), matching the same "expected empty result vs.
    system failure" distinction used throughout every earlier phase.
    """


class RepositoryIntelligenceError(RepoPilotError):
    """
    Raised for failures in Phase 8's repository-level analysis: a
    missing/invalid RepositoryModel, an LLM provider failure, or a
    failure while building the deterministic RepositoryOverview (e.g.
    malformed import data). An empty repository (zero files) is NOT an
    error — it's a valid, if minimal, RepositoryOverview — matching the
    same "expected empty state vs. system failure" distinction used
    throughout the rest of RepoPilot.
    """


class ActionError(RepoPilotError):
    """
    Raised for failures in Phase 9's developer-action pipeline: an
    unknown action type, a missing ActionRequest or empty request text,
    a required context source not being available for the requested
    action (e.g. CHANGE_PLAN needs a RepositoryModel; other actions need
    a SearchService), an LLM provider failure, or LLM output that isn't
    valid JSON / isn't a JSON object. Malformed LLM output is
    deliberately NOT silently patched or guessed at — an action's
    generated content is exactly the kind of thing that must fail
    loudly rather than return something subtly wrong to a developer.
    """


class ModificationError(RepoPilotError):
    """Raised when a Phase 10 change is malformed, unsafe, or cannot be applied."""


class ValidationError(RepoPilotError):
    """Raised when deterministic validation rejects an applied workspace change."""


class TestExecutionError(RepoPilotError):
    """Raised when no safe, deterministic test command can be selected."""


class ReviewError(RepoPilotError):
    """Raised for failures in Phase 11's intelligent code review pipeline."""


class GitHubWorkflowError(RepoPilotError):
    """Base exception for Phase 12 GitHub developer workflow failures."""


class AuthenticationError(GitHubWorkflowError):
    """Raised when GitHub token authentication fails or is missing."""


class BranchError(GitHubWorkflowError):
    """Raised when local or remote Git branch operations fail."""


class PullRequestError(GitHubWorkflowError):
    """Raised when GitHub API operations (e.g. PR creation) fail."""


