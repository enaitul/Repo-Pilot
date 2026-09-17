"""
RepoPilot — AI Developer Productivity Agent

Public API surface for this package. Downstream phases (chunking,
embeddings, agent tools) should import from here rather than reaching
into submodules directly — this keeps the internal module layout free
to change without breaking callers.

Phase 1/2: Repository Ingestion (clone, walk, fingerprint)
Phase 3:   Code Parsing (classes/functions/methods + line ranges)
Phase 4:   Code Chunking (symbol-aware, embedding-ready text pieces)
Phase 5:   Embeddings (chunks -> vectors, via a swappable provider)
Phase 6:   Vector Database + Semantic Retrieval (FAISS-backed search)
Phase 7:   RAG (retrieval-grounded Q&A, via a swappable LLM provider)
Phase 8:   Repository Intelligence & Architecture Analysis (deterministic
           repo facts + LLM interpretation)
Phase 9:   Intelligent Developer Actions (test generation, docs,
           refactoring, bug analysis, change planning — read-only,
           advisory, human-reviewed)
Phase 10:  Controlled agentic code modification in a disposable workspace
"""

from repopilot.models import (
    FileMetadata,
    RepositoryModel,
    SymbolType,
    ParseStatus,
    CodeSymbol,
    ParsedFile,
    CodeChunk,
    EmbeddedChunk,
    SearchResult,
    Source,
    RAGResponse,
    Technology,
    DependencyEdge,
    DependencyGraph,
    RepositoryOverview,
    RepositoryAnalysis,
    ActionType,
    ActionRequest,
    Finding,
    ActionResult,
    ChangeOperation,
    ProposedChange,
    FileDiff,
    ValidationResult,
    TestResult,
    AgentState,
    AgentResult,
    ReviewMode,
    ReviewSeverity,
    ReviewCategory,
    FindingKind,
    ReviewFinding,
    ReviewRequest,
    ReviewReport,
    GitHubCredentials,
    GitBranchDetails,
    PullRequestDetails,
    GitHubWorkflowRequest,
    GitHubWorkflowResult,
)
from repopilot.ingestion_service import IngestionService
from repopilot.parsing_service import ParsingService
from repopilot.parser_factory import ParserFactory
from repopilot.chunker import Chunker
from repopilot.chunking_service import ChunkingService
from repopilot.embedding_provider import EmbeddingProvider
from repopilot.sentence_transformer_provider import SentenceTransformerProvider
from repopilot.embedding_service import EmbeddingService
from repopilot.vector_store import VectorStore
from repopilot.search_service import SearchService

from repopilot.llm_provider import LLMProvider
from repopilot.groq_provider import GroqProvider
from repopilot.context_builder import ContextBuilder
from repopilot.prompt_builder import PromptBuilder
from repopilot.rag_service import RAGService
from repopilot.technology_detector import TechnologyDetector
from repopilot.important_file_detector import ImportantFileDetector
from repopilot.dependency_graph_builder import DependencyGraphBuilder
from repopilot.repository_overview_builder import RepositoryOverviewBuilder
from repopilot.architecture_context_builder import ArchitectureContextBuilder
from repopilot.repository_intelligence_service import RepositoryIntelligenceService
from repopilot.action_context_builder import ActionContextBuilder
from repopilot.action_router import ActionRouter
from repopilot.action_service import ActionService
from repopilot.workspace_tools import WorkspaceTools
from repopilot.validation_service import ValidationService
from repopilot.controlled_test_runner import TestRunner
from repopilot.modification_service import ModificationService
from repopilot.review_context_builder import ReviewContextBuilder
from repopilot.review_service import ReviewService
from repopilot.git_service import GitService
from repopilot.github_api_service import GitHubAPIService
from repopilot.github_workflow_service import GitHubWorkflowService
from repopilot.exceptions import (
    RepoPilotError,
    InvalidRepoURLError,
    CloneError,
    IngestionError,
    ParsingError,
    EmbeddingError,
    VectorStoreError,
    RAGError,
    RepositoryIntelligenceError,
    ActionError,
    ModificationError,
    ValidationError,
    TestExecutionError,
    ReviewError,
    GitHubWorkflowError,
    AuthenticationError,
    BranchError,
    PullRequestError,
)

__all__ = [
    # Phase 1/2
    "FileMetadata",
    "RepositoryModel",
    "IngestionService",
    # Phase 3
    "SymbolType",
    "ParseStatus",
    "CodeSymbol",
    "ParsedFile",
    "ParsingService",
    "ParserFactory",
    # Phase 4
    "CodeChunk",
    "Chunker",
    "ChunkingService",
    # Phase 5
    "EmbeddedChunk",
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "EmbeddingService",
    # Phase 6
    "SearchResult",
    "VectorStore",
    "SearchService",
    # Phase 7
    "Source",
    "RAGResponse",
    "LLMProvider",
    "GroqProvider",
    "ContextBuilder",
    "PromptBuilder",
    "RAGService",
    # Phase 8
    "Technology",
    "DependencyEdge",
    "DependencyGraph",
    "RepositoryOverview",
    "RepositoryAnalysis",
    "TechnologyDetector",
    "ImportantFileDetector",
    "DependencyGraphBuilder",
    "RepositoryOverviewBuilder",
    "ArchitectureContextBuilder",
    "RepositoryIntelligenceService",
    # Phase 9
    "ActionType",
    "ActionRequest",
    "Finding",
    "ActionResult",
    "ActionContextBuilder",
    "ActionRouter",
    "ActionService",
    # Phase 10
    "ChangeOperation",
    "ProposedChange",
    "FileDiff",
    "ValidationResult",
    "TestResult",
    "AgentState",
    "AgentResult",
    "WorkspaceTools",
    "ValidationService",
    "TestRunner",
    "ModificationService",
    # Phase 11
    "ReviewMode",
    "ReviewSeverity",
    "ReviewCategory",
    "FindingKind",
    "ReviewFinding",
    "ReviewRequest",
    "ReviewReport",
    "ReviewContextBuilder",
    "ReviewService",
    # Phase 12
    "GitHubCredentials",
    "GitBranchDetails",
    "PullRequestDetails",
    "GitHubWorkflowRequest",
    "GitHubWorkflowResult",
    "GitService",
    "GitHubAPIService",
    "GitHubWorkflowService",
    # Exceptions
    "RepoPilotError",
    "InvalidRepoURLError",
    "CloneError",
    "IngestionError",
    "ParsingError",
    "EmbeddingError",
    "VectorStoreError",
    "RAGError",
    "RepositoryIntelligenceError",
    "ActionError",
    "ModificationError",
    "ValidationError",
    "TestExecutionError",
    "ReviewError",
    "GitHubWorkflowError",
    "AuthenticationError",
    "BranchError",
    "PullRequestError",
]

