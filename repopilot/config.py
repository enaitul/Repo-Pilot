"""
Central configuration for the ingestion subsystem.

Keeping tunables in one module (rather than scattered magic numbers/strings
across the codebase) means changing "what counts as ignorable" or "how big
is too big" is a one-line change, and it's the first place a reviewer or
interviewer would look to understand the system's policy.
"""

from __future__ import annotations

# Directories we never walk into. Matched by exact directory name.
# Pruned in-place during os.walk so we never even descend into them —
# this matters a lot for node_modules-sized directories.
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "target",       # Rust/Java build output
        ".next",        # Next.js build output
        "coverage",
        ".tox",
        "egg-info",
    }
)

# File extensions that are always skipped outright (binary / non-source),
# even before we bother sniffing file content.
IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
        ".pyc", ".pyo", ".class", ".jar",
        ".woff", ".woff2", ".ttf", ".eot",
        ".mp3", ".mp4", ".mov", ".avi", ".wav",
        ".db", ".sqlite", ".sqlite3",
        ".lock",  # package-lock.json etc: real data, but not "source code"
    }
)

# Extension -> language name. Used by LanguageDetector.
# Only extensions present here are treated as "source code" for the
# purposes of metadata extraction + import parsing.
LANGUAGE_EXTENSION_MAP: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".sql": "SQL",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".md": "Markdown",
    ".html": "HTML",
    ".css": "CSS",
}

# Files larger than this are recorded as metadata-only: we still report
# path/language/size, but we do NOT read their content (no import
# extraction). Protects against pathological huge files (data dumps,
# checked-in logs) blowing up memory/time.
MAX_FILE_SIZE_BYTES: int = 1_000_000  # 1 MB

# Number of bytes read from the start of a file to sniff for binary
# content (null-byte heuristic — the same trick `git` uses internally).
BINARY_SNIFF_BYTES: int = 8192

# Hard timeout for the `git clone` subprocess. Prevents a hung network
# call or a pathologically large repo from blocking the pipeline forever.
CLONE_TIMEOUT_SECONDS: int = 120

# Repos above this size (measured after clone) are rejected rather than
# ingested, to keep Phase 1 predictable and fast. This is a policy
# decision, not a hard technical limit — easy to raise later.
MAX_REPO_SIZE_BYTES: int = 500_000_000  # 500 MB

# ---------------------------------------------------------------------------
# Phase 4: Chunking
# ---------------------------------------------------------------------------

# Rough ceiling on how large a single chunk's text is allowed to be, in
# characters. This is a cheap PROXY for "will roughly fit in one embedding
# call" — real embedding models count in tokens, not characters, but
# measuring exact tokens requires pulling in a tokenizer library. This is
# a deliberate approximation for Phase 4; a more precise version would
# swap this for a real token count.
MAX_CHUNK_CHARS: int = 2000

# When a single function/method is too large to fit in one chunk even
# after we've exhausted structural splitting (class -> methods), we fall
# back to a raw line-based sliding window. Consecutive windows share this
# many lines of overlap so that code sitting right at a cut boundary isn't
# fully lost from both resulting chunks.
CHUNK_OVERLAP_LINES: int = 3

# ---------------------------------------------------------------------------
# Phase 5: Embeddings
# ---------------------------------------------------------------------------

# Which sentence-transformers model to use for turning code chunks into
# vectors. Chosen for Phase 5 specifically because it's small (~80MB,
# downloads once and is cached locally forever), fast on CPU, requires no
# API key/account, and produces good general-purpose semantic embeddings.
# Swapping to a different local model, or to a hosted API instead, is a
# one-file change (a new EmbeddingProvider) — nothing else in the
# pipeline needs to know which model produced the vectors.
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# How many chunks' worth of text to hand to the embedding model in one
# call. sentence-transformers processes a batch together more
# efficiently than one-at-a-time; this caps memory use for very large
# repositories with thousands of chunks.
EMBEDDING_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Phase 6: Vector Database + Semantic Retrieval
# ---------------------------------------------------------------------------

# Where the FAISS index and its chunk-ID mapping are saved on disk, so a
# vector store built once can be reloaded on the next run instead of
# re-embedding everything from scratch. Relative to wherever the CLI is
# run from, matching the "local, no external service" philosophy of the
# rest of this project.
VECTOR_STORE_DIR: str = "repopilot_vector_store"

# Default number of results a semantic search returns when the caller
# doesn't specify one. Kept small on purpose — a handful of highly
# relevant chunks is far more useful downstream (e.g. to a future RAG
# step with limited context space) than ranking the entire repository.
DEFAULT_TOP_K: int = 5

# ---------------------------------------------------------------------------
# Phase 7: RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

# Which model to request from Groq. Chosen as a capable, current
# general-purpose open-weight model available on Groq's free tier.
# Swapping to a different Groq model, or to an entirely different LLM
# provider, is a one-file change (a new LLMProvider) — RAGService itself
# never needs to change.
GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Phase 10: Controlled code modification
# ---------------------------------------------------------------------------

# Modification is intentionally limited to text files RepoPilot already
# understands, plus the small set of common project/test configuration files.
MODIFICATION_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_EXTENSION_MAP) | frozenset(
    {".toml", ".ini", ".cfg", ".txt"}
)
MAX_MODIFIED_FILE_SIZE_BYTES: int = 1_000_000
TEST_TIMEOUT_SECONDS: int = 60
MAX_TEST_OUTPUT_CHARS: int = 20_000

# ---------------------------------------------------------------------------
# Phase 8: Repository Intelligence & Architecture Analysis
# ---------------------------------------------------------------------------

# Filenames commonly used as an application's actual starting point.
# Deliberately a SMALL, maintainable list of widely-used conventions —
# not an attempt to catch every possible entry-point naming style. This
# is a heuristic ("files that are OFTEN entry points"), not ground truth
# about any specific repository.
ENTRY_POINT_FILENAMES: frozenset[str] = frozenset(
    {
        "main.py", "app.py", "__main__.py", "manage.py",
        "server.js", "app.js", "index.js", "index.ts",
        "App.jsx", "App.tsx", "main.jsx", "main.tsx",
    }
)

# Filenames worth flagging as "probably important to read early" —
# entry points PLUS common project-level config/manifest files. Also a
# heuristic: a repository is free to organize itself completely
# differently, and this list won't catch that.
IMPORTANT_FILE_PATTERNS: frozenset[str] = ENTRY_POINT_FILENAMES | frozenset(
    {
        "package.json", "pyproject.toml", "requirements.txt", "setup.py",
        "Dockerfile", "docker-compose.yml", "pom.xml", "build.gradle",
    }
)

# Filename -> technology name, for the HIGH-CONFIDENCE "detected" tier:
# finding one of these files is direct, unambiguous evidence a
# technology/ecosystem is in use (a real manifest file naming it).
TECHNOLOGY_MANIFEST_FILES: dict[str, str] = {
    "requirements.txt": "Python (pip)",
    "pyproject.toml": "Python (packaging)",
    "package.json": "Node.js",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
}

# Import-string substring -> technology name, for the LOWER-CONFIDENCE
# "inferred" tier: seeing a framework imported repeatedly is real
# evidence, but weaker than a manifest file explicitly naming it (a
# manifest could be missing, out of date, or the import could be an
# unrelated same-named local module).
TECHNOLOGY_IMPORT_HINTS: dict[str, str] = {
    "flask": "Flask",
    "fastapi": "FastAPI",
    "django": "Django",
    "react": "React",
    "express": "Express",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
}

# ---------------------------------------------------------------------------
# Phase 12: GitHub Developer Workflow Integration
# ---------------------------------------------------------------------------
GITHUB_API_BASE_URL: str = "https://api.github.com"
DEFAULT_BRANCH_PREFIX: str = "feature/repopilot-"
DEFAULT_REMOTE_NAME: str = "origin"
DEFAULT_BASE_BRANCH: str = "main"
GITHUB_REQUEST_TIMEOUT_SECONDS: int = 30

