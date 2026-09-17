# RepoPilot

RepoPilot is an AI-powered developer productivity agent that understands public GitHub repositories using code-aware parsing, semantic retrieval, RAG, repository intelligence, controlled agentic code modification, validation, testing, and repository-level code review.

## What it does

- Clones and inventories a public GitHub HTTPS repository.
- Parses supported source files, creates symbol-aware chunks, and indexes them locally with sentence-transformers and FAISS.
- Answers repository questions with RAG and source locations.
- Builds deterministic architecture facts: languages, technologies, entry points, important files, and internal dependency edges.
- Provides Phase 9 developer actions: test generation, documentation, refactoring, bug analysis, and change planning.
- Runs Phase 10 structured modifications only in a disposable workspace, returning diffs, validation, and allowlisted test output.
- Runs Phase 11 LLM-assisted repository or diff review with severity-ranked findings.
- Supports Phase 12 local branch/commit workflow and, after explicit approval plus a server-side GitHub token, optional PR creation.

## Architecture

```
Browser UI → FastAPI adapter → existing RepoPilot services
                              ├─ clone → walk → parse → chunk
                              ├─ local embeddings → FAISS → RAG
                              ├─ repository intelligence → developer actions
                              ├─ controlled clone modification → validation → tests
                              └─ code review → approved GitHub workflow
```

The FastAPI layer in `api.py` is intentionally thin. It orchestrates the existing Phase 1–12 services instead of duplicating their logic. A session owns one controlled clone; it is removed at expiry or server shutdown.

## Technology stack

Python 3.12, FastAPI, Uvicorn, Git, sentence-transformers, FAISS, Groq, and a static responsive HTML/CSS/JavaScript interface.

## Phase summary

Phases 1–2 ingest and safely walk a shallow public clone; 3 parses code; 4 chunks symbols; 5 embeds; 6 retrieves with FAISS; 7 provides RAG; 8 computes repository intelligence; 9 generates advisory developer actions; 10 applies controlled structured changes; 11 reviews code; and 12 manages the approval-gated GitHub workflow.

## Security boundaries

- Only validated public `https://github.com/owner/repo` URLs reach `git clone`.
- Git uses argument lists, shallow cloning, and a timeout.
- A workspace guard rejects absolute paths, traversal, sensitive filenames, unsupported file types, and oversized modifications.
- Test execution is selected from a fixed allowlist; LLM output never supplies a shell command.
- Groq BYOK values are sent only with an AI request, used by an ephemeral provider instance, and are not stored, logged, or returned.
- GitHub tokens remain server-side environment variables.
- Original repositories are never modified. The Phase 12 remote workflow requires explicit user approval.

## Local setup

```bash
cd repopilot_phase4
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000`. The interactive API documentation is at `/docs`.

Set environment values in your deployment platform or shell (the application intentionally does not load `.env` itself):

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | No | Optional local-development fallback. Public users can enter a request-scoped Groq key in the UI. |
| `GITHUB_TOKEN` | Only for remote PR creation | GitHub token used only after approval |
| `REPOPILOT_SESSION_TTL_SECONDS` | No | Controlled workspace lifetime; defaults to 3600 |
| `REPOPILOT_ALLOWED_ORIGINS` | No | Comma-separated origins for a separately hosted frontend |

## Deployment

The included `Dockerfile` packages the single FastAPI service and its static UI. It is suitable for Render, Railway, Fly.io, or any container host with enough ephemeral disk for a temporary clone and enough memory for the local embedding model. `render.yaml` provides a minimal Render blueprint; optionally add `GROQ_API_KEY` as a local-development fallback; public users can instead supply their own Groq key in the UI. The service starts with:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Example workflow

1. Submit a public GitHub URL and inspect the detected overview.
2. Ask a code question; the answer lists retrieved file/line sources.
3. Generate a change plan or another developer action.
4. Request a controlled modification and inspect its diff, validation, and test result.
5. Run a review. If changes are satisfactory, explicitly approve the GitHub workflow.

## Limitations

The first analysis downloads the local embedding model if it is not cached. Repositories are limited by the existing clone and file-size safeguards. AI operations require a valid Groq key and may fail due to provider availability or malformed model output; failures are returned to the UI rather than silently treated as success. This deployment configuration is ready to deploy, but a public URL can only be created from a hosting account with the required secret configured.
