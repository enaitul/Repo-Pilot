"""
Orchestrates the full ingestion pipeline: clone -> walk -> build model.

This is the only module that knows the END-TO-END pipeline order. Both
the CLI (cli.py) and, later, any HTTP endpoint or agent tool should call
into THIS module rather than re-wiring GitCloner + RepoWalker themselves.
That's the point of having an orchestrator: pipeline policy lives in one
place, not duplicated across every caller.
"""

from __future__ import annotations

from repopilot.cloner import GitCloner
from repopilot.walker import RepoWalker
from repopilot.models import RepositoryModel
from repopilot.exceptions import RepoPilotError, IngestionError


class IngestionService:
    """High-level entry point for repository ingestion."""

    def __init__(self, cloner: GitCloner | None = None, walker: RepoWalker | None = None):
        # Constructor injection of collaborators — lets tests substitute
        # a fake/mock cloner without ever touching the network, and
        # keeps this class from being hard-wired to concrete implementations.
        self._cloner = cloner or GitCloner()
        self._walker = walker or RepoWalker()

    def ingest(self, repo_url: str, cleanup: bool = True) -> RepositoryModel:
        """
        Run the full ingestion pipeline for `repo_url`.

        If `cleanup` is True (default), the cloned repo is deleted from
        disk after metadata extraction — Phase 1 only needs the
        structured representation, not a persistent local checkout.
        Set cleanup=False if a later phase will need the raw files too.
        """
        local_path = None
        try:
            local_path = self._cloner.clone(repo_url)
            files = self._walker.walk(local_path)
            return RepositoryModel(
                repo_url=repo_url,
                local_path=local_path,
                files=files,
            )
        except RepoPilotError:
            # Already a well-typed domain error (InvalidRepoURLError,
            # CloneError, WalkError) — let it propagate as-is so callers
            # can distinguish failure modes.
            raise
        except Exception as exc:
            # Anything unexpected gets wrapped so callers only ever see
            # RepoPilotError subtypes from this package's public API.
            raise IngestionError(
                f"Unexpected failure ingesting '{repo_url}': {exc}"
            ) from exc
        finally:
            if cleanup and local_path:
                self._cloner.cleanup(local_path)
