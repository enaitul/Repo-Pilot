"""
Orchestrates Phase 9: turns a structured ActionRequest into a
structured, sourced ActionResult.

    ActionRequest
        |
        v
    ActionRouter.get_prompt_builder()      <- deterministic, NEW
        |
        v
    ActionContextBuilder.build()           <- reuses Phase 6/7/8, NEW glue
        |
        v
    (context_text, search_results)
        |
        v
    prompt_builder(context_text, request)  <- Phase 7's PromptBuilder, extended
        |
        v
    LLMProvider.generate()                 <- Phase 7's GroqProvider, UNCHANGED
        |
        v
    parsed JSON  ->  ActionResult (sources built from search_results, NOT from the LLM)

Phase 9 is READ-ONLY and advisory: nothing in this file (or anywhere
else in Phase 9) writes to the filesystem, runs a command, or touches
git. `generated_content` is a PROPOSAL for human review — it is never
written to disk automatically.
"""

from __future__ import annotations

import json

from repopilot.action_context_builder import ActionContextBuilder
from repopilot.action_router import ActionRouter
from repopilot.config import DEFAULT_TOP_K
from repopilot.exceptions import ActionError
from repopilot.groq_provider import GroqProvider
from repopilot.llm_provider import LLMProvider
from repopilot.models import ActionRequest, ActionResult, Finding, RepositoryModel, Source


class ActionService:
    """High-level entry point for Phase 9: intelligent developer actions."""

    def __init__(
        self,
        context_builder: ActionContextBuilder | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        # Same constructor-injection pattern as every earlier service —
        # lets tests substitute fakes with zero real LLM calls and zero
        # real retrieval.
        self._context_builder = context_builder or ActionContextBuilder()
        self._llm_provider = llm_provider or GroqProvider()

    def execute(
        self,
        request: ActionRequest,
        repo_model: RepositoryModel | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> ActionResult:
        """
        Execute `request`, returning a structured ActionResult. Never
        modifies the repository, runs code, or writes files — the result
        is a proposal for a human to review.
        """
        if request is None:
            raise ActionError("An ActionRequest is required.")
        if not request.request or not request.request.strip():
            raise ActionError("Action request text must not be empty.")

        prompt_builder = ActionRouter.get_prompt_builder(request.action_type)
        context_text, search_results = self._context_builder.build(request, repo_model, top_k)
        system_prompt, user_prompt = prompt_builder(context_text, request)

        try:
            raw_response = self._llm_provider.generate(system_prompt, user_prompt)
        except ActionError:
            raise
        except Exception as exc:
            raise ActionError(f"Action generation failed: {exc}") from exc

        parsed = self._parse_llm_json(raw_response)

        # Sources are built from OUR OWN retrieval metadata, never from
        # whatever the LLM claims — this is what keeps them trustworthy.
        # CHANGE_PLAN has no per-chunk search_results (see
        # ActionContextBuilder), so it simply has no sources here; its
        # grounding instead lives in the architecture facts already
        # embedded in the prompt.
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

        return ActionResult(
            action_type=request.action_type.value,
            summary=parsed.get("summary", ""),
            generated_content=parsed.get("generated_content"),
            findings=[Finding.from_dict(f) for f in parsed.get("findings", []) if isinstance(f, dict)],
            assumptions=[a for a in parsed.get("assumptions", []) if isinstance(a, str)],
            warnings=[w for w in parsed.get("warnings", []) if isinstance(w, str)],
            sources=sources,
        )

    @staticmethod
    def _parse_llm_json(raw_response: str) -> dict:
        """
        Defensively parse the LLM's JSON response. Tolerates the common
        case of a model wrapping its JSON in a markdown code fence
        despite being told not to; anything that still isn't valid JSON,
        or isn't a JSON object, raises ActionError rather than being
        silently patched or guessed at — malformed structured output
        from an action is exactly the kind of failure that must be loud.
        """
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ActionError(f"LLM response was not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ActionError("LLM response JSON must be an object with the expected action fields.")

        return parsed
