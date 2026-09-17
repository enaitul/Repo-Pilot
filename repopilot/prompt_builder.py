"""
Builds the (system_prompt, user_prompt) pair sent to an LLMProvider.

The system prompt is deliberately short and direct rather than a long
list of rules — an overloaded system prompt is itself a source of the
model losing track of instructions. Each instruction line exists for a
specific reason (see the docstring on each), targeting the exact
failure modes RAG is meant to reduce (see the module docstring on
context_builder.py and Phase 7's design notes for the fuller reasoning):

- "rely primarily on the provided context" -> pushes toward grounding
- "do not invent repository details"        -> directly targets hallucination
- "say so explicitly if context is insufficient" -> prevents confident
  wrong answers when evidence is missing (e.g. "does this use Redis?"
  when nothing in the retrieved code mentions Redis)
- "reference relevant files/functions"      -> keeps the answer traceable
- "distinguish facts from inference"        -> some reasonable reasoning
  beyond the literal code is fine, as long as it's labeled as such
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a specific software "
    "repository. You will be given code retrieved from that repository as "
    "context.\n\n"
    "Rely primarily on the provided context to answer. Do not invent "
    "details about the repository that are not shown in the context. If "
    "the context does not contain enough information to answer "
    "confidently, say so explicitly rather than guessing.\n\n"
    "When useful, reference the specific files or functions the context "
    "came from. Clearly distinguish facts stated directly in the code "
    "from any reasonable inference you are making beyond it."
)

_ARCHITECTURE_SYSTEM_PROMPT = (
    "You are an assistant explaining the architecture of a specific "
    "software repository to a developer. You will be given STRUCTURED, "
    "DETERMINISTICALLY COMPUTED facts about the repository: its "
    "languages, detected/inferred technologies, important files, entry "
    "points, and an internal file dependency graph built from real "
    "import statements.\n\n"
    "Rely ONLY on the provided facts. Do not invent files, technologies, "
    "frameworks, or relationships that are not listed. If the facts are "
    "insufficient to answer part of the question, say so explicitly "
    "rather than guessing.\n\n"
    "Clearly distinguish facts that are directly listed from any "
    "reasonable architectural interpretation you are adding on top of "
    "them. The dependency graph reflects import statements only — it is "
    "NOT a runtime call graph, and you should not claim to know exact "
    "runtime behavior from it alone."
)

# ---------------------------------------------------------------------------
# Phase 9: Intelligent Developer Actions
# ---------------------------------------------------------------------------
#
# Every action prompt shares one JSON output contract (so ActionService
# can parse all 5 the same way) and one grounding backbone (so
# hallucination-reduction rules are never accidentally different between
# actions). Each action then adds its OWN specific instructions on top —
# one shared giant prompt would produce vague results for all 5, per
# Phase 9's design notes.

_ACTION_JSON_CONTRACT = (
    "\n\nRespond with ONLY a single valid JSON object (no markdown code "
    "fences, no extra prose before or after it) with exactly these keys:\n"
    '  "summary": a string explaining your findings/approach in prose.\n'
    '  "generated_content": a string with any generated code/text this '
    "action produces, or null if this action doesn't generate content.\n"
    '  "findings": a list of objects, each with "category" (exactly '
    '"fact", "inference", or "suggestion"), "description", '
    '"affected_file" (or null), "affected_symbol" (or null), and '
    '"severity_or_confidence" (or null).\n'
    '  "assumptions": a list of strings — anything you assumed due to '
    "missing information.\n"
    '  "warnings": a list of strings — anything the developer should be '
    "cautious about.\n"
)

_ACTION_GROUNDING_RULES = (
    "Rely primarily on the provided repository context. Do not invent "
    "files, functions, classes, or dependencies that are not shown in "
    "the context. If the context is insufficient, say so explicitly in "
    '"assumptions" or "warnings" rather than guessing. Every "findings" '
    'entry MUST be labeled "fact" (directly supported by the provided '
    'context), "inference" (a reasonable conclusion beyond it), or '
    '"suggestion" (a proposed action, not a claim about current '
    "reality) — never blur these together. You are not able to execute "
    "code, modify files, or verify your output actually works — treat "
    "everything you produce as a proposal for human review, and say so "
    "if relevant."
)

_TEST_GENERATION_SYSTEM_PROMPT = (
    "You are an assistant that proposes unit tests for code in a "
    "specific software repository, based on retrieved repository "
    "context.\n\n"
    + _ACTION_GROUNDING_RULES
    + "\n\nDetect the testing framework already used in the repository "
    "from the retrieved context (e.g. an existing test file's imports). "
    "If no existing tests are visible in the context and you cannot "
    "confidently determine a framework, say so explicitly in "
    '"assumptions" instead of inventing one. Put the actual proposed '
    'test code in "generated_content", and describe your test strategy '
    'and what cases it covers in "summary". Prefer patterns already '
    "present in the repository's existing tests when they're visible in "
    "the context."
    + _ACTION_JSON_CONTRACT
)

_DOCUMENTATION_SYSTEM_PROMPT = (
    "You are an assistant that writes documentation for code in a "
    "specific software repository, based on retrieved repository "
    "context.\n\n"
    + _ACTION_GROUNDING_RULES
    + "\n\nDescribe the module/symbol's purpose, its inputs/outputs, its "
    "responsibilities, and its dependencies, using ONLY what's shown in "
    'the retrieved context. Put the documentation text in "summary" (or '
    '"generated_content" if producing a formatted doc block/docstring). '
    "Do not invent public methods, parameters, or return values that "
    "aren't shown in the context."
    + _ACTION_JSON_CONTRACT
)

_REFACTORING_SYSTEM_PROMPT = (
    "You are an assistant that analyzes code in a specific software "
    "repository for refactoring opportunities, based on retrieved "
    "repository context.\n\n"
    + _ACTION_GROUNDING_RULES
    + "\n\nLook for complexity, duplicated logic, large functions, "
    "unclear responsibilities, coupling, naming issues, and other "
    'maintainability concerns VISIBLE in the retrieved context. Put '
    'each issue as its own entry in "findings" — a genuinely observed '
    'problem should be "fact" or "inference" depending on how directly '
    'the context supports it, while a proposed fix approach is a '
    '"suggestion". Do not claim something is a problem without pointing '
    "to what in the context supports that claim."
    + _ACTION_JSON_CONTRACT
)

_BUG_ANALYSIS_SYSTEM_PROMPT = (
    "You are an assistant that investigates a described bug/error using "
    "retrieved repository context, based on retrieved repository "
    "context.\n\n"
    + _ACTION_GROUNDING_RULES
    + "\n\nGiven the error description and the retrieved code, identify "
    "likely causes and rank them by how strongly the retrieved context "
    'supports each one. Each hypothesis is a "findings" entry — label '
    'it "inference" (not "fact") unless the retrieved context directly '
    "shows the exact bug. Include suggested debugging/verification "
    'steps in "summary". Do not claim the bug is definitively identified '
    "unless the context directly demonstrates it."
    + _ACTION_JSON_CONTRACT
)

_CHANGE_PLAN_SYSTEM_PROMPT = (
    "You are an assistant that creates an implementation plan for a "
    "requested change to a specific software repository, based on "
    "structured, deterministically computed repository architecture "
    "facts (languages, technologies, important files, entry points, and "
    "an internal dependency graph built from real import statements).\n\n"
    + _ACTION_GROUNDING_RULES
    + "\n\nClearly separate EXISTING repository facts (what's actually "
    "there now, from the provided architecture facts) from PROPOSED "
    'changes (what the plan suggests adding/modifying) — use "fact" '
    'findings for the former and "suggestion" findings for the latter. '
    'Put the step-by-step plan in "summary". The dependency graph '
    "reflects import statements only, not runtime behavior — do not "
    "claim more certainty about existing behavior than the facts "
    "support."
    + _ACTION_JSON_CONTRACT
)


class PromptBuilder:
    """Combines retrieved context and a user's question into an LLM prompt."""

    @staticmethod
    def build(context_text: str, question: str) -> tuple[str, str]:
        """
        Return (system_prompt, user_prompt) for a Phase 7 RAG question
        grounded in retrieved code chunks. The system prompt is fixed
        (see module docstring); the user prompt weaves the retrieved
        context and the actual question together so the model sees both
        in one place.
        """
        user_prompt = f"Repository context:\n\n{context_text}\n\nQuestion: {question}"
        return _SYSTEM_PROMPT, user_prompt

    @staticmethod
    def build_architecture(context_text: str, question: str) -> tuple[str, str]:
        """
        Return (system_prompt, user_prompt) for a Phase 8 repository-level
        architecture question, grounded in a RepositoryOverview's
        deterministic facts (via ArchitectureContextBuilder) rather than
        retrieved code snippets. Kept as a separate system prompt from
        `build()` above because the grounding rules differ in an
        important way: this one explicitly warns that the dependency
        graph is import-based, not a runtime call graph — a distinction
        that doesn't apply to Phase 7's per-code-chunk questions at all.
        """
        user_prompt = f"Repository facts:\n\n{context_text}\n\nQuestion: {question}"
        return _ARCHITECTURE_SYSTEM_PROMPT, user_prompt

    # -- Phase 9: action-specific prompts ------------------------------

    @staticmethod
    def _build_action_user_prompt(context_text: str, request) -> str:
        lines = [f"Repository context:\n\n{context_text}", "", f"Request: {request.request}"]
        if request.target_file:
            lines.append(f"Target file: {request.target_file}")
        if request.target_symbol:
            lines.append(f"Target symbol: {request.target_symbol}")
        if request.error_message:
            lines.append(f"Error message: {request.error_message}")
        if request.additional_context:
            lines.append(f"Additional context from the developer: {request.additional_context}")
        return "\n".join(lines)

    @staticmethod
    def build_test_generation(context_text: str, request) -> tuple[str, str]:
        return _TEST_GENERATION_SYSTEM_PROMPT, PromptBuilder._build_action_user_prompt(context_text, request)

    @staticmethod
    def build_documentation(context_text: str, request) -> tuple[str, str]:
        return _DOCUMENTATION_SYSTEM_PROMPT, PromptBuilder._build_action_user_prompt(context_text, request)

    @staticmethod
    def build_refactoring(context_text: str, request) -> tuple[str, str]:
        return _REFACTORING_SYSTEM_PROMPT, PromptBuilder._build_action_user_prompt(context_text, request)

    @staticmethod
    def build_bug_analysis(context_text: str, request) -> tuple[str, str]:
        return _BUG_ANALYSIS_SYSTEM_PROMPT, PromptBuilder._build_action_user_prompt(context_text, request)

    @staticmethod
    def build_change_plan(context_text: str, request) -> tuple[str, str]:
        return _CHANGE_PLAN_SYSTEM_PROMPT, PromptBuilder._build_action_user_prompt(context_text, request)

    # -- Phase 11: code review prompts ---------------------------------

    @staticmethod
    def build_review_diff(context_text: str, request) -> tuple[str, str]:
        user_prompt = (
            f"Review Context and Diffs:\n\n{context_text}\n\n"
            f"Perform a thorough, evidence-based code review of these changes. "
            f"Analyze correctness, security, potential regressions in dependent files, "
            f"and test coverage."
        )
        if getattr(request, "focus_categories", None):
            user_prompt += f"\nPlease pay special attention to categories: {', '.join(request.focus_categories)}."
        if getattr(request, "custom_instructions", None):
            user_prompt += f"\nAdditional developer instructions: {request.custom_instructions}."
        return _REVIEW_DIFF_SYSTEM_PROMPT, user_prompt

    @staticmethod
    def build_review_full_repo(context_text: str, request) -> tuple[str, str]:
        user_prompt = (
            f"Repository Architectural Facts:\n\n{context_text}\n\n"
            f"Perform a repository-level architectural and code health review."
        )
        if getattr(request, "focus_categories", None):
            user_prompt += f"\nPlease pay special attention to categories: {', '.join(request.focus_categories)}."
        if getattr(request, "custom_instructions", None):
            user_prompt += f"\nAdditional developer instructions: {request.custom_instructions}."
        return _REVIEW_FULL_REPO_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Phase 11: Review System Prompts
# ---------------------------------------------------------------------------

_REVIEW_DIFF_SYSTEM_PROMPT = (
    "You are a principal software engineer performing an intelligent, evidence-based code review "
    "on code changes and diffs.\n\n"
    "Your review must be grounded ONLY in the supplied repository context, diffs, reverse dependencies, "
    "and test execution results. Do not invent non-existent files, functions, or vulnerabilities.\n\n"
    "You must return ONLY a valid JSON object with exactly these keys:\n"
    '- "summary": A concise overall review assessment of the changes.\n'
    '- "findings": A list of objects. Each finding must have:\n'
    '    * "category": One of "SECURITY", "BUG", "ARCHITECTURE", "TESTING", "ERROR_HANDLING", "MAINTAINABILITY", "PERFORMANCE", "CODE_QUALITY", "DOCUMENTATION"\n'
    '    * "severity": One of "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"\n'
    '    * "title": Short descriptive title\n'
    '    * "description": Detailed explanation of the issue or insight\n'
    '    * "file_path": The specific file path affected (or null if repo-wide)\n'
    '    * "line_number": Relevant line number (or null)\n'
    '    * "start_line": Starting line number if a range (or null)\n'
    '    * "end_line": Ending line number if a range (or null)\n'
    '    * "snippet": Brief relevant code snippet or diff hunk\n'
    '    * "evidence": Concrete factual evidence from the context supporting this finding\n'
    '    * "finding_kind": One of "FACT" (directly observable), "INFERENCE" (logical deduction), or "SUGGESTION" (improvement)\n'
    '    * "recommendation": Actionable remediation advice\n'
    '    * "confidence": "HIGH", "MEDIUM", or "LOW"\n'
    '- "passed_checks": A list of strings describing verified aspects that look good or passed checks.\n'
    '- "warnings": A list of strings describing any limitations, missing test info, or areas requiring human validation.\n\n'
    "Return raw JSON only, with no markdown code fences and no conversational text."
)

_REVIEW_FULL_REPO_SYSTEM_PROMPT = (
    "You are a principal software architect performing a repository-level code and architecture audit.\n\n"
    "Your review must be grounded ONLY in the supplied repository overview, dependency facts, "
    "and sampled files. Do not invent non-existent files or relationships.\n\n"
    "You must return ONLY a valid JSON object with exactly these keys:\n"
    '- "summary": An executive summary of the repository health and architecture.\n'
    '- "findings": A list of findings following the exact same schema (category, severity, title, description, file_path, line_number, start_line, end_line, snippet, evidence, finding_kind, recommendation, confidence).\n'
    '- "passed_checks": A list of strings describing solid architectural patterns or good practices found.\n'
    '- "warnings": A list of warnings or blind spots due to sampled or unread files.\n\n'
    "Return raw JSON only, with no markdown code fences and no conversational text."
)

