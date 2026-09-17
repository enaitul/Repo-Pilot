"""
Phase 12: GitHub Workflow Service.

Orchestrates Phase 10 modifications, Phase 11 code review findings, local Git branch
management, human approval enforcement, and remote GitHub Pull Request creation.
"""

from __future__ import annotations

import os
from typing import Optional

from repopilot.config import DEFAULT_BASE_BRANCH, DEFAULT_BRANCH_PREFIX, DEFAULT_REMOTE_NAME
from repopilot.exceptions import GitHubWorkflowError
from repopilot.git_service import GitService
from repopilot.github_api_service import GitHubAPIService
from repopilot.models import (
    AgentResult,
    GitBranchDetails,
    GitHubWorkflowRequest,
    GitHubWorkflowResult,
    PullRequestDetails,
    RepositoryModel,
    ReviewReport,
)


class GitHubWorkflowService:
    """Orchestrates Phase 12 GitHub developer workflow integration."""

    def __init__(self, git_service: Optional[GitService] = None, api_service: Optional[GitHubAPIService] = None):
        self._git_service = git_service or GitService()
        self._api_service = api_service or GitHubAPIService()

    def execute(
        self,
        request: GitHubWorkflowRequest,
        repo_model: RepositoryModel,
        agent_result: Optional[AgentResult] = None,
        review_report: Optional[ReviewReport] = None,
    ) -> GitHubWorkflowResult:
        """Executes the Phase 12 workflow."""
        if request is None:
            raise GitHubWorkflowError("A non-empty GitHubWorkflowRequest is required.")
        if repo_model is None or not repo_model.local_path:
            raise GitHubWorkflowError("A RepositoryModel with a valid local path is required.")

        warnings: list[str] = []
        errors: list[str] = []

        # Determine token from request or environment
        token = request.token or os.getenv("GITHUB_TOKEN")

        # Step 1: Branch Creation & Staging
        branch_name = request.branch_name or f"{DEFAULT_BRANCH_PREFIX}update"
        base_branch = request.base_branch or DEFAULT_BASE_BRANCH

        # Local branch management
        try:
            created_branch = self._git_service.create_feature_branch(
                repo_model.local_path, branch_name=branch_name, base_branch=base_branch
            )
        except Exception as exc:
            return GitHubWorkflowResult(
                status="failed",
                approved=request.approved,
                warnings=warnings,
                errors=[f"Failed to create local feature branch: {str(exc)}"],
            )

        # File staging & committing
        modified_files: list[str] = []
        if agent_result and agent_result.applied_changes:
            modified_files = list({c.file_path for c in agent_result.applied_changes})

        commit_msg = (
            f"repo-pilot: {request.pr_title}"
            if request.pr_title
            else f"repo-pilot: apply changes for {request.repo_name}"
        )

        try:
            branch_details = self._git_service.stage_and_commit(
                repo_path=repo_model.local_path,
                file_paths=modified_files,
                commit_message=commit_msg,
                branch_name=created_branch,
            )
        except Exception as exc:
            return GitHubWorkflowResult(
                status="failed",
                approved=request.approved,
                warnings=warnings,
                errors=[f"Failed to stage and commit workspace changes: {str(exc)}"],
            )

        # Step 2: Enforce Human Approval Gate
        if not request.approved:
            warnings.append(
                "Human approval was not granted (`approved=False`). Changes remain saved in local feature branch."
            )
            return GitHubWorkflowResult(
                status="missing_approval",
                approved=False,
                branch_details=branch_details,
                pr_details=None,
                warnings=warnings,
                errors=errors,
            )

        # If no token available, return branch_only state
        if not token:
            warnings.append("No GitHub authentication token supplied. Remote push and PR creation skipped.")
            return GitHubWorkflowResult(
                status="branch_only",
                approved=True,
                branch_details=branch_details,
                pr_details=None,
                warnings=warnings,
                errors=errors,
            )

        # Step 3: Remote Push & PR Creation (Approved + Token present)
        try:
            self._git_service.push_branch(
                repo_path=repo_model.local_path,
                branch_name=created_branch,
                remote_name=DEFAULT_REMOTE_NAME,
                token=token,
            )
        except Exception as exc:
            warnings.append(f"Remote push failed: {str(exc)}")

        # Format PR description
        change_plan_summary = ""
        if agent_result and agent_result.change_plan:
            change_plan_summary = agent_result.change_plan.summary

        test_status = "not_available"
        test_details = ""
        if agent_result and agent_result.test_result:
            test_status = agent_result.test_result.status
            test_details = agent_result.test_result.stdout or agent_result.test_result.stderr

        pr_title = request.pr_title or f"RepoPilot Edits for {request.repo_name}"
        pr_body = self._api_service.format_pr_description(
            change_plan_summary=change_plan_summary,
            test_status=test_status,
            test_details=test_details,
            review_report=review_report,
        )

        try:
            pr_details = self._api_service.create_pull_request(
                owner=request.repo_owner,
                repo=request.repo_name,
                title=pr_title,
                body=pr_body,
                head_branch=created_branch,
                base_branch=base_branch,
                token=token,
            )
            return GitHubWorkflowResult(
                status="approved_and_created",
                approved=True,
                branch_details=branch_details,
                pr_details=pr_details,
                warnings=warnings,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"Failed to create GitHub Pull Request: {str(exc)}")
            return GitHubWorkflowResult(
                status="failed",
                approved=True,
                branch_details=branch_details,
                pr_details=None,
                warnings=warnings,
                errors=errors,
            )
