"""
Phase 12: GitHub API Service.

Interacts with the GitHub REST API using Python's standard library (urllib.request)
to verify credentials, post Pull Requests, and format PR descriptions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from repopilot.config import GITHUB_API_BASE_URL, GITHUB_REQUEST_TIMEOUT_SECONDS
from repopilot.exceptions import AuthenticationError, PullRequestError
from repopilot.models import PullRequestDetails, ReviewReport


class GitHubAPIService:
    """Interacts with GitHub REST API endpoint using stdlib urllib."""

    def __init__(self, api_base_url: str = GITHUB_API_BASE_URL, timeout_seconds: int = GITHUB_REQUEST_TIMEOUT_SECONDS):
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def _make_request(
        self, endpoint: str, method: str = "GET", token: Optional[str] = None, payload: Optional[dict] = None
    ) -> dict:
        """Helper to send HTTP request to GitHub API."""
        url = f"{self._api_base_url}{endpoint}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RepoPilot-Agent",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        data_bytes = None
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                resp_bytes = resp.read()
                if resp_bytes:
                    return json.loads(resp_bytes.decode("utf-8"))
                return {}
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="replace")
            if exc.code in (401, 403):
                raise AuthenticationError(f"GitHub authentication failed (HTTP {exc.code}): {err_msg}") from exc
            raise PullRequestError(f"GitHub API error (HTTP {exc.code}) for {endpoint}: {err_msg}") from exc
        except urllib.error.URLError as exc:
            raise PullRequestError(f"Network error communicating with GitHub API: {exc.reason}") from exc
        except Exception as exc:
            raise PullRequestError(f"Unexpected error during GitHub API request: {str(exc)}") from exc

    def verify_token(self, token: str) -> dict:
        """Verifies that the provided GitHub token is valid."""
        if not token or not token.strip():
            raise AuthenticationError("A non-empty GitHub token is required for authentication.")
        return self._make_request("/user", method="GET", token=token)

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        token: Optional[str] = None,
    ) -> PullRequestDetails:
        """Sends POST request to GitHub API to open a Pull Request."""
        if not token:
            raise AuthenticationError("GitHub token is required to create a Pull Request.")

        endpoint = f"/repos/{owner}/{repo}/pulls"
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
        }

        resp = self._make_request(endpoint, method="POST", token=token, payload=payload)

        html_url = resp.get("html_url", f"https://github.com/{owner}/{repo}/pull/{resp.get('number', '')}")
        pr_number = resp.get("number")

        return PullRequestDetails(
            title=title,
            body=body,
            head_branch=head_branch,
            base_branch=base_branch,
            html_url=html_url,
            number=pr_number,
        )

    def format_pr_description(
        self,
        change_plan_summary: str,
        test_status: str,
        test_details: str,
        review_report: Optional[ReviewReport] = None,
    ) -> str:
        """Formats a structured Pull Request markdown body."""
        lines = [
            "## Summary of Changes (RepoPilot AI)",
            change_plan_summary if change_plan_summary else "Automated changes applied by RepoPilot.",
            "",
            "## Automated Test Execution (Phase 10)",
            f"- **Status**: `{test_status.upper()}`",
        ]

        if test_details:
            lines.extend(["```", test_details[:1000], "```", ""])

        if review_report:
            lines.extend([
                "## Intelligent Code Review Summary (Phase 11)",
                f"- **Review Mode**: `{review_report.mode}`",
                f"- **Summary**: {review_report.summary}",
                "- **Findings Count by Severity**:",
            ])
            for sev, cnt in review_report.severity_counts.items():
                if cnt > 0:
                    lines.append(f"  - `{sev}`: {cnt}")

            if review_report.findings:
                lines.extend(["", "### Key Findings:"])
                for f in review_report.findings[:5]:  # Top 5 findings
                    lines.append(f"- **[{f.severity}] {f.category}**: {f.title} ({f.file_path or 'General'})")

        lines.extend([
            "",
            "---",
            "*Generated automatically by RepoPilot Phase 12 Developer Workflow.*",
        ])

        return "\n".join(lines)
