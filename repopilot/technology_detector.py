"""
Detects technologies/frameworks in use, explicitly split into two
confidence tiers so a strong signal is never blurred with a weaker one:

  DETECTED - a real manifest/config file was found (requirements.txt,
             package.json, Dockerfile, ...) - direct, unambiguous evidence.
  INFERRED - a framework name showed up repeatedly in import statements -
             real evidence, but weaker (could be a same-named local
             module, or a manifest that's missing/out of date).

Deliberately does NOT attempt to detect every technology that exists —
only the small set defined in config.py's TECHNOLOGY_MANIFEST_FILES and
TECHNOLOGY_IMPORT_HINTS. Extending detection later means adding entries
to those config dicts, not touching this module's logic.
"""

from __future__ import annotations

from repopilot.config import TECHNOLOGY_IMPORT_HINTS, TECHNOLOGY_MANIFEST_FILES
from repopilot.models import RepositoryModel, Technology


class TechnologyDetector:
    """Detects technologies from manifest files (high confidence) and imports (lower confidence)."""

    def detect(self, repo_model: RepositoryModel) -> list[Technology]:
        technologies: dict[str, Technology] = {}

        filenames = {self._basename(f.path) for f in repo_model.files}
        for filename, tech_name in TECHNOLOGY_MANIFEST_FILES.items():
            if filename in filenames:
                technologies[tech_name] = Technology(
                    name=tech_name, confidence="detected", evidence=f"found {filename}"
                )

        import_counts: dict[str, int] = {}
        for file_meta in repo_model.files:
            for import_name in file_meta.imports:
                lowered = import_name.lower()
                for hint, tech_name in TECHNOLOGY_IMPORT_HINTS.items():
                    if hint in lowered:
                        import_counts[tech_name] = import_counts.get(tech_name, 0) + 1

        for tech_name, count in import_counts.items():
            if tech_name not in technologies:
                # Never let a lower-confidence "inferred" finding
                # overwrite a "detected" one already found above.
                plural = "file" if count == 1 else "files"
                technologies[tech_name] = Technology(
                    name=tech_name, confidence="inferred",
                    evidence=f"imported in {count} {plural}",
                )

        return sorted(technologies.values(), key=lambda t: (t.confidence, t.name))

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rsplit("/", 1)[-1]
