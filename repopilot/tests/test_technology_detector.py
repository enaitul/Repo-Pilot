from repopilot.models import FileMetadata, RepositoryModel
from repopilot.technology_detector import TechnologyDetector


def _repo(files):
    return RepositoryModel(repo_url="https://github.com/fake/repo", local_path="/tmp/fake", files=files)


def _file(path, imports=None):
    return FileMetadata(path=path, language="Python", size_bytes=100, imports=imports or [])


def test_manifest_file_is_detected_with_high_confidence():
    repo = _repo([_file("requirements.txt")])

    technologies = TechnologyDetector().detect(repo)

    match = next(t for t in technologies if t.name == "Python (pip)")
    assert match.confidence == "detected"
    assert "requirements.txt" in match.evidence


def test_import_hint_is_inferred_with_lower_confidence():
    repo = _repo([_file("app.py", imports=["flask"])])

    technologies = TechnologyDetector().detect(repo)

    match = next(t for t in technologies if t.name == "Flask")
    assert match.confidence == "inferred"


def test_detected_is_never_overwritten_by_inferred():
    # A manifest exists AND the framework is imported — detected wins,
    # not blurred into two separate entries for the same technology.
    repo = _repo([
        _file("package.json"),
        _file("index.js", imports=["react"]),
    ])

    technologies = TechnologyDetector().detect(repo)

    names = [t.name for t in technologies]
    assert names.count("Node.js") == 1


def test_import_count_is_recorded_in_evidence():
    repo = _repo([
        _file("a.py", imports=["flask"]),
        _file("b.py", imports=["flask"]),
        _file("c.py", imports=["flask"]),
    ])

    technologies = TechnologyDetector().detect(repo)

    match = next(t for t in technologies if t.name == "Flask")
    assert "3" in match.evidence


def test_no_technologies_detected_returns_empty_list():
    repo = _repo([_file("notes.md")])
    assert TechnologyDetector().detect(repo) == []


def test_unrelated_imports_are_ignored():
    repo = _repo([_file("main.py", imports=["os", "sys", "json"])])
    assert TechnologyDetector().detect(repo) == []
