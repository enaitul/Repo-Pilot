from repopilot.important_file_detector import ImportantFileDetector
from repopilot.models import FileMetadata, RepositoryModel


def _repo(files):
    return RepositoryModel(repo_url="https://github.com/fake/repo", local_path="/tmp/fake", files=files)


def _file(path):
    return FileMetadata(path=path, language="Python", size_bytes=100, imports=[])


def test_known_entry_point_filenames_are_detected():
    repo = _repo([_file("main.py"), _file("utils.py")])

    entry_points = ImportantFileDetector().detect_entry_points(repo)

    assert entry_points == ["main.py"]


def test_nested_entry_point_is_still_matched_by_basename():
    repo = _repo([_file("backend/app.py")])

    entry_points = ImportantFileDetector().detect_entry_points(repo)

    assert entry_points == ["backend/app.py"]


def test_config_files_are_important_but_not_entry_points():
    repo = _repo([_file("package.json")])

    important = ImportantFileDetector().detect_important_files(repo)
    entry_points = ImportantFileDetector().detect_entry_points(repo)

    assert important == ["package.json"]
    assert entry_points == []


def test_unrecognized_filenames_are_not_flagged():
    repo = _repo([_file("utils.py"), _file("helpers.py")])

    assert ImportantFileDetector().detect_important_files(repo) == []
    assert ImportantFileDetector().detect_entry_points(repo) == []


def test_entry_points_are_a_subset_of_important_files():
    repo = _repo([_file("main.py"), _file("package.json"), _file("utils.py")])

    important = ImportantFileDetector().detect_important_files(repo)
    entry_points = ImportantFileDetector().detect_entry_points(repo)

    assert set(entry_points).issubset(set(important))


def test_empty_repository_returns_empty_lists():
    repo = _repo([])
    assert ImportantFileDetector().detect_important_files(repo) == []
    assert ImportantFileDetector().detect_entry_points(repo) == []
