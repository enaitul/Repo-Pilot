from repopilot.dependency_graph_builder import DependencyGraphBuilder
from repopilot.models import FileMetadata, RepositoryModel


def _repo(files):
    return RepositoryModel(repo_url="https://github.com/fake/repo", local_path="/tmp/fake", files=files)


def _file(path, imports=None, language="Python"):
    return FileMetadata(path=path, language=language, size_bytes=100, imports=imports or [])


# --- basic resolution ----------------------------------------------------

def test_internal_import_becomes_an_edge():
    repo = _repo([
        _file("api.py", imports=["auth"]),
        _file("auth.py", imports=[]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert any(e.source == "api.py" and e.target == "auth.py" for e in graph.edges)


def test_external_import_does_not_become_an_edge():
    repo = _repo([
        _file("main.py", imports=["os", "sys", "requests"]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert graph.edges == []


def test_dotted_import_resolves_to_nested_file():
    repo = _repo([
        _file("tests/test_simple.py", imports=["sample.simple"]),
        _file("src/sample/simple.py", imports=[]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert any(e.target == "src/sample/simple.py" for e in graph.edges)


def test_dotted_import_resolves_to_package_init():
    repo = _repo([
        _file("main.py", imports=["utils"]),
        _file("utils/__init__.py", imports=[]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert any(e.target == "utils/__init__.py" for e in graph.edges)


# --- nodes, isolation, cycles --------------------------------------------

def test_all_files_appear_as_nodes_even_with_no_edges():
    repo = _repo([_file("a.py"), _file("b.py"), _file("c.py")])

    graph = DependencyGraphBuilder().build(repo)

    assert set(graph.nodes) == {"a.py", "b.py", "c.py"}


def test_isolated_files_are_detected():
    repo = _repo([
        _file("api.py", imports=["auth"]),
        _file("auth.py"),
        _file("standalone_script.py"),  # imports nothing, nothing imports it
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert "standalone_script.py" in graph.isolated_nodes
    assert "api.py" not in graph.isolated_nodes
    assert "auth.py" not in graph.isolated_nodes


def test_dependency_cycle_is_represented_not_broken():
    repo = _repo([
        _file("a.py", imports=["b"]),
        _file("b.py", imports=["a"]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert any(e.source == "a.py" and e.target == "b.py" for e in graph.edges)
    assert any(e.source == "b.py" and e.target == "a.py" for e in graph.edges)
    assert graph.isolated_nodes == []


def test_windows_backslash_paths_are_normalized():
    repo = _repo([
        _file("src\\sample\\simple.py"),
        _file("tests\\test_simple.py", imports=["sample.simple"]),
    ])

    graph = DependencyGraphBuilder().build(repo)

    assert "src/sample/simple.py" in graph.nodes
    assert any(e.target == "src/sample/simple.py" for e in graph.edges)


def test_empty_repository_produces_empty_graph():
    graph = DependencyGraphBuilder().build(_repo([]))

    assert graph.nodes == []
    assert graph.edges == []
    assert graph.isolated_nodes == []


def test_duplicate_imports_do_not_produce_duplicate_edges():
    repo = _repo([
        _file("a.py", imports=["b", "b", "b"]),
        _file("b.py"),
    ])

    graph = DependencyGraphBuilder().build(repo)

    matching = [e for e in graph.edges if e.source == "a.py" and e.target == "b.py"]
    assert len(matching) == 1
