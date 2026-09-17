from repopilot.architecture_context_builder import ArchitectureContextBuilder
from repopilot.models import DependencyEdge, DependencyGraph, RepositoryOverview, Technology


def _overview(**overrides):
    defaults = dict(
        repo_url="https://github.com/fake/repo",
        languages={"Python": 3},
        technologies=[Technology(name="Flask", confidence="inferred", evidence="imported in 2 files")],
        important_files=["main.py"],
        entry_points=["main.py"],
        dependency_graph=DependencyGraph(nodes=["main.py", "auth.py"], edges=[DependencyEdge("main.py", "auth.py")]),
    )
    defaults.update(overrides)
    return RepositoryOverview(**defaults)


def test_includes_repo_url():
    text = ArchitectureContextBuilder.build(_overview())
    assert "https://github.com/fake/repo" in text


def test_includes_languages():
    text = ArchitectureContextBuilder.build(_overview(languages={"Python": 5, "YAML": 2}))
    assert "Python: 5 file(s)" in text
    assert "YAML: 2 file(s)" in text


def test_includes_technology_with_confidence_label():
    text = ArchitectureContextBuilder.build(_overview())
    assert "Flask" in text
    assert "[inferred]" in text


def test_includes_entry_points_and_important_files():
    text = ArchitectureContextBuilder.build(_overview())
    assert "main.py" in text


def test_includes_dependency_edges():
    text = ArchitectureContextBuilder.build(_overview())
    assert "main.py -> auth.py" in text


def test_mentions_isolated_files():
    graph = DependencyGraph(nodes=["a.py", "b.py", "standalone.py"], edges=[DependencyEdge("a.py", "b.py")])
    text = ArchitectureContextBuilder.build(_overview(dependency_graph=graph))
    assert "standalone.py" in text
    assert "no detected internal dependency connections" in text.lower()


def test_states_dependency_graph_is_not_a_call_graph():
    text = ArchitectureContextBuilder.build(_overview())
    assert "not a runtime call graph" in text.lower()


def test_empty_overview_shows_none_detected_placeholders():
    empty = RepositoryOverview(
        repo_url="https://github.com/fake/empty",
        languages={},
        technologies=[],
        important_files=[],
        entry_points=[],
        dependency_graph=DependencyGraph(nodes=[], edges=[]),
    )

    text = ArchitectureContextBuilder.build(empty)

    assert "(none detected)" in text
    assert "(no internal dependencies resolved)" in text
