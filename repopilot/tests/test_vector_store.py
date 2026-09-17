import pytest

from repopilot.exceptions import VectorStoreError
from repopilot.models import CodeChunk, EmbeddedChunk
from repopilot.vector_store import VectorStore


def _chunk(chunk_id, symbol_name):
    return CodeChunk(
        chunk_id=chunk_id,
        file_path="a.py",
        language="Python",
        symbol_name=symbol_name,
        symbol_type="function",
        parent=None,
        start_line=1,
        end_line=2,
        content=f"def {symbol_name}(): pass",
    )


def _embedded(chunk_id, symbol_name, vector, model_name="fake-model"):
    return EmbeddedChunk(
        chunk=_chunk(chunk_id, symbol_name),
        vector=vector,
        model_name=model_name,
        dimensions=len(vector),
    )


# --- adding vectors ----------------------------------------------------

def test_adding_to_empty_list_is_a_noop():
    store = VectorStore()
    store.add([])
    assert store.size == 0


def test_add_increases_size():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "add_one", [1.0, 0.0, 0.0])])
    assert store.size == 1


def test_add_multiple_batches_accumulates():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])
    store.add([_embedded("a.py:3-4", "b", [0.0, 1.0, 0.0])])
    assert store.size == 2


def test_dimension_mismatch_raises():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])

    with pytest.raises(VectorStoreError):
        store.add([_embedded("a.py:3-4", "b", [1.0, 0.0])])  # wrong dimension


# --- searching, with known deterministic vectors ------------------------

def test_exact_match_ranks_first():
    store = VectorStore()
    store.add([
        _embedded("a.py:1-2", "login", [1.0, 0.0, 0.0]),
        _embedded("a.py:3-4", "logout", [0.0, 1.0, 0.0]),
        _embedded("a.py:5-6", "unrelated", [0.0, 0.0, 1.0]),
    ])

    results = store.search([1.0, 0.0, 0.0], top_k=3)

    assert results[0].chunk.symbol_name == "login"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_ranking_orders_by_similarity():
    store = VectorStore()
    store.add([
        _embedded("a.py:1-2", "close", [1.0, 0.1, 0.0]),
        _embedded("a.py:3-4", "far", [0.0, 0.0, 1.0]),
    ])

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert results[0].chunk.symbol_name == "close"
    assert results[1].chunk.symbol_name == "far"
    assert results[0].score > results[1].score


def test_top_k_limits_number_of_results():
    store = VectorStore()
    store.add([
        _embedded(f"a.py:{i}", f"fn{i}", v)
        for i, v in enumerate([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]])
    ])

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_top_k_larger_than_store_size_returns_everything():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "only_one", [1.0, 0.0, 0.0])])

    results = store.search([1.0, 0.0, 0.0], top_k=50)

    assert len(results) == 1


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_search_with_invalid_top_k_raises():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])

    with pytest.raises(VectorStoreError):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_search_with_wrong_dimension_query_raises():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])

    with pytest.raises(VectorStoreError):
        store.search([1.0, 0.0], top_k=1)  # 2 dims, store has 3


# --- metadata correctness -------------------------------------------------

def test_search_result_contains_full_original_chunk():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "login", [1.0, 0.0, 0.0])])

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert results[0].chunk.file_path == "a.py"
    assert results[0].chunk.content == "def login(): pass"


# --- rebuild / clear ----------------------------------------------------

def test_rebuild_replaces_previous_contents():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "old", [1.0, 0.0, 0.0])])

    store.rebuild([_embedded("a.py:3-4", "new", [0.0, 1.0, 0.0])])

    assert store.size == 1
    results = store.search([0.0, 1.0, 0.0], top_k=1)
    assert results[0].chunk.symbol_name == "new"


def test_clear_empties_the_store():
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])

    store.clear()

    assert store.size == 0
    assert store.search([1.0, 0.0, 0.0], top_k=1) == []


# --- persistence ----------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    store = VectorStore()
    store.add([
        _embedded("a.py:1-2", "login", [1.0, 0.0, 0.0]),
        _embedded("a.py:3-4", "logout", [0.0, 1.0, 0.0]),
    ])

    store.save(str(tmp_path))
    loaded = VectorStore.load(str(tmp_path))

    assert loaded.size == 2
    results = loaded.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk.symbol_name == "login"


def test_save_creates_both_files(tmp_path):
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])

    store.save(str(tmp_path))

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_save_empty_store_raises():
    store = VectorStore()
    with pytest.raises(VectorStoreError):
        store.save("/tmp/should_not_be_created")


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(VectorStoreError):
        VectorStore.load(str(tmp_path / "does_not_exist"))


def test_load_corrupted_metadata_raises(tmp_path):
    store = VectorStore()
    store.add([_embedded("a.py:1-2", "a", [1.0, 0.0, 0.0])])
    store.save(str(tmp_path))

    (tmp_path / "metadata.json").write_text("not valid json{{{")

    with pytest.raises(VectorStoreError):
        VectorStore.load(str(tmp_path))
