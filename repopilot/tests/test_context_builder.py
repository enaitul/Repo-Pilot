from repopilot.context_builder import ContextBuilder
from repopilot.models import CodeChunk, SearchResult


def _result(file_path="auth.py", symbol_name="login", parent=None, score=0.9, content="def login(): pass"):
    chunk = CodeChunk(
        chunk_id=f"{file_path}:1-2", file_path=file_path, language="Python",
        symbol_name=symbol_name, symbol_type="function", parent=parent,
        start_line=1, end_line=2, content=content,
    )
    return SearchResult(chunk=chunk, score=score)


def test_empty_results_returns_explicit_placeholder():
    text = ContextBuilder.build([])
    assert "no relevant code" in text.lower()


def test_context_contains_file_path():
    text = ContextBuilder.build([_result(file_path="backend/auth.py")])
    assert "backend/auth.py" in text


def test_context_contains_symbol_name():
    text = ContextBuilder.build([_result(symbol_name="login")])
    assert "login" in text


def test_context_contains_parent_qualified_symbol():
    text = ContextBuilder.build([_result(symbol_name="login", parent="AuthService")])
    assert "AuthService.login" in text


def test_context_contains_line_range():
    text = ContextBuilder.build([_result()])
    assert "1-2" in text


def test_context_contains_actual_code():
    text = ContextBuilder.build([_result(content="def login(username, password):\n    return True")])
    assert "def login(username, password):" in text


def test_context_contains_language():
    text = ContextBuilder.build([_result()])
    assert "Python" in text


def test_multiple_results_are_all_included_in_order():
    results = [
        _result(file_path="auth.py", symbol_name="login"),
        _result(file_path="payment.py", symbol_name="process_payment"),
    ]

    text = ContextBuilder.build(results)

    assert text.index("auth.py") < text.index("payment.py")


def test_module_level_chunk_with_no_symbol_name():
    result = _result(symbol_name=None)
    text = ContextBuilder.build([result])
    assert "module-level" in text.lower()
