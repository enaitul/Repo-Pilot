from repopilot.chunker import Chunker
from repopilot.models import CodeSymbol, ParsedFile, ParseStatus, SymbolType


def _symbol(name, symbol_type, start, end, parent=None):
    return CodeSymbol(name=name, symbol_type=symbol_type, start_line=start, end_line=end, parent=parent)


def _parsed_file(path, symbols, language="Python", status=ParseStatus.SUCCESS):
    return ParsedFile(path=path, language=language, status=status, symbols=symbols)


# --- basic behavior --------------------------------------------------------

def test_no_symbols_returns_no_chunks():
    parsed_file = _parsed_file("empty.py", symbols=[])
    chunks = Chunker().chunk_file(parsed_file, content="")
    assert chunks == []


def test_small_function_becomes_one_chunk():
    content = "def greet():\n    print('hi')\n"
    parsed_file = _parsed_file(
        "greet.py",
        symbols=[_symbol("greet", SymbolType.FUNCTION, 1, 2)],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol_name == "greet"
    assert chunk.symbol_type == "function"
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert "def greet():" in chunk.content
    assert "print('hi')" in chunk.content
    assert chunk.chunk_index == 0
    assert chunk.chunk_count == 1


def test_chunk_id_encodes_path_and_line_range():
    content = "def greet():\n    pass\n"
    parsed_file = _parsed_file(
        "pkg/greet.py",
        symbols=[_symbol("greet", SymbolType.FUNCTION, 1, 2)],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    assert chunks[0].chunk_id == "pkg/greet.py:1-2"


def test_small_class_stays_as_one_chunk_even_with_methods_tracked():
    content = (
        "class UserService:\n"
        "    def login(self):\n"
        "        pass\n"
        "\n"
        "    def logout(self):\n"
        "        pass\n"
    )
    parsed_file = _parsed_file(
        "user_service.py",
        symbols=[
            _symbol("UserService", SymbolType.CLASS, 1, 6),
            _symbol("login", SymbolType.METHOD, 2, 3, parent="UserService"),
            _symbol("logout", SymbolType.METHOD, 5, 6, parent="UserService"),
        ],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    # The whole class fits comfortably under MAX_CHUNK_CHARS, so methods
    # must NOT also be emitted as separate chunks (that would duplicate
    # their text across two chunks).
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "UserService"
    assert chunks[0].symbol_type == "class"


def test_only_top_level_symbols_are_chunked_directly():
    # A method with a parent should never produce its own top-level chunk
    # unless reached through recursion from an oversized class.
    content = "class A:\n    def m(self):\n        pass\n"
    parsed_file = _parsed_file(
        "a.py",
        symbols=[
            _symbol("A", SymbolType.CLASS, 1, 3),
            _symbol("m", SymbolType.METHOD, 2, 3, parent="A"),
        ],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    assert len(chunks) == 1
    assert chunks[0].symbol_name == "A"


# --- oversized class: split into methods -----------------------------------

def test_oversized_class_splits_into_its_methods():
    # Build a class whose full text exceeds MAX_CHUNK_CHARS, with two
    # small methods inside it.
    padding_line = "        x = 1  # padding to inflate size\n"
    method_a_body = padding_line * 150  # comfortably over MAX_CHUNK_CHARS on its own
    content_lines = ["class Big:", "    def method_a(self):"]
    content_lines += [padding_line.rstrip("\n")] * 150
    content_lines += ["    def method_b(self):", "        return 1"]
    content = "\n".join(content_lines) + "\n"

    total_lines = len(content_lines)
    method_a_end = 2 + 150  # header + 150 padding lines
    method_b_start = method_a_end + 1
    method_b_end = total_lines

    parsed_file = _parsed_file(
        "big.py",
        symbols=[
            _symbol("Big", SymbolType.CLASS, 1, total_lines),
            _symbol("method_a", SymbolType.METHOD, 2, method_a_end, parent="Big"),
            _symbol("method_b", SymbolType.METHOD, method_b_start, method_b_end, parent="Big"),
        ],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    # Should NOT be a single giant chunk of the whole class.
    symbol_names = {c.symbol_name for c in chunks}
    assert "method_a" in symbol_names or any(
        c.parent == "Big" for c in chunks
    ), "expected the oversized class to be split into its methods"
    # Nothing produced should be the whole 300+ line class as one chunk.
    assert all(c.symbol_name != "Big" for c in chunks) or len(chunks) > 1


# --- oversized function with nothing smaller to fall back on ---------------

def test_oversized_function_falls_back_to_sliding_window():
    padding_line = "    x = 1  # padding to inflate size beyond the chunk limit"
    body_lines = [padding_line] * 400
    content = "def huge():\n" + "\n".join(body_lines) + "\n"
    total_lines = 1 + len(body_lines)

    parsed_file = _parsed_file(
        "huge.py",
        symbols=[_symbol("huge", SymbolType.FUNCTION, 1, total_lines)],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    assert len(chunks) > 1, "an oversized single function with no smaller unit must be split"
    for chunk in chunks:
        assert chunk.symbol_name == "huge"
        assert chunk.chunk_count == len(chunks)
    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every window (except possibly the last) should respect the size cap
    from repopilot.config import MAX_CHUNK_CHARS
    for chunk in chunks[:-1]:
        assert len(chunk.content) <= MAX_CHUNK_CHARS


def test_sliding_window_chunks_overlap_at_boundaries():
    padding_line = "    x = 1  # padding line to force a split across windows"
    body_lines = [f"{padding_line} {i}" for i in range(400)]
    content = "def huge():\n" + "\n".join(body_lines) + "\n"
    total_lines = 1 + len(body_lines)

    parsed_file = _parsed_file(
        "huge.py",
        symbols=[_symbol("huge", SymbolType.FUNCTION, 1, total_lines)],
    )

    chunks = Chunker().chunk_file(parsed_file, content)

    assert len(chunks) > 1
    # Consecutive windows should overlap: the next chunk's start_line
    # should be <= the previous chunk's end_line (not strictly after it).
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start_line <= earlier.end_line


# --- statuses that should never produce chunks ------------------------------

def test_unsupported_language_with_no_symbols_produces_no_chunks():
    parsed_file = _parsed_file(
        "app.rb", symbols=[], language="Ruby", status=ParseStatus.UNSUPPORTED_LANGUAGE
    )
    assert Chunker().chunk_file(parsed_file, content="puts 'hi'") == []
