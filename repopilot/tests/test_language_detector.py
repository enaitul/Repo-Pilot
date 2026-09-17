from repopilot.language_detector import detect_language, is_recognized_source_file


def test_detects_python():
    assert detect_language("src/main.py") == "Python"


def test_detects_typescript():
    assert detect_language("components/App.tsx") == "TypeScript"


def test_unknown_extension_returns_unknown():
    assert detect_language("README.weirdext") == "Unknown"


def test_is_recognized_source_file():
    assert is_recognized_source_file("main.go") is True
    assert is_recognized_source_file("photo.png") is False


def test_case_insensitive_extension():
    assert detect_language("Main.PY") == "Python"
