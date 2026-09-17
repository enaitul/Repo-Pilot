from repopilot.import_extractor import extract_imports


def test_python_imports():
    content = """
import os
import sys as system
from typing import List, Optional
from . import sibling
"""
    imports = extract_imports(content, "Python")
    assert "os" in imports
    assert "sys" in imports
    assert "typing" in imports


def test_javascript_imports():
    content = """
import React from 'react';
import { useState } from "react";
const fs = require('fs');
"""
    imports = extract_imports(content, "JavaScript")
    assert "react" in imports
    assert "fs" in imports


def test_java_imports():
    content = """
import java.util.List;
import static java.lang.Math.max;
public class Foo {}
"""
    imports = extract_imports(content, "Java")
    assert "java.util.List" in imports
    assert "java.lang.Math.max" in imports


def test_unknown_language_returns_empty_list():
    assert extract_imports("import os", "Unknown") == []


def test_no_imports_returns_empty_list():
    assert extract_imports("print('hello world')", "Python") == []


def test_deduplicates_imports():
    content = "import os\nimport os\nimport os"
    assert extract_imports(content, "Python") == ["os"]


def test_never_raises_on_malformed_content():
    # Should not raise even on odd/garbage input
    result = extract_imports("\x00\x01 not real code ][{", "Python")
    assert isinstance(result, list)
