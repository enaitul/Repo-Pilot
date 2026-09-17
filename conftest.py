import sys
from unittest.mock import MagicMock

# Ensure faiss is mocked if not installed so that all test modules can import repopilot
try:
    import faiss
except ImportError:
    sys.modules["faiss"] = MagicMock()
