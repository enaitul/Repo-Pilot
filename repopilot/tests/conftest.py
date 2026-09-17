import sys
from unittest.mock import MagicMock

# Gracefully provide a mock for faiss if not installed in the current environment
try:
    import faiss
except ImportError:
    sys.modules["faiss"] = MagicMock()
