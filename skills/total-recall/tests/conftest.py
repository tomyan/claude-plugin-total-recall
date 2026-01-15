"""Pytest configuration and fixtures for total-recall tests."""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Set a test database path to avoid touching the real database
os.environ.setdefault("TOTAL_RECALL_DB_PATH", "/tmp/test_total_recall.db")
