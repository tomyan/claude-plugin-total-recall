"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def temp_db_path(tmp_path, monkeypatch):
    """Create a temporary database path and patch all modules that use DB_PATH.

    This fixture patches DB_PATH in config module. Modules that use
    `import config; config.DB_PATH` will see the patched value.

    Modules that do `from config import DB_PATH` need separate patching.
    """
    db_path = tmp_path / "memory.db"

    # Patch config module (source of truth)
    monkeypatch.setattr("config.DB_PATH", db_path)

    # Patch modules that import DB_PATH as a local name
    monkeypatch.setattr("memory_db.DB_PATH", db_path)

    # Also patch modules that might import it directly
    try:
        monkeypatch.setattr("backfill.DB_PATH", db_path)
    except AttributeError:
        pass
    try:
        monkeypatch.setattr("indexer.DB_PATH", db_path)
    except AttributeError:
        pass

    return db_path
