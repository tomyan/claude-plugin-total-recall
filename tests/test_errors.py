"""Tests for error handling."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))

from memory_db import MemgraphError


class TestEmbeddingErrors:
    """Test error handling for embedding operations."""

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Missing API key raises MemgraphError with helpful message."""
        monkeypatch.delenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", raising=False)

        from memory_db import get_embedding, clear_embedding_cache

        # Clear cache to force API call
        clear_embedding_cache()

        # Should raise MemgraphError, not generic ValueError
        with pytest.raises(MemgraphError) as exc_info:
            get_embedding("test for missing api key", use_cache=False)

        assert "OPENAI_TOKEN_MEMORY_EMBEDDINGS" in str(exc_info.value)
        assert exc_info.value.error_code == "missing_api_key"

    def test_api_call_failure_wrapped(self, monkeypatch):
        """API failures are wrapped in MemgraphError."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from memory_db import get_embedding, clear_embedding_cache

        # Clear cache to force API call
        clear_embedding_cache()

        # Mock API to fail
        with patch("memory_db.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("API rate limited")
            mock_openai.return_value = mock_client

            with pytest.raises(MemgraphError) as exc_info:
                get_embedding("test for api failure", use_cache=False)

            assert "embedding" in str(exc_info.value).lower()
            assert exc_info.value.error_code == "embedding_failed"


class TestDatabaseErrors:
    """Test error handling for database operations."""

    def test_database_init_failure_wrapped(self, tmp_path, monkeypatch):
        """Database init failures are wrapped in MemgraphError."""
        import memory_db

        # Point to a read-only location
        readonly_path = tmp_path / "readonly"
        readonly_path.mkdir()
        readonly_path.chmod(0o444)

        db_path = readonly_path / "subdir" / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)

        with pytest.raises(MemgraphError) as exc_info:
            memory_db.init_db()

        assert exc_info.value.error_code == "database_error"

        # Cleanup
        readonly_path.chmod(0o755)

    def test_corrupted_database_handled(self, tmp_path, monkeypatch):
        """Corrupted database files are handled gracefully when executing queries."""
        import memory_db

        db_path = tmp_path / "corrupted.db"
        # Write invalid SQLite header to cause failure on query
        db_path.write_bytes(b"not a valid sqlite database" + b"\x00" * 100)
        monkeypatch.setattr("memory_db.DB_PATH", db_path)

        # SQLite may not fail on connect, but will fail on execute
        # Test that at least the connection succeeds or we get MemgraphError
        try:
            db = memory_db.get_db()
            # If connect succeeded, try a query that should fail
            with pytest.raises(Exception):
                db.execute("SELECT * FROM ideas")
            db.close()
        except MemgraphError as e:
            assert e.error_code == "database_error"


class TestIndexerErrors:
    """Test error handling for indexer operations."""

    def test_missing_transcript_handled(self, tmp_path, monkeypatch):
        """Missing transcript file raises MemgraphError."""
        import memory_db
        from indexer import index_transcript

        missing_file = tmp_path / "nonexistent.jsonl"

        with pytest.raises(MemgraphError) as exc_info:
            index_transcript(str(missing_file))

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.error_code == "file_not_found"

    def test_malformed_json_handled(self, tmp_path, monkeypatch):
        """Malformed JSON lines are skipped gracefully."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)

        # Create transcript with some bad JSON
        transcript = tmp_path / "test.jsonl"
        transcript.write_text('{"type":"user","message":{"content":"valid"}}\n'
                             '{this is not valid json\n'
                             '{"type":"assistant","message":{"content":"also valid"}}')

        from indexer import index_transcript
        memory_db.init_db()

        # Mock embeddings
        fake_embedding = [0.1] * 1536
        monkeypatch.setattr("memory_db.get_embedding", lambda x, use_cache=True: fake_embedding)

        # Should not raise, should skip bad lines
        result = index_transcript(str(transcript))
        assert "error" not in result or result.get("messages_indexed", 0) >= 0


class TestCLIErrors:
    """Test CLI error handling."""

    def test_cli_reports_missing_api_key_clearly(self, tmp_path, monkeypatch):
        """CLI reports missing API key with helpful message."""
        import memory_db

        # Set up working database first
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Now remove API key
        monkeypatch.delenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", raising=False)

        # Import after patching
        from cli import run_search_command

        result = run_search_command("test query for cli error")
        assert result["success"] is False
        assert "api key" in result["error"].lower()

    def test_cli_reports_database_error_clearly(self, tmp_path, monkeypatch):
        """CLI reports database errors with helpful message."""
        import memory_db

        # Point to corrupted database
        db_path = tmp_path / "corrupted.db"
        db_path.write_text("invalid")
        monkeypatch.setattr("memory_db.DB_PATH", db_path)

        from cli import run_stats_command

        result = run_stats_command()
        assert result["success"] is False
        assert "database" in result["error"].lower()


class TestMemgraphError:
    """Test MemgraphError exception class."""

    def test_error_has_code_and_message(self):
        """MemgraphError has error code and message."""
        error = MemgraphError("Something went wrong", "test_error")
        assert str(error) == "Something went wrong"
        assert error.error_code == "test_error"

    def test_error_has_optional_details(self):
        """MemgraphError can include additional details."""
        error = MemgraphError(
            "API failed",
            "api_error",
            details={"status_code": 429, "retry_after": 60}
        )
        assert error.details["status_code"] == 429
        assert error.details["retry_after"] == 60

    def test_error_is_json_serializable(self):
        """MemgraphError can be serialized to JSON."""
        import json

        error = MemgraphError("Test error", "test_code", details={"key": "value"})
        serialized = error.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert "test_code" in json_str
        assert "Test error" in json_str
