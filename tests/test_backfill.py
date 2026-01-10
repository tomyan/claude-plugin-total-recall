"""Tests for backfill functionality."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))

from backfill import backfill_transcript, get_progress


class TestBackfillTranscript:
    """Test backfilling transcripts into the memory database."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path to use tmp directory."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        monkeypatch.setattr("backfill.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock OpenAI embeddings to avoid API calls."""
        fake_embedding = [0.1] * 1536
        mock_get = MagicMock(return_value=fake_embedding)
        monkeypatch.setattr("memory_db.get_embedding", mock_get)
        return mock_get

    @pytest.fixture
    def sample_transcript(self, tmp_path):
        """Create a sample transcript file."""
        transcript = tmp_path / "session.jsonl"
        lines = [
            {"type": "user", "message": {"content": "hello"}, "timestamp": "T1"},
            {"type": "user", "message": {"content": "Help me implement a REST API with authentication"}, "timestamp": "T2"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll help you implement a REST API. For authentication, I recommend JWT tokens with refresh rotation."}]}, "timestamp": "T3"},
            {"type": "user", "message": {"content": "ok"}, "timestamp": "T4"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))
        return transcript

    def test_backfill_stores_indexable_messages(self, mock_db, mock_embeddings, sample_transcript):
        """Backfill stores indexable messages as ideas."""
        import memory_db
        memory_db.init_db()

        result = backfill_transcript(str(sample_transcript))

        assert result["messages_indexed"] == 2  # 2 indexable messages
        assert result["file_path"] == str(sample_transcript)

        # Verify ideas were stored
        stats = memory_db.get_stats()
        assert stats["total_ideas"] == 2

    def test_backfill_tracks_source_location(self, mock_db, mock_embeddings, sample_transcript):
        """Each idea tracks its source file and line."""
        import memory_db
        memory_db.init_db()

        backfill_transcript(str(sample_transcript))

        db = memory_db.get_db()
        cursor = db.execute("SELECT source_file, source_line FROM ideas ORDER BY source_line")
        rows = cursor.fetchall()
        db.close()

        assert len(rows) == 2
        assert rows[0]["source_file"] == str(sample_transcript)
        assert rows[0]["source_line"] == 2  # "Help me implement..."
        assert rows[1]["source_line"] == 3  # Assistant response

    def test_backfill_updates_index_state(self, mock_db, mock_embeddings, sample_transcript):
        """Backfill updates the index state for incremental indexing."""
        import memory_db
        memory_db.init_db()

        backfill_transcript(str(sample_transcript))

        last_line = memory_db.get_last_indexed_line(str(sample_transcript))
        assert last_line == 4  # Last line of the file

    def test_backfill_incremental(self, mock_db, mock_embeddings, tmp_path):
        """Backfill only indexes new content on subsequent runs."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "session.jsonl"
        # Initial content
        lines = [
            {"type": "user", "message": {"content": "First substantive message about database design"}, "timestamp": "T1"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        result1 = backfill_transcript(str(transcript))
        assert result1["messages_indexed"] == 1

        # Append more content
        with open(transcript, "a") as f:
            f.write("\n" + json.dumps({"type": "user", "message": {"content": "Second substantive message about API design"}, "timestamp": "T2"}))

        result2 = backfill_transcript(str(transcript))
        assert result2["messages_indexed"] == 1  # Only the new message

        # Total should be 2
        stats = memory_db.get_stats()
        assert stats["total_ideas"] == 2

    def test_backfill_empty_file(self, mock_db, mock_embeddings, tmp_path):
        """Backfill handles empty files gracefully."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")

        result = backfill_transcript(str(transcript))
        assert result["messages_indexed"] == 0

    def test_backfill_no_indexable_content(self, mock_db, mock_embeddings, tmp_path):
        """Backfill handles files with only greetings/acks."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "greetings.jsonl"
        lines = [
            {"type": "user", "message": {"content": "hi"}, "timestamp": "T1"},
            {"type": "user", "message": {"content": "ok"}, "timestamp": "T2"},
            {"type": "user", "message": {"content": "thanks"}, "timestamp": "T3"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        result = backfill_transcript(str(transcript))
        assert result["messages_indexed"] == 0


class TestGetProgress:
    """Test progress tracking for backfill."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        monkeypatch.setattr("backfill.DB_PATH", db_path)
        return db_path

    def test_get_progress_unindexed_file(self, mock_db, tmp_path):
        """Progress shows 0 for unindexed files."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "test.jsonl"
        transcript.write_text('{"type": "user", "message": {"content": "test"}, "timestamp": "T1"}')

        progress = get_progress(str(transcript))
        assert progress["last_indexed_line"] == 0
        assert progress["total_lines"] == 1

    def test_get_progress_partially_indexed(self, mock_db, tmp_path):
        """Progress shows correct state for partially indexed files."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "test.jsonl"
        lines = [
            {"type": "user", "message": {"content": "First substantive message about architecture"}, "timestamp": "T1"},
            {"type": "user", "message": {"content": "Second substantive message about testing"}, "timestamp": "T2"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        # Simulate partial indexing
        memory_db.update_index_state(str(transcript), 1)

        progress = get_progress(str(transcript))
        assert progress["last_indexed_line"] == 1
        assert progress["total_lines"] == 2
