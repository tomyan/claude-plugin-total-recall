"""Tests for backfill integration - Slice 12."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def test_db():
    """Create a fresh test database for each test."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    os.environ["TOTAL_RECALL_DB_PATH"] = db_path

    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith(('config', 'db.', 'backfill', 'batch_processor', 'context', 'batcher', 'protocol', 'executor')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


def make_transcript(messages: list[tuple[str, str, datetime]]) -> str:
    """Create transcript JSONL content."""
    lines = []
    for role, content, ts in messages:
        lines.append(json.dumps({
            "type": role,
            "message": {"content": content},
            "timestamp": ts.isoformat()
        }))
    return "\n".join(lines) + "\n"


@pytest.fixture
def transcript_files():
    """Create multiple temp transcript files."""
    files = []
    for i in range(3):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)
        files.append(path)

    yield files

    for path in files:
        Path(path).unlink(missing_ok=True)


class TestBackfillEnqueue:
    """Tests for enqueuing files."""

    def test_enqueues_single_file(self, test_db, transcript_files):
        """Should enqueue a single file."""
        from backfill import enqueue_file

        Path(transcript_files[0]).write_text("test content")

        result = enqueue_file(transcript_files[0])

        assert result["status"] == "queued"
        assert result["file_path"] == transcript_files[0]

    def test_reports_already_queued(self, test_db, transcript_files):
        """Should report when file is already queued."""
        from backfill import enqueue_file

        Path(transcript_files[0]).write_text("test content")

        enqueue_file(transcript_files[0])
        result = enqueue_file(transcript_files[0])

        assert result["status"] == "already_queued"

    def test_handles_missing_file(self, test_db):
        """Should handle non-existent file."""
        from backfill import enqueue_file

        result = enqueue_file("/nonexistent/path.jsonl")

        assert "error" in result

    def test_enqueues_multiple_files(self, test_db, transcript_files):
        """Should enqueue multiple files."""
        from backfill import enqueue_file
        from db.connection import get_db

        for path in transcript_files:
            Path(path).write_text("content")
            enqueue_file(path)

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 3


class TestBackfillProgress:
    """Tests for progress tracking."""

    def test_reports_zero_progress_when_empty(self, test_db):
        """Should report zero progress when nothing indexed."""
        from backfill import get_progress

        progress = get_progress()

        assert progress["queue"]["pending_files"] == 0
        assert progress["indexed"]["files"] == 0

    def test_reports_queued_files(self, test_db, transcript_files):
        """Should report queued file count."""
        from backfill import enqueue_file, get_progress

        for path in transcript_files:
            Path(path).write_text("content")
            enqueue_file(path)

        progress = get_progress()

        assert progress["queue"]["pending_files"] == 3

    def test_reports_indexed_bytes(self, test_db, transcript_files):
        """Should report indexed byte count."""
        from backfill import get_progress
        from db.connection import get_db

        # Create content
        content = "test content here"
        Path(transcript_files[0]).write_text(content)

        # Mark as partially indexed
        db = get_db()
        db.execute("""
            INSERT INTO index_state (file_path, byte_position)
            VALUES (?, ?)
        """, (transcript_files[0], len(content)))
        db.commit()
        db.close()

        progress = get_progress()

        assert progress["indexed"]["bytes"] == len(content)
        assert progress["indexed"]["files"] == 1


class TestBackfillProcessing:
    """Tests for batch processing integration."""

    def test_processes_queued_files(self, test_db, transcript_files):
        """Should process files from the queue."""
        from backfill import enqueue_file, process_queue
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Test message", base_time),
        ]

        for path in transcript_files:
            Path(path).write_text(make_transcript(messages))
            enqueue_file(path)

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            result = process_queue(limit=3)

        assert result["files_processed"] == 3

        # Queue should be empty
        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 0

    def test_limits_batch_size(self, test_db, transcript_files):
        """Should respect batch size limit."""
        from backfill import enqueue_file, process_queue
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [("user", "Test", base_time)]

        for path in transcript_files:
            Path(path).write_text(make_transcript(messages))
            enqueue_file(path)

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            result = process_queue(limit=1)

        assert result["files_processed"] == 1

        # 2 files should remain in queue
        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 2

    def test_handles_processing_errors(self, test_db, transcript_files):
        """Should handle errors gracefully."""
        from backfill import enqueue_file, process_queue
        from batch_processor import ProcessingError
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [("user", "Test", base_time)]

        Path(transcript_files[0]).write_text(make_transcript(messages))
        enqueue_file(transcript_files[0])

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.side_effect = ProcessingError("API error")
            result = process_queue(limit=1)

        assert result["files_processed"] == 0
        assert result["errors"] == 1

    def test_deduplicates_ideas(self, test_db, transcript_files):
        """Should not create duplicate ideas for same source."""
        from backfill import enqueue_file, process_queue
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Test message", base_time),
        ]

        Path(transcript_files[0]).write_text(make_transcript(messages))
        enqueue_file(transcript_files[0])

        mock_llm_response = {
            "items": [
                {"type": "decision", "content": "Test decision", "source_line": 1, "confidence": 0.9}
            ]
        }

        # Process twice
        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_queue(limit=1)

        # Re-enqueue and process again
        enqueue_file(transcript_files[0])

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_queue(limit=1)

        # Should only have one idea (deduplicated by source_file, source_line)
        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM ideas")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 1

    def test_handles_large_transcript(self, test_db, transcript_files):
        """Should handle large transcripts via batching."""
        from backfill import enqueue_file, process_queue

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        # Create 50 messages spanning 10+ batches
        messages = []
        for i in range(50):
            ts = base_time + timedelta(seconds=i * 10)  # 10 second gaps
            messages.append(("user", f"Message {i}", ts))
            messages.append(("assistant", f"Response {i}", ts + timedelta(seconds=2)))

        Path(transcript_files[0]).write_text(make_transcript(messages))
        enqueue_file(transcript_files[0])

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            result = process_queue(limit=1)

        # Should have processed multiple batches
        assert mock_llm.call_count >= 10
        assert result["files_processed"] == 1

    def test_generates_session_from_path(self, test_db, transcript_files):
        """Should generate session ID from file path."""
        from backfill import enqueue_file, process_queue
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [("user", "Test", base_time)]

        Path(transcript_files[0]).write_text(make_transcript(messages))
        enqueue_file(transcript_files[0])

        mock_llm_response = {"items": [], "topic_update": {"name": "Test", "summary": "Test"}}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_queue(limit=1)

        # Check span was created with session from file path
        db = get_db()
        cursor = db.execute("SELECT session FROM spans")
        span = cursor.fetchone()
        db.close()

        assert span is not None
        # Session should be derived from file path (e.g., filename without extension)
        assert len(span["session"]) > 0
