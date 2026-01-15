"""Tests for batch processor integration - Slice 9."""

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
        if mod.startswith(('config', 'db.', 'batch_processor', 'batcher', 'context', 'protocol', 'executor', 'embeddings')):
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
def transcript_file():
    """Create a temp transcript file."""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


class TestBatchProcessor:
    """Tests for the batch processor integration."""

    def test_processes_transcript_file(self, test_db, transcript_file):
        """Should process transcript and store ideas."""
        from batch_processor import process_transcript
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "I want to implement user authentication", base_time),
            ("assistant", "Great, we should use JWT tokens for auth", base_time + timedelta(seconds=2)),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        # Mock LLM response
        mock_llm_response = {
            "items": [
                {"type": "decision", "content": "Using JWT for auth", "source_line": 2, "confidence": 0.9}
            ],
            "topic_update": {"name": "Authentication", "summary": "Implementing auth"},
        }

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            result = process_transcript(transcript_file, session="test-session")

        assert result["batches_processed"] >= 1

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM ideas")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count >= 1

    def test_updates_byte_position(self, test_db, transcript_file):
        """Should update byte position after processing."""
        from batch_processor import process_transcript
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Test message", base_time),
        ]

        content = make_transcript(messages)
        Path(transcript_file).write_text(content)

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_transcript(transcript_file, session="test-session")

        db = get_db()
        cursor = db.execute("SELECT byte_position FROM index_state WHERE file_path = ?", (transcript_file,))
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["byte_position"] == len(content)

    def test_skips_already_processed(self, test_db, transcript_file):
        """Should skip already processed content."""
        from batch_processor import process_transcript
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First message", base_time),
        ]

        content = make_transcript(messages)
        Path(transcript_file).write_text(content)

        # Mark as already processed
        db = get_db()
        db.execute("""
            INSERT INTO index_state (file_path, byte_position)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        with patch('batch_processor.call_llm') as mock_llm:
            result = process_transcript(transcript_file, session="test-session")

        # LLM should not be called
        mock_llm.assert_not_called()
        assert result["batches_processed"] == 0

    def test_handles_llm_failure(self, test_db, transcript_file):
        """Should handle LLM failures gracefully."""
        from batch_processor import process_transcript, ProcessingError

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Test message", base_time),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.side_effect = ProcessingError("LLM API error")

            with pytest.raises(ProcessingError):
                process_transcript(transcript_file, session="test-session")

    def test_creates_span_on_first_batch(self, test_db, transcript_file):
        """Should create a span for the first batch."""
        from batch_processor import process_transcript
        from db.connection import get_db

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Start of conversation", base_time),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        mock_llm_response = {
            "items": [],
            "topic_update": {"name": "New Topic", "summary": "Starting fresh"}
        }

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_transcript(transcript_file, session="test-session")

        db = get_db()
        cursor = db.execute("SELECT * FROM spans WHERE session = 'test-session'")
        span = cursor.fetchone()
        db.close()

        assert span is not None

    def test_processes_multiple_batches(self, test_db, transcript_file):
        """Should process multiple time-separated batches."""
        from batch_processor import process_transcript

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First batch message", base_time),
            ("assistant", "First response", base_time + timedelta(seconds=2)),
            # Gap > 5 seconds
            ("user", "Second batch message", base_time + timedelta(seconds=20)),
            ("assistant", "Second response", base_time + timedelta(seconds=22)),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            result = process_transcript(transcript_file, session="test-session")

        # Should have called LLM twice (once per batch)
        assert mock_llm.call_count == 2
        assert result["batches_processed"] == 2

    def test_includes_context_in_llm_call(self, test_db, transcript_file):
        """Should include hierarchy context in LLM call."""
        from batch_processor import process_transcript
        from db.connection import get_db

        # Create existing topic and span
        db = get_db()
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Existing Topic', 'existing topic', 'Some context')
        """)
        topic_id = cursor.lastrowid

        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, topic_id)
            VALUES ('test-session', 'Existing Span', 'Previous work', 1, 0, ?)
        """, (topic_id,))
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Continue with the work", base_time),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        mock_llm_response = {"items": []}

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_transcript(transcript_file, session="test-session", span_id=span_id)

        # Check that context was passed to LLM
        call_args = mock_llm.call_args[0][0]
        assert "hierarchy" in call_args
        assert call_args["hierarchy"]["topic"] is not None

    def test_executes_relations(self, test_db, transcript_file):
        """Should execute relations from LLM response."""
        from batch_processor import process_transcript
        from db.connection import get_db

        # Create existing idea to relate to
        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Old decision', 'decision', '/old.jsonl', 1)
        """)
        old_idea_id = cursor.lastrowid
        db.commit()
        db.close()

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Let's update that decision", base_time),
        ]

        Path(transcript_file).write_text(make_transcript(messages))

        mock_llm_response = {
            "items": [
                {"type": "decision", "content": "New decision", "source_line": 1, "confidence": 0.9}
            ],
            "relations": [
                {"from_line": 1, "to_idea_id": old_idea_id, "type": "supersedes"}
            ]
        }

        with patch('batch_processor.call_llm') as mock_llm:
            mock_llm.return_value = mock_llm_response
            process_transcript(transcript_file, session="test-session")

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM relations")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 1
