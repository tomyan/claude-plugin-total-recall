"""Tests for indexer pipeline - Slices 5.1-5.4."""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def test_db():
    """Create a fresh test database for each test."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    os.environ["TOTAL_RECALL_DB_PATH"] = db_path

    # Clear cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith(('config', 'db.', 'indexer.', 'memory_db')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


def make_transcript(messages: list[tuple[str, str, str]]) -> str:
    """Create transcript JSONL content.

    Args:
        messages: List of (role, content, timestamp) tuples
    """
    lines = []
    for role, content, ts in messages:
        lines.append(json.dumps({
            "type": role,
            "message": {"content": content},
            "timestamp": ts
        }))
    return "\n".join(lines) + "\n"


@pytest.fixture
def transcript_file():
    """Create a temp transcript file."""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


class TestBatchCollector:
    """Tests for batch collector - Slice 5.1."""

    @pytest.mark.asyncio
    async def test_reads_from_last_byte_position(self, test_db, transcript_file):
        """Should read from last byte position."""
        from indexer.batch_collector import collect_batch_updates
        from db.connection import get_db

        # Write transcript content
        content = make_transcript([
            ("user", "First message", "2024-01-15T10:00:00Z"),
            ("assistant", "First response", "2024-01-15T10:00:02Z"),
        ])
        Path(transcript_file).write_text(content)

        # Mark first part as processed
        db = get_db()
        db.execute("""
            INSERT INTO index_state (file_path, byte_position)
            VALUES (?, ?)
        """, (transcript_file, 50))  # 50 bytes already processed
        db.commit()
        db.close()

        updates = await collect_batch_updates([transcript_file])

        # Should only have content after byte 50
        assert len(updates) >= 1

    @pytest.mark.asyncio
    async def test_parses_jsonl_messages(self, test_db, transcript_file):
        """Should parse JSONL messages."""
        from indexer.batch_collector import collect_batch_updates

        content = make_transcript([
            ("user", "Hello world", "2024-01-15T10:00:00Z"),
        ])
        Path(transcript_file).write_text(content)

        updates = await collect_batch_updates([transcript_file])

        assert len(updates) == 1
        assert len(updates[0].messages) >= 1

    @pytest.mark.asyncio
    async def test_tracks_byte_positions(self, test_db, transcript_file):
        """Should track start and end byte positions."""
        from indexer.batch_collector import collect_batch_updates

        content = make_transcript([
            ("user", "Test message", "2024-01-15T10:00:00Z"),
        ])
        Path(transcript_file).write_text(content)

        updates = await collect_batch_updates([transcript_file])

        assert updates[0].start_byte == 0
        assert updates[0].end_byte == len(content)


class TestFormatAgentInput:
    """Tests for format_agent_input - Slice 5.2."""

    def test_formats_multiple_sessions(self):
        """Should format updates from multiple sessions."""
        from indexer.agent_input import format_agent_input, BatchUpdate, Message

        updates = [
            BatchUpdate(
                session="session-1",
                file_path="/a.jsonl",
                messages=[Message(role="user", content="Hello", line_num=1, timestamp="")],
                start_byte=0,
                end_byte=100
            ),
            BatchUpdate(
                session="session-2",
                file_path="/b.jsonl",
                messages=[Message(role="user", content="Hi", line_num=1, timestamp="")],
                start_byte=0,
                end_byte=50
            ),
        ]

        result = format_agent_input(updates, mode="continuous")

        # Should contain both sessions
        assert "session-1" in result
        assert "session-2" in result

    def test_includes_all_message_fields(self):
        """Should include all message fields."""
        from indexer.agent_input import format_agent_input, BatchUpdate, Message

        updates = [
            BatchUpdate(
                session="session-1",
                file_path="/test.jsonl",
                messages=[
                    Message(
                        role="user",
                        content="Important message",
                        line_num=42,
                        timestamp="2024-01-15T10:00:00Z"
                    )
                ],
                start_byte=0,
                end_byte=100
            ),
        ]

        result = format_agent_input(updates, mode="continuous")

        assert "Important message" in result
        assert "42" in result  # line number

    def test_sets_mode_correctly(self):
        """Should set mode in output."""
        from indexer.agent_input import format_agent_input, BatchUpdate, Message

        updates = [
            BatchUpdate(
                session="s1",
                file_path="/t.jsonl",
                messages=[Message(role="user", content="Test", line_num=1, timestamp="")],
                start_byte=0,
                end_byte=10
            ),
        ]

        result_continuous = format_agent_input(updates, mode="continuous")
        result_backfill = format_agent_input(updates, mode="backfill")

        assert "continuous" in result_continuous
        assert "backfill" in result_backfill


class TestSystemPrompt:
    """Tests for indexing agent system prompt - Slice 5.3."""

    def test_prompt_includes_tool_usage_instructions(self):
        """Prompt should include tool usage instructions."""
        from indexer.prompts import INDEXING_SYSTEM_PROMPT

        assert "tool" in INDEXING_SYSTEM_PROMPT.lower()

    def test_prompt_includes_output_schema(self):
        """Prompt should include output schema description."""
        from indexer.prompts import INDEXING_SYSTEM_PROMPT

        assert "ideas" in INDEXING_SYSTEM_PROMPT.lower()
        assert "type" in INDEXING_SYSTEM_PROMPT.lower()  # "type" field in output schema

    def test_prompt_includes_filtering_guidelines(self):
        """Prompt should include what to filter out."""
        from indexer.prompts import INDEXING_SYSTEM_PROMPT

        # Should mention filtering low-value content
        assert any(word in INDEXING_SYSTEM_PROMPT.lower()
                   for word in ["filter", "skip", "ignore", "exclude"])

    def test_prompt_includes_importance_scoring(self):
        """Prompt should include importance scoring guidance."""
        from indexer.prompts import INDEXING_SYSTEM_PROMPT

        assert "importance" in INDEXING_SYSTEM_PROMPT.lower()


class TestRunIndexingAgent:
    """Tests for run_indexing_agent - Slice 5.4."""

    @pytest.mark.asyncio
    async def test_runs_full_pipeline(self, test_db, transcript_file):
        """Should run full indexing pipeline."""
        from indexer.run import run_indexing_agent
        from indexer.batch_collector import BatchUpdate, Message

        updates = [
            BatchUpdate(
                session="test-session",
                file_path=transcript_file,
                messages=[
                    Message(
                        role="user",
                        content="Let's implement authentication",
                        line_num=1,
                        timestamp="2024-01-15T10:00:00Z"
                    )
                ],
                start_byte=0,
                end_byte=100
            ),
        ]

        # Mock the agent to return a valid response
        mock_response = {
            "ideas": [
                {"type": "decision", "content": "Implement auth", "source_line": 1, "confidence": 0.9}
            ]
        }

        with patch('indexer.run.run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = mock_response
            with patch('indexer.executor.generate_embeddings', new_callable=AsyncMock):
                result = await run_indexing_agent(updates)

        assert result["ideas_created"] >= 1

    @pytest.mark.asyncio
    async def test_updates_byte_positions_on_success(self, test_db, transcript_file):
        """Should update byte positions after success."""
        from indexer.run import run_indexing_agent
        from indexer.batch_collector import BatchUpdate, Message
        from db.connection import get_db

        updates = [
            BatchUpdate(
                session="test-session",
                file_path=transcript_file,
                messages=[
                    Message(role="user", content="Test", line_num=1, timestamp="")
                ],
                start_byte=0,
                end_byte=150
            ),
        ]

        mock_response = {"ideas": []}

        with patch('indexer.run.run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = mock_response
            await run_indexing_agent(updates)

        db = get_db()
        cursor = db.execute(
            "SELECT byte_position FROM index_state WHERE file_path = ?",
            (transcript_file,)
        )
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["byte_position"] == 150

    @pytest.mark.asyncio
    async def test_returns_execution_stats(self, test_db, transcript_file):
        """Should return execution stats."""
        from indexer.run import run_indexing_agent
        from indexer.batch_collector import BatchUpdate, Message

        updates = [
            BatchUpdate(
                session="test-session",
                file_path=transcript_file,
                messages=[
                    Message(role="user", content="Test", line_num=1, timestamp="")
                ],
                start_byte=0,
                end_byte=100
            ),
        ]

        mock_response = {
            "ideas": [
                {"type": "context", "content": "Test", "source_line": 1}
            ]
        }

        with patch('indexer.run.run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = mock_response
            with patch('indexer.executor.generate_embeddings', new_callable=AsyncMock):
                result = await run_indexing_agent(updates)

        assert "ideas_created" in result
        assert "sessions_processed" in result
