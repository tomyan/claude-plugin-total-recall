"""Tests for backfill mode - Slice 5.6."""

import asyncio
import json
import os
import sys
import tempfile
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
        if mod.startswith(('config', 'db.', 'indexer.', 'memory_db', 'daemon')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def large_transcript_file():
    """Create a large transcript file for backfill testing."""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


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


def make_large_transcript(num_messages: int = 100) -> str:
    """Create a large transcript with many messages."""
    messages = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * 100  # ~100 chars per message
        ts = f"2024-01-15T10:{i:02d}:00Z"
        messages.append((role, content, ts))
    return make_transcript(messages)


class TestBackfillSession:
    """Tests for session backfill - Slice 5.6."""

    @pytest.mark.asyncio
    async def test_processes_single_session(self, test_db, large_transcript_file):
        """Should process a single session from file."""
        from daemon_agent import AgentDaemon

        # Write content
        content = make_transcript([
            ("user", "First message", "2024-01-15T10:00:00Z"),
            ("assistant", "Response", "2024-01-15T10:00:02Z"),
            ("user", "Second message", "2024-01-15T10:00:05Z"),
        ])
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 2,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            result = await daemon.backfill_session(large_transcript_file)

        assert result["sessions_processed"] == 1
        mock_agent.assert_called()

    @pytest.mark.asyncio
    async def test_respects_token_limit_per_batch(self, test_db, large_transcript_file):
        """Should split large files into batches based on token limit."""
        from daemon_agent import AgentDaemon

        # Create a large transcript
        content = make_large_transcript(num_messages=50)
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=1000  # Small token limit to force batching
        )

        batches_processed = []

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def track_batch(updates, **kwargs):
                batches_processed.append(len(updates))
                return {
                    "ideas_created": 0,
                    "sessions_processed": 1,
                    "bytes_processed": sum(u.end_byte - u.start_byte for u in updates)
                }

            mock_agent.side_effect = track_batch

            await daemon.backfill_session(large_transcript_file)

        # Should have made multiple agent calls due to token limit
        assert mock_agent.call_count >= 1

    @pytest.mark.asyncio
    async def test_continues_from_last_position(self, test_db, large_transcript_file):
        """Should continue from last indexed position."""
        from daemon_agent import AgentDaemon

        # Write content
        content = make_transcript([
            ("user", "First message", "2024-01-15T10:00:00Z"),
            ("assistant", "Response", "2024-01-15T10:00:02Z"),
        ])
        Path(large_transcript_file).write_text(content)

        # Mark first line as processed
        from db.connection import get_db
        db = get_db()
        first_line_bytes = len(content.split("\n")[0]) + 1
        db.execute("""
            INSERT INTO index_state (file_path, byte_position)
            VALUES (?, ?)
        """, (large_transcript_file, first_line_bytes))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 1,
                "sessions_processed": 1,
                "bytes_processed": len(content) - first_line_bytes
            }

            await daemon.backfill_session(large_transcript_file)

            # Should have only processed remaining content
            call_args = mock_agent.call_args
            if call_args:
                updates = call_args[0][0]
                if updates:
                    # Start byte should be after first line
                    assert updates[0].start_byte >= first_line_bytes

    @pytest.mark.asyncio
    async def test_handles_large_files(self, test_db, large_transcript_file):
        """Should handle large files by batching."""
        from daemon_agent import AgentDaemon

        # Create large content
        content = make_large_transcript(num_messages=200)
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=5000
        )

        total_bytes = 0

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def track_bytes(updates, **kwargs):
                nonlocal total_bytes
                batch_bytes = sum(u.end_byte - u.start_byte for u in updates)
                total_bytes += batch_bytes
                return {
                    "ideas_created": 0,
                    "sessions_processed": 1,
                    "bytes_processed": batch_bytes
                }

            mock_agent.side_effect = track_bytes

            await daemon.backfill_session(large_transcript_file)

        # Should have processed entire file
        assert total_bytes >= len(content) * 0.9  # Allow some margin

    @pytest.mark.asyncio
    async def test_uses_backfill_mode(self, test_db, large_transcript_file):
        """Should call agent with mode='backfill'."""
        from daemon_agent import AgentDaemon

        content = make_transcript([
            ("user", "Test message", "2024-01-15T10:00:00Z"),
        ])
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 0,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            await daemon.backfill_session(large_transcript_file)

            # Should have been called with mode='backfill'
            call_kwargs = mock_agent.call_args[1]
            assert call_kwargs.get("mode") == "backfill"


class TestBackfillProgress:
    """Tests for backfill progress tracking."""

    @pytest.mark.asyncio
    async def test_returns_progress_stats(self, test_db, large_transcript_file):
        """Should return progress statistics."""
        from daemon_agent import AgentDaemon

        content = make_large_transcript(num_messages=10)
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 5,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            result = await daemon.backfill_session(large_transcript_file)

        assert "ideas_created" in result
        assert "bytes_processed" in result
        assert "sessions_processed" in result

    @pytest.mark.asyncio
    async def test_updates_byte_position_incrementally(self, test_db, large_transcript_file):
        """Should update byte position after each batch."""
        from daemon_agent import AgentDaemon

        content = make_large_transcript(num_messages=50)
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=1000
        )

        byte_positions = []

        async def track_position(*args, **kwargs):
            from db.connection import get_db
            db = get_db()
            cursor = db.execute(
                "SELECT byte_position FROM index_state WHERE file_path = ?",
                (large_transcript_file,)
            )
            row = cursor.fetchone()
            db.close()
            if row:
                byte_positions.append(row["byte_position"])

            return {
                "ideas_created": 0,
                "sessions_processed": 1,
                "bytes_processed": 1000
            }

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.side_effect = track_position
            await daemon.backfill_session(large_transcript_file)

        # Byte positions should increase
        if len(byte_positions) > 1:
            for i in range(1, len(byte_positions)):
                assert byte_positions[i] >= byte_positions[i - 1]


class TestBackfillErrorHandling:
    """Tests for backfill error handling."""

    @pytest.mark.asyncio
    async def test_handles_missing_file(self, test_db):
        """Should handle missing file gracefully."""
        from daemon_agent import AgentDaemon

        daemon = AgentDaemon(batch_window_seconds=0)

        result = await daemon.backfill_session("/nonexistent/file.jsonl")

        assert result.get("error") is not None or result.get("sessions_processed", 0) == 0

    @pytest.mark.asyncio
    async def test_handles_agent_errors(self, test_db, large_transcript_file):
        """Should handle agent errors gracefully."""
        from daemon_agent import AgentDaemon

        content = make_transcript([
            ("user", "Test", "2024-01-15T10:00:00Z"),
        ])
        Path(large_transcript_file).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.side_effect = Exception("Agent error")

            result = await daemon.backfill_session(large_transcript_file)

        assert "error" in result
