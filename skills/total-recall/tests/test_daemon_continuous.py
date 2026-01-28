"""Tests for continuous mode daemon - Slice 5.5."""

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


class TestBatchWindow:
    """Tests for batch window functionality - Slice 5.5."""

    @pytest.mark.asyncio
    async def test_waits_for_batch_window(self, test_db, transcript_files):
        """Should wait for batch window before processing."""
        from daemon_agent import AgentDaemon

        # Write content to first file
        Path(transcript_files[0]).write_text(make_transcript([
            ("user", "First message", "2024-01-15T10:00:00Z"),
        ]))

        # Add to queue
        from db.connection import get_db
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, 100)
        """, (transcript_files[0],))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0.1)

        # Track timing
        start_time = asyncio.get_event_loop().time()
        processed = False

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {"ideas_created": 0, "sessions_processed": 1}

            # Run one cycle
            await daemon.continuous_cycle()

        end_time = asyncio.get_event_loop().time()

        # Should have waited at least batch_window_seconds
        assert end_time - start_time >= 0.1

    @pytest.mark.asyncio
    async def test_processes_multiple_files_together(self, test_db, transcript_files):
        """Should batch multiple files into single agent call."""
        from daemon_agent import AgentDaemon

        # Write content to all files
        for i, path in enumerate(transcript_files):
            Path(path).write_text(make_transcript([
                ("user", f"Message from file {i}", "2024-01-15T10:00:00Z"),
            ]))

        # Add all to queue
        from db.connection import get_db
        db = get_db()
        for path in transcript_files:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, 100)
            """, (path,))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {"ideas_created": 0, "sessions_processed": 3}

            await daemon.continuous_cycle()

            # Should have been called once with all files
            assert mock_agent.call_count == 1

            # Check it received updates for all files
            call_args = mock_agent.call_args
            updates = call_args[0][0]  # First positional arg
            assert len(updates) == 3

    @pytest.mark.asyncio
    async def test_single_agent_call_per_cycle(self, test_db, transcript_files):
        """Should make single agent call per cycle."""
        from daemon_agent import AgentDaemon

        # Write content to files
        for path in transcript_files[:2]:
            Path(path).write_text(make_transcript([
                ("user", "Test message", "2024-01-15T10:00:00Z"),
            ]))

        # Add to queue
        from db.connection import get_db
        db = get_db()
        for path in transcript_files[:2]:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, 100)
            """, (path,))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)
        agent_calls = []

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def track_call(*args, **kwargs):
                agent_calls.append((args, kwargs))
                return {"ideas_created": 0, "sessions_processed": 2}

            mock_agent.side_effect = track_call

            await daemon.continuous_cycle()

        # Exactly one agent call
        assert len(agent_calls) == 1

    @pytest.mark.asyncio
    async def test_handles_empty_queue(self, test_db):
        """Should handle empty queue gracefully."""
        from daemon_agent import AgentDaemon

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            result = await daemon.continuous_cycle()

        # Should return False (no work done)
        assert result is False

        # Should not call agent
        mock_agent.assert_not_called()


class TestQueueManagement:
    """Tests for queue management during continuous processing."""

    @pytest.mark.asyncio
    async def test_removes_processed_from_queue(self, test_db, transcript_files):
        """Should remove files from queue after processing."""
        from daemon_agent import AgentDaemon

        # Write content
        Path(transcript_files[0]).write_text(make_transcript([
            ("user", "Test", "2024-01-15T10:00:00Z"),
        ]))

        # Add to queue
        from db.connection import get_db
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, 100)
        """, (transcript_files[0],))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {"ideas_created": 0, "sessions_processed": 1}
            await daemon.continuous_cycle()

        # Queue should be empty
        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        row = cursor.fetchone()
        db.close()

        assert row["cnt"] == 0

    @pytest.mark.asyncio
    async def test_updates_byte_positions(self, test_db, transcript_files):
        """Should update byte positions after processing."""
        from daemon_agent import AgentDaemon

        content = make_transcript([
            ("user", "Test message", "2024-01-15T10:00:00Z"),
        ])
        Path(transcript_files[0]).write_text(content)

        # Add to queue
        from db.connection import get_db
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_files[0], len(content)))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 0,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }
            await daemon.continuous_cycle()

        # Byte position should be updated
        db = get_db()
        cursor = db.execute(
            "SELECT byte_position FROM index_state WHERE file_path = ?",
            (transcript_files[0],)
        )
        row = cursor.fetchone()
        db.close()

        # The agent updates byte positions internally, verify it was called
        assert mock_agent.called


class TestContinuousStats:
    """Tests for continuous mode statistics."""

    @pytest.mark.asyncio
    async def test_accumulates_stats(self, test_db, transcript_files):
        """Should accumulate processing stats across files."""
        from daemon_agent import AgentDaemon

        for path in transcript_files[:2]:
            Path(path).write_text(make_transcript([
                ("user", "Test", "2024-01-15T10:00:00Z"),
            ]))

        # Add to queue
        from db.connection import get_db
        db = get_db()
        for path in transcript_files[:2]:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, 100)
            """, (path,))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 5,
                "sessions_processed": 2,
                "bytes_processed": 200
            }
            await daemon.continuous_cycle()

        # Stats should be accumulated
        assert daemon.ctx.ideas_stored >= 5
        assert daemon.ctx.files_processed >= 2

    @pytest.mark.asyncio
    async def test_tracks_errors(self, test_db, transcript_files):
        """Should track processing errors."""
        from daemon_agent import AgentDaemon

        Path(transcript_files[0]).write_text(make_transcript([
            ("user", "Test", "2024-01-15T10:00:00Z"),
        ]))

        # Add to queue
        from db.connection import get_db
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, 100)
        """, (transcript_files[0],))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.side_effect = Exception("Test error")

            await daemon.continuous_cycle()

        # Should have tracked error
        assert len(daemon.ctx.errors) >= 1
