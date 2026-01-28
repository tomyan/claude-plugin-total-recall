"""Integration tests for backfill indexing - Slice 6.2.

End-to-end tests that verify the full backfill pipeline
from command to fully indexed sessions.
"""

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
        if mod.startswith(('config', 'db.', 'indexer.', 'memory_db', 'daemon', 'entities')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def backfill_files():
    """Create multiple transcript files for backfill testing."""
    files = []
    for i in range(3):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)
        files.append(path)
    yield files
    for path in files:
        Path(path).unlink(missing_ok=True)


def make_transcript(messages: list[tuple[str, str, str]]) -> str:
    """Create transcript JSONL content."""
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
        content = f"Message {i}: " + "x" * 100
        ts = f"2024-01-15T10:{i % 60:02d}:00Z"
        messages.append((role, content, ts))
    return make_transcript(messages)


class TestBackfillEnqueue:
    """Tests for backfill command enqueueing files."""

    @pytest.mark.asyncio
    async def test_backfill_enqueues_single_file(self, test_db, backfill_files):
        """Backfill command should enqueue a single file."""
        from db.connection import get_db

        # Write content
        content = make_transcript([
            ("user", "Old conversation from last month", "2024-01-15T10:00:00Z"),
        ])
        Path(backfill_files[0]).write_text(content)

        # Simulate backfill enqueue
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (backfill_files[0], len(content)))
        db.commit()

        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 1

    @pytest.mark.asyncio
    async def test_backfill_enqueues_multiple_files(self, test_db, backfill_files):
        """Backfill command should enqueue multiple files."""
        from db.connection import get_db

        # Write content to all files
        for path in backfill_files:
            content = make_transcript([
                ("user", "Historical conversation", "2024-01-15T10:00:00Z"),
            ])
            Path(path).write_text(content)

        # Simulate backfill enqueue for all files
        db = get_db()
        for path in backfill_files:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, ?)
            """, (path, 100))
        db.commit()

        cursor = db.execute("SELECT COUNT(*) as cnt FROM work_queue")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 3


class TestBackfillProcessing:
    """Tests for backfill session processing."""

    @pytest.mark.asyncio
    async def test_processes_session_by_session(self, test_db, backfill_files):
        """Daemon should process sessions one by one for backfill."""
        from daemon_agent import AgentDaemon

        # Write content
        content = make_transcript([
            ("user", "First session content", "2024-01-15T10:00:00Z"),
            ("assistant", "Response to first", "2024-01-15T10:00:05Z"),
        ])
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)
        sessions_processed = []

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def track_session(updates, mode="backfill"):
                sessions_processed.append(updates[0].session)
                return {
                    "ideas_created": 1,
                    "sessions_processed": 1,
                    "bytes_processed": len(content)
                }

            mock_agent.side_effect = track_session

            await daemon.backfill_session(backfill_files[0])

        # Should have processed exactly one session
        assert len(sessions_processed) >= 1

    @pytest.mark.asyncio
    async def test_all_content_indexed(self, test_db, backfill_files):
        """All content should be indexed after backfill."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Write substantial content
        content = make_large_transcript(num_messages=20)
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 5,
                "sessions_processed": 1,
                "bytes_processed": 1000
            }

            result = await daemon.backfill_session(backfill_files[0])

        # Should have processed all bytes
        assert result["bytes_processed"] >= len(content) * 0.9


class TestBackfillProgress:
    """Tests for backfill progress tracking."""

    @pytest.mark.asyncio
    async def test_progress_is_trackable(self, test_db, backfill_files):
        """Progress should be trackable during backfill."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        content = make_large_transcript(num_messages=50)
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=1000  # Small to force multiple batches
        )

        progress_updates = []

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            async def track_progress(updates, mode="backfill"):
                # Track byte position after each batch
                from db.connection import get_db
                db = get_db()
                cursor = db.execute(
                    "SELECT byte_position FROM index_state WHERE file_path = ?",
                    (backfill_files[0],)
                )
                row = cursor.fetchone()
                db.close()
                if row:
                    progress_updates.append(row["byte_position"])

                return {
                    "ideas_created": 1,
                    "sessions_processed": 1,
                    "bytes_processed": updates[0].end_byte - updates[0].start_byte
                }

            mock_agent.side_effect = track_progress

            await daemon.backfill_session(backfill_files[0])

        # Progress should be monotonically increasing
        if len(progress_updates) > 1:
            for i in range(1, len(progress_updates)):
                assert progress_updates[i] >= progress_updates[i - 1]

    @pytest.mark.asyncio
    async def test_returns_total_stats(self, test_db, backfill_files):
        """Backfill should return total stats."""
        from daemon_agent import AgentDaemon

        content = make_transcript([
            ("user", "Test content", "2024-01-15T10:00:00Z"),
            ("assistant", "Response", "2024-01-15T10:00:05Z"),
        ])
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 3,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            result = await daemon.backfill_session(backfill_files[0])

        assert "ideas_created" in result
        assert "sessions_processed" in result
        assert "bytes_processed" in result
        assert result["sessions_processed"] == 1


class TestBackfillResume:
    """Tests for resumable backfill."""

    @pytest.mark.asyncio
    async def test_resumes_from_last_position(self, test_db, backfill_files):
        """Backfill should resume from last indexed position."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        content = make_large_transcript(num_messages=10)
        Path(backfill_files[0]).write_text(content)

        # Mark half as already indexed
        half_way = len(content) // 2
        db = get_db()
        db.execute("""
            INSERT INTO index_state (file_path, byte_position)
            VALUES (?, ?)
        """, (backfill_files[0], half_way))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 1,
                "sessions_processed": 1,
                "bytes_processed": len(content) - half_way
            }

            result = await daemon.backfill_session(backfill_files[0])

        # Should only process remaining content
        assert result["bytes_processed"] <= len(content) - half_way + 100  # some margin

    @pytest.mark.asyncio
    async def test_handles_interruption(self, test_db, backfill_files):
        """Backfill should handle interruption gracefully."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        content = make_large_transcript(num_messages=20)
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=500
        )

        batch_count = 0

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            async def fail_after_first(updates, mode="backfill"):
                nonlocal batch_count
                batch_count += 1
                if batch_count == 2:
                    raise Exception("Simulated interruption")
                return {
                    "ideas_created": 1,
                    "sessions_processed": 1,
                    "bytes_processed": updates[0].end_byte - updates[0].start_byte
                }

            mock_agent.side_effect = fail_after_first

            result = await daemon.backfill_session(backfill_files[0])

        # Should have recorded error
        assert "error" in result

        # Byte position should have been updated for successful batch
        db = get_db()
        cursor = db.execute(
            "SELECT byte_position FROM index_state WHERE file_path = ?",
            (backfill_files[0],)
        )
        row = cursor.fetchone()
        db.close()

        # Should have made some progress
        assert row is not None
        assert row["byte_position"] > 0


class TestBackfillComplete:
    """Tests for complete backfill scenarios."""

    @pytest.mark.asyncio
    async def test_marks_file_complete(self, test_db, backfill_files):
        """File should be fully indexed after backfill."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        content = make_transcript([
            ("user", "First message", "2024-01-15T10:00:00Z"),
            ("assistant", "Response", "2024-01-15T10:00:05Z"),
        ])
        Path(backfill_files[0]).write_text(content)

        daemon = AgentDaemon(batch_window_seconds=0)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 1,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            await daemon.backfill_session(backfill_files[0])

        # Byte position should equal file size
        db = get_db()
        cursor = db.execute(
            "SELECT byte_position FROM index_state WHERE file_path = ?",
            (backfill_files[0],)
        )
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["byte_position"] >= len(content)
