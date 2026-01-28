"""Integration tests for continuous indexing - Slice 6.1.

End-to-end tests that verify the full continuous indexing pipeline
from hook enqueue to searchable ideas.
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
def transcript_file():
    """Create a temp transcript file."""
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    yield path
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


class TestHookEnqueue:
    """Tests for hook triggering queue addition."""

    @pytest.mark.asyncio
    async def test_hook_enqueues_file(self, test_db, transcript_file):
        """Hook should enqueue file to work queue."""
        from db.connection import get_db

        # Write transcript content
        content = make_transcript([
            ("user", "Let's implement a new feature", "2024-01-15T10:00:00Z"),
            ("assistant", "Sure, let's start with the database schema", "2024-01-15T10:00:02Z"),
        ])
        Path(transcript_file).write_text(content)

        # Simulate hook enqueue (what the hook does)
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()

        # Verify enqueue
        cursor = db.execute("SELECT file_path FROM work_queue")
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["file_path"] == transcript_file


class TestDaemonPickup:
    """Tests for daemon picking up queued files."""

    @pytest.mark.asyncio
    async def test_daemon_picks_up_after_batch_window(self, test_db, transcript_file):
        """Daemon should pick up file after batch window."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Write content
        content = make_transcript([
            ("user", "Implement user authentication", "2024-01-15T10:00:00Z"),
        ])
        Path(transcript_file).write_text(content)

        # Enqueue file
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0.05)
        processed = False

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 1,
                "sessions_processed": 1,
                "bytes_processed": len(content)
            }

            # Run one cycle
            result = await daemon.continuous_cycle()
            processed = result

        assert processed is True
        mock_agent.assert_called_once()


class TestAgentExtraction:
    """Tests for agent extracting ideas from content."""

    @pytest.mark.asyncio
    async def test_agent_extracts_ideas(self, test_db, transcript_file):
        """Agent should extract meaningful ideas from content."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Write content with extractable ideas
        content = make_transcript([
            ("user", "We need to implement rate limiting for the API", "2024-01-15T10:00:00Z"),
            ("assistant", "I'll use a token bucket algorithm. Decision: Use Redis for distributed rate limiting", "2024-01-15T10:00:05Z"),
            ("user", "Great! Also add logging for rate limit hits", "2024-01-15T10:00:10Z"),
        ])
        Path(transcript_file).write_text(content)

        # Enqueue file
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        # Mock agent to return structured ideas
        mock_response = {
            "ideas_created": 2,
            "sessions_processed": 1,
            "bytes_processed": len(content)
        }

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = mock_response
            await daemon.continuous_cycle()

        # Verify agent was called with updates containing our messages
        call_args = mock_agent.call_args
        assert call_args is not None
        updates = call_args[0][0]
        assert len(updates) >= 1
        assert any("rate limiting" in m.content.lower() for u in updates for m in u.messages)


class TestIdeasSearchable:
    """Tests for ideas being searchable after indexing."""

    @pytest.mark.asyncio
    async def test_ideas_are_searchable_after_indexing(self, test_db, transcript_file):
        """Ideas should be searchable after indexing."""
        from indexer.batch_collector import BatchUpdate, Message
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput
        from db.connection import get_db

        # Create a span first
        db = get_db()
        db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('test-session', 'Test', '', 1, 0)
        """)
        span_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()
        db.close()

        # Execute ideas directly
        ideas = [
            IdeaOutput(
                intent="decision",
                content="Use PostgreSQL for the main database",
                source_line=1,
                confidence=0.9,
                importance=0.8,
                entities=["PostgreSQL"]
            ),
            IdeaOutput(
                intent="todo",
                content="Set up database migrations",
                source_line=2,
                confidence=0.8,
                importance=0.7,
                entities=[]
            ),
        ]

        idea_ids = await execute_ideas(
            ideas=ideas,
            session="test-session",
            source_file=transcript_file,
            span_id=span_id
        )

        # Verify ideas are stored
        assert len(idea_ids) == 2

        # Verify ideas are queryable
        db = get_db()
        cursor = db.execute("""
            SELECT id, content, intent FROM ideas
            WHERE content LIKE '%PostgreSQL%'
        """)
        rows = cursor.fetchall()
        db.close()

        assert len(rows) >= 1
        assert any("PostgreSQL" in row["content"] for row in rows)


class TestFullPipeline:
    """End-to-end tests for the full continuous pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_hook_to_search(self, test_db, transcript_file):
        """Full pipeline: hook enqueue -> daemon pickup -> agent extract -> searchable."""
        from daemon_agent import AgentDaemon
        from indexer.batch_collector import BatchUpdate, Message
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput
        from db.connection import get_db

        # 1. Write transcript content
        content = make_transcript([
            ("user", "Let's use GraphQL for the API", "2024-01-15T10:00:00Z"),
            ("assistant", "Good choice. I'll set up Apollo Server", "2024-01-15T10:00:05Z"),
        ])
        Path(transcript_file).write_text(content)

        # 2. Simulate hook enqueue
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        # 3. Run daemon cycle with mocked agent that stores real ideas
        daemon = AgentDaemon(batch_window_seconds=0)

        async def mock_agent_with_storage(updates, mode="continuous"):
            """Mock that actually stores ideas."""
            from db.connection import get_db

            # Create span
            db = get_db()
            db.execute("""
                INSERT INTO spans (session, name, summary, start_line, depth)
                VALUES (?, 'GraphQL Setup', '', 1, 0)
            """, (updates[0].session,))
            span_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            db.commit()
            db.close()

            # Store ideas
            ideas = [
                IdeaOutput(
                    intent="decision",
                    content="Use GraphQL for the API",
                    source_line=1,
                    confidence=0.9,
                    importance=0.8,
                    entities=["GraphQL"]
                ),
            ]
            await execute_ideas(ideas, updates[0].session, updates[0].file_path, span_id)

            return {"ideas_created": 1, "sessions_processed": 1, "bytes_processed": len(content)}

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.side_effect = mock_agent_with_storage
            await daemon.continuous_cycle()

        # 4. Verify idea is searchable
        db = get_db()
        cursor = db.execute("""
            SELECT id, content, intent FROM ideas
            WHERE content LIKE '%GraphQL%'
        """)
        rows = cursor.fetchall()
        db.close()

        assert len(rows) >= 1
        assert rows[0]["content"] == "Use GraphQL for the API"
        assert rows[0]["intent"] == "decision"


class TestErrorRecovery:
    """Tests for error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_daemon_continues_after_error(self, test_db, transcript_file):
        """Daemon should continue processing after errors."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Write content
        content = make_transcript([
            ("user", "Test message", "2024-01-15T10:00:00Z"),
        ])
        Path(transcript_file).write_text(content)

        # Enqueue file
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)

        # First call fails, second succeeds
        call_count = 0

        async def failing_then_success(updates, mode="continuous"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return {"ideas_created": 1, "sessions_processed": 1, "bytes_processed": 100}

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.side_effect = failing_then_success

            # First cycle should handle error gracefully
            await daemon.continuous_cycle()

        # Error should be tracked
        assert len(daemon.ctx.errors) >= 1
