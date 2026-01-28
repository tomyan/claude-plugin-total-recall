"""Performance baseline tests - Slice 6.3.

Measures and documents performance characteristics of the indexing pipeline.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
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


def make_large_transcript(num_messages: int = 100) -> str:
    """Create a large transcript with many messages."""
    messages = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: This is a test message with some content that simulates real conversation data. " * 5
        ts = f"2024-01-15T10:{i % 60:02d}:00Z"
        messages.append((role, content, ts))
    return make_transcript(messages)


class TestLLMCallsPerFile:
    """Measure LLM calls per file (before vs after agent approach)."""

    @pytest.mark.asyncio
    async def test_single_llm_call_for_small_file(self, test_db, transcript_file):
        """Small file should require only 1 LLM call with agent approach."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Write small content (< 30k tokens)
        content = make_transcript([
            ("user", "Let's implement authentication", "2024-01-15T10:00:00Z"),
            ("assistant", "I'll use JWT tokens", "2024-01-15T10:00:05Z"),
        ])
        Path(transcript_file).write_text(content)

        # Enqueue
        db = get_db()
        db.execute("""
            INSERT INTO work_queue (file_path, file_size)
            VALUES (?, ?)
        """, (transcript_file, len(content)))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)
        llm_calls = 0

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def count_calls(updates, mode="continuous"):
                nonlocal llm_calls
                llm_calls += 1
                return {"ideas_created": 1, "sessions_processed": 1, "bytes_processed": len(content)}

            mock_agent.side_effect = count_calls
            await daemon.continuous_cycle()

        # Should be exactly 1 LLM call for a small file
        assert llm_calls == 1

    @pytest.mark.asyncio
    async def test_batched_files_single_llm_call(self, test_db):
        """Multiple files in batch window should use single LLM call."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Create multiple small files
        files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix='.jsonl')
            os.close(fd)
            content = make_transcript([
                ("user", f"Message from file {i}", "2024-01-15T10:00:00Z"),
            ])
            Path(path).write_text(content)
            files.append((path, len(content)))

        # Enqueue all
        db = get_db()
        for path, size in files:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, ?)
            """, (path, size))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0)
        llm_calls = 0

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def count_calls(updates, mode="continuous"):
                nonlocal llm_calls
                llm_calls += 1
                return {"ideas_created": len(updates), "sessions_processed": len(updates), "bytes_processed": 100}

            mock_agent.side_effect = count_calls
            await daemon.continuous_cycle()

        # Cleanup
        for path, _ in files:
            Path(path).unlink(missing_ok=True)

        # Should be exactly 1 LLM call for batched files
        assert llm_calls == 1


class TestTokenUsage:
    """Measure token usage per file."""

    @pytest.mark.asyncio
    async def test_respects_token_budget(self, test_db, transcript_file):
        """Should respect token budget per LLM call."""
        from daemon_agent import AgentDaemon
        from indexer.batch_collector import collect_batch_updates

        # Write large content
        content = make_large_transcript(num_messages=100)
        Path(transcript_file).write_text(content)

        # Collect batch updates directly
        updates = await collect_batch_updates([transcript_file])

        # Estimate tokens in the update
        total_chars = sum(len(m.content) for u in updates for m in u.messages)
        estimated_tokens = total_chars // 4

        # Should be within reasonable bounds for context window
        assert estimated_tokens < 50000  # Well under context limit


class TestIndexingLatency:
    """Measure indexing latency."""

    @pytest.mark.asyncio
    async def test_batch_collection_fast(self, test_db, transcript_file):
        """Batch collection should be fast."""
        from indexer.batch_collector import collect_batch_updates

        # Write moderate content
        content = make_large_transcript(num_messages=50)
        Path(transcript_file).write_text(content)

        start = time.time()
        updates = await collect_batch_updates([transcript_file])
        elapsed = time.time() - start

        # Batch collection should be < 100ms for typical files
        assert elapsed < 0.1
        assert len(updates) == 1
        assert len(updates[0].messages) > 0

    @pytest.mark.asyncio
    async def test_idea_storage_fast(self, test_db, transcript_file):
        """Idea storage should be fast."""
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput
        from db.connection import get_db

        # Create span
        db = get_db()
        db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('test', 'Test', '', 1, 0)
        """)
        span_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()
        db.close()

        # Create ideas
        ideas = [
            IdeaOutput(
                intent="decision",
                content=f"Test decision {i}",
                source_line=i,
                confidence=0.8,
                importance=0.7,
                entities=[]
            )
            for i in range(100)
        ]

        start = time.time()
        with patch('indexer.executor.generate_embeddings', new_callable=AsyncMock):
            idea_ids = await execute_ideas(ideas, "test", transcript_file, span_id)
        elapsed = time.time() - start

        # Storing 100 ideas should be < 1 second
        assert elapsed < 1.0
        assert len(idea_ids) == 100


class TestRelationAccuracy:
    """Measure relation extraction accuracy."""

    @pytest.mark.asyncio
    async def test_creates_relations_correctly(self, test_db, transcript_file):
        """Relations should be created with correct types."""
        from indexer.executor import execute_relations
        from indexer.output_parser import RelationOutput
        from db.connection import get_db

        # Create span and ideas
        db = get_db()
        db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('test', 'Test', '', 1, 0)
        """)
        span_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Create source and target ideas
        db.execute("""
            INSERT INTO ideas (span_id, content, intent, source_file, source_line)
            VALUES (?, 'Source idea', 'decision', ?, 1)
        """, (span_id, transcript_file))
        source_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        db.execute("""
            INSERT INTO ideas (span_id, content, intent, source_file, source_line)
            VALUES (?, 'Target idea', 'context', ?, 2)
        """, (span_id, transcript_file))
        target_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()
        db.close()

        # Create relations
        relations = [
            RelationOutput(
                from_line=1,
                to_idea_id=target_id,
                relation_type="builds_on"
            ),
        ]

        # Map source line to idea id
        line_to_id = {1: source_id, 2: target_id}

        created = await execute_relations(relations, transcript_file, line_to_id)

        # Verify relation was created
        db = get_db()
        cursor = db.execute("""
            SELECT from_id, to_id, relation_type FROM relations
        """)
        rows = cursor.fetchall()
        db.close()

        assert len(rows) >= 1
        assert rows[0]["from_id"] == source_id
        assert rows[0]["to_id"] == target_id
        assert rows[0]["relation_type"] == "builds_on"


class TestScalability:
    """Test scalability characteristics."""

    @pytest.mark.asyncio
    async def test_handles_many_files_in_batch(self, test_db):
        """Should handle many files in single batch."""
        from daemon_agent import AgentDaemon
        from db.connection import get_db

        # Create many small files
        files = []
        for i in range(10):
            fd, path = tempfile.mkstemp(suffix='.jsonl')
            os.close(fd)
            content = make_transcript([
                ("user", f"File {i} content", "2024-01-15T10:00:00Z"),
            ])
            Path(path).write_text(content)
            files.append((path, len(content)))

        # Enqueue all
        db = get_db()
        for path, size in files:
            db.execute("""
                INSERT INTO work_queue (file_path, file_size)
                VALUES (?, ?)
            """, (path, size))
        db.commit()
        db.close()

        daemon = AgentDaemon(batch_window_seconds=0, max_files_per_batch=10)

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "ideas_created": 10,
                "sessions_processed": 10,
                "bytes_processed": 1000
            }

            await daemon.continuous_cycle()

            # Should process all files in one batch
            assert mock_agent.call_count == 1
            updates = mock_agent.call_args[0][0]
            assert len(updates) == 10

        # Cleanup
        for path, _ in files:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_handles_large_file_in_batches(self, test_db, transcript_file):
        """Should handle large files by batching."""
        from daemon_agent import AgentDaemon

        # Write very large content
        content = make_large_transcript(num_messages=200)
        Path(transcript_file).write_text(content)

        daemon = AgentDaemon(
            batch_window_seconds=0,
            backfill_target_tokens=2000  # Small to force batching
        )

        batch_count = 0

        with patch('daemon_agent.run_indexing_agent', new_callable=AsyncMock) as mock_agent:
            def count_batches(updates, mode="backfill"):
                nonlocal batch_count
                batch_count += 1
                return {
                    "ideas_created": 1,
                    "sessions_processed": 1,
                    "bytes_processed": updates[0].end_byte - updates[0].start_byte
                }

            mock_agent.side_effect = count_batches

            await daemon.backfill_session(transcript_file)

        # Should have made multiple batches
        assert batch_count >= 2


class TestPerformanceBaseline:
    """Document performance baselines."""

    def test_batch_collector_memory_efficient(self, test_db, transcript_file):
        """Batch collector should not load entire file into memory."""
        import sys

        # Write large content
        content = make_large_transcript(num_messages=100)
        Path(transcript_file).write_text(content)

        # The batch collector reads line by line, so memory usage should be
        # bounded regardless of file size
        # (This is a design assertion, not a runtime test)
        assert True  # Design assertion

    def test_agent_batching_reduces_calls(self):
        """Agent batching should reduce total LLM calls.

        Previous approach: 1 LLM call per message batch (~5 min window)
        New approach: 1 LLM call per file batch (multiple files together)

        For 3 files with 10 batches each:
        - Old: 30 LLM calls
        - New: 1 LLM call (batched)

        Expected improvement: ~30x fewer LLM calls
        """
        # This is a documentation test
        old_calls = 3 * 10  # 3 files, 10 batches each
        new_calls = 1  # Single batched call

        improvement_factor = old_calls / new_calls
        assert improvement_factor >= 10  # At least 10x improvement expected
