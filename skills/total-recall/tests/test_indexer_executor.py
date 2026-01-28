"""Tests for indexer executor - Slices 4.2-4.9."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
        if mod.startswith(('config', 'db.', 'indexer.', 'memory_db', 'entities')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_span(test_db):
    """Create a sample span for testing."""
    from db.connection import get_db

    db = get_db()
    cursor = db.execute("""
        INSERT INTO spans (session, name, summary, start_line, depth)
        VALUES ('test-session', 'Test Span', 'A test span', 1, 0)
    """)
    span_id = cursor.lastrowid
    db.commit()
    db.close()

    return span_id


class TestExecuteIdeas:
    """Tests for execute_ideas - Slice 4.2."""

    @pytest.mark.asyncio
    async def test_creates_idea_with_all_fields(self, test_db, sample_span):
        """Should create idea with all fields."""
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput
        from db.connection import get_db

        ideas = [
            IdeaOutput(
                intent="decision",
                content="Use JWT for authentication",
                source_line=10,
                confidence=0.9,
                importance=0.8
            )
        ]

        result = await execute_ideas(
            ideas=ideas,
            session="test-session",
            source_file="/test.jsonl",
            span_id=sample_span
        )

        assert len(result) == 1

        db = get_db()
        cursor = db.execute("SELECT * FROM ideas WHERE id = ?", (result[0],))
        row = cursor.fetchone()
        db.close()

        assert row["content"] == "Use JWT for authentication"
        assert row["intent"] == "decision"
        assert row["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_sets_importance_from_output(self, test_db, sample_span):
        """Should set importance from output."""
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput
        from db.connection import get_db

        ideas = [
            IdeaOutput(intent="decision", content="Test", source_line=5, importance=0.95)
        ]

        result = await execute_ideas(
            ideas=ideas,
            session="test-session",
            source_file="/test.jsonl",
            span_id=sample_span
        )

        db = get_db()
        cursor = db.execute("SELECT importance FROM ideas WHERE id = ?", (result[0],))
        row = cursor.fetchone()
        db.close()

        assert row["importance"] == 0.95

    @pytest.mark.asyncio
    async def test_handles_duplicate_source_line(self, test_db, sample_span):
        """Should handle duplicate source_file/source_line gracefully."""
        from indexer.executor import execute_ideas
        from indexer.output_parser import IdeaOutput

        ideas = [IdeaOutput(intent="decision", content="First", source_line=10)]

        # Insert first
        await execute_ideas(ideas, "test-session", "/test.jsonl", sample_span)

        # Insert duplicate
        ideas2 = [IdeaOutput(intent="context", content="Second", source_line=10)]
        result = await execute_ideas(ideas2, "test-session", "/test.jsonl", sample_span)

        # Should return empty (duplicate skipped) or existing ID
        assert len(result) <= 1


class TestExecuteTopicUpdates:
    """Tests for execute_topic_updates - Slice 4.3."""

    @pytest.mark.asyncio
    async def test_updates_span_name(self, test_db, sample_span):
        """Should update span name."""
        from indexer.executor import execute_topic_updates
        from indexer.output_parser import TopicUpdate
        from db.connection import get_db

        updates = [
            TopicUpdate(span_id=sample_span, name="New Name")
        ]

        await execute_topic_updates(updates)

        db = get_db()
        cursor = db.execute("SELECT name FROM spans WHERE id = ?", (sample_span,))
        row = cursor.fetchone()
        db.close()

        assert row["name"] == "New Name"

    @pytest.mark.asyncio
    async def test_updates_span_summary(self, test_db, sample_span):
        """Should update span summary."""
        from indexer.executor import execute_topic_updates
        from indexer.output_parser import TopicUpdate
        from db.connection import get_db

        updates = [
            TopicUpdate(span_id=sample_span, summary="New summary content")
        ]

        await execute_topic_updates(updates)

        db = get_db()
        cursor = db.execute("SELECT summary FROM spans WHERE id = ?", (sample_span,))
        row = cursor.fetchone()
        db.close()

        assert row["summary"] == "New summary content"

    @pytest.mark.asyncio
    async def test_handles_nonexistent_span(self, test_db):
        """Should handle non-existent span gracefully."""
        from indexer.executor import execute_topic_updates
        from indexer.output_parser import TopicUpdate

        updates = [
            TopicUpdate(span_id=99999, name="Won't work")
        ]

        # Should not raise
        await execute_topic_updates(updates)


class TestExecuteTopicChanges:
    """Tests for execute_topic_changes - Slice 4.4."""

    @pytest.mark.asyncio
    async def test_creates_new_span(self, test_db, sample_span):
        """Should create new span for topic change."""
        from indexer.executor import execute_topic_changes
        from indexer.output_parser import TopicChange
        from db.connection import get_db

        changes = [
            TopicChange(
                from_span_id=sample_span,
                new_name="New Topic",
                reason="Topic shifted",
                at_line=50
            )
        ]

        result = await execute_topic_changes(changes, session="test-session")

        assert len(result) == 1

        db = get_db()
        cursor = db.execute("SELECT * FROM spans WHERE id = ?", (result[0],))
        row = cursor.fetchone()
        db.close()

        assert row["name"] == "New Topic"
        assert row["start_line"] == 50

    @pytest.mark.asyncio
    async def test_sets_parent_to_from_span(self, test_db, sample_span):
        """Should set parent_id to from_span_id."""
        from indexer.executor import execute_topic_changes
        from indexer.output_parser import TopicChange
        from db.connection import get_db

        changes = [
            TopicChange(
                from_span_id=sample_span,
                new_name="Child Topic",
                reason="Subtopic",
                at_line=100
            )
        ]

        result = await execute_topic_changes(changes, session="test-session")

        db = get_db()
        cursor = db.execute("SELECT parent_id FROM spans WHERE id = ?", (result[0],))
        row = cursor.fetchone()
        db.close()

        assert row["parent_id"] == sample_span


class TestExecuteAnsweredQuestions:
    """Tests for execute_answered_questions - Slice 4.5."""

    @pytest.fixture
    def sample_question(self, test_db, sample_span):
        """Create a sample question idea."""
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line, span_id, session)
            VALUES ('How do we authenticate?', 'question', '/test.jsonl', 5, ?, 'test-session')
        """, (sample_span,))
        question_id = cursor.lastrowid
        db.commit()
        db.close()

        return question_id

    @pytest.mark.asyncio
    async def test_sets_answered_true(self, sample_question):
        """Should set answered=TRUE on question."""
        from indexer.executor import execute_answered_questions
        from indexer.output_parser import AnsweredQuestion
        from db.connection import get_db

        answers = [
            AnsweredQuestion(question_id=sample_question, answer_line=20)
        ]

        await execute_answered_questions(answers)

        db = get_db()
        cursor = db.execute("SELECT answered FROM ideas WHERE id = ?", (sample_question,))
        row = cursor.fetchone()
        db.close()

        assert row["answered"] == True

    @pytest.mark.asyncio
    async def test_handles_nonexistent_question(self, test_db):
        """Should handle non-existent question gracefully."""
        from indexer.executor import execute_answered_questions
        from indexer.output_parser import AnsweredQuestion

        answers = [
            AnsweredQuestion(question_id=99999, answer_line=20)
        ]

        # Should not raise
        await execute_answered_questions(answers)


class TestExecuteRelations:
    """Tests for execute_relations - Slice 4.6."""

    @pytest.fixture
    def sample_ideas(self, test_db, sample_span):
        """Create sample ideas for testing relations."""
        from db.connection import get_db

        db = get_db()

        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line, span_id, session)
            VALUES ('Old decision', 'decision', '/test.jsonl', 5, ?, 'test-session')
        """, (sample_span,))
        old_id = cursor.lastrowid

        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line, span_id, session)
            VALUES ('New decision', 'decision', '/test.jsonl', 20, ?, 'test-session')
        """, (sample_span,))
        new_id = cursor.lastrowid

        db.commit()
        db.close()

        return {"old": old_id, "new": new_id}

    @pytest.mark.asyncio
    async def test_creates_relation(self, sample_ideas):
        """Should create relation with correct type."""
        from indexer.executor import execute_relations
        from indexer.output_parser import RelationOutput
        from db.connection import get_db

        relations = [
            RelationOutput(
                from_line=20,  # Line of new decision
                to_idea_id=sample_ideas["old"],
                relation_type="supersedes"
            )
        ]

        # Map line to idea id
        line_map = {20: sample_ideas["new"]}

        count = await execute_relations(relations, "/test.jsonl", line_map)

        assert count == 1

        db = get_db()
        cursor = db.execute("SELECT * FROM relations WHERE from_id = ?", (sample_ideas["new"],))
        row = cursor.fetchone()
        db.close()

        assert row["to_id"] == sample_ideas["old"]
        assert row["relation_type"] == "supersedes"


class TestExecuteAgentOutput:
    """Tests for execute_agent_output - Slice 4.9."""

    @pytest.mark.asyncio
    async def test_executes_all_output_types(self, test_db, sample_span):
        """Should execute all output types."""
        from indexer.executor import execute_agent_output
        from indexer.output_parser import AgentOutput, IdeaOutput

        output = AgentOutput(
            ideas=[
                IdeaOutput(intent="decision", content="Test decision", source_line=10)
            ]
        )

        with patch('indexer.executor.generate_embeddings', new_callable=AsyncMock):
            result = await execute_agent_output(
                output=output,
                session="test-session",
                source_file="/test.jsonl",
                span_id=sample_span
            )

        assert result["ideas_created"] == 1

    @pytest.mark.asyncio
    async def test_returns_execution_stats(self, test_db, sample_span):
        """Should return count of each type executed."""
        from indexer.executor import execute_agent_output
        from indexer.output_parser import AgentOutput, IdeaOutput, TopicUpdate

        output = AgentOutput(
            ideas=[
                IdeaOutput(intent="decision", content="Decision 1", source_line=10),
                IdeaOutput(intent="context", content="Context 1", source_line=15),
            ],
            topic_updates=[
                TopicUpdate(span_id=sample_span, summary="Updated")
            ]
        )

        with patch('indexer.executor.generate_embeddings', new_callable=AsyncMock):
            result = await execute_agent_output(
                output=output,
                session="test-session",
                source_file="/test.jsonl",
                span_id=sample_span
            )

        assert result["ideas_created"] == 2
        assert result["topic_updates"] == 1
