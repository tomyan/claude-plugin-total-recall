"""Tests for indexing agent tools - Slices 2.2-2.8."""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

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
        if mod.startswith(('config', 'db.', 'llm.indexing_tools', 'memory_db', 'search', 'embeddings', 'entities')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_ideas(test_db):
    """Create sample ideas for testing."""
    from db.connection import get_db

    db = get_db()

    # Create a topic and span first
    cursor = db.execute("""
        INSERT INTO topics (name, canonical_name, summary)
        VALUES ('Test Topic', 'test topic', 'A test topic')
    """)
    topic_id = cursor.lastrowid

    cursor = db.execute("""
        INSERT INTO spans (session, name, summary, start_line, depth, topic_id)
        VALUES ('test-session', 'Test Span', 'A test span', 1, 0, ?)
    """, (topic_id,))
    span_id = cursor.lastrowid

    # Create sample ideas
    ideas = [
        ("Using JWT tokens for authentication", "decision", "session-1", "/test1.jsonl", 10, span_id),
        ("Implement OAuth 2.0 flow", "decision", "session-1", "/test1.jsonl", 20, span_id),
        ("Database migration needed", "context", "session-2", "/test2.jsonl", 5, span_id),
        ("How should we handle rate limiting?", "question", "session-1", "/test1.jsonl", 30, span_id),
        ("Completed user registration feature", "observation", "session-2", "/test2.jsonl", 15, span_id),
    ]

    idea_ids = []
    for content, intent, session, source_file, source_line, span_id in ideas:
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line, span_id, session)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content, intent, source_file, source_line, span_id, session))
        idea_ids.append(cursor.lastrowid)

    db.commit()
    db.close()

    return idea_ids


class TestSearchIdeasTool:
    """Tests for search_ideas tool - Slice 2.2."""

    @pytest.mark.asyncio
    async def test_returns_semantically_similar_ideas(self, sample_ideas):
        """Should return ideas semantically similar to query."""
        from llm.indexing_tools import tool_search_ideas

        # Mock embedding search to return our ideas
        with patch('llm.indexing_tools.search_ideas') as mock_search:
            mock_search.return_value = [
                {"id": sample_ideas[0], "content": "Using JWT tokens for authentication", "score": 0.9},
                {"id": sample_ideas[1], "content": "Implement OAuth 2.0 flow", "score": 0.8},
            ]

            results = await tool_search_ideas("authentication methods")

        assert len(results) == 2
        assert "JWT" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_ideas):
        """Should respect the limit parameter."""
        from llm.indexing_tools import tool_search_ideas

        with patch('llm.indexing_tools.search_ideas') as mock_search:
            mock_search.return_value = [
                {"id": sample_ideas[0], "content": "Result 1", "score": 0.9},
            ]

            results = await tool_search_ideas("test", limit=1)

        # Verify limit was passed to search
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("limit") == 1

    @pytest.mark.asyncio
    async def test_filters_by_session(self, sample_ideas):
        """Should filter by session when provided."""
        from llm.indexing_tools import tool_search_ideas

        with patch('llm.indexing_tools.search_ideas') as mock_search:
            mock_search.return_value = []

            await tool_search_ideas("test", session="session-1")

        # Verify session filter was passed
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("session") == "session-1"

    @pytest.mark.asyncio
    async def test_filters_by_intent(self, sample_ideas):
        """Should filter by intent when provided."""
        from llm.indexing_tools import tool_search_ideas

        with patch('llm.indexing_tools.search_ideas') as mock_search:
            mock_search.return_value = []

            await tool_search_ideas("test", intent="decision")

        # Verify intent filter was passed
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("intent") == "decision"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_matches(self, test_db):
        """Should return empty list when no matches found."""
        from llm.indexing_tools import tool_search_ideas

        with patch('llm.indexing_tools.search_ideas') as mock_search:
            mock_search.return_value = []

            results = await tool_search_ideas("nonexistent query")

        assert results == []


class TestGetOpenQuestionsTool:
    """Tests for get_open_questions tool - Slice 2.3."""

    @pytest.mark.asyncio
    async def test_returns_unanswered_questions(self, sample_ideas):
        """Should return questions with answered=FALSE."""
        from llm.indexing_tools import tool_get_open_questions

        results = await tool_get_open_questions(session="session-1")

        # Should find the question we created
        assert len(results) >= 1
        assert any("rate limiting" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_filters_by_session(self, sample_ideas):
        """Should filter by session."""
        from llm.indexing_tools import tool_get_open_questions

        results = await tool_get_open_questions(session="session-1")

        # All results should be from session-1
        for r in results:
            assert r.get("session") == "session-1" or "session" not in r

    @pytest.mark.asyncio
    async def test_orders_by_recency(self, sample_ideas):
        """Should order by recency (most recent first)."""
        from llm.indexing_tools import tool_get_open_questions

        results = await tool_get_open_questions(session="session-1")

        # Results should be ordered by source_line descending (as proxy for recency)
        if len(results) > 1:
            assert results[0]["source_line"] >= results[-1]["source_line"]

    @pytest.mark.asyncio
    async def test_returns_id_content_source_line(self, sample_ideas):
        """Should return id, content, and source_line."""
        from llm.indexing_tools import tool_get_open_questions

        results = await tool_get_open_questions(session="session-1")

        if results:
            assert "id" in results[0]
            assert "content" in results[0]
            assert "source_line" in results[0]

    @pytest.mark.asyncio
    async def test_excludes_answered_questions(self, test_db, sample_ideas):
        """Should exclude questions that have been answered."""
        from llm.indexing_tools import tool_get_open_questions
        from db.connection import get_db

        # Mark the question as answered
        db = get_db()
        db.execute("UPDATE ideas SET answered = TRUE WHERE intent = 'question'")
        db.commit()
        db.close()

        results = await tool_get_open_questions(session="session-1")

        # Should not find any questions
        assert len(results) == 0


class TestGetOpenTodosTool:
    """Tests for get_open_todos tool - Slice 2.4."""

    @pytest.fixture
    def sample_todos(self, test_db):
        """Create sample todo ideas."""
        from db.connection import get_db

        db = get_db()

        # Create a span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Test', 'Test', 1, 0)
        """)
        span_id = cursor.lastrowid

        # Create todos
        todos = [
            ("TODO: Add error handling", False, "session-1", 10),
            ("TODO: Write tests", False, "session-1", 20),
            ("TODO: Completed task", True, "session-1", 5),
            ("TODO: Other session", False, "session-2", 15),
        ]

        for content, completed, session, line in todos:
            db.execute("""
                INSERT INTO ideas (content, intent, completed, source_file, source_line, span_id, session)
                VALUES (?, 'todo', ?, '/test.jsonl', ?, ?, ?)
            """, (content, completed, line, span_id, session))

        db.commit()
        db.close()

    @pytest.mark.asyncio
    async def test_returns_incomplete_todos(self, sample_todos):
        """Should return todos that aren't completed."""
        from llm.indexing_tools import tool_get_open_todos

        results = await tool_get_open_todos(session="session-1")

        # Should find the incomplete todos
        assert len(results) == 2
        assert all("completed" not in r["content"].lower() or not r.get("completed", False)
                   for r in results)

    @pytest.mark.asyncio
    async def test_filters_by_session(self, sample_todos):
        """Should filter by session."""
        from llm.indexing_tools import tool_get_open_todos

        results = await tool_get_open_todos(session="session-1")

        # Should not include session-2 todo
        assert all("Other session" not in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_orders_by_recency(self, sample_todos):
        """Should order by recency."""
        from llm.indexing_tools import tool_get_open_todos

        results = await tool_get_open_todos(session="session-1")

        # Higher source_line = more recent
        if len(results) > 1:
            assert results[0]["source_line"] >= results[-1]["source_line"]


class TestGetCurrentSpanTool:
    """Tests for get_current_span tool - Slice 2.5."""

    @pytest.mark.asyncio
    async def test_returns_most_recent_span(self, sample_ideas):
        """Should return most recent span for session."""
        from llm.indexing_tools import tool_get_current_span

        result = await tool_get_current_span(session="test-session")

        assert result is not None
        assert result["name"] == "Test Span"

    @pytest.mark.asyncio
    async def test_includes_name_summary_start_line(self, sample_ideas):
        """Should include name, summary, and start_line."""
        from llm.indexing_tools import tool_get_current_span

        result = await tool_get_current_span(session="test-session")

        assert "name" in result
        assert "summary" in result
        assert "start_line" in result

    @pytest.mark.asyncio
    async def test_returns_none_for_new_session(self, test_db):
        """Should return None for session with no spans."""
        from llm.indexing_tools import tool_get_current_span

        result = await tool_get_current_span(session="nonexistent-session")

        assert result is None


class TestListSessionSpansTool:
    """Tests for list_session_spans tool - Slice 2.6."""

    @pytest.fixture
    def multiple_spans(self, test_db):
        """Create multiple spans for a session."""
        from db.connection import get_db

        db = get_db()

        # Create hierarchical spans
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES ('session-1', 'Root Span', 'The root', 1, 0, NULL)
        """)
        root_id = cursor.lastrowid

        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES ('session-1', 'Child Span', 'A child', 50, 1, ?)
        """, (root_id,))
        child_id = cursor.lastrowid

        db.commit()
        db.close()

        return {"root": root_id, "child": child_id}

    @pytest.mark.asyncio
    async def test_returns_all_spans_for_session(self, multiple_spans):
        """Should return all spans for the session."""
        from llm.indexing_tools import tool_list_session_spans

        results = await tool_list_session_spans(session="session-1")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_ordered_by_start_line(self, multiple_spans):
        """Should be ordered by start_line."""
        from llm.indexing_tools import tool_list_session_spans

        results = await tool_list_session_spans(session="session-1")

        # Root span (line 1) should come before child (line 50)
        assert results[0]["start_line"] < results[1]["start_line"]

    @pytest.mark.asyncio
    async def test_includes_hierarchy(self, multiple_spans):
        """Should include parent_id and depth."""
        from llm.indexing_tools import tool_list_session_spans

        results = await tool_list_session_spans(session="session-1")

        # Find the child span
        child = next(r for r in results if r["name"] == "Child Span")
        assert "parent_id" in child
        assert "depth" in child
        assert child["depth"] == 1


class TestSearchEntitiesTool:
    """Tests for search_entities tool - Slice 2.7."""

    @pytest.fixture
    def sample_entities(self, test_db):
        """Create sample golden entities with mentions."""
        from db.connection import get_db
        import uuid

        db = get_db()

        # Create golden entities (using UUID for test simplicity)
        golden1_id = str(uuid.uuid4())
        golden2_id = str(uuid.uuid4())

        db.execute("""
            INSERT INTO golden_entities (id, canonical_name, metadata)
            VALUES (?, 'Anthropic', '{"type": "organization"}')
        """, (golden1_id,))

        db.execute("""
            INSERT INTO golden_entities (id, canonical_name, metadata)
            VALUES (?, 'Claude', '{"type": "product"}')
        """, (golden2_id,))

        # Create mentions
        for i in range(3):
            mention_id = str(uuid.uuid4())
            db.execute("""
                INSERT INTO entity_mentions (id, name, golden_id, source_file, source_line)
                VALUES (?, 'Anthropic', ?, '/test.jsonl', ?)
            """, (mention_id, golden1_id, i))

        mention_id = str(uuid.uuid4())
        db.execute("""
            INSERT INTO entity_mentions (id, name, golden_id, source_file, source_line)
            VALUES (?, 'Claude', ?, '/test.jsonl', 10)
        """, (mention_id, golden2_id))

        db.commit()
        db.close()

        return {"anthropic": golden1_id, "claude": golden2_id}

    @pytest.mark.asyncio
    async def test_fuzzy_matches_entity_names(self, sample_entities):
        """Should fuzzy match entity names."""
        from llm.indexing_tools import tool_search_entities

        # Slight misspelling should still match
        results = await tool_search_entities(name="Antropic")

        assert len(results) >= 1
        assert any(r["canonical_name"] == "Anthropic" for r in results)

    @pytest.mark.asyncio
    async def test_filters_by_type(self, sample_entities):
        """Should filter by type in metadata."""
        from llm.indexing_tools import tool_search_entities

        results = await tool_search_entities(name="Claude", type="product")

        assert len(results) >= 1
        assert all(r.get("type") == "product" for r in results)

    @pytest.mark.asyncio
    async def test_returns_golden_entities(self, sample_entities):
        """Should return golden entities."""
        from llm.indexing_tools import tool_search_entities

        results = await tool_search_entities(name="Anthropic")

        assert len(results) >= 1
        assert "id" in results[0]
        assert "canonical_name" in results[0]

    @pytest.mark.asyncio
    async def test_returns_mention_count(self, sample_entities):
        """Should return mention count per golden."""
        from llm.indexing_tools import tool_search_entities

        results = await tool_search_entities(name="Anthropic")

        anthropic = next(r for r in results if r["canonical_name"] == "Anthropic")
        assert anthropic["mention_count"] == 3


class TestGetRecentIdeasTool:
    """Tests for get_recent_ideas tool - Slice 2.8."""

    @pytest.mark.asyncio
    async def test_returns_ideas_ordered_by_recency(self, sample_ideas):
        """Should return ideas ordered by recency."""
        from llm.indexing_tools import tool_get_recent_ideas

        results = await tool_get_recent_ideas(session="session-1")

        # Higher source_line = more recent (proxy for created_at)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["source_line"] >= results[i + 1]["source_line"]

    @pytest.mark.asyncio
    async def test_filters_by_session(self, sample_ideas):
        """Should filter by session."""
        from llm.indexing_tools import tool_get_recent_ideas

        results = await tool_get_recent_ideas(session="session-1")

        # All should be from session-1
        assert all(r.get("session") == "session-1" for r in results)

    @pytest.mark.asyncio
    async def test_filters_by_intent(self, sample_ideas):
        """Should filter by intent."""
        from llm.indexing_tools import tool_get_recent_ideas

        results = await tool_get_recent_ideas(session="session-1", intent="decision")

        assert len(results) >= 1
        assert all(r["intent"] == "decision" for r in results)

    @pytest.mark.asyncio
    async def test_includes_content_intent_source_line(self, sample_ideas):
        """Should include content, intent, and source_line."""
        from llm.indexing_tools import tool_get_recent_ideas

        results = await tool_get_recent_ideas(session="session-1")

        if results:
            assert "content" in results[0]
            assert "intent" in results[0]
            assert "source_line" in results[0]

    @pytest.mark.asyncio
    async def test_respects_limit(self, sample_ideas):
        """Should respect the limit parameter."""
        from llm.indexing_tools import tool_get_recent_ideas

        results = await tool_get_recent_ideas(session="session-1", limit=1)

        assert len(results) <= 1
