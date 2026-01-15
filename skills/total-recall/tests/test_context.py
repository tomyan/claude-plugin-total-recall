"""Tests for context builder - Slice 2."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_db():
    """Create a fresh test database for each test."""
    # Create temp file for test database
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    # Point config to test database
    os.environ["TOTAL_RECALL_DB_PATH"] = db_path

    # Clear any cached imports that may have loaded the old config
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith(('config', 'db.', 'context')):
            del sys.modules[mod]

    # Initialize fresh database
    from db.schema import init_db
    init_db()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestContextBuilder:
    """Tests for the context builder."""

    def test_returns_empty_hierarchy_for_new_session(self, test_db):
        """New session with no spans should return empty hierarchy."""
        from context import build_context

        ctx = build_context(session="new-session-123", span_id=None)

        assert ctx["project"] is None
        assert ctx["topic"] is None
        assert ctx["parent_spans"] == []
        assert ctx["current_span"] is None

    def test_returns_span_info_when_span_exists(self, test_db):
        """Should return current span info when span exists."""
        from context import build_context
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Test Span', 'A test span', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=span_id)

        assert ctx["current_span"] is not None
        assert ctx["current_span"]["name"] == "Test Span"
        assert ctx["current_span"]["summary"] == "A test span"

    def test_includes_topic_when_span_has_topic(self, test_db):
        """Should include topic when span is linked to one."""
        from context import build_context
        from db.connection import get_db

        db = get_db()

        # Create topic
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Authentication', 'authentication', 'User auth system')
        """)
        topic_id = cursor.lastrowid

        # Create span linked to topic
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, topic_id)
            VALUES ('session-1', 'Login Flow', 'Implementing login', 1, 0, ?)
        """, (topic_id,))
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=span_id)

        assert ctx["topic"] is not None
        assert ctx["topic"]["name"] == "Authentication"
        assert ctx["topic"]["summary"] == "User auth system"

    def test_includes_project_when_topic_has_project(self, test_db):
        """Should include project when topic is linked to one."""
        from context import build_context
        from db.connection import get_db

        db = get_db()

        # Create project
        cursor = db.execute("""
            INSERT INTO projects (name, description)
            VALUES ('MyApp', 'A sample application')
        """)
        project_id = cursor.lastrowid

        # Create topic linked to project
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary, project_id)
            VALUES ('Auth', 'auth', 'Authentication system', ?)
        """, (project_id,))
        topic_id = cursor.lastrowid

        # Create span linked to topic
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, topic_id)
            VALUES ('session-1', 'JWT Setup', 'Setting up JWT', 1, 0, ?)
        """, (topic_id,))
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=span_id)

        assert ctx["project"] is not None
        assert ctx["project"]["name"] == "MyApp"
        assert ctx["project"]["description"] == "A sample application"

    def test_includes_parent_spans_when_nested(self, test_db):
        """Should include parent spans when span is nested."""
        from context import build_context
        from db.connection import get_db

        db = get_db()

        # Create parent span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Parent Span', 'Top level', 1, 0)
        """)
        parent_id = cursor.lastrowid

        # Create child span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES ('session-1', 'Child Span', 'Nested work', 10, 1, ?)
        """, (parent_id,))
        child_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=child_id)

        assert len(ctx["parent_spans"]) == 1
        assert ctx["parent_spans"][0]["name"] == "Parent Span"
        assert ctx["current_span"]["name"] == "Child Span"

    def test_includes_deeply_nested_parent_chain(self, test_db):
        """Should include full parent chain for deeply nested spans."""
        from context import build_context
        from db.connection import get_db

        db = get_db()

        # Create grandparent span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Grandparent', 'Level 0', 1, 0)
        """)
        grandparent_id = cursor.lastrowid

        # Create parent span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES ('session-1', 'Parent', 'Level 1', 10, 1, ?)
        """, (grandparent_id,))
        parent_id = cursor.lastrowid

        # Create child span
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES ('session-1', 'Child', 'Level 2', 20, 2, ?)
        """, (parent_id,))
        child_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=child_id)

        # Parent chain should be in order from root to immediate parent
        assert len(ctx["parent_spans"]) == 2
        assert ctx["parent_spans"][0]["name"] == "Grandparent"
        assert ctx["parent_spans"][1]["name"] == "Parent"
        assert ctx["current_span"]["name"] == "Child"

    def test_handles_missing_span_gracefully(self, test_db):
        """Should return empty context when span_id doesn't exist."""
        from context import build_context

        ctx = build_context(session="session-1", span_id=99999)

        assert ctx["current_span"] is None
        assert ctx["topic"] is None
        assert ctx["project"] is None
        assert ctx["parent_spans"] == []

    def test_handles_orphan_span_without_topic(self, test_db):
        """Should work with spans that have no topic assigned."""
        from context import build_context
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Orphan Span', 'No topic', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=span_id)

        assert ctx["current_span"]["name"] == "Orphan Span"
        assert ctx["topic"] is None
        assert ctx["project"] is None

    def test_returns_recent_spans_for_session(self, test_db):
        """Should include recent spans from session as context."""
        from context import build_context
        from db.connection import get_db

        db = get_db()

        # Create a few spans in the session
        for i in range(3):
            db.execute("""
                INSERT INTO spans (session, name, summary, start_line, depth)
                VALUES ('session-1', ?, ?, ?, 0)
            """, (f"Span {i}", f"Summary {i}", i * 10))
        db.commit()
        db.close()

        ctx = build_context(session="session-1", span_id=None)

        # Should have recent_spans with session history
        assert "recent_spans" in ctx
        assert len(ctx["recent_spans"]) == 3


class TestCrossSessionContext:
    """Tests for cross-session topic context - Slice 10."""

    def test_finds_related_topics_from_other_sessions(self, test_db):
        """Should find related topics from other sessions."""
        from context import get_related_topics
        from db.connection import get_db

        db = get_db()

        # Create topics in different sessions
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Authentication', 'authentication', 'User auth')
        """)
        topic1_id = cursor.lastrowid

        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Authorization', 'authorization', 'Access control')
        """)
        topic2_id = cursor.lastrowid

        # Link them via topic_links
        db.execute("""
            INSERT INTO topic_links (topic_id, related_topic_id, similarity)
            VALUES (?, ?, 0.8)
        """, (topic1_id, topic2_id))
        db.commit()
        db.close()

        related = get_related_topics(topic_id=topic1_id, limit=3)

        assert len(related) >= 1
        assert any(t["name"] == "Authorization" for t in related)

    def test_limits_to_top_n_related(self, test_db):
        """Should limit to top N related topics."""
        from context import get_related_topics
        from db.connection import get_db

        db = get_db()

        # Create main topic
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Main Topic', 'main topic', 'The main one')
        """)
        main_id = cursor.lastrowid

        # Create 5 related topics
        for i in range(5):
            cursor = db.execute("""
                INSERT INTO topics (name, canonical_name, summary)
                VALUES (?, ?, ?)
            """, (f"Related {i}", f"related {i}", f"Related topic {i}"))
            related_id = cursor.lastrowid
            db.execute("""
                INSERT INTO topic_links (topic_id, related_topic_id, similarity)
                VALUES (?, ?, ?)
            """, (main_id, related_id, 0.9 - i * 0.1))

        db.commit()
        db.close()

        related = get_related_topics(topic_id=main_id, limit=3)

        assert len(related) == 3
        # Should be sorted by similarity (highest first)
        assert related[0]["name"] == "Related 0"

    def test_excludes_current_topic(self, test_db):
        """Should not include the current topic in related list."""
        from context import get_related_topics
        from db.connection import get_db

        db = get_db()

        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Topic A', 'topic a', 'First topic')
        """)
        topic_id = cursor.lastrowid
        db.commit()
        db.close()

        related = get_related_topics(topic_id=topic_id, limit=3)

        # Current topic should not appear in related
        assert not any(t.get("id") == topic_id for t in related)

    def test_handles_no_related_topics(self, test_db):
        """Should return empty list when no related topics."""
        from context import get_related_topics
        from db.connection import get_db

        db = get_db()

        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Lonely Topic', 'lonely topic', 'No friends')
        """)
        topic_id = cursor.lastrowid
        db.commit()
        db.close()

        related = get_related_topics(topic_id=topic_id, limit=3)

        assert related == []

    def test_handles_none_topic_id(self, test_db):
        """Should return empty list when topic_id is None."""
        from context import get_related_topics

        related = get_related_topics(topic_id=None, limit=3)

        assert related == []
