"""Tests for action executor - Slices 5, 6, 7."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_db():
    """Create a fresh test database for each test."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    os.environ["TOTAL_RECALL_DB_PATH"] = db_path

    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith(('config', 'db.', 'executor', 'context')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class TestIdeaExecutor:
    """Tests for idea storage - Slice 5."""

    def test_stores_idea_with_correct_intent(self, test_db):
        """Should store idea with the specified intent type."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {
                "type": "decision",
                "content": "Using PostgreSQL for the database",
                "confidence": 0.9,
                "source_line": 42,
                "entities": ["PostgreSQL"]
            }
        ]

        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT * FROM ideas WHERE source_line = 42")
        idea = cursor.fetchone()
        db.close()

        assert idea is not None
        assert idea["intent"] == "decision"
        assert idea["content"] == "Using PostgreSQL for the database"
        assert idea["confidence"] == 0.9

    def test_links_idea_to_span(self, test_db):
        """Should link idea to the current span."""
        from executor import execute_ideas
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Test Span', 'Testing', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        items = [
            {
                "type": "context",
                "content": "Setting up the project",
                "source_line": 1
            }
        ]

        execute_ideas(items, span_id=span_id, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT span_id FROM ideas WHERE source_line = 1")
        idea = cursor.fetchone()
        db.close()

        assert idea["span_id"] == span_id

    def test_handles_duplicate_source_line(self, test_db):
        """Should update existing idea on duplicate source_line."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {
                "type": "decision",
                "content": "Original content",
                "source_line": 10
            }
        ]
        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        # Insert again with different content
        items[0]["content"] = "Updated content"
        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("""
            SELECT COUNT(*) as cnt FROM ideas
            WHERE source_file = '/test/transcript.jsonl' AND source_line = 10
        """)
        count = cursor.fetchone()["cnt"]

        cursor = db.execute("""
            SELECT content FROM ideas
            WHERE source_file = '/test/transcript.jsonl' AND source_line = 10
        """)
        idea = cursor.fetchone()
        db.close()

        assert count == 1
        assert idea["content"] == "Updated content"

    def test_records_confidence(self, test_db):
        """Should store the confidence value."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {
                "type": "conclusion",
                "content": "The API is stable",
                "confidence": 0.75,
                "source_line": 5
            }
        ]

        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT confidence FROM ideas WHERE source_line = 5")
        idea = cursor.fetchone()
        db.close()

        assert idea["confidence"] == 0.75

    def test_stores_multiple_ideas(self, test_db):
        """Should store multiple ideas from a batch."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {"type": "question", "content": "What about caching?", "source_line": 1},
            {"type": "decision", "content": "Use Redis", "source_line": 2},
            {"type": "todo", "content": "Implement cache layer", "source_line": 3},
        ]

        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM ideas WHERE source_file = '/test/transcript.jsonl'")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 3

    def test_stores_entities(self, test_db):
        """Should store entities linked to ideas."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {
                "type": "decision",
                "content": "Using React with TypeScript",
                "source_line": 1,
                "entities": ["React", "TypeScript"]
            }
        ]

        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("""
            SELECT e.name FROM entities e
            JOIN idea_entities ie ON e.id = ie.entity_id
            JOIN ideas i ON ie.idea_id = i.id
            WHERE i.source_line = 1
        """)
        entities = [row["name"] for row in cursor.fetchall()]
        db.close()

        assert "React" in entities
        assert "TypeScript" in entities

    def test_defaults_confidence_to_half(self, test_db):
        """Should default confidence to 0.5 if not provided."""
        from executor import execute_ideas
        from db.connection import get_db

        items = [
            {
                "type": "context",
                "content": "Just some context",
                "source_line": 1
            }
        ]

        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT confidence FROM ideas WHERE source_line = 1")
        idea = cursor.fetchone()
        db.close()

        assert idea["confidence"] == 0.5


class TestTopicExecutor:
    """Tests for topic/span updates - Slice 6."""

    def test_updates_span_name(self, test_db):
        """Should update span name from topic_update."""
        from executor import execute_topic_update
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Old Name', 'Old summary', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        topic_update = {
            "name": "New Name",
            "summary": "New summary"
        }

        execute_topic_update(topic_update, span_id=span_id)

        db = get_db()
        cursor = db.execute("SELECT name, summary FROM spans WHERE id = ?", (span_id,))
        span = cursor.fetchone()
        db.close()

        assert span["name"] == "New Name"
        assert span["summary"] == "New summary"

    def test_creates_child_span_on_topic_shift(self, test_db):
        """Should create child span when new_span is provided."""
        from executor import execute_new_span
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Parent Span', 'Parent', 1, 0)
        """)
        parent_id = cursor.lastrowid
        db.commit()
        db.close()

        new_span = {
            "name": "Child Span",
            "reason": "Topic shifted to new area"
        }

        child_id = execute_new_span(
            new_span,
            session="session-1",
            parent_id=parent_id,
            start_line=50
        )

        db = get_db()
        cursor = db.execute("SELECT * FROM spans WHERE id = ?", (child_id,))
        span = cursor.fetchone()
        db.close()

        assert span is not None
        assert span["name"] == "Child Span"
        assert span["parent_id"] == parent_id
        assert span["depth"] == 1

    def test_links_span_to_existing_topic(self, test_db):
        """Should link span to existing topic if name matches."""
        from executor import execute_topic_update
        from db.connection import get_db

        db = get_db()
        # Create topic
        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, summary)
            VALUES ('Authentication', 'authentication', 'Auth work')
        """)
        topic_id = cursor.lastrowid

        # Create span without topic
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Login', 'Login work', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        topic_update = {
            "name": "Authentication",
            "summary": "Working on auth"
        }

        execute_topic_update(topic_update, span_id=span_id)

        db = get_db()
        cursor = db.execute("SELECT topic_id FROM spans WHERE id = ?", (span_id,))
        span = cursor.fetchone()
        db.close()

        assert span["topic_id"] == topic_id

    def test_creates_new_topic_if_not_exists(self, test_db):
        """Should create new topic if name doesn't match existing."""
        from executor import execute_topic_update
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Test', 'Testing', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        topic_update = {
            "name": "Brand New Topic",
            "summary": "Something new"
        }

        execute_topic_update(topic_update, span_id=span_id)

        db = get_db()
        cursor = db.execute("SELECT topic_id FROM spans WHERE id = ?", (span_id,))
        span = cursor.fetchone()

        cursor = db.execute("SELECT name FROM topics WHERE id = ?", (span["topic_id"],))
        topic = cursor.fetchone()
        db.close()

        assert topic is not None
        assert topic["name"] == "Brand New Topic"


class TestRelationsExecutor:
    """Tests for relations between ideas - Slice 7."""

    def test_creates_relation_between_ideas(self, test_db):
        """Should create relation from new idea to existing idea."""
        from executor import execute_ideas, execute_relations
        from db.connection import get_db

        # Create existing idea
        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Original decision', 'decision', '/test/old.jsonl', 100)
        """)
        existing_id = cursor.lastrowid
        db.commit()
        db.close()

        # Create new idea
        items = [
            {"type": "decision", "content": "Updated decision", "source_line": 42}
        ]
        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        # Create relation
        relations = [
            {"from_line": 42, "to_idea_id": existing_id, "type": "supersedes"}
        ]
        execute_relations(relations, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("""
            SELECT r.relation_type, r.to_id
            FROM relations r
            JOIN ideas i ON r.from_id = i.id
            WHERE i.source_line = 42
        """)
        relation = cursor.fetchone()
        db.close()

        assert relation is not None
        assert relation["relation_type"] == "supersedes"
        assert relation["to_id"] == existing_id

    def test_handles_missing_source_idea(self, test_db):
        """Should skip relation if source idea doesn't exist."""
        from executor import execute_relations
        from db.connection import get_db

        relations = [
            {"from_line": 999, "to_idea_id": 1, "type": "builds_on"}
        ]

        # Should not raise
        execute_relations(relations, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM relations")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 0

    def test_handles_missing_target_idea(self, test_db):
        """Should skip relation if target idea doesn't exist."""
        from executor import execute_ideas, execute_relations
        from db.connection import get_db

        items = [{"type": "decision", "content": "Test", "source_line": 1}]
        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        relations = [
            {"from_line": 1, "to_idea_id": 99999, "type": "builds_on"}
        ]

        # Should not raise
        execute_relations(relations, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM relations")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 0

    def test_creates_multiple_relations(self, test_db):
        """Should create multiple relations."""
        from executor import execute_ideas, execute_relations
        from db.connection import get_db

        # Create target ideas
        db = get_db()
        db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Idea A', 'decision', '/old.jsonl', 1)
        """)
        db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Idea B', 'decision', '/old.jsonl', 2)
        """)
        db.commit()
        db.close()

        # Create source idea
        items = [{"type": "conclusion", "content": "Final answer", "source_line": 10}]
        execute_ideas(items, span_id=None, source_file="/test/transcript.jsonl")

        # Get target IDs
        db = get_db()
        cursor = db.execute("SELECT id FROM ideas WHERE source_file = '/old.jsonl' ORDER BY source_line")
        targets = [row["id"] for row in cursor.fetchall()]
        db.close()

        relations = [
            {"from_line": 10, "to_idea_id": targets[0], "type": "builds_on"},
            {"from_line": 10, "to_idea_id": targets[1], "type": "relates_to"}
        ]
        execute_relations(relations, source_file="/test/transcript.jsonl")

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM relations")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 2
