"""Tests for working memory - Slice 11."""

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
        if mod.startswith(('config', 'db.', 'working_memory')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class TestWorkingMemory:
    """Tests for working memory activation."""

    def test_activates_mentioned_ideas(self, test_db):
        """Should activate ideas mentioned in content."""
        from working_memory import activate_ideas
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Use JWT for authentication', 'decision', '/test.jsonl', 1)
        """)
        idea_id = cursor.lastrowid
        db.commit()
        db.close()

        # Activate based on content mentioning JWT
        activate_ideas(
            session="session-1",
            content="We discussed JWT tokens",
            idea_ids=[idea_id]
        )

        db = get_db()
        cursor = db.execute("""
            SELECT activation FROM working_memory
            WHERE session = 'session-1' AND idea_id = ?
        """, (idea_id,))
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["activation"] > 0

    def test_decays_old_activations(self, test_db):
        """Should decay activations over time."""
        from working_memory import activate_ideas, decay_activations
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Test idea', 'context', '/test.jsonl', 1)
        """)
        idea_id = cursor.lastrowid
        db.commit()
        db.close()

        # Activate
        activate_ideas(session="session-1", content="Test", idea_ids=[idea_id])

        # Get initial activation
        db = get_db()
        cursor = db.execute("""
            SELECT activation FROM working_memory
            WHERE session = 'session-1' AND idea_id = ?
        """, (idea_id,))
        initial = cursor.fetchone()["activation"]
        db.close()

        # Decay
        decay_activations(session="session-1", factor=0.5)

        # Check decayed
        db = get_db()
        cursor = db.execute("""
            SELECT activation FROM working_memory
            WHERE session = 'session-1' AND idea_id = ?
        """, (idea_id,))
        decayed = cursor.fetchone()["activation"]
        db.close()

        assert decayed < initial
        assert decayed == pytest.approx(initial * 0.5, rel=0.01)

    def test_prunes_low_activations(self, test_db):
        """Should remove very low activations."""
        from working_memory import activate_ideas, decay_activations, prune_activations
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Test idea', 'context', '/test.jsonl', 1)
        """)
        idea_id = cursor.lastrowid
        db.commit()
        db.close()

        # Activate with low value
        activate_ideas(session="session-1", content="", idea_ids=[idea_id])

        # Decay multiple times
        for _ in range(10):
            decay_activations(session="session-1", factor=0.5)

        # Prune
        prune_activations(session="session-1", threshold=0.01)

        # Should be removed
        db = get_db()
        cursor = db.execute("""
            SELECT COUNT(*) as cnt FROM working_memory
            WHERE session = 'session-1'
        """)
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 0

    def test_respects_session_scope(self, test_db):
        """Should only affect ideas in the specified session."""
        from working_memory import activate_ideas, decay_activations
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Idea 1', 'context', '/test.jsonl', 1)
        """)
        idea1 = cursor.lastrowid
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Idea 2', 'context', '/test.jsonl', 2)
        """)
        idea2 = cursor.lastrowid
        db.commit()
        db.close()

        # Activate in different sessions
        activate_ideas(session="session-1", content="", idea_ids=[idea1])
        activate_ideas(session="session-2", content="", idea_ids=[idea2])

        # Decay only session-1
        decay_activations(session="session-1", factor=0.5)

        # Check session-2 unaffected
        db = get_db()
        cursor = db.execute("""
            SELECT activation FROM working_memory
            WHERE session = 'session-2' AND idea_id = ?
        """, (idea2,))
        row = cursor.fetchone()
        db.close()

        assert row["activation"] == 1.0

    def test_get_active_ideas(self, test_db):
        """Should retrieve currently active ideas."""
        from working_memory import activate_ideas, get_active_ideas
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES ('Active idea', 'decision', '/test.jsonl', 1)
        """)
        idea_id = cursor.lastrowid
        db.commit()
        db.close()

        activate_ideas(session="session-1", content="", idea_ids=[idea_id])

        active = get_active_ideas(session="session-1", limit=10)

        assert len(active) == 1
        assert active[0]["content"] == "Active idea"
