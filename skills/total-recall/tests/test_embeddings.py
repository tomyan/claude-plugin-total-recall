"""Tests for batch embeddings - Slice 8."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def test_db():
    """Create a fresh test database for each test."""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    os.environ["TOTAL_RECALL_DB_PATH"] = db_path

    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith(('config', 'db.', 'embeddings')):
            del sys.modules[mod]

    from db.schema import init_db
    init_db()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


def create_test_ideas(db, count: int) -> list[int]:
    """Helper to create test ideas."""
    ids = []
    for i in range(count):
        cursor = db.execute("""
            INSERT INTO ideas (content, intent, source_file, source_line)
            VALUES (?, 'context', '/test.jsonl', ?)
        """, (f"Test idea {i}", i + 1))
        ids.append(cursor.lastrowid)
    db.commit()
    return ids


class TestBatchEmbeddings:
    """Tests for batch embedding functionality."""

    def test_embeds_ideas_in_batch(self, test_db):
        """Should embed multiple ideas in a single API call."""
        from embeddings.batch import embed_ideas
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 3)
        db.close()

        # Mock the embedding API
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.return_value = mock_embeddings
            result = embed_ideas(idea_ids)

        assert result == 3
        mock_get.assert_called_once()
        # Verify batch size
        call_args = mock_get.call_args[0][0]
        assert len(call_args) == 3

    def test_batches_large_sets(self, test_db):
        """Should batch ideas into chunks of 100."""
        from embeddings.batch import embed_ideas, BATCH_SIZE
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 150)
        db.close()

        mock_embeddings_100 = [[0.1] * 1536] * 100
        mock_embeddings_50 = [[0.1] * 1536] * 50

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.side_effect = [mock_embeddings_100, mock_embeddings_50]
            result = embed_ideas(idea_ids)

        assert result == 150
        assert mock_get.call_count == 2

    def test_skips_already_embedded(self, test_db):
        """Should skip ideas that already have embeddings."""
        from embeddings.batch import embed_ideas
        from embeddings.serialize import serialize_embedding
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 3)

        # Pre-embed first idea
        embedding = [0.5] * 1536
        db.execute("""
            INSERT INTO idea_embeddings (idea_id, embedding)
            VALUES (?, ?)
        """, (idea_ids[0], serialize_embedding(embedding)))
        db.commit()
        db.close()

        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.return_value = mock_embeddings
            result = embed_ideas(idea_ids)

        # Should only embed 2 new ideas
        assert result == 2
        call_args = mock_get.call_args[0][0]
        assert len(call_args) == 2

    def test_stores_embeddings_in_db(self, test_db):
        """Should store embeddings in idea_embeddings table."""
        from embeddings.batch import embed_ideas
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 2)
        db.close()

        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.return_value = mock_embeddings
            embed_ideas(idea_ids)

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM idea_embeddings")
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 2

    def test_handles_api_failure(self, test_db):
        """Should handle API failures gracefully."""
        from embeddings.batch import embed_ideas, EmbeddingError
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 2)
        db.close()

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.side_effect = Exception("API error")

            with pytest.raises(EmbeddingError):
                embed_ideas(idea_ids)

    def test_embeds_span_summary(self, test_db):
        """Should embed span summary when provided."""
        from embeddings.batch import embed_span
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth)
            VALUES ('session-1', 'Test Span', 'This is a test summary', 1, 0)
        """)
        span_id = cursor.lastrowid
        db.commit()
        db.close()

        mock_embedding = [0.1] * 1536

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.return_value = [mock_embedding]
            embed_span(span_id)

        db = get_db()
        cursor = db.execute("SELECT COUNT(*) as cnt FROM span_embeddings WHERE span_id = ?", (span_id,))
        count = cursor.fetchone()["cnt"]
        db.close()

        assert count == 1

    def test_empty_list_returns_zero(self, test_db):
        """Should return 0 for empty idea list."""
        from embeddings.batch import embed_ideas

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            result = embed_ideas([])

        assert result == 0
        mock_get.assert_not_called()

    def test_updates_existing_embedding(self, test_db):
        """Should update embedding if force=True."""
        from embeddings.batch import embed_ideas
        from embeddings.serialize import serialize_embedding
        from db.connection import get_db

        db = get_db()
        idea_ids = create_test_ideas(db, 1)

        # Pre-embed with old embedding
        old_embedding = [0.5] * 1536
        db.execute("""
            INSERT INTO idea_embeddings (idea_id, embedding)
            VALUES (?, ?)
        """, (idea_ids[0], serialize_embedding(old_embedding)))
        db.commit()
        db.close()

        new_embedding = [0.9] * 1536

        with patch('embeddings.batch.get_embeddings_batch_sync') as mock_get:
            mock_get.return_value = [new_embedding]
            # Force re-embed
            result = embed_ideas(idea_ids, force=True)

        assert result == 1
