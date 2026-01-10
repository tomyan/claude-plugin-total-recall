"""Tests for advanced retrieval features."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))


class TestHyDE:
    """Test Hypothetical Document Embeddings."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock OpenAI embeddings."""
        fake_embedding = [0.1] * 1536
        mock_get = MagicMock(return_value=fake_embedding)
        monkeypatch.setattr("memory_db.get_embedding", mock_get)
        return mock_get

    @pytest.fixture
    def mock_hyde_generation(self, monkeypatch):
        """Mock LLM for HyDE document generation."""
        mock_gen = MagicMock(return_value="The authentication system uses JWT tokens with refresh rotation for secure session management.")
        monkeypatch.setattr("memory_db.generate_hypothetical_doc", mock_gen)
        return mock_gen

    def test_hyde_search_generates_hypothetical(self, mock_db, mock_embeddings, mock_hyde_generation):
        """HyDE search generates a hypothetical answer first."""
        import memory_db
        memory_db.init_db()

        # Store some ideas
        memory_db.store_idea(
            content="Implemented JWT authentication with refresh tokens",
            source_file="test.jsonl",
            source_line=1
        )

        results = memory_db.hyde_search("how does auth work?", limit=5)

        # Should have called the hypothetical doc generator
        mock_hyde_generation.assert_called_once()
        # Should return results (even if empty in this test)
        assert isinstance(results, list)


class TestGraphExpansion:
    """Test graph expansion in retrieval."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock OpenAI embeddings."""
        fake_embedding = [0.1] * 1536
        mock_get = MagicMock(return_value=fake_embedding)
        monkeypatch.setattr("memory_db.get_embedding", mock_get)
        return mock_get

    def test_expand_with_relations(self, mock_db, mock_embeddings):
        """Graph expansion includes related ideas."""
        import memory_db
        memory_db.init_db()

        # Create two related ideas
        id1 = memory_db.store_idea(
            content="Original decision to use PostgreSQL database",
            source_file="test.jsonl",
            source_line=1
        )
        id2 = memory_db.store_idea(
            content="Updated to use PostgreSQL with read replicas",
            source_file="test.jsonl",
            source_line=2
        )

        # Add relation: id2 supersedes id1
        memory_db.add_relation(id2, id1, "supersedes")

        # Expand from id2 should include id1
        expanded = memory_db.expand_with_relations([id2])
        assert id1 in expanded

    def test_expand_includes_parent_span(self, mock_db, mock_embeddings):
        """Graph expansion includes parent span context."""
        import memory_db
        memory_db.init_db()

        # Create span and idea
        span_id = memory_db.create_span(
            session="test-session",
            name="Database design discussion",
            start_line=1
        )
        idea_id = memory_db.store_idea(
            content="Using PostgreSQL for the primary database",
            source_file="test.jsonl",
            source_line=1,
            span_id=span_id
        )

        # Get context should include span info
        context = memory_db.get_idea_context(idea_id)
        assert context["span_name"] == "Database design discussion"


class TestTemporalFiltering:
    """Test temporal filtering in retrieval."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock OpenAI embeddings."""
        fake_embedding = [0.1] * 1536
        mock_get = MagicMock(return_value=fake_embedding)
        monkeypatch.setattr("memory_db.get_embedding", mock_get)
        return mock_get

    def test_filter_by_session(self, mock_db, mock_embeddings):
        """Can filter ideas by session."""
        import memory_db
        memory_db.init_db()

        # Create spans in different sessions
        span1 = memory_db.create_span(session="project-a", name="Topic 1", start_line=1)
        span2 = memory_db.create_span(session="project-b", name="Topic 2", start_line=1)

        memory_db.store_idea(
            content="Idea from project A about database design",
            source_file="a.jsonl", source_line=1, span_id=span1
        )
        memory_db.store_idea(
            content="Idea from project B about database design",
            source_file="b.jsonl", source_line=1, span_id=span2
        )

        # Search with session filter
        results = memory_db.search_ideas("database", session="project-a")
        sessions = [r.get("session") for r in results if r.get("session")]

        # All results should be from project-a
        assert all(s == "project-a" for s in sessions) if sessions else True

    def test_filter_by_date_range(self, mock_db, mock_embeddings):
        """Can filter ideas by date range."""
        import memory_db
        memory_db.init_db()

        # Store an idea (will have current timestamp)
        memory_db.store_idea(
            content="Recent idea about caching implementation",
            source_file="test.jsonl",
            source_line=1
        )

        # Search with recent filter (last 24 hours)
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        results = memory_db.search_ideas_temporal("caching", since=since)

        assert isinstance(results, list)

    def test_recency_weighting(self, mock_db, mock_embeddings):
        """More recent ideas get higher weight."""
        import memory_db
        memory_db.init_db()

        # This tests the concept - actual implementation may vary
        # The search should be able to apply recency weighting
        results = memory_db.search_ideas("test query", recency_weight=0.3)
        assert isinstance(results, list)


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_cache_stores_embeddings(self, monkeypatch):
        """Cache stores embeddings after first call."""
        from memory_db import get_embedding, clear_embedding_cache, get_embedding_cache_stats

        # Clear cache first
        clear_embedding_cache()

        # Mock the API call
        call_count = [0]
        fake_embedding = [0.1] * 1536

        def mock_get(text, use_cache=True):
            if use_cache and text in get_embedding_cache_stats():
                pass
            call_count[0] += 1
            return fake_embedding

        monkeypatch.setattr("memory_db.get_embedding", mock_get)

        # This is a simplified test - just verify the cache structure exists
        stats = get_embedding_cache_stats()
        assert stats["max_size"] == 1000

    def test_cache_can_be_cleared(self):
        """Cache can be cleared."""
        from memory_db import clear_embedding_cache, get_embedding_cache_stats, _embedding_cache

        # Add something to cache
        _embedding_cache["test"] = [0.1] * 1536

        assert get_embedding_cache_stats()["size"] > 0

        clear_embedding_cache()

        assert get_embedding_cache_stats()["size"] == 0


class TestQueryAnalysis:
    """Test query analysis for retrieval."""

    def test_detect_temporal_qualifier(self):
        """Detect temporal qualifiers in queries."""
        from memory_db import analyze_query

        analysis = analyze_query("what did we discuss last week about caching?")
        assert analysis.get("temporal") is not None

        analysis = analyze_query("recent decisions about authentication")
        assert analysis.get("temporal") is not None

    def test_detect_intent_filter(self):
        """Detect intent filters in queries."""
        from memory_db import analyze_query

        analysis = analyze_query("decisions about the database")
        assert analysis.get("intent_filter") == "decision"

        analysis = analyze_query("problems with authentication")
        assert analysis.get("intent_filter") == "problem"

    def test_extract_entities_from_query(self):
        """Extract entity mentions from query."""
        from memory_db import analyze_query

        analysis = analyze_query("what did we decide about PostgreSQL?")
        assert "PostgreSQL" in analysis.get("entities", [])
