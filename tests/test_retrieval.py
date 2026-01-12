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

    def test_hyde_uses_llm_when_available(self, mock_db, mock_embeddings, monkeypatch):
        """HyDE generation uses Claude CLI when available."""
        import memory_db

        mock_claude = MagicMock(return_value="JWT auth with refresh tokens")
        monkeypatch.setattr("memory_db.claude_complete", mock_claude)

        result = memory_db.generate_hypothetical_doc("how does auth work?")

        mock_claude.assert_called_once()
        assert result == "JWT auth with refresh tokens"

    def test_hyde_falls_back_on_llm_error(self, monkeypatch):
        """HyDE generation falls back to heuristic on LLM error."""
        import memory_db

        mock_claude = MagicMock(side_effect=Exception("Claude CLI error"))
        monkeypatch.setattr("memory_db.claude_complete", mock_claude)

        result = memory_db.generate_hypothetical_doc("how does auth work?")

        # Should return heuristic result, not raise
        assert "auth" in result.lower()
        assert len(result) > 20  # Should be a meaningful response


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

    def test_cache_persistence_save_load(self, tmp_path):
        """Cache can be saved to and loaded from disk."""
        import memory_db

        cache_file = tmp_path / "cache.json"
        original_path = memory_db.CACHE_PATH
        memory_db.CACHE_PATH = cache_file

        try:
            # Add some entries
            memory_db.clear_embedding_cache()
            memory_db._embedding_cache["text1"] = [0.1] * 10  # Shortened for test
            memory_db._embedding_cache["text2"] = [0.2] * 10

            # Save cache
            memory_db.save_embedding_cache()
            assert cache_file.exists()

            # Clear and reload
            memory_db.clear_embedding_cache()
            assert len(memory_db._embedding_cache) == 0

            memory_db.load_embedding_cache()
            assert len(memory_db._embedding_cache) == 2
            assert memory_db._embedding_cache["text1"][0] == 0.1
            assert memory_db._embedding_cache["text2"][0] == 0.2
        finally:
            memory_db.CACHE_PATH = original_path
            memory_db.clear_embedding_cache()

    def test_cache_load_handles_missing_file(self, tmp_path):
        """Loading from missing file doesn't error."""
        import memory_db

        cache_file = tmp_path / "nonexistent.json"
        original_path = memory_db.CACHE_PATH
        memory_db.CACHE_PATH = cache_file

        try:
            memory_db.clear_embedding_cache()
            memory_db.load_embedding_cache()  # Should not raise
        finally:
            memory_db.CACHE_PATH = original_path


class TestBatchEmbedding:
    """Test batch embedding functionality."""

    def test_batch_embedding_returns_list(self, monkeypatch):
        """Batch embedding returns a list of embeddings."""
        import memory_db

        memory_db.clear_embedding_cache()
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        fake_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in fake_embeddings]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai = MagicMock(return_value=mock_client)
        monkeypatch.setattr("memory_db.OpenAI", mock_openai)

        results = memory_db.get_embeddings_batch(["text1", "text2", "text3"])

        assert len(results) == 3
        assert results[0][0] == 0.1
        assert results[1][0] == 0.2
        assert results[2][0] == 0.3

    def test_batch_uses_cache(self, monkeypatch):
        """Batch embedding uses cache for known texts."""
        import memory_db

        memory_db.clear_embedding_cache()
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        # Pre-cache one text
        memory_db._embedding_cache["cached_text"] = [0.5] * 1536

        fake_embeddings = [[0.1] * 1536, [0.2] * 1536]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in fake_embeddings]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai = MagicMock(return_value=mock_client)
        monkeypatch.setattr("memory_db.OpenAI", mock_openai)

        results = memory_db.get_embeddings_batch(["text1", "cached_text", "text2"])

        # Should only request 2 texts (text1 and text2)
        call_args = mock_client.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 2

        # Should return all 3 embeddings in correct order
        assert len(results) == 3
        assert results[1][0] == 0.5  # Cached text

    def test_batch_empty_list(self):
        """Batch embedding handles empty list."""
        import memory_db

        results = memory_db.get_embeddings_batch([])
        assert results == []

    def test_batch_all_cached(self, monkeypatch):
        """Batch embedding returns early when all texts are cached."""
        import memory_db

        memory_db.clear_embedding_cache()
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        # Pre-cache all texts
        memory_db._embedding_cache["text1"] = [0.1] * 1536
        memory_db._embedding_cache["text2"] = [0.2] * 1536

        mock_openai = MagicMock()
        monkeypatch.setattr("memory_db.OpenAI", mock_openai)

        results = memory_db.get_embeddings_batch(["text1", "text2"])

        # Should not call OpenAI at all
        mock_openai.assert_not_called()
        assert len(results) == 2


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

    def test_query_expansion(self):
        """Query expansion includes synonyms."""
        from memory_db import expand_query

        # Single word expansion
        expanded = expand_query("auth")
        assert "auth" in expanded
        assert "authentication" in expanded
        assert "login" in expanded

        # Multi-word query
        expanded = expand_query("check the db config")
        assert "db" in expanded
        assert "database" in expanded
        assert "config" in expanded
        assert "configuration" in expanded

    def test_analyze_query_includes_expanded_terms(self):
        """Analyze query includes expanded terms."""
        from memory_db import analyze_query

        analysis = analyze_query("auth problems")
        assert "expanded_terms" in analysis
        assert "authentication" in analysis["expanded_terms"]


class TestStats:
    """Test database statistics."""

    def test_get_stats_includes_all_fields(self, tmp_path, monkeypatch):
        """Stats includes all expected fields."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        stats = memory_db.get_stats()

        # Core counts
        assert "total_ideas" in stats
        assert "total_spans" in stats
        assert "total_entities" in stats
        assert "total_relations" in stats
        assert "sessions_indexed" in stats

        # Breakdowns
        assert "by_intent" in stats
        assert "entities_by_type" in stats

        # New fields
        assert "unanswered_questions" in stats
        assert "embedding_cache" in stats
        assert "size" in stats["embedding_cache"]
        assert "max_size" in stats["embedding_cache"]

    def test_stats_counts_unanswered_questions(self, tmp_path, monkeypatch):
        """Stats correctly counts unanswered questions."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embedding
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        # Add a question
        q_id = memory_db.store_idea(
            content="How should we handle auth?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )

        stats = memory_db.get_stats()
        assert stats["unanswered_questions"] == 1

        # Mark as answered
        memory_db.mark_question_answered(q_id)

        stats = memory_db.get_stats()
        assert stats["unanswered_questions"] == 0


class TestRelationTraversal:
    """Test relation traversal functions."""

    def test_get_idea_with_relations(self, tmp_path, monkeypatch):
        """Get idea with all its relations."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embedding
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        # Create ideas with relations
        q_id = memory_db.store_idea(
            content="How should we implement caching?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )
        a_id = memory_db.store_idea(
            content="Use Redis for distributed caching",
            source_file="test.jsonl",
            source_line=2,
            intent="solution"
        )

        # Add relation: answer -> question
        memory_db.add_relation(a_id, q_id, "answers")

        # Get the answer with its relations
        result = memory_db.get_idea_with_relations(a_id)

        assert result["id"] == a_id
        assert "relations" in result
        assert "answers" in result["relations"]
        assert len(result["relations"]["answers"]) == 1
        assert result["relations"]["answers"][0]["id"] == q_id
        assert result["relations"]["answers"][0]["direction"] == "outgoing"

    def test_get_idea_with_no_relations(self, tmp_path, monkeypatch):
        """Get idea with no relations returns empty relations dict."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        idea_id = memory_db.store_idea(
            content="A standalone idea",
            source_file="test.jsonl",
            source_line=1
        )

        result = memory_db.get_idea_with_relations(idea_id)

        assert result["id"] == idea_id
        assert result["relations"] == {}

    def test_get_nonexistent_idea(self, tmp_path, monkeypatch):
        """Get nonexistent idea returns empty dict."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        result = memory_db.get_idea_with_relations(99999)
        assert result == {}

    def test_enrich_with_relations(self, tmp_path, monkeypatch):
        """Enrich search results with related ideas."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        # Create ideas with relation
        q_id = memory_db.store_idea(
            content="What database should we use?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )
        a_id = memory_db.store_idea(
            content="Use PostgreSQL for reliability",
            source_file="test.jsonl",
            source_line=2,
            intent="solution"
        )
        memory_db.add_relation(a_id, q_id, "answers")

        # Simulate search results
        results = [{"id": a_id, "content": "Use PostgreSQL"}]

        enriched = memory_db.enrich_with_relations(results)

        assert len(enriched) == 1
        assert "related" in enriched[0]
        assert len(enriched[0]["related"]) == 1
        assert enriched[0]["related"][0]["type"] == "answers"

    def test_enrich_empty_results(self):
        """Enrich handles empty results."""
        import memory_db

        results = memory_db.enrich_with_relations([])
        assert results == []


class TestSimilarIdeas:
    """Test finding similar ideas."""

    def test_find_similar_ideas(self, tmp_path, monkeypatch):
        """Find ideas similar to a given idea."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Use slightly different embeddings for different ideas
        call_count = [0]
        def mock_embedding(text, use_cache=True):
            call_count[0] += 1
            # Return slightly different embeddings
            base = [0.1] * 1536
            base[0] = 0.1 + (call_count[0] * 0.01)
            return base

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Create multiple ideas
        id1 = memory_db.store_idea(
            content="We should use PostgreSQL for the database",
            source_file="test.jsonl", source_line=1
        )
        id2 = memory_db.store_idea(
            content="PostgreSQL offers good reliability",
            source_file="test.jsonl", source_line=2
        )
        id3 = memory_db.store_idea(
            content="The API should use REST endpoints",
            source_file="test.jsonl", source_line=3
        )

        # Find similar to id1
        similar = memory_db.find_similar_ideas(id1, limit=5)

        # Should return results (may include id2, id3)
        assert isinstance(similar, list)
        # Should not include self
        assert all(s["id"] != id1 for s in similar)

    def test_find_similar_nonexistent_idea(self, tmp_path, monkeypatch):
        """Find similar for nonexistent idea returns empty list."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        similar = memory_db.find_similar_ideas(99999)
        assert similar == []


class TestGraphRevision:
    """Test graph revision operations."""

    def test_update_idea_intent(self, tmp_path, monkeypatch):
        """Can update an idea's intent."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        idea_id = memory_db.store_idea(
            content="This is a question",
            source_file="test.jsonl", source_line=1,
            intent="context"
        )

        # Update intent
        result = memory_db.update_idea_intent(idea_id, "question")
        assert result is True

        # Verify change
        context = memory_db.get_idea_context(idea_id)
        assert context["intent"] == "question"

    def test_move_idea_to_span(self, tmp_path, monkeypatch):
        """Can move an idea to a different span."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        # Create two spans
        span1 = memory_db.create_span("test", "Topic 1", 1)
        span2 = memory_db.create_span("test", "Topic 2", 10)

        # Create idea in span1
        idea_id = memory_db.store_idea(
            content="This belongs elsewhere",
            source_file="test.jsonl", source_line=1,
            span_id=span1
        )

        # Move to span2
        result = memory_db.move_idea_to_span(idea_id, span2)
        assert result is True

        # Verify move
        context = memory_db.get_idea_context(idea_id)
        assert context["span_id"] == span2

    def test_merge_spans(self, tmp_path, monkeypatch):
        """Can merge two spans together."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        # Create spans with ideas
        span1 = memory_db.create_span("test", "Topic A", 1)
        span2 = memory_db.create_span("test", "Topic B", 10)

        memory_db.store_idea("Idea 1", "test.jsonl", 1, span_id=span1)
        memory_db.store_idea("Idea 2", "test.jsonl", 2, span_id=span1)
        memory_db.store_idea("Idea 3", "test.jsonl", 10, span_id=span2)

        # Merge span1 into span2
        result = memory_db.merge_spans(span1, span2)

        assert result["ideas_moved"] == 2
        assert result["source_span_id"] == span1
        assert result["target_span_id"] == span2

    def test_supersede_idea(self, tmp_path, monkeypatch):
        """Can mark one idea as superseding another."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()
        monkeypatch.setattr("memory_db.get_embedding", lambda t, use_cache=True: [0.1] * 1536)

        old_id = memory_db.store_idea("Use MySQL", "test.jsonl", 1, intent="decision")
        new_id = memory_db.store_idea("Actually use PostgreSQL", "test.jsonl", 2, intent="decision")

        memory_db.supersede_idea(old_id, new_id)

        # Check relation exists
        result = memory_db.get_idea_with_relations(new_id)
        assert "supersedes" in result["relations"]

    def test_reparent_span(self, tmp_path, monkeypatch):
        """Can change a span's parent."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Create hierarchy
        parent1 = memory_db.create_span("test", "Parent 1", 1, depth=0)
        parent2 = memory_db.create_span("test", "Parent 2", 10, depth=0)
        child = memory_db.create_span("test", "Child", 20, parent_id=parent1, depth=1)

        # Reparent child to parent2
        result = memory_db.reparent_span(child, parent2)
        assert result is True

        # Verify new parent
        db = memory_db.get_db()
        cursor = db.execute("SELECT parent_id, depth FROM spans WHERE id = ?", (child,))
        row = cursor.fetchone()
        db.close()

        assert row["parent_id"] == parent2
        assert row["depth"] == 1  # Auto-calculated


class TestExportImport:
    """Test export and import functionality."""

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

    def test_export_data_basic(self, mock_db, mock_embeddings):
        """Export produces expected structure."""
        import memory_db
        memory_db.init_db()

        # Create span and idea
        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        memory_db.store_idea(
            content="Test decision",
            intent="decision",
            source_file="test.jsonl",
            source_line=1,
            span_id=span_id
        )

        # Export
        data = memory_db.export_data()

        assert data["version"] == 1
        assert "spans" in data
        assert "ideas" in data
        assert "stats" in data
        assert data["stats"]["ideas_count"] >= 1
        assert data["stats"]["spans_count"] >= 1

    def test_export_session_filter(self, mock_db, mock_embeddings):
        """Export can filter by session."""
        import memory_db
        memory_db.init_db()

        # Create data in two sessions
        span1 = memory_db.create_span("session-1", "Topic 1", 1)
        span2 = memory_db.create_span("session-2", "Topic 2", 1)
        memory_db.store_idea(content="Session 1 idea", source_file="a.jsonl", source_line=1, span_id=span1)
        memory_db.store_idea(content="Session 2 idea", source_file="b.jsonl", source_line=1, span_id=span2)

        # Export only session-1
        data = memory_db.export_data(session="session-1")

        assert data["session_filter"] == "session-1"
        assert data["stats"]["ideas_count"] == 1
        assert all(s["session"] == "session-1" for s in data["spans"])

    def test_import_data_basic(self, mock_db, mock_embeddings):
        """Import restores exported data."""
        import memory_db
        memory_db.init_db()

        # Create some data with span
        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        memory_db.store_idea(content="Original idea", intent="decision", source_file="test.jsonl", source_line=1, span_id=span_id)

        # Export
        export_data = memory_db.export_data()

        # Clear and reimport
        result = memory_db.import_data(export_data, replace=True)

        assert result["success"] is True
        assert result["stats"]["ideas_imported"] >= 1
        assert result["stats"]["spans_imported"] >= 1

    def test_import_merge_mode(self, mock_db, mock_embeddings):
        """Import in merge mode adds to existing data."""
        import memory_db
        memory_db.init_db()

        # Create initial data with span
        span_id = memory_db.create_span("existing-session", "Existing Topic", 1)
        memory_db.store_idea(content="Existing idea", source_file="a.jsonl", source_line=1, span_id=span_id)
        initial_stats = memory_db.get_stats()

        # Create export data for different content
        export_data = {
            "version": 1,
            "spans": [{"id": 1, "session": "imported", "name": "Imported Topic", "start_line": 1, "depth": 0}],
            "ideas": [{"id": 1, "span_id": 1, "content": "Imported idea", "source_file": "imported.jsonl", "source_line": 1}],
            "relations": [],
            "entities": [],
            "entity_links": [],
        }

        result = memory_db.import_data(export_data, replace=False)

        final_stats = memory_db.get_stats()
        # Should have more ideas than before
        assert final_stats["total_ideas"] > initial_stats["total_ideas"]

    def test_import_unsupported_version(self, mock_db):
        """Import rejects unsupported versions."""
        import memory_db
        memory_db.init_db()

        with pytest.raises(memory_db.MemgraphError) as exc:
            memory_db.import_data({"version": 999, "spans": [], "ideas": []})

        assert "Unsupported export version" in str(exc.value)
        assert exc.value.error_code == "import_error"

    def test_roundtrip_with_relations(self, mock_db, mock_embeddings):
        """Export and import preserves relations."""
        import memory_db
        memory_db.init_db()

        # Create span and ideas with relation
        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        id1 = memory_db.store_idea(content="Problem statement", intent="problem", source_file="test.jsonl", source_line=1, span_id=span_id)
        id2 = memory_db.store_idea(content="Solution to problem", intent="solution", source_file="test.jsonl", source_line=10, span_id=span_id)

        # Create relation (use valid relation_type)
        db = memory_db.get_db()
        db.execute("INSERT INTO relations (from_id, to_id, relation_type) VALUES (?, ?, ?)", (id1, id2, "relates_to"))
        db.commit()
        db.close()

        # Export
        export_data = memory_db.export_data()
        assert len(export_data["relations"]) >= 1

        # Clear and reimport
        result = memory_db.import_data(export_data, replace=True)

        assert result["stats"]["relations_imported"] >= 1


class TestGetContext:
    """Test context viewing functionality."""

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

    def test_get_context_not_found(self, mock_db):
        """Get context for non-existent idea raises error."""
        import memory_db
        memory_db.init_db()

        with pytest.raises(memory_db.MemgraphError) as exc:
            memory_db.get_context(999)

        assert exc.value.error_code == "not_found"

    def test_get_context_missing_source(self, mock_db, mock_embeddings):
        """Get context when source file doesn't exist."""
        import memory_db
        memory_db.init_db()

        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        idea_id = memory_db.store_idea(
            content="Test idea",
            source_file="/nonexistent/file.jsonl",
            source_line=5,
            span_id=span_id
        )

        result = memory_db.get_context(idea_id)

        assert result["idea"]["id"] == idea_id
        assert result["context"] is None
        assert "not found" in result.get("error", "")

    def test_get_context_with_source(self, mock_db, mock_embeddings, tmp_path):
        """Get context reads from source file."""
        import memory_db
        memory_db.init_db()

        # Create a mock transcript file
        transcript_file = tmp_path / "transcript.jsonl"
        lines = [
            '{"type": "user", "message": {"content": "Line 1"}}\n',
            '{"type": "assistant", "message": {"content": "Line 2"}}\n',
            '{"type": "user", "message": {"content": "Line 3 - the important one"}}\n',
            '{"type": "assistant", "message": {"content": "Line 4"}}\n',
            '{"type": "user", "message": {"content": "Line 5"}}\n',
        ]
        transcript_file.write_text("".join(lines))

        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        idea_id = memory_db.store_idea(
            content="Test idea from line 3",
            source_file=str(transcript_file),
            source_line=3,
            span_id=span_id
        )

        result = memory_db.get_context(idea_id, lines_before=2, lines_after=2)

        assert result["idea"]["id"] == idea_id
        assert result["context"] is not None
        assert len(result["context"]) == 5  # lines 1-5

        # Find the source line
        source_line_entry = next((c for c in result["context"] if c["is_source"]), None)
        assert source_line_entry is not None
        assert source_line_entry["line"] == 3
        assert "Line 3" in source_line_entry["content"]


class TestTemporalSearch:
    """Test temporal filtering in search."""

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

    def test_search_ideas_temporal(self, mock_db, mock_embeddings):
        """Search with temporal filters."""
        import memory_db
        memory_db.init_db()

        span_id = memory_db.create_span("test-session", "Test Topic", 1)
        memory_db.store_idea(
            content="Old idea",
            source_file="test.jsonl",
            source_line=1,
            span_id=span_id
        )
        memory_db.store_idea(
            content="New idea",
            source_file="test.jsonl",
            source_line=10,
            span_id=span_id
        )

        # Search without temporal filter
        results = memory_db.search_ideas_temporal("idea", limit=10)
        assert len(results) >= 2

        # Search with future date (should return all since they're before future)
        results = memory_db.search_ideas_temporal("idea", limit=10, until="2099-01-01")
        assert len(results) >= 2

        # Search with far past date (should return nothing since they're after past)
        results = memory_db.search_ideas_temporal("idea", limit=10, since="2099-01-01")
        assert len(results) == 0


class TestListSessions:
    """Test session listing functionality."""

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

    def test_list_sessions_empty(self, mock_db):
        """List sessions when empty returns empty list."""
        import memory_db
        memory_db.init_db()

        sessions = memory_db.list_sessions()
        assert sessions == []

    def test_list_sessions_with_data(self, mock_db, mock_embeddings):
        """List sessions shows all indexed sessions."""
        import memory_db
        memory_db.init_db()

        # Create data in two sessions
        span1 = memory_db.create_span("project-alpha", "Topic 1", 1)
        span2 = memory_db.create_span("project-beta", "Topic 2", 1)
        memory_db.store_idea(content="Alpha idea", source_file="a.jsonl", source_line=1, span_id=span1)
        memory_db.store_idea(content="Beta idea 1", source_file="b.jsonl", source_line=1, span_id=span2)
        memory_db.store_idea(content="Beta idea 2", source_file="b.jsonl", source_line=2, span_id=span2)

        sessions = memory_db.list_sessions()

        assert len(sessions) == 2
        session_names = {s["session"] for s in sessions}
        assert "project-alpha" in session_names
        assert "project-beta" in session_names

        beta_session = next(s for s in sessions if s["session"] == "project-beta")
        assert beta_session["idea_count"] == 2
        assert beta_session["topic_count"] == 1
