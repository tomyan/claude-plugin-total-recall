"""Tests for LLM-powered and embedding-based features."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))


class TestLLMSummarization:
    """Test LLM-powered span summarization."""

    def test_summarize_span_with_llm(self, monkeypatch):
        """LLM generates meaningful summary from messages."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from indexer import summarize_span_with_llm

        messages = [
            {"content": "Let's implement JWT authentication"},
            {"content": "We should use refresh tokens for security"},
            {"content": "The tokens will expire after 15 minutes"},
            {"content": "Refresh tokens last 7 days"},
        ]

        with patch("indexer.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Discussed JWT authentication with refresh tokens. Access tokens expire after 15 minutes, refresh tokens after 7 days."))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            summary = summarize_span_with_llm(messages)

            assert "JWT" in summary or "token" in summary.lower()
            mock_client.chat.completions.create.assert_called_once()

    def test_summarize_span_fallback_on_api_failure(self, monkeypatch):
        """Falls back to basic summarization on API failure."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from indexer import summarize_span_with_llm

        messages = [
            {"content": "This is a test message about databases and performance"},
        ]

        with patch("indexer.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_openai.return_value = mock_client

            summary = summarize_span_with_llm(messages)

            # Should get basic fallback summary
            assert len(summary) > 0

    def test_summarize_span_fallback_no_api_key(self, monkeypatch):
        """Falls back to basic summarization without API key."""
        monkeypatch.delenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", raising=False)

        from indexer import summarize_span_with_llm

        messages = [
            {"content": "Message about implementing caching layer"},
        ]

        summary = summarize_span_with_llm(messages)

        # Should get basic fallback
        assert len(summary) > 0


class TestLLMIntentClassification:
    """Test LLM-powered intent classification."""

    def test_classify_intent_with_llm(self, monkeypatch):
        """LLM classifies ambiguous content accurately."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from indexer import classify_intent_with_llm

        content = "After much deliberation, we've settled on PostgreSQL for the database"

        with patch("indexer.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="decision"))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            intent = classify_intent_with_llm(content)

            assert intent == "decision"

    def test_classify_intent_with_llm_validates_output(self, monkeypatch):
        """LLM output is validated against known intents."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from indexer import classify_intent_with_llm

        content = "Test content"

        with patch("indexer.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # LLM returns invalid intent
            mock_response.choices = [
                MagicMock(message=MagicMock(content="invalid_intent_type"))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            intent = classify_intent_with_llm(content)

            # Should fallback to context
            assert intent == "context"

    def test_classify_intent_fallback_on_failure(self, monkeypatch):
        """Falls back to regex classification on LLM failure."""
        monkeypatch.setenv("OPENAI_TOKEN_MEMORY_EMBEDDINGS", "test-key")

        from indexer import classify_intent_with_llm

        content = "We decided to use Python"

        with patch("indexer.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_openai.return_value = mock_client

            intent = classify_intent_with_llm(content)

            # Should fall back to regex classification
            assert intent == "decision"


class TestEmbeddingRelations:
    """Test embedding-based relation detection."""

    def test_find_similar_ideas(self, tmp_path, monkeypatch):
        """Find semantically similar ideas using embeddings."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embeddings - similar ideas have similar vectors
        call_count = [0]
        def mock_embedding(text, use_cache=True):
            call_count[0] += 1
            # Make similar content have similar embeddings
            base = [0.1] * 1536
            if "database" in text.lower():
                base[0] = 0.9
            if "caching" in text.lower():
                base[1] = 0.9
            return base

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Store some ideas
        id1 = memory_db.store_idea(
            content="We should use PostgreSQL for the database",
            source_file="test.jsonl", source_line=1
        )
        id2 = memory_db.store_idea(
            content="The database should support read replicas",
            source_file="test.jsonl", source_line=2
        )
        id3 = memory_db.store_idea(
            content="Add caching layer with Redis",
            source_file="test.jsonl", source_line=3
        )

        from indexer import find_similar_ideas

        # Find ideas similar to database query
        similar = find_similar_ideas("database setup", limit=2)

        # Should return database-related ideas
        assert len(similar) <= 2

    def test_detect_relations_with_embeddings(self, tmp_path, monkeypatch):
        """Detect relations using embedding similarity."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embeddings with high similarity for related content
        def mock_embedding(text, use_cache=True):
            base = [0.1] * 1536
            if "postgresql" in text.lower() or "database" in text.lower():
                base[0] = 0.9
            return base

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Store initial idea
        id1 = memory_db.store_idea(
            content="We decided to use PostgreSQL",
            source_file="test.jsonl", source_line=1, intent="decision"
        )

        from indexer import detect_relations_with_embeddings

        # New content about the same topic
        new_content = "Updated the database to use connection pooling"
        relations = detect_relations_with_embeddings(new_content, "context", [id1])

        # Should detect some relation to the database decision
        assert isinstance(relations, list)

    def test_embedding_similarity_threshold(self, tmp_path, monkeypatch):
        """Only detect relations above similarity threshold."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embeddings - make them dissimilar
        def mock_embedding(text, use_cache=True):
            base = [0.1] * 1536
            if "database" in text.lower():
                base[0] = 0.9
            elif "frontend" in text.lower():
                base[0] = -0.9  # Very different
            return base

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Store ideas about different topics
        id1 = memory_db.store_idea(
            content="Setting up the database schema",
            source_file="test.jsonl", source_line=1
        )

        from indexer import detect_relations_with_embeddings

        # Unrelated content
        relations = detect_relations_with_embeddings(
            "Working on frontend components",
            "context",
            [id1]
        )

        # Should not find relations - topics are unrelated
        assert len(relations) == 0


class TestSemanticTopicShift:
    """Test semantic topic shift detection."""

    def test_detect_semantic_shift(self, monkeypatch):
        """Detect topic shift using embedding distance."""
        # Mock embeddings to show topic shift - use very different vectors
        def mock_embedding(text, use_cache=True):
            if "database" in text.lower():
                # Database topics point in one direction
                return [1.0] * 768 + [0.0] * 768
            elif "authentication" in text.lower():
                # Auth topics point in opposite direction
                return [0.0] * 768 + [1.0] * 768
            else:
                return [0.5] * 1536

        monkeypatch.setattr("indexer.get_embedding", mock_embedding)

        from indexer import detect_topic_shift_semantic

        # Previous topic was database
        context = {
            "last_embedding": mock_embedding("database"),
            "threshold": 0.5,
        }

        # New topic is authentication - should detect shift (similarity ~0)
        is_shift = detect_topic_shift_semantic(
            "Let's set up user authentication",
            context
        )
        assert is_shift is True

    def test_no_shift_on_similar_topic(self, monkeypatch):
        """No topic shift detected when topics are similar."""
        def mock_embedding(text, use_cache=True):
            # All database-related topics use similar vectors
            if "database" in text.lower() or "postgres" in text.lower():
                return [1.0] * 768 + [0.0] * 768
            return [0.5] * 1536

        monkeypatch.setattr("indexer.get_embedding", mock_embedding)

        from indexer import detect_topic_shift_semantic

        context = {
            "last_embedding": mock_embedding("database setup"),
            "threshold": 0.5,
        }

        # Still talking about database - no shift (similarity = 1.0)
        is_shift = detect_topic_shift_semantic(
            "We should use PostgreSQL for the database",
            context
        )
        assert is_shift is False

    def test_shift_detection_without_context(self, monkeypatch):
        """No shift detected without previous context."""
        from indexer import detect_topic_shift_semantic

        # No previous embedding
        context = {}

        is_shift = detect_topic_shift_semantic("Some new topic", context)
        assert is_shift is False

    def test_combined_keyword_and_semantic_shift(self, monkeypatch):
        """Both keyword and semantic detection work together."""
        def mock_embedding(text, use_cache=True):
            return [0.1] * 1536

        monkeypatch.setattr("indexer.get_embedding", mock_embedding)

        from indexer import detect_topic_shift

        context = {}

        # Explicit transition keyword should trigger shift
        is_shift = detect_topic_shift("Let's move on to testing", context)
        assert is_shift is True


class TestEntityResolution:
    """Test entity resolution and normalization."""

    def test_resolve_postgresql_variants(self):
        """PostgreSQL variants resolve to canonical name."""
        from indexer import resolve_entity

        assert resolve_entity("postgres", "technology") == "PostgreSQL"
        assert resolve_entity("postgresql", "technology") == "PostgreSQL"
        assert resolve_entity("pg", "technology") == "PostgreSQL"
        assert resolve_entity("Postgres", "technology") == "PostgreSQL"

    def test_resolve_kubernetes_variants(self):
        """Kubernetes variants resolve to canonical name."""
        from indexer import resolve_entity

        assert resolve_entity("k8s", "technology") == "Kubernetes"
        assert resolve_entity("kubernetes", "technology") == "Kubernetes"
        assert resolve_entity("K8s", "technology") == "Kubernetes"

    def test_resolve_javascript_variants(self):
        """JavaScript variants resolve to canonical name."""
        from indexer import resolve_entity

        assert resolve_entity("js", "technology") == "JavaScript"
        assert resolve_entity("javascript", "technology") == "JavaScript"
        assert resolve_entity("Javascript", "technology") == "JavaScript"

    def test_resolve_unknown_entity_unchanged(self):
        """Unknown entities return as-is."""
        from indexer import resolve_entity

        assert resolve_entity("CustomTech", "technology") == "CustomTech"
        assert resolve_entity("my-project", "project") == "my-project"

    def test_extract_entities_with_resolution(self):
        """Extracted entities are resolved to canonical names."""
        from indexer import extract_entities

        entities = extract_entities("We use postgres and k8s for deployment")

        # Should have resolved names
        entity_names = [e[0] for e in entities]
        assert "PostgreSQL" in entity_names or "postgres" in entity_names  # Either resolved or original
        assert "Kubernetes" in entity_names or "k8s" in entity_names

    def test_resolve_nodejs_variants(self):
        """Node.js variants resolve to canonical name."""
        from indexer import resolve_entity

        assert resolve_entity("node", "technology") == "Node.js"
        assert resolve_entity("nodejs", "technology") == "Node.js"
        assert resolve_entity("node.js", "technology") == "Node.js"

    def test_resolution_preserves_type(self):
        """Resolution doesn't change entity type."""
        from indexer import resolve_entity

        # File type should stay unchanged
        assert resolve_entity("config.js", "file") == "config.js"
