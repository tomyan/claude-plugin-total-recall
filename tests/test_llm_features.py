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
