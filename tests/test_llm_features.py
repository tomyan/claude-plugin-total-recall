"""Tests for LLM-powered features."""

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
