"""Tests for the indexer module - topic tracking and idea extraction."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))


class TestTopicDetection:
    """Test topic shift detection."""

    def test_explicit_transition_detected(self):
        """Detect explicit topic transitions."""
        from indexer import detect_topic_shift

        # Explicit transitions
        assert detect_topic_shift("okay, let's move on to the database design", {}) is True
        assert detect_topic_shift("back to the authentication issue", {}) is True
        assert detect_topic_shift("now let's discuss the API", {}) is True
        assert detect_topic_shift("switching to frontend work", {}) is True

    def test_no_shift_for_continuation(self):
        """No shift detected for continuing discussion."""
        from indexer import detect_topic_shift

        assert detect_topic_shift("yes, that makes sense", {}) is False
        assert detect_topic_shift("can you explain more about that?", {}) is False
        assert detect_topic_shift("I agree with that approach", {}) is False

    def test_domain_change_detected(self):
        """Detect domain/technology shifts via embedding distance."""
        from indexer import detect_topic_shift

        # When embedding distance is high, it's a topic shift
        context = {"last_embedding": [0.1] * 1536, "threshold": 0.5}
        # Mock: if content mentions completely different domain
        # This would be detected by embedding comparison in real implementation
        # For now, test the explicit patterns
        assert detect_topic_shift("let's work on the React components now", {}) is True


class TestIntentClassification:
    """Test classifying message intent."""

    def test_classify_decision(self):
        """Identify decision statements."""
        from indexer import classify_intent

        assert classify_intent("We decided to use PostgreSQL for the database") == "decision"
        assert classify_intent("Going with JWT tokens for authentication") == "decision"
        assert classify_intent("I'll use Redis for caching") == "decision"

    def test_classify_question(self):
        """Identify questions."""
        from indexer import classify_intent

        assert classify_intent("How should we handle rate limiting?") == "question"
        assert classify_intent("What's the best approach for caching?") == "question"
        assert classify_intent("Should we use WebSockets or SSE?") == "question"

    def test_classify_problem(self):
        """Identify problem statements."""
        from indexer import classify_intent

        assert classify_intent("The issue is that connections are timing out") == "problem"
        assert classify_intent("The problem with this approach is scalability") == "problem"
        assert classify_intent("We're running into memory issues") == "problem"

    def test_classify_solution(self):
        """Identify solutions."""
        from indexer import classify_intent

        assert classify_intent("Fixed by increasing the connection pool size") == "solution"
        assert classify_intent("The solution is to add a retry mechanism") == "solution"
        assert classify_intent("Resolved this by using batch processing") == "solution"

    def test_classify_conclusion(self):
        """Identify conclusions/insights."""
        from indexer import classify_intent

        assert classify_intent("The key insight is that we need async processing") == "conclusion"
        assert classify_intent("In conclusion, the hybrid approach works best") == "conclusion"
        assert classify_intent("Learned that caching significantly improves performance") == "conclusion"

    def test_classify_todo(self):
        """Identify TODOs."""
        from indexer import classify_intent

        assert classify_intent("Need to implement error handling") == "todo"
        assert classify_intent("TODO: add unit tests for the API") == "todo"
        assert classify_intent("Should implement rate limiting next") == "todo"

    def test_classify_context(self):
        """Default to context for general statements."""
        from indexer import classify_intent

        assert classify_intent("The API uses REST endpoints for data access") == "context"
        assert classify_intent("Users authenticate via OAuth2") == "context"


class TestEntityExtraction:
    """Test extracting entities from content."""

    def test_extract_technology_entities(self):
        """Extract technology mentions."""
        from indexer import extract_entities

        entities = extract_entities("We're using PostgreSQL with Redis for caching")
        tech_names = [e[0] for e in entities if e[1] == "technology"]
        assert "PostgreSQL" in tech_names
        assert "Redis" in tech_names

    def test_extract_file_entities(self):
        """Extract file path mentions."""
        from indexer import extract_entities

        entities = extract_entities("Updated the src/api/auth.py file")
        file_names = [e[0] for e in entities if e[1] == "file"]
        assert "src/api/auth.py" in file_names

    def test_extract_concept_entities(self):
        """Extract concept mentions."""
        from indexer import extract_entities

        entities = extract_entities("Implementing rate limiting with a sliding window algorithm")
        concept_names = [e[0] for e in entities if e[1] == "concept"]
        assert "rate limiting" in concept_names or "sliding window" in concept_names


class TestConfidenceAssessment:
    """Test confidence scoring."""

    def test_firm_decision_high_confidence(self):
        """Firm decisions get high confidence."""
        from indexer import assess_confidence

        assert assess_confidence("We decided to use PostgreSQL", "decision") >= 0.8
        assert assess_confidence("The final solution is to use caching", "solution") >= 0.8

    def test_tentative_lower_confidence(self):
        """Tentative statements get lower confidence."""
        from indexer import assess_confidence

        assert assess_confidence("Maybe we should try Redis?", "decision") < 0.7
        assert assess_confidence("I think the issue might be memory", "problem") < 0.7

    def test_questions_moderate_confidence(self):
        """Questions get moderate confidence."""
        from indexer import assess_confidence

        conf = assess_confidence("How should we handle this?", "question")
        assert 0.4 <= conf <= 0.6


class TestRelationDetection:
    """Test relation detection between ideas."""

    def test_detect_supersession(self):
        """Detect when new idea supersedes old."""
        from indexer import detect_relations

        recent = [{"id": 1, "content": "Using PostgreSQL for the database", "intent": "decision"}]

        # Supersession with "instead"
        relations = detect_relations(
            "Using MySQL instead for better compatibility",
            "decision",
            recent
        )
        assert (1, "supersedes") in relations

    def test_detect_builds_on(self):
        """Detect when new idea builds on old."""
        from indexer import detect_relations

        recent = [{"id": 1, "content": "Implementing caching layer with Redis cluster", "intent": "decision"}]

        relations = detect_relations(
            "Additionally, we're adding caching invalidation for the Redis cluster",
            "decision",
            recent
        )
        assert (1, "builds_on") in relations

    def test_detect_solution_answers_question(self):
        """Detect when solution answers a question."""
        from indexer import detect_relations

        recent = [{"id": 1, "content": "How should we handle authentication tokens?", "intent": "question"}]

        relations = detect_relations(
            "Fixed the authentication issue by using refresh tokens",
            "solution",
            recent
        )
        assert (1, "answers") in relations

    def test_no_relation_for_unrelated(self):
        """No relation for unrelated content."""
        from indexer import detect_relations

        recent = [{"id": 1, "content": "Database schema design", "intent": "context"}]

        relations = detect_relations(
            "The frontend uses React components",
            "context",
            recent
        )
        assert len(relations) == 0


class TestIndexerIntegration:
    """Integration tests for the full indexer."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        monkeypatch.setattr("indexer.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock OpenAI embeddings."""
        fake_embedding = [0.1] * 1536
        mock_get = MagicMock(return_value=fake_embedding)
        monkeypatch.setattr("memory_db.get_embedding", mock_get)
        monkeypatch.setattr("indexer.get_embedding", mock_get)
        return mock_get

    @pytest.fixture
    def mock_llm(self, monkeypatch):
        """Mock LLM calls for summarization."""
        mock_summarize = MagicMock(return_value="Summary of the topic discussion")
        monkeypatch.setattr("indexer.summarize_span", mock_summarize)
        return mock_summarize

    def test_index_transcript_creates_spans(self, mock_db, mock_embeddings, mock_llm, tmp_path):
        """Indexing creates topic spans."""
        import memory_db
        from indexer import index_transcript

        memory_db.init_db()

        transcript = tmp_path / "session.jsonl"
        lines = [
            {"type": "user", "message": {"content": "Help me design a REST API for user management"}, "timestamp": "T1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll help you design a REST API. Let's start with the endpoints."}]}, "timestamp": "T2"},
            {"type": "user", "message": {"content": "okay, let's move on to database design"}, "timestamp": "T3"},
            {"type": "user", "message": {"content": "What schema should we use for the users table?"}, "timestamp": "T4"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        result = index_transcript(str(transcript))

        # Should detect topic shift at line 3
        assert result["spans_created"] >= 1

    def test_index_transcript_extracts_ideas_with_intent(self, mock_db, mock_embeddings, mock_llm, tmp_path):
        """Indexing extracts ideas with intent classification."""
        import memory_db
        from indexer import index_transcript

        memory_db.init_db()

        transcript = tmp_path / "session.jsonl"
        lines = [
            {"type": "user", "message": {"content": "We decided to use PostgreSQL for the database"}, "timestamp": "T1"},
            {"type": "user", "message": {"content": "How should we handle authentication?"}, "timestamp": "T2"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        result = index_transcript(str(transcript))

        # Check ideas were stored with intents
        db = memory_db.get_db()
        cursor = db.execute("SELECT content, intent FROM ideas ORDER BY source_line")
        rows = cursor.fetchall()
        db.close()

        assert len(rows) == 2
        assert rows[0]["intent"] == "decision"
        assert rows[1]["intent"] == "question"
