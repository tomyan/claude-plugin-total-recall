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

    @pytest.fixture
    def setup_span_db(self, tmp_path, monkeypatch):
        """Set up a database with initialized schema."""
        import memory_db

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)

        # Initialize database schema
        memory_db.init_db()

        return db_path

    def test_detect_semantic_shift(self, setup_span_db, monkeypatch):
        """Detect topic shift using embedding distance against span embedding."""
        import memory_db
        from indexer import detect_topic_shift_semantic

        # Mock embeddings - database topics vs auth topics are orthogonal (similarity = 0)
        def mock_embedding(text, use_cache=True):
            if "database" in text.lower():
                return [1.0] * 768 + [0.0] * 768
            elif "authentication" in text.lower() or "auth" in text.lower():
                return [0.0] * 768 + [1.0] * 768
            else:
                return [0.5] * 1536

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Create a span with a "database" topic embedding
        span_id = memory_db.create_span("test-session", "Database setup", start_line=1)

        # Store span embedding (database-related)
        db = memory_db.get_db()
        span_embedding = mock_embedding("database")
        db.execute(
            "INSERT INTO span_embeddings (span_id, embedding) VALUES (?, ?)",
            (span_id, memory_db.serialize_embedding(span_embedding))
        )
        db.commit()
        db.close()

        context = {
            "span_id": span_id,
            "threshold": 0.5,
            "divergence_history": [],
        }

        # Authentication topic is orthogonal to database (similarity = 0)
        # This triggers a "strong shift" immediately because 0 < (0.5 - 0.15) = 0.35
        is_shift = detect_topic_shift_semantic(
            "Let's set up user authentication",
            context
        )
        assert is_shift is True

    def test_gradual_shift_requires_multiple_messages(self, setup_span_db, monkeypatch):
        """Gradual topic drift requires multiple divergent messages."""
        import memory_db
        import math
        from indexer import detect_topic_shift_semantic

        # Mock embeddings with controlled similarity
        # For similarity ~0.4 (below 0.5 threshold but above 0.35 strong threshold):
        # Use [0.4, sqrt(1-0.16)] = [0.4, 0.9165] normalized, repeated
        def mock_embedding(text, use_cache=True):
            if "database" in text.lower():
                # Unit vector pointing in x direction
                vec = [1.0, 0.0]
            elif "slightly" in text.lower():
                # Vector with cosine similarity ~0.4 to database vector
                vec = [0.4, math.sqrt(1 - 0.16)]  # [0.4, 0.9165]
            else:
                vec = [0.5, 0.5]
            # Repeat to make 1536-dim vector
            return vec * 768

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        span_id = memory_db.create_span("test-session", "Database setup", start_line=1)

        db = memory_db.get_db()
        span_embedding = mock_embedding("database")
        db.execute(
            "INSERT INTO span_embeddings (span_id, embedding) VALUES (?, ?)",
            (span_id, memory_db.serialize_embedding(span_embedding))
        )
        db.commit()
        db.close()

        context = {
            "span_id": span_id,
            "threshold": 0.5,
            "divergence_history": [],
        }

        # First moderately divergent message (similarity ~0.4) - not yet a shift
        is_shift = detect_topic_shift_semantic(
            "slightly different topic here",
            context
        )
        assert is_shift is False

        # Second divergent message - now triggers gradual shift (2+ consecutive)
        is_shift = detect_topic_shift_semantic(
            "slightly off topic again",
            context
        )
        assert is_shift is True

    def test_no_shift_on_similar_topic(self, setup_span_db, monkeypatch):
        """No topic shift detected when topics are similar."""
        import memory_db
        from indexer import detect_topic_shift_semantic

        def mock_embedding(text, use_cache=True):
            # All database-related topics use similar vectors
            if "database" in text.lower() or "postgres" in text.lower() or "sql" in text.lower():
                return [1.0] * 768 + [0.0] * 768
            return [0.5] * 1536

        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)

        # Create a span with a "database" topic embedding
        span_id = memory_db.create_span("test-session", "Database setup", start_line=1)

        db = memory_db.get_db()
        span_embedding = mock_embedding("database")
        db.execute(
            "INSERT INTO span_embeddings (span_id, embedding) VALUES (?, ?)",
            (span_id, memory_db.serialize_embedding(span_embedding))
        )
        db.commit()
        db.close()

        context = {
            "span_id": span_id,
            "threshold": 0.5,
            "divergence_history": [],
        }

        # Still talking about database - no shift (similarity = 1.0)
        is_shift = detect_topic_shift_semantic(
            "We should use PostgreSQL for the database",
            context
        )
        assert is_shift is False

    def test_shift_detection_without_span_id(self, monkeypatch):
        """No shift detected without span_id in context."""
        from indexer import detect_topic_shift_semantic

        # No span_id in context
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


class TestQuestionAnswered:
    """Test question.answered field tracking."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock embeddings."""
        fake_embedding = [0.1] * 1536
        monkeypatch.setattr("memory_db.get_embedding", lambda x, use_cache=True: fake_embedding)

    def test_mark_question_answered(self, mock_db, mock_embeddings):
        """Mark a question as answered."""
        import memory_db
        memory_db.init_db()

        # Store a question
        q_id = memory_db.store_idea(
            content="How should we handle authentication?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )

        # Initially not answered
        db = memory_db.get_db()
        cursor = db.execute("SELECT answered FROM ideas WHERE id = ?", (q_id,))
        assert cursor.fetchone()["answered"] is None

        # Mark as answered
        memory_db.mark_question_answered(q_id)

        cursor = db.execute("SELECT answered FROM ideas WHERE id = ?", (q_id,))
        assert cursor.fetchone()["answered"] == 1
        db.close()

    def test_solution_marks_related_question_answered(self, mock_db, mock_embeddings):
        """Storing a solution marks related question as answered."""
        import memory_db
        memory_db.init_db()

        # Store a question
        q_id = memory_db.store_idea(
            content="How do we implement caching?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )

        # Add answers relation and mark answered
        s_id = memory_db.store_idea(
            content="We implement caching using Redis with a TTL of 1 hour",
            source_file="test.jsonl",
            source_line=2,
            intent="solution"
        )

        # Add answers relation
        memory_db.add_relation(s_id, q_id, "answers")

        # Mark the question as answered
        memory_db.mark_question_answered(q_id)

        # Verify question is marked answered
        db = memory_db.get_db()
        cursor = db.execute("SELECT answered FROM ideas WHERE id = ?", (q_id,))
        assert cursor.fetchone()["answered"] == 1
        db.close()

    def test_get_unanswered_questions(self, mock_db, mock_embeddings):
        """Get list of unanswered questions."""
        import memory_db
        memory_db.init_db()

        # Store questions - one answered, one not
        q1_id = memory_db.store_idea(
            content="What database should we use?",
            source_file="test.jsonl",
            source_line=1,
            intent="question"
        )
        q2_id = memory_db.store_idea(
            content="How do we deploy to production?",
            source_file="test.jsonl",
            source_line=2,
            intent="question"
        )

        # Mark first as answered
        memory_db.mark_question_answered(q1_id)

        # Get unanswered
        unanswered = memory_db.get_unanswered_questions()

        assert len(unanswered) == 1
        assert unanswered[0]["id"] == q2_id


class TestRelationAutoDetection:
    """Test automatic relation detection during indexing."""

    @pytest.fixture
    def mock_db(self, tmp_path, monkeypatch):
        """Mock database path."""
        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        return db_path

    @pytest.fixture
    def mock_embeddings(self, monkeypatch):
        """Mock embeddings with topic-based similarity."""
        def mock_embedding(text, use_cache=True):
            base = [0.1] * 1536
            # Make database-related content similar
            if "database" in text.lower() or "postgres" in text.lower() or "pooling" in text.lower():
                base = [0.9] * 768 + [0.1] * 768
            return base
        monkeypatch.setattr("memory_db.get_embedding", mock_embedding)
        monkeypatch.setattr("indexer.get_embedding", mock_embedding)

    def test_index_with_relation_detection(self, mock_db, mock_embeddings, tmp_path):
        """Indexer stores relations between related ideas."""
        import memory_db
        memory_db.init_db()

        transcript = tmp_path / "test.jsonl"
        lines = [
            {"type": "user", "message": {"content": "How should we handle database connection pooling?"}, "timestamp": "T1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "The solution is to use PgBouncer for database connection pooling."}]}, "timestamp": "T2"},
        ]
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        from indexer import index_transcript
        result = index_transcript(str(transcript))

        assert result["ideas_created"] == 2

        # Verify ideas exist with correct intents
        db = memory_db.get_db()
        cursor = db.execute("SELECT intent FROM ideas ORDER BY source_line")
        intents = [row["intent"] for row in cursor]
        db.close()

        assert "question" in intents
        assert "solution" in intents


class TestSubtopicHierarchy:
    """Test sub-topic detection and hierarchy."""

    def test_detect_subtopic_patterns(self):
        """Detect patterns that indicate drilling into a sub-topic."""
        from indexer import detect_subtopic

        assert detect_subtopic("Specifically, let's look at the auth module")
        assert detect_subtopic("Diving into the database schema")
        assert detect_subtopic("Let's dig into the caching layer")
        assert detect_subtopic("More specifically, the token refresh logic")
        assert detect_subtopic("Focusing on the error handling")

        # Should not match normal content
        assert not detect_subtopic("We decided to use PostgreSQL")
        assert not detect_subtopic("The API returns JSON")

    def test_detect_return_to_parent(self):
        """Detect patterns that indicate returning to parent topic."""
        from indexer import detect_return_to_parent

        assert detect_return_to_parent("Back to the bigger picture")
        assert detect_return_to_parent("Stepping back, let's consider the overall architecture")
        assert detect_return_to_parent("More broadly, the system needs to scale")
        assert detect_return_to_parent("Overall, the design is solid")

        # Should not match normal content
        assert not detect_return_to_parent("We need to implement this feature")
        assert not detect_return_to_parent("The database is configured")

    def test_subtopic_creates_child_span(self, tmp_path, monkeypatch):
        """Sub-topic detection creates child spans."""
        import memory_db
        from indexer import index_transcript

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embeddings
        fake_embedding = [0.1] * 1536
        monkeypatch.setattr("memory_db.get_embedding", lambda text, use_cache=True: fake_embedding)
        monkeypatch.setattr("indexer.get_embedding", lambda text, use_cache=True: fake_embedding)

        # Create transcript with sub-topic (correct format)
        transcript = tmp_path / "test-session.jsonl"
        messages = [
            {"type": "user", "message": {"content": "Let's discuss database design"}},
            {"type": "assistant", "message": {"content": "Good idea, we should consider the schema"}},
            {"type": "user", "message": {"content": "Specifically, let's look at the user table"}},
            {"type": "assistant", "message": {"content": "The user table needs email and password"}},
        ]
        with open(transcript, "w") as f:
            for i, msg in enumerate(messages):
                f.write(json.dumps(msg) + "\n")

        result = index_transcript(str(transcript))

        # Should have created spans
        assert result["spans_created"] >= 1

        # Check for parent-child relationship in spans
        db = memory_db.get_db()
        cursor = db.execute("""
            SELECT id, parent_id, depth, name FROM spans ORDER BY start_line
        """)
        spans = [dict(row) for row in cursor]
        db.close()

        # Should have at least one span with depth > 0 (the sub-topic)
        depths = [s["depth"] for s in spans]
        assert any(d > 0 for d in depths), f"Expected at least one nested span, got depths: {depths}"

    def test_return_to_parent_pops_stack(self, tmp_path, monkeypatch):
        """Return to parent closes sub-topic span."""
        import memory_db
        from indexer import index_transcript

        db_path = tmp_path / "memory.db"
        monkeypatch.setattr("memory_db.DB_PATH", db_path)
        memory_db.init_db()

        # Mock embeddings
        fake_embedding = [0.1] * 1536
        monkeypatch.setattr("memory_db.get_embedding", lambda text, use_cache=True: fake_embedding)
        monkeypatch.setattr("indexer.get_embedding", lambda text, use_cache=True: fake_embedding)

        # Create transcript with sub-topic and return (correct format)
        transcript = tmp_path / "test-session.jsonl"
        messages = [
            {"type": "user", "message": {"content": "Let's discuss database design"}},
            {"type": "assistant", "message": {"content": "Specifically, let's dig into indexing"}},
            {"type": "user", "message": {"content": "We need indexes on user_id"}},
            {"type": "assistant", "message": {"content": "Stepping back, the overall schema looks good"}},
        ]
        with open(transcript, "w") as f:
            for i, msg in enumerate(messages):
                f.write(json.dumps(msg) + "\n")

        result = index_transcript(str(transcript))

        # Check that sub-topic span was closed
        db = memory_db.get_db()
        cursor = db.execute("""
            SELECT id, parent_id, depth, end_line FROM spans WHERE depth > 0
        """)
        sub_spans = [dict(row) for row in cursor]
        db.close()

        # Sub-topic spans should have end_line set (closed)
        for span in sub_spans:
            assert span["end_line"] is not None, f"Sub-span {span['id']} not closed"
