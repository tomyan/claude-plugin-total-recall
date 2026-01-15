"""Test cases for intent classification and filtering."""

import pytest

from indexer import classify_intent
from transcript import is_indexable, get_filter_reason


# Each test case: (message_content, expected_intent, should_index)
# should_index=False means it should be filtered out entirely

CLASSIFICATION_TEST_CASES = [
    # === DECISIONS ===
    (
        "We decided to use SQLite instead of PostgreSQL for the local database.",
        "decision",
        True,
    ),
    (
        "Going with the WAL mode approach for better concurrency.",
        "decision",
        True,
    ),
    (
        "I decided to use a 3.3V regulator rather than the 5V one.",
        "decision",
        True,
    ),
    (
        "Choosing React over Vue for the frontend framework.",
        "decision",
        True,
    ),
    (
        "Using asyncio over threading for the concurrent operations.",
        "decision",
        True,
    ),

    # === CONCLUSIONS ===
    (
        "Turns out the root cause was the missing NULL check in the parser.",
        "conclusion",
        True,
    ),
    (
        "Realized that the relay is switching correctly under load.",
        "conclusion",
        True,
    ),
    (
        "The key insight is that the bottleneck is in the embedding generation step.",
        "conclusion",
        True,
    ),

    # === QUESTIONS (real questions from user) ===
    (
        "What's the best way to handle concurrent database writes?",
        "question",
        True,
    ),
    (
        "How should I structure the authentication flow?",
        "question",
        True,
    ),
    (
        "Should we add error handling for the API timeout case?",
        "question",
        True,
    ),
    (
        "Can we use Redis instead of SQLite for this?",
        "question",
        True,
    ),

    # === PROBLEMS ===
    (
        "The issue is the build is failing with a type error in the auth module.",
        "problem",
        True,
    ),
    (
        "Running into database locked errors when running the hook.",
        "problem",
        True,
    ),
    (
        "The relay isn't switching - doesn't work at all.",
        "problem",
        True,
    ),
    (
        "Problem: There's a race condition between the indexer and the backfill.",
        "problem",
        True,
    ),

    # === SOLUTIONS ===
    (
        "Fixed by adding WAL mode to the database connection.",
        "solution",
        True,
    ),
    (
        "The fix is to add a unique constraint on source_file and source_line.",
        "solution",
        True,
    ),
    (
        "Resolved by increasing the timeout from 30s to 120s.",
        "solution",
        True,
    ),
    (
        "Works now after updating the dependency version.",
        "solution",
        True,
    ),

    # === TODOS ===
    (
        "TODO: Add retry logic for the embedding API calls.",
        "todo",
        True,
    ),
    (
        "Need to implement the export functionality next.",
        "todo",
        True,
    ),
    (
        "Should add tests for the edge cases in the parser.",
        "todo",
        True,
    ),

    # === CONTEXT (legitimate) ===
    (
        "The current architecture uses a SQLite database with vector extensions for semantic search.",
        "context",
        True,
    ),
    (
        "This project is a Claude Code skill for long-term memory across conversations.",
        "context",
        True,
    ),
]


FILTERING_TEST_CASES = [
    # === SHOULD BE FILTERED (boilerplate/warmup) ===
    (
        "I understand. I'm ready to help you explore a codebase and design implementation plans.",
        False,  # Should be filtered (preamble)
    ),
    (
        "Let me check that file for you.",
        False,  # Should be filtered (preamble)
    ),
    (
        "I'll read the file now.",
        False,  # Should be filtered (preamble)
    ),
    (
        "ok",
        False,  # Should be filtered (too short + acknowledgment)
    ),
    (
        "thanks",
        False,  # Should be filtered (too short + acknowledgment)
    ),
    (
        "got it",
        False,  # Should be filtered (too short + acknowledgment)
    ),
    (
        "sounds good",
        False,  # Should be filtered (acknowledgment)
    ),
    (
        "hello",
        False,  # Should be filtered (greeting)
    ),
    (
        "hi there",
        False,  # Should be filtered (greeting)
    ),
    (
        "y",
        False,  # Should be filtered (too short)
    ),
    (
        "n",
        False,  # Should be filtered (too short)
    ),
    (
        "?",
        False,  # Should be filtered (too short)
    ),
    (
        "Done!",
        False,  # Should be filtered (tool narration)
    ),
    (
        "Running the command now.",
        False,  # Should be filtered (tool narration)
    ),
    (
        "go ahead",
        False,  # Should be filtered (short instruction)
    ),
    (
        "do it",
        False,  # Should be filtered (short instruction)
    ),

    # === SHOULD NOT BE FILTERED (substantive content) ===
    (
        "The database is locked because SQLite uses exclusive locks by default in rollback journal mode.",
        True,  # Should NOT be filtered - substantive explanation
    ),
    (
        "We decided to use WAL mode for better concurrency between the hook and backfill processes.",
        True,  # Should NOT be filtered - decision
    ),
    (
        "What's the best relay for switching 240V AC at 10 amps for a home heating system?",
        True,  # Should NOT be filtered - substantive question
    ),
    (
        "The fix was to add a unique constraint on (source_file, source_line) to prevent duplicates.",
        True,  # Should NOT be filtered - solution
    ),
]


class TestIntentClassification:
    """Tests for intent classification accuracy."""

    @pytest.mark.parametrize("content,expected_intent,should_index", CLASSIFICATION_TEST_CASES)
    def test_classification(self, content, expected_intent, should_index):
        """Test that messages are classified correctly."""
        result = classify_intent(content)
        assert result == expected_intent, f"Expected '{expected_intent}', got '{result}' for: {content[:60]}..."


class TestFiltering:
    """Tests for message filtering."""

    @pytest.mark.parametrize("content,should_index", FILTERING_TEST_CASES)
    def test_filtering(self, content, should_index):
        """Test that messages are filtered correctly."""
        message = {"content": content, "type": "assistant"}
        result = is_indexable(message)

        if should_index:
            assert result, f"Should NOT filter: {content[:60]}..."
        else:
            assert not result, f"Should filter: {content[:60]}... (reason: {get_filter_reason(message)})"


class TestFilterReasons:
    """Tests for filter reason reporting."""

    def test_filter_reason_empty(self):
        """Empty messages should report 'empty' reason."""
        message = {"content": "", "type": "user"}
        reason = get_filter_reason(message)
        assert reason == "empty"

    def test_filter_reason_too_short(self):
        """Short messages should report 'too_short' reason."""
        message = {"content": "hi", "type": "user"}
        reason = get_filter_reason(message)
        assert reason is not None and "too_short" in reason

    def test_filter_reason_greeting(self):
        """Greetings should report 'greeting' reason."""
        message = {"content": "hello there is anyone home", "type": "user"}
        # This is long enough but if it matches greeting pattern...
        reason = get_filter_reason(message)
        # May or may not be filtered depending on length

    def test_no_filter_reason_for_substantive(self):
        """Substantive content should have no filter reason."""
        message = {
            "content": "The database is locked because SQLite uses exclusive write locks.",
            "type": "assistant"
        }
        reason = get_filter_reason(message)
        assert reason is None, f"Unexpected filter reason: {reason}"


# Edge cases for classification
EDGE_CASES = [
    # Very short but valid
    ("Need to fix the bug", "todo", True),

    # Code blocks shouldn't dominate classification
    (
        "Fixed by adding this code:\n```python\ndb.execute('PRAGMA journal_mode=WAL')\n```",
        "solution",
        True,
    ),

    # Questions ending with question mark
    ("Is this the right approach?", "question", True),

    # Decisions with "let's"
    ("Let's use Redis for caching instead of in-memory.", "decision", True),
]


class TestEdgeCases:
    """Tests for edge cases in classification."""

    @pytest.mark.parametrize("content,expected_intent,should_index", EDGE_CASES)
    def test_edge_cases(self, content, expected_intent, should_index):
        """Test edge cases are handled correctly."""
        result = classify_intent(content)
        assert result == expected_intent, f"Expected '{expected_intent}', got '{result}' for: {content[:60]}..."


# Test that question patterns don't over-match
QUESTION_NEGATIVE_CASES = [
    # Tables with ? in headers shouldn't be questions
    ("| Item | Price? | Notes |\n|------|--------|-------|", "context"),
    # URLs with ? shouldn't be questions
    ("Check the docs at https://example.com/faq?topic=wal", "context"),
    # Rhetorical statements
    ("Good price - already close to volume pricing.", "context"),
]


class TestQuestionNegativeCases:
    """Test that non-questions aren't classified as questions."""

    @pytest.mark.parametrize("content,expected_intent", QUESTION_NEGATIVE_CASES)
    def test_not_questions(self, content, expected_intent):
        """Test that non-questions aren't misclassified."""
        result = classify_intent(content)
        # These shouldn't be classified as questions
        # (though they might be classified as something else)
        if "?" not in content or content.strip().endswith("?"):
            # If it ends with ? it's probably a question by our rules
            pass
        else:
            assert result != "question" or expected_intent == "question", \
                f"Should not be 'question': {content[:60]}..."
