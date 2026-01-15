"""Tests for LLM protocol - Slice 3 & 4."""

import json
from datetime import datetime, timedelta

import pytest

from batcher import Batch, Message


def make_messages(count: int, base_time: datetime = None) -> list[Message]:
    """Create a list of test messages."""
    if base_time is None:
        base_time = datetime(2024, 1, 15, 10, 0, 0)

    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(
            role=role,
            content=f"Message {i}",
            line_num=i + 1,
            timestamp=(base_time + timedelta(seconds=i)).isoformat()
        ))
    return messages


class TestLLMInputProtocol:
    """Tests for LLM input protocol formatting."""

    def test_formats_basic_batch(self):
        """Should format a simple batch with messages."""
        from protocol import format_llm_input

        batch = Batch(
            messages=make_messages(2),
            end_byte=100
        )
        context = {
            "project": None,
            "topic": None,
            "parent_spans": [],
            "current_span": None,
            "recent_spans": [],
        }

        result = format_llm_input(batch, context, recent_messages=[])

        assert "new_messages" in result
        assert len(result["new_messages"]) == 2
        assert result["new_messages"][0]["role"] == "user"
        assert result["new_messages"][0]["content"] == "Message 0"
        assert result["new_messages"][0]["line"] == 1

    def test_includes_hierarchy_levels(self):
        """Should include all hierarchy levels in prompt."""
        from protocol import format_llm_input

        batch = Batch(messages=make_messages(1), end_byte=50)
        context = {
            "project": {"name": "MyProject", "description": "A test project"},
            "topic": {"name": "Auth", "summary": "Authentication work"},
            "parent_spans": [{"name": "Setup", "summary": "Initial setup"}],
            "current_span": {"name": "JWT", "summary": "JWT implementation"},
            "recent_spans": [],
        }

        result = format_llm_input(batch, context, recent_messages=[])

        assert result["hierarchy"]["project"]["name"] == "MyProject"
        assert result["hierarchy"]["topic"]["name"] == "Auth"
        assert result["hierarchy"]["span"]["name"] == "JWT"
        assert len(result["hierarchy"]["parent_spans"]) == 1

    def test_includes_recent_messages(self):
        """Should include recent messages for context."""
        from protocol import format_llm_input

        batch = Batch(messages=make_messages(1), end_byte=50)
        context = {
            "project": None,
            "topic": None,
            "parent_spans": [],
            "current_span": None,
            "recent_spans": [],
        }
        recent = [
            {"role": "user", "content": "Earlier message", "timestamp": "2024-01-15T09:00:00"},
            {"role": "assistant", "content": "Earlier response", "timestamp": "2024-01-15T09:00:01"},
        ]

        result = format_llm_input(batch, context, recent_messages=recent)

        assert "recent_messages" in result
        assert len(result["recent_messages"]) == 2
        assert result["recent_messages"][0]["content"] == "Earlier message"

    def test_truncates_recent_messages(self):
        """Should truncate recent messages to max limit."""
        from protocol import format_llm_input

        batch = Batch(messages=make_messages(1), end_byte=50)
        context = {
            "project": None, "topic": None, "parent_spans": [],
            "current_span": None, "recent_spans": [],
        }
        # Create 20 recent messages
        recent = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"Recent {i}",
             "timestamp": f"2024-01-15T09:00:{i:02d}"}
            for i in range(20)
        ]

        result = format_llm_input(batch, context, recent_messages=recent, max_recent=10)

        assert len(result["recent_messages"]) == 10
        # Should keep most recent
        assert result["recent_messages"][-1]["content"] == "Recent 19"

    def test_new_messages_include_line_numbers(self):
        """New messages should include their source line numbers."""
        from protocol import format_llm_input

        messages = [
            Message(role="user", content="Test", line_num=42, timestamp="2024-01-15T10:00:00"),
            Message(role="assistant", content="Response", line_num=43, timestamp="2024-01-15T10:00:01"),
        ]
        batch = Batch(messages=messages, end_byte=100)
        context = {
            "project": None, "topic": None, "parent_spans": [],
            "current_span": None, "recent_spans": [],
        }

        result = format_llm_input(batch, context, recent_messages=[])

        assert result["new_messages"][0]["line"] == 42
        assert result["new_messages"][1]["line"] == 43

    def test_empty_context_handled(self):
        """Should handle completely empty context."""
        from protocol import format_llm_input

        batch = Batch(messages=make_messages(1), end_byte=50)
        context = {
            "project": None,
            "topic": None,
            "parent_spans": [],
            "current_span": None,
            "recent_spans": [],
        }

        result = format_llm_input(batch, context, recent_messages=[])

        assert result["hierarchy"]["project"] is None
        assert result["hierarchy"]["topic"] is None
        assert result["hierarchy"]["span"] is None

    def test_output_is_json_serializable(self):
        """The output should be JSON serializable."""
        from protocol import format_llm_input

        batch = Batch(
            messages=make_messages(2),
            end_byte=100
        )
        context = {
            "project": {"name": "Test", "description": "Test project"},
            "topic": {"name": "Topic", "summary": "Topic summary"},
            "parent_spans": [],
            "current_span": {"name": "Span", "summary": "Span summary"},
            "recent_spans": [],
        }
        recent = [{"role": "user", "content": "Hi", "timestamp": "2024-01-15T09:00:00"}]

        result = format_llm_input(batch, context, recent_messages=recent)

        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result

    def test_preserves_message_timestamps(self):
        """Should preserve original timestamps in new messages."""
        from protocol import format_llm_input

        messages = [
            Message(role="user", content="Test", line_num=1,
                    timestamp="2024-01-15T10:30:45.123456"),
        ]
        batch = Batch(messages=messages, end_byte=50)
        context = {
            "project": None, "topic": None, "parent_spans": [],
            "current_span": None, "recent_spans": [],
        }

        result = format_llm_input(batch, context, recent_messages=[])

        assert result["new_messages"][0]["timestamp"] == "2024-01-15T10:30:45.123456"


class TestLLMOutputProtocol:
    """Tests for LLM output protocol parsing - Slice 4."""

    def test_parses_valid_response(self):
        """Should parse a complete valid response."""
        from protocol import parse_llm_output

        response = {
            "topic_update": {
                "name": "Auth implementation",
                "summary": "Working on authentication"
            },
            "items": [
                {
                    "type": "decision",
                    "content": "Using JWT for auth",
                    "confidence": 0.9,
                    "source_line": 42,
                    "entities": ["JWT"]
                }
            ],
            "relations": [],
            "skip_lines": []
        }

        result = parse_llm_output(response)

        assert result.topic_update is not None
        assert result.topic_update["name"] == "Auth implementation"
        assert len(result.items) == 1
        assert result.items[0]["type"] == "decision"

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields gracefully."""
        from protocol import parse_llm_output

        # Minimal response with only items
        response = {
            "items": [
                {
                    "type": "context",
                    "content": "User is setting up project",
                    "source_line": 1
                }
            ]
        }

        result = parse_llm_output(response)

        assert result.topic_update is None
        assert result.new_span is None
        assert result.relations == []
        assert result.skip_lines == []
        assert len(result.items) == 1

    def test_validates_intent_types(self):
        """Should validate item intent types."""
        from protocol import parse_llm_output, ProtocolError

        response = {
            "items": [
                {
                    "type": "invalid_type",
                    "content": "Test",
                    "source_line": 1
                }
            ]
        }

        with pytest.raises(ProtocolError) as exc:
            parse_llm_output(response)

        assert "invalid_type" in str(exc.value)

    def test_rejects_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        from protocol import parse_llm_output_str, ProtocolError

        malformed = "{ not valid json }"

        with pytest.raises(ProtocolError) as exc:
            parse_llm_output_str(malformed)

        assert "JSON" in str(exc.value) or "parse" in str(exc.value).lower()

    def test_handles_partial_response(self):
        """Should handle partial/incomplete responses."""
        from protocol import parse_llm_output

        # Response with only topic_update
        response = {
            "topic_update": {
                "name": "New topic",
                "summary": "Topic summary"
            }
        }

        result = parse_llm_output(response)

        assert result.topic_update is not None
        assert result.items == []

    def test_parses_new_span_action(self):
        """Should parse new_span for topic shifts."""
        from protocol import parse_llm_output

        response = {
            "new_span": {
                "name": "Debugging session",
                "reason": "Shifted from implementation to debugging"
            },
            "items": []
        }

        result = parse_llm_output(response)

        assert result.new_span is not None
        assert result.new_span["name"] == "Debugging session"
        assert result.new_span["reason"] == "Shifted from implementation to debugging"

    def test_parses_relations(self):
        """Should parse relations between ideas."""
        from protocol import parse_llm_output

        response = {
            "items": [],
            "relations": [
                {"from_line": 42, "to_idea_id": 156, "type": "supersedes"},
                {"from_line": 43, "to_idea_id": 100, "type": "builds_on"}
            ]
        }

        result = parse_llm_output(response)

        assert len(result.relations) == 2
        assert result.relations[0]["type"] == "supersedes"
        assert result.relations[1]["from_line"] == 43

    def test_parses_skip_lines(self):
        """Should parse lines to skip."""
        from protocol import parse_llm_output

        response = {
            "items": [],
            "skip_lines": [44, 45, 46]
        }

        result = parse_llm_output(response)

        assert result.skip_lines == [44, 45, 46]

    def test_validates_required_item_fields(self):
        """Should validate that items have required fields."""
        from protocol import parse_llm_output, ProtocolError

        response = {
            "items": [
                {
                    "type": "decision"
                    # Missing content and source_line
                }
            ]
        }

        with pytest.raises(ProtocolError) as exc:
            parse_llm_output(response)

        assert "content" in str(exc.value).lower() or "source_line" in str(exc.value).lower()

    def test_empty_response_returns_empty_result(self):
        """Empty response should return empty result."""
        from protocol import parse_llm_output

        response = {}

        result = parse_llm_output(response)

        assert result.topic_update is None
        assert result.new_span is None
        assert result.items == []
        assert result.relations == []
        assert result.skip_lines == []

    def test_confidence_defaults_to_half(self):
        """Items without confidence should default to 0.5."""
        from protocol import parse_llm_output

        response = {
            "items": [
                {
                    "type": "context",
                    "content": "Test content",
                    "source_line": 1
                }
            ]
        }

        result = parse_llm_output(response)

        assert result.items[0].get("confidence", 0.5) == 0.5
