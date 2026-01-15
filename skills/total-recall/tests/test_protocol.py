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
