"""Tests for transcript parsing."""

import json
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))

from transcript import (
    parse_transcript_line,
    extract_message_content,
    read_transcript,
    is_indexable,
    get_indexable_messages,
)


class TestParseTranscriptLine:
    """Test parsing individual JSON lines from Claude transcripts."""

    def test_parse_user_message(self):
        """Parse a user message line."""
        line = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "Hello, how are you?"},
            "timestamp": "2026-01-10T12:00:00Z"
        })

        result = parse_transcript_line(line)

        assert result["type"] == "user"
        assert result["content"] == "Hello, how are you?"
        assert result["timestamp"] == "2026-01-10T12:00:00Z"

    def test_parse_assistant_message(self):
        """Parse an assistant message with text content."""
        line = json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'm doing well, thanks!"}]
            },
            "timestamp": "2026-01-10T12:00:01Z"
        })

        result = parse_transcript_line(line)

        assert result["type"] == "assistant"
        assert result["content"] == "I'm doing well, thanks!"

    def test_parse_assistant_message_with_tool_use(self):
        """Parse an assistant message that includes tool use."""
        line = json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that file."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/test"}}
                ]
            },
            "timestamp": "2026-01-10T12:00:02Z"
        })

        result = parse_transcript_line(line)

        assert result["type"] == "assistant"
        assert result["content"] == "Let me check that file."
        assert result["has_tool_use"] is True

    def test_parse_invalid_json_returns_none(self):
        """Invalid JSON should return None."""
        result = parse_transcript_line("not valid json")
        assert result is None

    def test_parse_tool_result_skipped(self):
        """Tool results should be skipped (return None)."""
        line = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "tool_result", "content": "file contents"}]},
            "toolUseResult": {"stdout": "output"}
        })

        result = parse_transcript_line(line)
        assert result is None  # Tool results are not indexable

    def test_parse_assistant_only_tool_use_returns_empty_content(self):
        """Assistant message with only tool_use has empty text content."""
        line = json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]
            },
            "timestamp": "2026-01-10T12:00:03Z"
        })

        result = parse_transcript_line(line)
        # Should return None - no text content to index
        assert result is None


class TestReadTranscript:
    """Test reading and iterating transcript files."""

    def test_read_transcript_yields_parsed_lines_with_numbers(self, tmp_path):
        """Read transcript yields (line_number, parsed_dict) tuples."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "Hello"}, "timestamp": "T1"}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi!"}]}, "timestamp": "T2"}),
        ]))

        results = list(read_transcript(str(transcript)))

        assert len(results) == 2
        assert results[0][0] == 1  # line number
        assert results[0][1]["content"] == "Hello"
        assert results[1][0] == 2
        assert results[1][1]["content"] == "Hi!"

    def test_read_transcript_skips_unparseable_lines(self, tmp_path):
        """Unparseable lines are skipped."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "First"}, "timestamp": "T1"}),
            "invalid json line",
            json.dumps({"type": "user", "message": {"content": "Third"}, "timestamp": "T3"}),
        ]))

        results = list(read_transcript(str(transcript)))

        assert len(results) == 2
        assert results[0][0] == 1
        assert results[1][0] == 3  # Line 2 was skipped

    def test_read_transcript_from_offset(self, tmp_path):
        """Can start reading from a specific line offset."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "Line 1"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "Line 2"}, "timestamp": "T2"}),
            json.dumps({"type": "user", "message": {"content": "Line 3"}, "timestamp": "T3"}),
        ]))

        results = list(read_transcript(str(transcript), start_line=2))

        assert len(results) == 2
        assert results[0][0] == 2
        assert results[0][1]["content"] == "Line 2"


class TestExtractMessageContent:
    """Test extracting text content from message structures."""

    def test_string_content(self):
        """Simple string content."""
        msg = {"content": "Hello world"}
        assert extract_message_content(msg) == "Hello world"

    def test_list_content_with_text(self):
        """List content with text blocks."""
        msg = {"content": [
            {"type": "text", "text": "First part. "},
            {"type": "text", "text": "Second part."}
        ]}
        assert extract_message_content(msg) == "First part. Second part."

    def test_list_content_filters_tool_use(self):
        """Tool use blocks should be filtered out."""
        msg = {"content": [
            {"type": "text", "text": "Here's the result."},
            {"type": "tool_use", "name": "Bash", "input": {}}
        ]}
        assert extract_message_content(msg) == "Here's the result."

    def test_empty_content(self):
        """Empty content returns empty string."""
        msg = {"content": ""}
        assert extract_message_content(msg) == ""


class TestIsIndexable:
    """Test filtering of low-value content."""

    def test_substantive_user_message_is_indexable(self):
        """User messages with substantive content are indexable."""
        msg = {"type": "user", "content": "Can you help me implement a caching layer for the API?"}
        assert is_indexable(msg) is True

    def test_greeting_not_indexable(self):
        """Simple greetings are not indexable."""
        assert is_indexable({"type": "user", "content": "Hello"}) is False
        assert is_indexable({"type": "user", "content": "Hi there"}) is False
        assert is_indexable({"type": "user", "content": "hey"}) is False

    def test_acknowledgment_not_indexable(self):
        """Simple acknowledgments are not indexable."""
        assert is_indexable({"type": "user", "content": "ok"}) is False
        assert is_indexable({"type": "user", "content": "okay"}) is False
        assert is_indexable({"type": "user", "content": "yes"}) is False
        assert is_indexable({"type": "user", "content": "no"}) is False
        assert is_indexable({"type": "user", "content": "thanks"}) is False
        assert is_indexable({"type": "user", "content": "got it"}) is False

    def test_short_content_not_indexable(self):
        """Very short content (< 20 chars) is not indexable."""
        assert is_indexable({"type": "user", "content": "do it"}) is False
        assert is_indexable({"type": "user", "content": "go ahead"}) is False

    def test_assistant_explanation_is_indexable(self):
        """Assistant explanations are indexable."""
        msg = {
            "type": "assistant",
            "content": "The issue is that the TCXO needs to be enabled via DIO3 for reliable transmission."
        }
        assert is_indexable(msg) is True

    def test_assistant_tool_only_not_indexable(self):
        """Assistant messages with only tool use commentary are not indexable."""
        msg = {
            "type": "assistant",
            "content": "Let me check that file.",
            "has_tool_use": True
        }
        # Short content that's just tool preamble
        assert is_indexable(msg) is False

    def test_assistant_substantive_with_tool_is_indexable(self):
        """Assistant messages with substantive content + tool use ARE indexable."""
        msg = {
            "type": "assistant",
            "content": "The database schema needs to include a foreign key constraint to maintain referential integrity. Let me update the migration file.",
            "has_tool_use": True
        }
        assert is_indexable(msg) is True

    def test_empty_content_not_indexable(self):
        """Empty content is not indexable."""
        assert is_indexable({"type": "user", "content": ""}) is False
        assert is_indexable({"type": "user", "content": "   "}) is False


class TestGetIndexableMessages:
    """Test getting all indexable messages from a transcript."""

    def test_returns_only_indexable_messages(self, tmp_path):
        """Returns filtered list of indexable messages."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "hello"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "Can you help me design a database schema for user authentication?"}, "timestamp": "T2"}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll help you design an authentication schema. You'll need tables for users, sessions, and password reset tokens."}]}, "timestamp": "T3"}),
            json.dumps({"type": "user", "message": {"content": "ok"}, "timestamp": "T4"}),
        ]))

        messages = get_indexable_messages(str(transcript))

        assert len(messages) == 2
        assert messages[0]["line_num"] == 2
        assert "database schema" in messages[0]["content"]
        assert messages[1]["line_num"] == 3

    def test_includes_metadata(self, tmp_path):
        """Each message includes line_num, type, content, timestamp."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text(json.dumps({
            "type": "user",
            "message": {"content": "Implement a caching layer using Redis for the API responses"},
            "timestamp": "2026-01-10T12:00:00Z"
        }))

        messages = get_indexable_messages(str(transcript))

        assert len(messages) == 1
        msg = messages[0]
        assert msg["line_num"] == 1
        assert msg["type"] == "user"
        assert "caching" in msg["content"]
        assert msg["timestamp"] == "2026-01-10T12:00:00Z"

    def test_respects_start_line(self, tmp_path):
        """Can start from a specific line for incremental indexing."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "First substantive message about API design patterns"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "Second substantive message about database optimization"}, "timestamp": "T2"}),
            json.dumps({"type": "user", "message": {"content": "Third substantive message about caching strategies"}, "timestamp": "T3"}),
        ]))

        messages = get_indexable_messages(str(transcript), start_line=2)

        assert len(messages) == 2
        assert messages[0]["line_num"] == 2
        assert "database" in messages[0]["content"]
