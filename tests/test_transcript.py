"""Tests for transcript parsing."""

import json
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "memgraph" / "src"))

from transcript import parse_transcript_line, extract_message_content, read_transcript


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
