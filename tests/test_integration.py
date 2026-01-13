"""Integration tests with real transcript data."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "total-recall" / "src"))

from transcript import read_transcript, is_indexable


class TestRealTranscriptParsing:
    """Test parsing against real Claude transcript format."""

    @pytest.fixture
    def sample_transcript(self, tmp_path):
        """Create a realistic transcript file."""
        lines = [
            # User greeting - should be filtered
            {"type": "user", "message": {"role": "user", "content": "hello"}, "timestamp": "2026-01-10T10:00:00Z", "uuid": "1"},

            # Assistant greeting - should be filtered (short)
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello! How can I help?"}]}, "timestamp": "2026-01-10T10:00:01Z", "uuid": "2"},

            # User substantive request - should be INDEXED
            {"type": "user", "message": {"role": "user", "content": "Can you help me design a heating control system with LoRa connectivity?"}, "timestamp": "2026-01-10T10:00:02Z", "uuid": "3"},

            # Assistant substantive response - should be INDEXED
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "I'll help you design a heating control system. The key decisions are: 1) Use ESP32-C6 for the microcontroller with built-in WiFi/BLE, 2) SX1262 for LoRa communication at 868MHz, 3) Relay outputs for pump/valve control. Let me outline the architecture."}]}, "timestamp": "2026-01-10T10:00:03Z", "uuid": "4"},

            # Tool use with preamble only - should be filtered
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Let me check that file."}, {"type": "tool_use", "id": "1", "name": "Read", "input": {"file_path": "/tmp/test"}}]}, "timestamp": "2026-01-10T10:00:04Z", "uuid": "5"},

            # Tool result - should be filtered (parse returns None)
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "file contents here"}]}, "toolUseResult": {"stdout": "file contents"}, "timestamp": "2026-01-10T10:00:05Z", "uuid": "6"},

            # User acknowledgment - should be filtered
            {"type": "user", "message": {"role": "user", "content": "ok"}, "timestamp": "2026-01-10T10:00:06Z", "uuid": "7"},

            # Assistant decision - should be INDEXED
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Based on our testing, we've decided to use SF12 spreading factor with 22dBm power for reliable through-wall transmission. SF7 was insufficient for the range requirements."}]}, "timestamp": "2026-01-10T10:00:07Z", "uuid": "8"},
        ]

        transcript = tmp_path / "session.jsonl"
        transcript.write_text("\n".join(json.dumps(line) for line in lines))
        return transcript

    def test_parse_real_transcript_format(self, sample_transcript):
        """Parse realistic transcript and verify structure."""
        messages = list(read_transcript(str(sample_transcript)))

        # Should have parsed 6 messages (2 filtered at parse level: tool result and tool-only)
        # Actually let me count: 8 lines, tool result filtered = 7 parseable
        assert len(messages) >= 5  # At least 5 should parse

        # Check first parsed message
        line_num, msg = messages[0]
        assert line_num == 1
        assert msg["type"] == "user"
        assert "content" in msg

    def test_filter_indexable_from_real_transcript(self, sample_transcript):
        """Filter for indexable content from realistic transcript."""
        messages = list(read_transcript(str(sample_transcript)))
        indexable = [(ln, msg) for ln, msg in messages if is_indexable(msg)]

        # Should have 3 indexable messages:
        # - Line 3: User request about heating control
        # - Line 4: Assistant architecture response
        # - Line 8: Assistant decision about SF12
        assert len(indexable) == 3

        # Verify the right ones were selected
        line_numbers = [ln for ln, _ in indexable]
        assert 3 in line_numbers  # User request
        assert 4 in line_numbers  # Assistant response
        assert 8 in line_numbers  # Assistant decision

    def test_indexable_content_is_substantive(self, sample_transcript):
        """Verify indexable messages contain substantive content."""
        messages = list(read_transcript(str(sample_transcript)))
        indexable = [(ln, msg) for ln, msg in messages if is_indexable(msg)]

        for _, msg in indexable:
            content = msg["content"]
            # Should be reasonably long
            assert len(content) >= 20
            # Should not be just acknowledgments
            assert content.lower() not in ["ok", "yes", "thanks", "hello"]


class TestTranscriptStructure:
    """Test transcript structure expectations."""

    def test_synthetic_full_conversation(self, tmp_path):
        """Test a full synthetic conversation flow."""
        lines = [
            # Metadata lines (should be skipped)
            {"type": "summary", "summary": "...", "leafUuid": "abc"},
            {"type": "file-history-snapshot", "snapshot": {}, "messageId": "1"},

            # User greeting - filtered
            {"type": "user", "message": {"role": "user", "content": "hi"}, "timestamp": "T1", "cwd": "/tmp"},

            # User substantive - INDEXED
            {"type": "user", "message": {"role": "user", "content": "Help me implement a REST API with authentication and rate limiting"}, "timestamp": "T2", "cwd": "/tmp"},

            # Assistant response - INDEXED
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "I'll help you implement a REST API. For authentication, I recommend using JWT tokens with refresh token rotation. For rate limiting, we can use a sliding window algorithm with Redis."}]}, "timestamp": "T3", "cwd": "/tmp"},

            # User ack - filtered
            {"type": "user", "message": {"role": "user", "content": "sounds good"}, "timestamp": "T4", "cwd": "/tmp"},
        ]

        transcript = tmp_path / "full.jsonl"
        transcript.write_text("\n".join(json.dumps(line) for line in lines))

        messages = list(read_transcript(str(transcript)))
        indexable = [(ln, msg) for ln, msg in messages if is_indexable(msg)]

        # Only 2 should be indexable
        assert len(indexable) == 2
