"""Tests for message batcher - Slice 1."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def make_message(role: str, content: str, timestamp: datetime) -> str:
    """Create a JSONL line for a message."""
    return json.dumps({
        "type": role,
        "message": {"content": content},
        "timestamp": timestamp.isoformat()
    })


def make_transcript(messages: list[tuple[str, str, datetime]]) -> str:
    """Create transcript content from (role, content, timestamp) tuples."""
    lines = []
    for role, content, ts in messages:
        lines.append(make_message(role, content, ts))
    return "\n".join(lines) + "\n"


class TestMessageBatcher:
    """Tests for the message batcher."""

    def test_messages_within_window_batched_together(self):
        """Messages within 5s window should be in same batch."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First message", base_time),
            ("assistant", "Response one", base_time + timedelta(seconds=2)),
            ("user", "Follow up", base_time + timedelta(seconds=4)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 1
            assert len(batches[0].messages) == 3

        Path(f.name).unlink()

    def test_messages_outside_window_in_separate_batches(self):
        """Messages > 5s apart should be in different batches."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First message", base_time),
            ("assistant", "Response", base_time + timedelta(seconds=2)),
            # Gap of 10 seconds
            ("user", "Later message", base_time + timedelta(seconds=12)),
            ("assistant", "Later response", base_time + timedelta(seconds=14)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 2
            assert len(batches[0].messages) == 2
            assert len(batches[1].messages) == 2

        Path(f.name).unlink()

    def test_empty_file_returns_no_batches(self):
        """Empty file should return no batches."""
        from batcher import collect_batches

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("")
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 0

        Path(f.name).unlink()

    def test_start_byte_skips_earlier_content(self):
        """Starting from byte position should skip earlier messages."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First message", base_time),
            ("assistant", "Response", base_time + timedelta(seconds=2)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            # Get byte position after first line
            with open(f.name, 'r') as rf:
                first_line = rf.readline()
                start_byte = len(first_line.encode('utf-8'))

            batches = list(collect_batches(f.name, start_byte=start_byte))

            assert len(batches) == 1
            assert len(batches[0].messages) == 1
            assert batches[0].messages[0].role == "assistant"

        Path(f.name).unlink()

    def test_batch_includes_line_numbers(self):
        """Each message in batch should have its source line number."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First", base_time),
            ("assistant", "Second", base_time + timedelta(seconds=1)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert batches[0].messages[0].line_num == 1
            assert batches[0].messages[1].line_num == 2

        Path(f.name).unlink()

    def test_batch_includes_timestamps(self):
        """Each message should preserve its timestamp."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Hello", base_time),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert batches[0].messages[0].timestamp == base_time.isoformat()

        Path(f.name).unlink()

    def test_batch_returns_end_byte_position(self):
        """Batch should report the byte position after last message."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Hello", base_time),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            file_size = Path(f.name).stat().st_size

            batches = list(collect_batches(f.name, start_byte=0))

            assert batches[0].end_byte == file_size

        Path(f.name).unlink()

    def test_handles_malformed_json_gracefully(self):
        """Malformed JSON lines should be skipped, not crash."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_message("user", "Valid", base_time) + "\n")
            f.write("not valid json\n")
            f.write(make_message("assistant", "Also valid", base_time + timedelta(seconds=1)) + "\n")
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 1
            assert len(batches[0].messages) == 2

        Path(f.name).unlink()

    def test_handles_non_message_types(self):
        """Non user/assistant types should be skipped."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_message("user", "Hello", base_time) + "\n")
            f.write(json.dumps({"type": "system", "content": "ignored"}) + "\n")
            f.write(json.dumps({"type": "summary", "content": "also ignored"}) + "\n")
            f.write(make_message("assistant", "Hi", base_time + timedelta(seconds=1)) + "\n")
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 1
            assert len(batches[0].messages) == 2

        Path(f.name).unlink()

    def test_file_not_found_raises(self):
        """Missing file should raise appropriate error."""
        from batcher import collect_batches, BatcherError

        with pytest.raises(BatcherError):
            list(collect_batches("/nonexistent/file.jsonl", start_byte=0))

    def test_configurable_window_size(self):
        """Window size should be configurable."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First", base_time),
            ("assistant", "Second", base_time + timedelta(seconds=8)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            # Default 5s window - should be 2 batches
            batches_default = list(collect_batches(f.name, start_byte=0))
            assert len(batches_default) == 2

            # 10s window - should be 1 batch
            batches_wider = list(collect_batches(f.name, start_byte=0, window_seconds=10))
            assert len(batches_wider) == 1

        Path(f.name).unlink()

    def test_out_of_order_timestamps_stay_batched(self):
        """Out-of-order timestamps (negative gap) stay in same batch."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "First", base_time),
            # Timestamp is earlier than previous (clock skew, editing, etc)
            ("assistant", "Response", base_time - timedelta(seconds=2)),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            # Negative gap treated as "close together"
            assert len(batches) == 1
            assert len(batches[0].messages) == 2

        Path(f.name).unlink()

    def test_unicode_content_preserved(self):
        """Unicode content should be preserved correctly."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Hello 世界 🌍 café", base_time),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert batches[0].messages[0].content == "Hello 世界 🌍 café"

        Path(f.name).unlink()

    def test_single_message_yields_single_batch(self):
        """A single message should yield exactly one batch."""
        from batcher import collect_batches

        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            ("user", "Only message", base_time),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(make_transcript(messages))
            f.flush()

            batches = list(collect_batches(f.name, start_byte=0))

            assert len(batches) == 1
            assert len(batches[0].messages) == 1

        Path(f.name).unlink()
