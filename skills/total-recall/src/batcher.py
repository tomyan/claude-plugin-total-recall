"""Message batcher - groups messages within time windows."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator


class BatcherError(Exception):
    """Error during batch collection."""
    pass


@dataclass
class Message:
    """A single message from a transcript."""
    role: str
    content: str
    line_num: int
    timestamp: str


@dataclass
class Batch:
    """A batch of messages within a time window."""
    messages: list[Message] = field(default_factory=list)
    end_byte: int = 0


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    # Handle both with and without microseconds
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        # Try without microseconds
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def collect_batches(
    file_path: str,
    start_byte: int = 0,
    window_seconds: float = 5.0
) -> Generator[Batch, None, None]:
    """
    Collect messages from a transcript file into time-windowed batches.

    Args:
        file_path: Path to the JSONL transcript file
        start_byte: Byte position to start reading from
        window_seconds: Maximum time gap between messages in same batch

    Yields:
        Batch objects containing messages within the time window

    Raises:
        BatcherError: If file not found or other IO errors
    """
    path = Path(file_path)

    if not path.exists():
        raise BatcherError(f"File not found: {file_path}")

    try:
        with open(path, 'rb') as f:
            # Seek to start position
            f.seek(start_byte)

            current_batch = Batch()
            last_timestamp: datetime | None = None
            line_num = 0
            current_byte = start_byte

            # Count lines before start_byte to get correct line numbers
            if start_byte > 0:
                f.seek(0)
                while f.tell() < start_byte:
                    f.readline()
                    line_num += 1
                f.seek(start_byte)

            for line_bytes in f:
                line_num += 1
                current_byte += len(line_bytes)
                line = line_bytes.decode('utf-8').strip()

                if not line:
                    continue

                # Try to parse JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip non user/assistant messages
                msg_type = data.get("type", "")
                if msg_type not in ("user", "assistant"):
                    continue

                # Extract message content and timestamp
                message_data = data.get("message", {})
                content = message_data.get("content", "")
                timestamp_str = data.get("timestamp", "")

                if not timestamp_str:
                    continue

                try:
                    timestamp = parse_timestamp(timestamp_str)
                except (ValueError, TypeError):
                    continue

                # Check if we need to start a new batch
                if last_timestamp is not None:
                    gap = (timestamp - last_timestamp).total_seconds()
                    if gap > window_seconds:
                        # Yield current batch and start new one
                        if current_batch.messages:
                            yield current_batch
                        current_batch = Batch()

                # Add message to current batch
                msg = Message(
                    role=msg_type,
                    content=content,
                    line_num=line_num,
                    timestamp=timestamp_str
                )
                current_batch.messages.append(msg)
                current_batch.end_byte = current_byte
                last_timestamp = timestamp

            # Yield final batch if it has messages
            if current_batch.messages:
                yield current_batch

    except IOError as e:
        raise BatcherError(f"Error reading file: {e}")
