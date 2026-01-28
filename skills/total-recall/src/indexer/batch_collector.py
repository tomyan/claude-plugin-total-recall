"""Batch collector for indexing agent - Slice 5.1.

Collects new messages from transcript files for batch processing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from db.connection import get_db


@dataclass
class Message:
    """A parsed message from a transcript."""
    role: str
    content: str
    line_num: int
    timestamp: str


@dataclass
class BatchUpdate:
    """A batch of updates for a single session/file."""
    session: str
    file_path: str
    messages: list[Message]
    start_byte: int
    end_byte: int


def get_byte_position(file_path: str) -> int:
    """Get last processed byte position for a file.

    Args:
        file_path: Path to transcript file

    Returns:
        Last processed byte position, or 0 if not yet indexed
    """
    db = get_db()
    cursor = db.execute(
        "SELECT byte_position FROM index_state WHERE file_path = ?",
        (file_path,)
    )
    row = cursor.fetchone()
    db.close()

    return row["byte_position"] if row else 0


def get_session_from_path(file_path: str) -> str:
    """Extract session ID from file path.

    Args:
        file_path: Path to transcript file

    Returns:
        Session ID (typically filename without extension)
    """
    return Path(file_path).stem


async def collect_batch_updates(files: list[str]) -> list[BatchUpdate]:
    """Collect new messages from transcript files.

    Args:
        files: List of transcript file paths

    Returns:
        List of BatchUpdate objects with new messages
    """
    updates = []

    for file_path in files:
        if not Path(file_path).exists():
            continue

        start_byte = get_byte_position(file_path)
        session = get_session_from_path(file_path)

        messages = []
        end_byte = start_byte

        with open(file_path, "rb") as f:
            # Seek to last position
            f.seek(start_byte)

            line_num = 0
            # Count lines before start_byte for proper line numbering
            if start_byte > 0:
                f.seek(0)
                for _ in range(start_byte):
                    if f.read(1) == b"\n":
                        line_num += 1
                f.seek(start_byte)

            # Read new content
            for line in f:
                line_num += 1
                end_byte += len(line)

                try:
                    data = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                msg_type = data.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                message_data = data.get("message", {})
                content = message_data.get("content", "")

                # Handle content blocks
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = "\n".join(text_parts)

                if not content.strip():
                    continue

                messages.append(Message(
                    role=msg_type,
                    content=content,
                    line_num=line_num,
                    timestamp=data.get("timestamp", ""),
                ))

        if messages:
            updates.append(BatchUpdate(
                session=session,
                file_path=file_path,
                messages=messages,
                start_byte=start_byte,
                end_byte=end_byte,
            ))

    return updates
