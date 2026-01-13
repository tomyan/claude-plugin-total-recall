"""Backfill functionality for indexing existing transcripts."""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import memory_db
from memory_db import DB_PATH
from transcript import get_indexable_messages


def compute_content_hash(content: str) -> str:
    """Compute a hash of content for deduplication.

    Normalizes whitespace to catch near-duplicates.

    Args:
        content: Text content to hash

    Returns:
        SHA256 hash of normalized content
    """
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', content.strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def is_duplicate_idea(content: str) -> bool:
    """Check if an idea with this content already exists.

    Args:
        content: Content to check

    Returns:
        True if duplicate exists
    """
    db = memory_db.get_db()

    # Check for exact match
    cursor = db.execute(
        "SELECT id FROM ideas WHERE content = ? LIMIT 1",
        (content,)
    )
    exact_match = cursor.fetchone()
    db.close()

    return exact_match is not None


def backfill_transcript(
    file_path: str,
    start_line: Optional[int] = None
) -> dict:
    """Backfill a transcript file into the memory database.

    Indexes all indexable messages from the transcript, storing each as an idea.
    Supports incremental indexing - only processes lines after the last indexed line.
    Deduplicates content to avoid storing the same idea twice.

    Args:
        file_path: Path to the JSONL transcript file
        start_line: Optional override for start line (defaults to last indexed + 1)

    Returns:
        Dict with:
            - file_path: The processed file
            - messages_indexed: Count of messages stored
            - skipped_duplicates: Count of duplicates skipped
            - start_line: Line indexing started from
            - end_line: Last line processed
    """
    # Get start line from index state if not provided
    if start_line is None:
        last_indexed = memory_db.get_last_indexed_line(file_path)
        start_line = last_indexed + 1

    # Get indexable messages from start_line onwards
    messages = get_indexable_messages(file_path, start_line)

    # Extract session name from path
    session = memory_db.extract_session_from_path(file_path)

    # Get or create a span for this session
    span_id = None
    open_span = memory_db.get_open_span(session)
    if open_span:
        span_id = open_span["id"]

    # Store each message as an idea (with deduplication)
    messages_indexed = 0
    skipped_duplicates = 0
    last_line = start_line - 1

    for msg in messages:
        content = msg["content"]

        # Check for duplicate
        if is_duplicate_idea(content):
            skipped_duplicates += 1
            last_line = max(last_line, msg["line_num"])
            continue

        memory_db.store_idea(
            content=content,
            source_file=file_path,
            source_line=msg["line_num"],
            span_id=span_id,
            intent=None,  # Future: classify intent
            confidence=0.5
        )
        messages_indexed += 1
        last_line = max(last_line, msg["line_num"])

    # Count total lines in file
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    # Update index state to mark all lines as processed
    if total_lines > 0:
        memory_db.update_index_state(file_path, total_lines)

    return {
        "file_path": file_path,
        "messages_indexed": messages_indexed,
        "skipped_duplicates": skipped_duplicates,
        "start_line": start_line,
        "end_line": last_line if last_line >= start_line else start_line - 1
    }


def get_progress(file_path: str) -> dict:
    """Get indexing progress for a transcript file.

    Args:
        file_path: Path to the JSONL transcript file

    Returns:
        Dict with:
            - file_path: The file
            - last_indexed_line: Last line that was indexed
            - total_lines: Total lines in the file
    """
    last_indexed = memory_db.get_last_indexed_line(file_path)

    # Count total lines
    try:
        with open(file_path, "r") as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        total_lines = 0

    return {
        "file_path": file_path,
        "last_indexed_line": last_indexed,
        "total_lines": total_lines
    }


def main():
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Backfill transcripts into memory database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # backfill command
    bf_cmd = subparsers.add_parser("backfill", help="Backfill a transcript file")
    bf_cmd.add_argument("file", help="Path to JSONL transcript file")
    bf_cmd.add_argument("--start-line", type=int, help="Line number to start from")

    # progress command
    prog_cmd = subparsers.add_parser("progress", help="Get indexing progress for a file")
    prog_cmd.add_argument("file", help="Path to JSONL transcript file")

    args = parser.parse_args()

    if args.command == "backfill":
        result = backfill_transcript(args.file, args.start_line)
        print(json.dumps(result))

    elif args.command == "progress":
        result = get_progress(args.file)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
