"""Backfill functionality for indexing existing transcripts.

Enqueues transcript files for the daemon to process.
"""

import json
import os
from pathlib import Path

from db.connection import get_db


def find_transcripts() -> list[str]:
    """Find all transcript files in the Claude projects directory.

    Returns:
        List of absolute paths to transcript files
    """
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return []

    transcripts = []
    for jsonl in projects_dir.rglob("*.jsonl"):
        # Skip subagent transcripts
        if "subagents" in jsonl.parts:
            continue
        transcripts.append(str(jsonl))

    return sorted(transcripts)


def enqueue_file(file_path: str) -> dict:
    """Enqueue a single transcript file for processing.

    Args:
        file_path: Path to the JSONL transcript file

    Returns:
        Dict with file_path, file_size, and whether it was already queued
    """
    if not os.path.exists(file_path):
        return {"file_path": file_path, "error": "File not found"}

    file_size = os.path.getsize(file_path)

    db = get_db()

    # Check if already queued
    cursor = db.execute(
        "SELECT id FROM work_queue WHERE file_path = ?",
        (file_path,)
    )
    if cursor.fetchone():
        db.close()
        return {"file_path": file_path, "file_size": file_size, "status": "already_queued"}

    # Enqueue with file size
    db.execute(
        "INSERT INTO work_queue (file_path, file_size) VALUES (?, ?)",
        (file_path, file_size)
    )
    db.commit()
    db.close()

    return {"file_path": file_path, "file_size": file_size, "status": "queued"}


def enqueue_all() -> dict:
    """Find and enqueue all transcript files.

    Returns:
        Dict with counts of files found, queued, and already queued
    """
    transcripts = find_transcripts()

    queued = 0
    already_queued = 0
    total_size = 0

    for file_path in transcripts:
        result = enqueue_file(file_path)
        if result.get("status") == "queued":
            queued += 1
            total_size += result.get("file_size", 0)
        elif result.get("status") == "already_queued":
            already_queued += 1
            total_size += result.get("file_size", 0)

    return {
        "files_found": len(transcripts),
        "files_queued": queued,
        "files_already_queued": already_queued,
        "total_size_bytes": total_size
    }


def session_from_path(file_path: str) -> str:
    """Generate session ID from file path.

    Uses the file name (without extension) as session ID.

    Args:
        file_path: Path to transcript file

    Returns:
        Session identifier
    """
    return Path(file_path).stem


def process_queue(limit: int = 10) -> dict:
    """Process files from the work queue.

    Args:
        limit: Maximum number of files to process

    Returns:
        Dict with processing stats
    """
    from batch_processor import process_transcript, ProcessingError

    db = get_db()

    # Get pending files
    cursor = db.execute("""
        SELECT id, file_path FROM work_queue
        ORDER BY queued_at ASC
        LIMIT ?
    """, (limit,))
    pending = cursor.fetchall()
    db.close()

    files_processed = 0
    errors = 0
    batches_total = 0

    for row in pending:
        queue_id = row["id"]
        file_path = row["file_path"]
        session = session_from_path(file_path)

        try:
            result = process_transcript(file_path, session=session)
            batches_total += result.get("batches_processed", 0)
            files_processed += 1

            # Remove from queue on success
            db = get_db()
            db.execute("DELETE FROM work_queue WHERE id = ?", (queue_id,))
            db.commit()
            db.close()

        except ProcessingError:
            errors += 1
            # Leave in queue for retry

    return {
        "files_processed": files_processed,
        "batches_total": batches_total,
        "errors": errors
    }


def get_progress() -> dict:
    """Get overall indexing progress.

    Returns:
        Dict with queue stats and byte progress
    """
    db = get_db()

    # Queue stats
    cursor = db.execute("SELECT COUNT(*) as count, SUM(file_size) as size FROM work_queue")
    row = cursor.fetchone()
    queued_count = row["count"] or 0
    queued_size = row["size"] or 0

    # Index state stats
    cursor = db.execute("SELECT COUNT(*) as count, SUM(byte_position) as indexed FROM index_state")
    row = cursor.fetchone()
    files_indexed = row["count"] or 0
    bytes_indexed = row["indexed"] or 0

    # Get total size of all known files
    cursor = db.execute("SELECT file_path FROM index_state")
    total_size = 0
    for row in cursor:
        try:
            total_size += os.path.getsize(row["file_path"])
        except (FileNotFoundError, OSError):
            pass

    db.close()

    return {
        "queue": {
            "pending_files": queued_count,
            "pending_bytes": queued_size
        },
        "indexed": {
            "files": files_indexed,
            "bytes": bytes_indexed,
            "total_bytes": total_size,
            "percent": round(100 * bytes_indexed / total_size, 1) if total_size > 0 else 0
        }
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill transcripts into memory database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # enqueue command (single file)
    eq_cmd = subparsers.add_parser("enqueue", help="Enqueue a single transcript file")
    eq_cmd.add_argument("file", help="Path to JSONL transcript file")

    # enqueue-all command
    subparsers.add_parser("enqueue-all", help="Find and enqueue all transcript files")

    # progress command
    subparsers.add_parser("progress", help="Get overall indexing progress")

    args = parser.parse_args()

    if args.command == "enqueue":
        result = enqueue_file(args.file)
        print(json.dumps(result))

    elif args.command == "enqueue-all":
        result = enqueue_all()
        print(json.dumps(result, indent=2))

    elif args.command == "progress":
        result = get_progress()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
