"""Run indexing agent - Slice 5.4.

Top-level function to run the complete indexing pipeline.
"""

from typing import Any, Literal

from db.connection import get_db
from indexer.agent_input import format_agent_input
from indexer.batch_collector import BatchUpdate
from indexer.executor import execute_agent_output
from indexer.output_parser import parse_agent_output
from indexer.prompts import INDEXING_SYSTEM_PROMPT
from llm.agent_harness import run_agent
from llm.tool_registry import INDEXING_TOOLS


async def run_indexing_agent(
    updates: list[BatchUpdate],
    mode: Literal["continuous", "backfill"] = "continuous",
) -> dict[str, Any]:
    """Run the indexing agent on batch updates.

    Args:
        updates: List of batch updates to process
        mode: Processing mode

    Returns:
        Dict with execution stats
    """
    if not updates:
        return {
            "ideas_created": 0,
            "sessions_processed": 0,
            "bytes_processed": 0,
        }

    # Format input for agent
    user_input = format_agent_input(updates, mode=mode)

    # Run agent
    raw_output = await run_agent(
        system_prompt=INDEXING_SYSTEM_PROMPT,
        user_input=user_input,
        tools=INDEXING_TOOLS,
        max_turns=10,
    )

    # Handle error case
    if "error" in raw_output and "raw" in raw_output:
        # Agent returned non-JSON - try to continue with empty output
        raw_output = {"ideas": []}

    # Parse output
    output = parse_agent_output(raw_output)

    # Execute for each session
    total_stats = {
        "ideas_created": 0,
        "topic_updates": 0,
        "topic_changes": 0,
        "questions_answered": 0,
        "relations_created": 0,
        "sessions_processed": 0,
        "bytes_processed": 0,
    }

    for update in updates:
        # Get or create span for session
        span_id = await get_or_create_span(update.session, update.file_path)

        # Execute output
        stats = await execute_agent_output(
            output=output,
            session=update.session,
            source_file=update.file_path,
            span_id=span_id,
        )

        # Accumulate stats
        for key, value in stats.items():
            if key in total_stats:
                total_stats[key] += value

        # Update byte position
        await update_byte_position(update.file_path, update.end_byte)

        total_stats["sessions_processed"] += 1
        total_stats["bytes_processed"] += update.end_byte - update.start_byte

    return total_stats


async def get_or_create_span(session: str, file_path: str) -> int:
    """Get existing span for session or create a new one.

    Args:
        session: Session ID
        file_path: Source file path

    Returns:
        Span ID
    """
    db = get_db()

    # Try to find existing span
    cursor = db.execute(
        "SELECT id FROM spans WHERE session = ? ORDER BY start_line DESC LIMIT 1",
        (session,)
    )
    row = cursor.fetchone()

    if row:
        db.close()
        return row["id"]

    # Create new span
    cursor = db.execute("""
        INSERT INTO spans (session, name, summary, start_line, depth)
        VALUES (?, 'New Session', 'Auto-created span', 1, 0)
    """, (session,))
    span_id = cursor.lastrowid
    db.commit()
    db.close()

    return span_id


async def update_byte_position(file_path: str, byte_position: int) -> None:
    """Update the byte position for a file.

    Args:
        file_path: Path to transcript file
        byte_position: New byte position
    """
    db = get_db()
    db.execute("""
        INSERT INTO index_state (file_path, byte_position)
        VALUES (?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            byte_position = excluded.byte_position,
            last_indexed = datetime('now')
    """, (file_path, byte_position))
    db.commit()
    db.close()
