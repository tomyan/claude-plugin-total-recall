"""Batch processor - integrates batcher, LLM, executor, embeddings."""

import json
import os
from typing import Any, Optional

from openai import OpenAI

from batcher import collect_batches, BatcherError
from context import build_context
from db.connection import get_db
from executor import execute_ideas, execute_topic_update, execute_new_span, execute_relations
from protocol import format_llm_input, parse_llm_output, ProtocolError


class ProcessingError(Exception):
    """Error during batch processing."""
    pass


def get_byte_position(file_path: str) -> int:
    """Get last indexed byte position for a file."""
    db = get_db()
    cursor = db.execute(
        "SELECT byte_position FROM index_state WHERE file_path = ?",
        (file_path,)
    )
    row = cursor.fetchone()
    db.close()
    return row["byte_position"] if row else 0


def update_byte_position(file_path: str, byte_position: int):
    """Update byte position for a file."""
    db = get_db()
    db.execute("""
        INSERT INTO index_state (file_path, byte_position, last_indexed)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(file_path) DO UPDATE SET
            byte_position = excluded.byte_position,
            last_indexed = excluded.last_indexed
    """, (file_path, byte_position))
    db.commit()
    db.close()


def get_or_create_span(session: str, span_id: Optional[int], start_line: int) -> int:
    """Get existing span or create a new one."""
    if span_id:
        return span_id

    db = get_db()

    # Check for existing span in this session
    cursor = db.execute("""
        SELECT id FROM spans WHERE session = ? ORDER BY id DESC LIMIT 1
    """, (session,))
    row = cursor.fetchone()

    if row:
        db.close()
        return row["id"]

    # Create new span
    cursor = db.execute("""
        INSERT INTO spans (session, name, summary, start_line, depth)
        VALUES (?, 'New Conversation', '', ?, 0)
    """, (session, start_line))
    span_id = cursor.lastrowid
    db.commit()
    db.close()

    return span_id


def call_llm(prompt: dict[str, Any]) -> dict[str, Any]:
    """Call LLM with formatted prompt.

    Args:
        prompt: Formatted prompt from format_llm_input

    Returns:
        Parsed LLM response

    Raises:
        ProcessingError: If LLM call fails
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Return empty response if no API key
        return {"items": []}

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = """You are analyzing a conversation transcript to extract meaningful insights.

Given the context and new messages, return a JSON response with:
- topic_update: {name, summary} if the topic should be updated
- new_span: {name, reason} if there's a significant topic shift requiring a new span
- items: array of insights, each with:
  - type: one of "decision", "conclusion", "question", "problem", "solution", "todo", "context"
  - content: the insight text
  - source_line: line number from new_messages
  - confidence: 0.0-1.0
  - entities: array of relevant entity names (optional)
- relations: array of {from_line, to_idea_id, type} for connections to existing ideas
- skip_lines: array of line numbers to skip (greetings, acknowledgments, etc.)

Focus on extracting actionable insights, decisions, and important context."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt, indent=2)}
            ],
            response_format={"type": "json_object"},
            timeout=120
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        raise ProcessingError(f"LLM call failed: {e}") from e


def process_transcript(
    file_path: str,
    session: str,
    span_id: Optional[int] = None,
    window_seconds: float = 5.0
) -> dict[str, Any]:
    """
    Process a transcript file using batched LLM analysis.

    Args:
        file_path: Path to the transcript file
        session: Session identifier
        span_id: Optional current span ID
        window_seconds: Time window for batching messages

    Returns:
        Dict with processing stats

    Raises:
        ProcessingError: If processing fails
    """
    # Get starting position
    start_byte = get_byte_position(file_path)

    # Check if file has new content
    if not os.path.exists(file_path):
        return {"batches_processed": 0, "error": "file_not_found"}

    file_size = os.path.getsize(file_path)
    if start_byte >= file_size:
        return {"batches_processed": 0, "status": "already_indexed"}

    # Collect batches
    try:
        batches = list(collect_batches(file_path, start_byte, window_seconds))
    except BatcherError as e:
        raise ProcessingError(f"Failed to collect batches: {e}") from e

    if not batches:
        update_byte_position(file_path, file_size)
        return {"batches_processed": 0, "status": "no_messages"}

    # Get or create span
    first_line = batches[0].messages[0].line_num if batches[0].messages else 1
    current_span_id = get_or_create_span(session, span_id, first_line)

    batches_processed = 0
    ideas_stored = 0
    relations_created = 0

    recent_messages = []

    for batch in batches:
        # Build context
        context = build_context(session=session, span_id=current_span_id)

        # Format LLM input
        llm_input = format_llm_input(batch, context, recent_messages)

        # Call LLM
        try:
            llm_response = call_llm(llm_input)
        except ProcessingError:
            raise

        # Parse response
        try:
            parsed = parse_llm_output(llm_response)
        except ProtocolError as e:
            # Log but continue with empty result
            parsed = parse_llm_output({})

        # Execute actions
        if parsed.topic_update and current_span_id:
            execute_topic_update(parsed.topic_update, span_id=current_span_id)

        if parsed.new_span:
            current_span_id = execute_new_span(
                parsed.new_span,
                session=session,
                parent_id=current_span_id,
                start_line=batch.messages[0].line_num if batch.messages else 1
            )

        if parsed.items:
            execute_ideas(parsed.items, span_id=current_span_id, source_file=file_path)
            ideas_stored += len(parsed.items)

        if parsed.relations:
            relations_created += execute_relations(parsed.relations, source_file=file_path)

        # Update recent messages for next batch
        for msg in batch.messages:
            recent_messages.append({
                "role": msg.role,
                "content": msg.content[:500],
                "timestamp": msg.timestamp
            })
        # Keep only last 10
        recent_messages = recent_messages[-10:]

        # Update byte position after each batch
        update_byte_position(file_path, batch.end_byte)
        batches_processed += 1

    return {
        "batches_processed": batches_processed,
        "ideas_stored": ideas_stored,
        "relations_created": relations_created,
        "final_span_id": current_span_id
    }
