"""Batch processor - integrates batcher, LLM, executor, embeddings."""

import json
import os
from typing import Any, Optional

from batcher import collect_batches, BatcherError
from context import build_context
from db.connection import get_db
from executor import execute_ideas, execute_topic_update, execute_new_span, execute_relations, embed_ideas, embed_messages
from llm.claude import claude_complete
from protocol import format_llm_input, parse_llm_output, ProtocolError
from errors import TotalRecallError


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


def get_session_ideas(session: str, limit: int = 20) -> list[dict]:
    """Get recent ideas from this session for context."""
    db = get_db()
    cursor = db.execute("""
        SELECT content, intent as type FROM ideas i
        JOIN spans s ON i.span_id = s.id
        WHERE s.session = ?
        ORDER BY i.id DESC LIMIT ?
    """, (session, limit))
    ideas = [dict(row) for row in cursor.fetchall()]
    db.close()
    return ideas


def store_messages(messages: list, session: str, source_file: str) -> list[int]:
    """Store raw messages in the messages table for FTS/RAG.

    Returns:
        List of message IDs that were stored
    """
    if not messages:
        return []

    db = get_db()
    message_ids = []

    for msg in messages:
        try:
            db.execute("""
                INSERT INTO messages (session, line_num, role, content, timestamp, source_file)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_file, line_num) DO UPDATE SET
                    content = excluded.content,
                    timestamp = excluded.timestamp
            """, (
                session,
                msg.line_num,
                msg.role,
                msg.content,
                msg.timestamp,
                source_file
            ))
            # Get the message ID
            cursor = db.execute(
                "SELECT id FROM messages WHERE source_file = ? AND line_num = ?",
                (source_file, msg.line_num)
            )
            row = cursor.fetchone()
            if row:
                message_ids.append(row["id"])
        except Exception as e:
            # Log but continue - don't fail on message storage
            import logging
            logging.getLogger("total-recall").debug(f"Failed to store message: {e}")

    if message_ids:
        # Update FTS index
        try:
            db.execute("""
                INSERT INTO messages_fts(messages_fts) VALUES('rebuild')
            """)
        except Exception:
            pass  # FTS rebuild can fail if table doesn't exist yet

        db.commit()

    db.close()
    return message_ids


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


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


def search_related_ideas(batch_content: str, limit: int = 10) -> list[dict]:
    """Pre-search for ideas related to batch content."""
    import os
    try:
        from embeddings.openai import OpenAIEmbeddingProvider

        api_key = os.environ.get("OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS")
        if not api_key:
            return []

        # Create a summary query from batch content (first 1000 chars)
        query = batch_content[:1000]

        provider = OpenAIEmbeddingProvider(api_key=api_key)
        embedding = provider.embed(query)

        db = get_db()
        cursor = db.execute("""
            SELECT i.id, i.content, i.intent, s.name as topic,
                   vec_distance_cosine(i.embedding, ?) as distance
            FROM ideas i
            LEFT JOIN spans s ON i.span_id = s.id
            WHERE i.embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
        """, (json.dumps(embedding), limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "content": row["content"][:200],
                "type": row["intent"],
                "topic": row["topic"],
                "similarity": round(1 - row["distance"], 3)
            })
        db.close()
        return results
    except Exception:
        return []


def call_llm(prompt: dict[str, Any], session: str = None, related_ideas: list[dict] = None) -> dict[str, Any]:
    """Call Claude CLI with formatted prompt.

    Args:
        prompt: Formatted prompt from format_llm_input
        session: Session identifier
        related_ideas: Pre-searched related ideas for context

    Returns:
        Parsed LLM response

    Raises:
        ProcessingError: If LLM call fails
    """
    system_prompt = """You extract insights from conversation transcripts into structured JSON.

Identify these types:
- decision: Explicit choices ("we'll use X", "let's go with Y")
- conclusion: Findings reached ("it works because...", "the issue was...")
- question: Open questions needing answers
- problem: Issues, bugs, challenges identified
- solution: Fixes or resolutions
- todo: Action items or tasks
- context: Important background info

RULES:
- Each item MUST have source_line from new_messages
- Skip trivial messages (greetings, "ok", "thanks")
- Check EXISTING_IDEAS - don't duplicate, use related_to for links
- Output ONLY the JSON object, no other text"""

    # Build user prompt with related ideas context
    user_parts = []

    if related_ideas:
        user_parts.append("EXISTING_IDEAS (check for duplicates, link via related_to):")
        for idea in related_ideas:
            user_parts.append(f"  ID {idea['id']} [{idea['type']}]: {idea['content']}")
        user_parts.append("")

    user_parts.append("NEW_MESSAGES to analyze:")
    user_parts.append(json.dumps(prompt, indent=2))
    user_parts.append("")
    user_parts.append("Respond with ONLY a JSON object, no other text:")
    user_parts.append('{"topic_update": {"name": "topic name", "summary": "..."}, "items": [...]}')

    user_prompt = "\n".join(user_parts)

    try:
        response = claude_complete(user_prompt, system=system_prompt)

        # Parse JSON from response
        text = response.strip()

        # Handle markdown code blocks anywhere in response
        if "```" in text:
            # Find JSON block - look for ```json or just ```
            import re
            # Match ```json ... ``` or ``` ... ```
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try to find JSON object if text has preamble
        if not text.startswith("{"):
            # Look for first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

        result = json.loads(text)
        return result

    except TotalRecallError as e:
        raise ProcessingError(f"Claude CLI failed: {e}") from e
    except json.JSONDecodeError as e:
        # Log the bad response for debugging
        import logging
        logging.getLogger("total-recall").warning(f"Failed to parse LLM response: {text[:500] if text else 'empty'}")
        return {"items": []}


def process_transcript(
    file_path: str,
    session: str,
    span_id: Optional[int] = None,
    window_seconds: float = 300.0,
    target_tokens: int = 30000  # Conservative limit - models work better with headroom
) -> dict[str, Any]:
    """
    Process a transcript file using batched LLM analysis.

    Args:
        file_path: Path to the transcript file
        session: Session identifier
        span_id: Optional current span ID
        window_seconds: Time window for batching messages
        target_tokens: Target tokens per LLM call (for smart batching)

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

    # Smart batching: combine small batches until we hit target_tokens
    smart_batches = []
    current_batch_msgs = []
    current_tokens = 0
    current_end_byte = start_byte

    for batch in batches:
        batch_tokens = sum(estimate_tokens(m.content) for m in batch.messages)
        if current_tokens + batch_tokens > target_tokens and current_batch_msgs:
            # Save current accumulated batch
            smart_batches.append((current_batch_msgs, current_end_byte))
            current_batch_msgs = []
            current_tokens = 0

        current_batch_msgs.extend(batch.messages)
        current_tokens += batch_tokens
        current_end_byte = batch.end_byte

    if current_batch_msgs:
        smart_batches.append((current_batch_msgs, current_end_byte))

    # Get or create span
    first_line = smart_batches[0][0][0].line_num if smart_batches and smart_batches[0][0] else 1
    current_span_id = get_or_create_span(session, span_id, first_line)

    # Get existing ideas from session for context
    existing_ideas = get_session_ideas(session, limit=20)

    batches_processed = 0
    ideas_stored = 0
    relations_created = 0

    recent_messages = []

    for messages, end_byte in smart_batches:
        # Create a pseudo-batch for formatting
        from batcher import Batch
        batch = Batch(messages=messages, end_byte=end_byte)

        # Build context
        context = build_context(session=session, span_id=current_span_id)

        # Format LLM input
        llm_input = format_llm_input(batch, context, recent_messages)

        # Pre-search for related ideas to avoid duplicates
        batch_text = " ".join(m.content for m in messages)
        related_ideas = search_related_ideas(batch_text, limit=10)

        # Call LLM with related ideas context
        try:
            llm_response = call_llm(llm_input, session=session, related_ideas=related_ideas)
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
            idea_ids = execute_ideas(parsed.items, span_id=current_span_id, source_file=file_path)
            ideas_stored += len(idea_ids)
            # Generate embeddings for new ideas
            embed_ideas(idea_ids)

        if parsed.relations:
            relations_created += execute_relations(parsed.relations, source_file=file_path)

        # Store raw messages for FTS/RAG and generate embeddings
        message_ids = store_messages(batch.messages, session=session, source_file=file_path)
        embed_messages(message_ids)

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
