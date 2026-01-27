"""Async batch processor - integrates batcher, LLM, executor, embeddings."""

import asyncio
import json
import os
from typing import Any, Optional

from batcher import collect_batches, BatcherError
from context import build_context
from db.async_connection import get_async_db
from embeddings.openai import get_embedding
from embeddings.cache import cache_source
from llm.claude import claude_complete
from protocol import format_llm_input, parse_llm_output, ProtocolError
from errors import TotalRecallError
from utils.async_retry import retry_with_backoff


class ProcessingError(Exception):
    """Error during batch processing."""
    pass


async def get_byte_position(file_path: str) -> int:
    """Get last indexed byte position for a file."""
    async def do_query():
        db = await get_async_db()
        try:
            cursor = await db.execute(
                "SELECT byte_position FROM index_state WHERE file_path = ?",
                (file_path,)
            )
            row = await cursor.fetchone()
            return row["byte_position"] if row else 0
        finally:
            await db.close()
    return await retry_with_backoff(do_query)


async def update_byte_position(file_path: str, byte_position: int):
    """Update byte position for a file."""
    async def do_update():
        db = await get_async_db()
        try:
            await db.execute("""
                INSERT INTO index_state (file_path, byte_position, last_indexed)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    byte_position = excluded.byte_position,
                    last_indexed = excluded.last_indexed
            """, (file_path, byte_position))
            await db.commit()
        finally:
            await db.close()
    await retry_with_backoff(do_update)


async def get_session_ideas(session: str, limit: int = 20) -> list[dict]:
    """Get recent ideas from this session for context."""
    async def do_query():
        db = await get_async_db()
        try:
            cursor = await db.execute("""
                SELECT content, intent as type FROM ideas i
                JOIN spans s ON i.span_id = s.id
                WHERE s.session = ?
                ORDER BY i.id DESC LIMIT ?
            """, (session, limit))
            return [dict(row) for row in await cursor.fetchall()]
        finally:
            await db.close()
    return await retry_with_backoff(do_query)


async def store_messages(messages: list, session: str, source_file: str) -> list[int]:
    """Store raw messages in the messages table for FTS/RAG.

    Returns:
        List of message IDs that were stored
    """
    if not messages:
        return []

    async def do_store():
        db = await get_async_db()
        message_ids = []
        try:
            for msg in messages:
                try:
                    await db.execute("""
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
                    cursor = await db.execute(
                        "SELECT id FROM messages WHERE source_file = ? AND line_num = ?",
                        (source_file, msg.line_num)
                    )
                    row = await cursor.fetchone()
                    if row:
                        message_ids.append(row["id"])
                except Exception as e:
                    import logging
                    logging.getLogger("total-recall").debug(f"Failed to store message: {e}")

            if message_ids:
                # Update FTS index
                try:
                    await db.execute("""
                        INSERT INTO messages_fts(messages_fts) VALUES('rebuild')
                    """)
                except Exception:
                    pass  # FTS rebuild can fail if table doesn't exist yet

                await db.commit()

            return message_ids
        finally:
            await db.close()

    return await retry_with_backoff(do_store)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


async def get_or_create_span(session: str, span_id: Optional[int], start_line: int) -> int:
    """Get existing span or create a new one."""
    if span_id:
        return span_id

    async def do_get_or_create():
        db = await get_async_db()
        try:
            # Check for existing span in this session
            cursor = await db.execute("""
                SELECT id FROM spans WHERE session = ? ORDER BY id DESC LIMIT 1
            """, (session,))
            row = await cursor.fetchone()

            if row:
                return row["id"]

            # Create new span
            cursor = await db.execute("""
                INSERT INTO spans (session, name, summary, start_line, depth)
                VALUES (?, 'New Conversation', '', ?, 0)
            """, (session, start_line))
            span_id = cursor.lastrowid
            await db.commit()
            return span_id
        finally:
            await db.close()

    return await retry_with_backoff(do_get_or_create)


async def search_related_ideas(batch_content: str, limit: int = 10) -> list[dict]:
    """Pre-search for ideas related to batch content."""
    try:
        from config import get_openai_api_key

        api_key = get_openai_api_key()
        if not api_key:
            return []

        # Create a summary query from batch content (first 1000 chars)
        query = batch_content[:1000]

        # Use async embedding with cache
        async with cache_source("indexing"):
            embedding = await get_embedding(query)

        async def do_search():
            db = await get_async_db()
            try:
                from embeddings.serialize import serialize_embedding
                embedding_blob = serialize_embedding(embedding)

                cursor = await db.execute("""
                    SELECT i.id, i.content, i.intent, s.name as topic,
                           vec_distance_cosine(ie.embedding, ?) as distance
                    FROM ideas i
                    JOIN idea_embeddings ie ON ie.idea_id = i.id
                    LEFT JOIN spans s ON i.span_id = s.id
                    ORDER BY distance ASC
                    LIMIT ?
                """, (embedding_blob, limit))

                results = []
                async for row in cursor:
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:200],
                        "type": row["intent"],
                        "topic": row["topic"],
                        "similarity": round(1 - row["distance"], 3) if row["distance"] else 0
                    })
                return results
            finally:
                await db.close()

        return await retry_with_backoff(do_search)
    except Exception:
        return []


async def call_llm(prompt: dict[str, Any], session: str = None, related_ideas: list[dict] = None) -> dict[str, Any]:
    """Call Claude CLI with formatted prompt (async wrapper).

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
        # Run blocking LLM call in thread pool
        response = await asyncio.to_thread(claude_complete, user_prompt, system_prompt)

        # Parse JSON from response
        text = response.strip()

        # Handle markdown code blocks anywhere in response
        if "```" in text:
            import re
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Try to find JSON object if text has preamble
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]

        result = json.loads(text)
        return result

    except TotalRecallError as e:
        raise ProcessingError(f"Claude CLI failed: {e}") from e
    except json.JSONDecodeError as e:
        import logging
        logging.getLogger("total-recall").warning(f"Failed to parse LLM response: {text[:500] if text else 'empty'}")
        return {"items": []}


async def execute_ideas(
    items: list[dict[str, Any]],
    span_id: int | None,
    source_file: str
) -> list[int]:
    """Store ideas from parsed LLM response."""
    async def do_execute():
        db = await get_async_db()
        idea_ids = []
        try:
            for item in items:
                content = item.get("content")
                intent = item.get("type", "context")
                source_line = item.get("source_line")
                confidence = item.get("confidence", 0.5)
                entities = item.get("entities", [])

                if not content or not source_line:
                    continue

                cursor = await db.execute("""
                    INSERT INTO ideas (span_id, content, intent, confidence, source_file, source_line)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_file, source_line) DO UPDATE SET
                        content = excluded.content,
                        intent = excluded.intent,
                        confidence = excluded.confidence,
                        span_id = COALESCE(excluded.span_id, span_id)
                """, (span_id, content, intent, confidence, source_file, source_line))

                cursor = await db.execute("""
                    SELECT id FROM ideas WHERE source_file = ? AND source_line = ?
                """, (source_file, source_line))
                idea_id = (await cursor.fetchone())["id"]
                idea_ids.append(idea_id)

                # Store entities
                for entity_name in entities:
                    await db.execute("""
                        INSERT INTO entities (name, type)
                        VALUES (?, 'concept')
                        ON CONFLICT(name, type) DO UPDATE SET name = name
                    """, (entity_name,))

                    cursor = await db.execute("""
                        SELECT id FROM entities WHERE name = ? AND type = 'concept'
                    """, (entity_name,))
                    entity_id = (await cursor.fetchone())["id"]

                    await db.execute("""
                        INSERT OR IGNORE INTO idea_entities (idea_id, entity_id)
                        VALUES (?, ?)
                    """, (idea_id, entity_id))

            await db.commit()
            return idea_ids
        finally:
            await db.close()

    return await retry_with_backoff(do_execute)


async def execute_topic_update(topic_update: dict[str, str], span_id: int) -> None:
    """Update span name/summary and link to topic."""
    async def do_update():
        db = await get_async_db()
        try:
            name = topic_update.get("name")
            summary = topic_update.get("summary")

            if name or summary:
                if name and summary:
                    await db.execute("""
                        UPDATE spans SET name = ?, summary = ? WHERE id = ?
                    """, (name, summary, span_id))
                elif name:
                    await db.execute("UPDATE spans SET name = ? WHERE id = ?", (name, span_id))
                else:
                    await db.execute("UPDATE spans SET summary = ? WHERE id = ?", (summary, span_id))

            if name:
                canonical = name.lower().strip()[:50]
                cursor = await db.execute("""
                    SELECT id FROM topics WHERE canonical_name = ?
                """, (canonical,))
                row = await cursor.fetchone()

                if row:
                    topic_id = row["id"]
                else:
                    cursor = await db.execute("""
                        INSERT INTO topics (name, canonical_name, summary)
                        VALUES (?, ?, ?)
                    """, (name[:100], canonical, summary))
                    topic_id = cursor.lastrowid

                await db.execute("UPDATE spans SET topic_id = ? WHERE id = ?", (topic_id, span_id))

            await db.commit()
        finally:
            await db.close()

    await retry_with_backoff(do_update)


async def execute_new_span(
    new_span: dict[str, str],
    session: str,
    parent_id: int | None,
    start_line: int
) -> int:
    """Create a new child span for topic shift."""
    async def do_create():
        db = await get_async_db()
        try:
            name = new_span.get("name", "New Span")
            reason = new_span.get("reason", "")

            depth = 0
            if parent_id:
                cursor = await db.execute("SELECT depth FROM spans WHERE id = ?", (parent_id,))
                row = await cursor.fetchone()
                if row:
                    depth = row["depth"] + 1

            cursor = await db.execute("""
                INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session, name, reason, start_line, depth, parent_id))

            span_id = cursor.lastrowid
            await db.commit()
            return span_id
        finally:
            await db.close()

    return await retry_with_backoff(do_create)


async def execute_relations(relations: list[dict[str, Any]], source_file: str) -> int:
    """Create relations between ideas."""
    async def do_create():
        db = await get_async_db()
        created = 0
        try:
            for rel in relations:
                from_line = rel.get("from_line")
                to_idea_id = rel.get("to_idea_id")
                rel_type = rel.get("type", "related")

                if not from_line or not to_idea_id:
                    continue

                cursor = await db.execute("""
                    SELECT id FROM ideas WHERE source_file = ? AND source_line = ?
                """, (source_file, from_line))
                row = await cursor.fetchone()

                if not row:
                    continue

                from_id = row["id"]

                cursor = await db.execute("SELECT id FROM ideas WHERE id = ?", (to_idea_id,))
                if not await cursor.fetchone():
                    continue

                try:
                    await db.execute("""
                        INSERT INTO relations (from_id, to_id, relation_type)
                        VALUES (?, ?, ?)
                    """, (from_id, to_idea_id, rel_type))
                    created += 1
                except Exception:
                    pass

            await db.commit()
            return created
        finally:
            await db.close()

    return await retry_with_backoff(do_create)


async def embed_ideas(idea_ids: list[int]) -> int:
    """Generate embeddings for ideas asynchronously."""
    if not idea_ids:
        return 0

    from config import get_openai_api_key
    if not get_openai_api_key():
        return 0

    embedded = 0

    for idea_id in idea_ids:
        try:
            # Get idea content
            async def get_content():
                db = await get_async_db()
                try:
                    cursor = await db.execute("SELECT content FROM ideas WHERE id = ?", (idea_id,))
                    row = await cursor.fetchone()
                    if not row:
                        return None

                    # Check if already embedded
                    cursor = await db.execute("SELECT 1 FROM idea_embeddings WHERE idea_id = ?", (idea_id,))
                    if await cursor.fetchone():
                        return None

                    return row["content"]
                finally:
                    await db.close()

            content = await retry_with_backoff(get_content)
            if content is None:
                continue

            # Generate embedding
            async with cache_source("indexing"):
                embedding = await get_embedding(content)

            # Store embedding
            async def store_embedding():
                db = await get_async_db()
                try:
                    from embeddings.serialize import serialize_embedding
                    embedding_blob = serialize_embedding(embedding)
                    await db.execute("""
                        INSERT INTO idea_embeddings (idea_id, embedding)
                        VALUES (?, ?)
                    """, (idea_id, embedding_blob))
                    await db.commit()
                finally:
                    await db.close()

            await retry_with_backoff(store_embedding)
            embedded += 1
        except Exception:
            continue

    return embedded


async def embed_messages(message_ids: list[int]) -> int:
    """Generate embeddings for messages asynchronously."""
    if not message_ids:
        return 0

    from config import get_openai_api_key
    if not get_openai_api_key():
        return 0

    embedded = 0

    for message_id in message_ids:
        try:
            async def get_content():
                db = await get_async_db()
                try:
                    cursor = await db.execute("SELECT content FROM messages WHERE id = ?", (message_id,))
                    row = await cursor.fetchone()
                    if not row:
                        return None

                    cursor = await db.execute("SELECT 1 FROM message_embeddings WHERE message_id = ?", (message_id,))
                    if await cursor.fetchone():
                        return None

                    return row["content"]
                finally:
                    await db.close()

            content = await retry_with_backoff(get_content)
            if content is None:
                continue

            async with cache_source("indexing"):
                embedding = await get_embedding(content)

            async def store_embedding():
                db = await get_async_db()
                try:
                    from embeddings.serialize import serialize_embedding
                    embedding_blob = serialize_embedding(embedding)
                    await db.execute("""
                        INSERT INTO message_embeddings (message_id, embedding)
                        VALUES (?, ?)
                    """, (message_id, embedding_blob))
                    await db.commit()
                finally:
                    await db.close()

            await retry_with_backoff(store_embedding)
            embedded += 1
        except Exception:
            continue

    return embedded


async def _process_transcript_impl(
    file_path: str,
    session: str,
    span_id: Optional[int] = None,
    window_seconds: float = 300.0,
    target_tokens: int = 30000
) -> dict[str, Any]:
    """
    Process a transcript file using batched LLM analysis (async).

    Args:
        file_path: Path to the transcript file
        session: Session identifier
        span_id: Optional current span ID
        window_seconds: Time window for batching messages
        target_tokens: Target tokens per LLM call

    Returns:
        Dict with processing stats

    Raises:
        ProcessingError: If processing fails
    """
    # Get starting position
    start_byte = await get_byte_position(file_path)

    # Check if file has new content
    if not os.path.exists(file_path):
        return {"batches_processed": 0, "error": "file_not_found"}

    file_size = os.path.getsize(file_path)
    if start_byte >= file_size:
        return {"batches_processed": 0, "status": "already_indexed"}

    # Collect batches (sync operation - file I/O)
    try:
        batches = await asyncio.to_thread(
            lambda: list(collect_batches(file_path, start_byte, window_seconds))
        )
    except BatcherError as e:
        raise ProcessingError(f"Failed to collect batches: {e}") from e

    if not batches:
        await update_byte_position(file_path, file_size)
        return {"batches_processed": 0, "status": "no_messages"}

    # Smart batching: combine small batches until we hit target_tokens
    smart_batches = []
    current_batch_msgs = []
    current_tokens = 0
    current_end_byte = start_byte

    for batch in batches:
        batch_tokens = sum(estimate_tokens(m.content) for m in batch.messages)
        if current_tokens + batch_tokens > target_tokens and current_batch_msgs:
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
    current_span_id = await get_or_create_span(session, span_id, first_line)

    # Get existing ideas from session for context
    existing_ideas = await get_session_ideas(session, limit=20)

    batches_processed = 0
    ideas_stored = 0
    relations_created = 0

    recent_messages = []

    for messages, end_byte in smart_batches:
        # Create a pseudo-batch for formatting
        from batcher import Batch
        batch = Batch(messages=messages, end_byte=end_byte)

        # Build context (sync - just dict building)
        context = build_context(session=session, span_id=current_span_id)

        # Format LLM input
        llm_input = format_llm_input(batch, context, recent_messages)

        # Pre-search for related ideas to avoid duplicates
        batch_text = " ".join(m.content for m in messages)
        related_ideas = await search_related_ideas(batch_text, limit=10)

        # Call LLM with related ideas context
        try:
            llm_response = await call_llm(llm_input, session=session, related_ideas=related_ideas)
        except ProcessingError:
            raise

        # Parse response
        try:
            parsed = parse_llm_output(llm_response)
        except ProtocolError as e:
            parsed = parse_llm_output({})

        # Execute actions
        if parsed.topic_update and current_span_id:
            await execute_topic_update(parsed.topic_update, span_id=current_span_id)

        if parsed.new_span:
            current_span_id = await execute_new_span(
                parsed.new_span,
                session=session,
                parent_id=current_span_id,
                start_line=batch.messages[0].line_num if batch.messages else 1
            )

        if parsed.items:
            idea_ids = await execute_ideas(parsed.items, span_id=current_span_id, source_file=file_path)
            ideas_stored += len(idea_ids)
            # Generate embeddings for new ideas
            await embed_ideas(idea_ids)

        if parsed.relations:
            relations_created += await execute_relations(parsed.relations, source_file=file_path)

        # Store raw messages for FTS/RAG and generate embeddings
        message_ids = await store_messages(batch.messages, session=session, source_file=file_path)
        await embed_messages(message_ids)

        # Update recent messages for next batch
        for msg in batch.messages:
            recent_messages.append({
                "role": msg.role,
                "content": msg.content[:500],
                "timestamp": msg.timestamp
            })
        recent_messages = recent_messages[-10:]

        # Update byte position after each batch
        await update_byte_position(file_path, batch.end_byte)
        batches_processed += 1

    return {
        "batches_processed": batches_processed,
        "ideas_stored": ideas_stored,
        "relations_created": relations_created,
        "final_span_id": current_span_id
    }


# Alias for async callers (like daemon)
process_transcript_async = _process_transcript_impl


# Sync wrapper for simple use cases (CLI, scripts)
def process_transcript(
    file_path: str,
    session: str,
    span_id: Optional[int] = None,
    window_seconds: float = 300.0,
    target_tokens: int = 30000
) -> dict[str, Any]:
    """Sync wrapper for process_transcript."""
    return asyncio.run(_process_transcript_impl(
        file_path, session, span_id, window_seconds, target_tokens
    ))
