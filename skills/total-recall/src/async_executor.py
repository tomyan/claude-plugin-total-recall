"""Async executor for total-recall indexing operations.

Provides async wrappers for indexing operations that integrate with the
async database connections and embedding cache.
"""

import asyncio
from datetime import datetime
from typing import Optional

from db.async_connection import get_async_db
from embeddings.cache import cache_source, flush_write_queue
from embeddings.openai import get_embedding_async, get_embeddings_batch_async
from embeddings.serialize import serialize_embedding
from utils.async_retry import retry_with_backoff


async def store_idea_async(
    content: str,
    source_file: str,
    source_line: int,
    span_id: Optional[int] = None,
    intent: str = "context",
    confidence: float = 0.5,
    entities: Optional[list] = None,
    message_time: Optional[str] = None
) -> int:
    """Store an idea asynchronously with embedding.

    Args:
        content: The idea content
        source_file: Path to source file
        source_line: Line number in source file
        span_id: Optional span ID this idea belongs to
        intent: Intent classification
        confidence: Confidence score
        entities: Optional list of entities
        message_time: Optional ISO timestamp of the message

    Returns:
        The new idea's ID
    """
    # Get embedding asynchronously with cache
    async with cache_source("indexing"):
        embedding = await get_embedding_async(content)

    async def do_store():
        db = await get_async_db()
        try:
            now = datetime.utcnow().isoformat()

            # Insert idea
            cursor = await db.execute("""
                INSERT INTO ideas (content, source_file, source_line, span_id,
                                  intent, confidence, created_at, message_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (content, source_file, source_line, span_id,
                  intent, confidence, now, message_time))
            idea_id = cursor.lastrowid

            # Insert embedding
            embedding_blob = serialize_embedding(embedding)
            await db.execute("""
                INSERT INTO idea_embeddings (idea_id, embedding)
                VALUES (?, ?)
            """, (idea_id, embedding_blob))

            # Insert entities if provided
            if entities:
                for name, entity_type in entities:
                    # Insert or get the entity
                    await db.execute("""
                        INSERT OR IGNORE INTO entities (name, type)
                        VALUES (?, ?)
                    """, (name, entity_type))

                    # Get the entity_id
                    cursor = await db.execute(
                        "SELECT id FROM entities WHERE name = ?",
                        (name,)
                    )
                    entity_row = await cursor.fetchone()
                    if entity_row:
                        entity_id = entity_row["id"]
                        await db.execute("""
                            INSERT OR IGNORE INTO idea_entities (idea_id, entity_id)
                            VALUES (?, ?)
                        """, (idea_id, entity_id))

            await db.commit()
            return idea_id
        finally:
            await db.close()

    return await retry_with_backoff(do_store)


async def store_ideas_batch_async(
    ideas: list[dict]
) -> list[int]:
    """Store multiple ideas in a batch with batch embedding.

    More efficient than storing ideas one by one.

    Args:
        ideas: List of idea dicts with content, source_file, source_line, etc.

    Returns:
        List of new idea IDs
    """
    if not ideas:
        return []

    # Get all embeddings in a batch
    contents = [idea["content"] for idea in ideas]
    async with cache_source("indexing"):
        embeddings = await get_embeddings_batch_async(contents)

    async def do_store_batch():
        db = await get_async_db()
        try:
            now = datetime.utcnow().isoformat()
            idea_ids = []

            for idea, embedding in zip(ideas, embeddings):
                # Insert idea
                cursor = await db.execute("""
                    INSERT INTO ideas (content, source_file, source_line, span_id,
                                      intent, confidence, created_at, message_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    idea["content"],
                    idea.get("source_file"),
                    idea.get("source_line"),
                    idea.get("span_id"),
                    idea.get("intent", "context"),
                    idea.get("confidence", 0.5),
                    now,
                    idea.get("message_time")
                ))
                idea_id = cursor.lastrowid
                idea_ids.append(idea_id)

                # Insert embedding
                embedding_blob = serialize_embedding(embedding)
                await db.execute("""
                    INSERT INTO idea_embeddings (idea_id, embedding)
                    VALUES (?, ?)
                """, (idea_id, embedding_blob))

                # Insert entities if provided
                entities = idea.get("entities")
                if entities:
                    for name, entity_type in entities:
                        # Insert or get the entity
                        await db.execute("""
                            INSERT OR IGNORE INTO entities (name, type)
                            VALUES (?, ?)
                        """, (name, entity_type))

                        # Get the entity_id
                        cursor = await db.execute(
                            "SELECT id FROM entities WHERE name = ?",
                            (name,)
                        )
                        entity_row = await cursor.fetchone()
                        if entity_row:
                            entity_id = entity_row["id"]
                            await db.execute("""
                                INSERT OR IGNORE INTO idea_entities (idea_id, entity_id)
                                VALUES (?, ?)
                            """, (idea_id, entity_id))

            await db.commit()
            return idea_ids
        finally:
            await db.close()

    return await retry_with_backoff(do_store_batch)


async def create_span_async(
    session: str,
    name: str,
    start_line: int,
    parent_id: Optional[int] = None,
    depth: int = 0,
    start_time: Optional[str] = None
) -> int:
    """Create a new topic span asynchronously.

    Args:
        session: Session identifier
        name: Span name/topic
        start_line: Starting line number
        parent_id: Optional parent span ID
        depth: Hierarchy depth (0 = top level)
        start_time: Optional ISO timestamp

    Returns:
        New span ID
    """
    async def do_create():
        db = await get_async_db()
        try:
            now = datetime.utcnow().isoformat()
            cursor = await db.execute("""
                INSERT INTO spans (session, name, start_line, parent_id, depth,
                                  created_at, start_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session, name, start_line, parent_id, depth, now, start_time))
            span_id = cursor.lastrowid
            await db.commit()
            return span_id
        finally:
            await db.close()

    return await retry_with_backoff(do_create)


async def close_span_async(
    span_id: int,
    end_line: int,
    summary: str = "",
    end_time: Optional[str] = None
) -> None:
    """Close a span with end line and summary.

    Args:
        span_id: ID of the span to close
        end_line: Ending line number
        summary: Optional summary of the span
        end_time: Optional ISO timestamp
    """
    async def do_close():
        db = await get_async_db()
        try:
            await db.execute("""
                UPDATE spans
                SET end_line = ?, summary = ?, end_time = ?
                WHERE id = ?
            """, (end_line, summary, end_time, span_id))
            await db.commit()
        finally:
            await db.close()

    await retry_with_backoff(do_close)


async def update_span_embedding_async(
    span_id: int,
    include_ideas: bool = True
) -> None:
    """Update a span's embedding based on its content and ideas.

    Args:
        span_id: Span ID to update
        include_ideas: Whether to include idea embeddings
    """
    async def do_update():
        db = await get_async_db()
        try:
            # Get span name and summary
            cursor = await db.execute("""
                SELECT name, summary FROM spans WHERE id = ?
            """, (span_id,))
            row = await cursor.fetchone()
            if not row:
                return

            # Build content for embedding
            content_parts = [row["name"]]
            if row["summary"]:
                content_parts.append(row["summary"])

            if include_ideas:
                # Get top ideas from this span
                cursor = await db.execute("""
                    SELECT content FROM ideas
                    WHERE span_id = ?
                    ORDER BY confidence DESC
                    LIMIT 10
                """, (span_id,))
                rows = await cursor.fetchall()
                for r in rows:
                    content_parts.append(r["content"][:200])

            # Generate embedding
            combined = "\n".join(content_parts)
            async with cache_source("indexing"):
                embedding = await get_embedding_async(combined)

            embedding_blob = serialize_embedding(embedding)

            # Delete existing then insert (virtual tables don't support UPSERT)
            await db.execute(
                "DELETE FROM span_embeddings WHERE span_id = ?",
                (span_id,)
            )
            await db.execute("""
                INSERT INTO span_embeddings (span_id, embedding)
                VALUES (?, ?)
            """, (span_id, embedding_blob))

            await db.commit()
        finally:
            await db.close()

    await retry_with_backoff(do_update)


async def add_relation_async(
    from_id: int,
    to_id: int,
    relation_type: str
) -> None:
    """Add a relation between two ideas asynchronously.

    Args:
        from_id: Source idea ID
        to_id: Target idea ID
        relation_type: Type of relation
    """
    async def do_add():
        db = await get_async_db()
        try:
            await db.execute("""
                INSERT OR IGNORE INTO relations (from_id, to_id, relation_type)
                VALUES (?, ?, ?)
            """, (from_id, to_id, relation_type))
            await db.commit()
        finally:
            await db.close()

    await retry_with_backoff(do_add)


async def get_open_span_async(session: str) -> Optional[dict]:
    """Get the currently open span for a session.

    Args:
        session: Session identifier

    Returns:
        Span dict or None if no open span
    """
    async def do_get():
        db = await get_async_db()
        try:
            cursor = await db.execute("""
                SELECT id, name, start_line, depth, parent_id
                FROM spans
                WHERE session = ? AND end_line IS NULL
                ORDER BY depth DESC, id DESC
                LIMIT 1
            """, (session,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    return await retry_with_backoff(do_get)


async def detect_semantic_topic_shift_async(
    span_id: int,
    content: str,
    threshold: float = 0.55,
    divergence_history: Optional[list] = None
) -> tuple[bool, float, list]:
    """Detect semantic topic shift using embedding similarity.

    Args:
        span_id: Current span ID
        content: Message content to check
        threshold: Similarity threshold for shift detection
        divergence_history: Previous divergence scores

    Returns:
        Tuple of (is_shift, similarity, updated_history)
    """
    if divergence_history is None:
        divergence_history = []

    async def do_detect():
        db = await get_async_db()
        try:
            # Get span embedding
            cursor = await db.execute("""
                SELECT embedding FROM span_embeddings WHERE span_id = ?
            """, (span_id,))
            row = await cursor.fetchone()
            if not row:
                return False, 1.0, divergence_history

            # Get content embedding
            async with cache_source("indexing"):
                content_embedding = await get_embedding_async(content)

            # Deserialize span embedding
            import struct
            span_embedding = list(struct.unpack(f'{1536}f', row["embedding"]))

            # Calculate similarity
            similarity = _cosine_similarity(content_embedding, span_embedding)

            # Update history
            new_history = divergence_history[-2:] + [similarity]  # Keep last 3

            # Detect shift if consistently below threshold
            is_shift = (
                len(new_history) >= 3 and
                all(s < threshold for s in new_history)
            )

            return is_shift, similarity, new_history
        finally:
            await db.close()

    return await retry_with_backoff(do_detect)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


async def flush_all_async():
    """Flush all pending writes and close connections.

    Call this at the end of indexing to ensure all data is persisted.
    """
    await flush_write_queue()
