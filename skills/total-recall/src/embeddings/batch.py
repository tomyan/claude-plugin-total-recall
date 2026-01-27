"""Batch embedding for ideas and spans."""

import asyncio
from typing import Any

from db.connection import get_db
from embeddings.openai import get_embeddings_batch_async
from embeddings.serialize import serialize_embedding


def get_embeddings_batch(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """Sync wrapper for get_embeddings_batch_async."""
    return asyncio.run(get_embeddings_batch_async(texts, use_cache))


# Maximum texts per API call
BATCH_SIZE = 100


class EmbeddingError(Exception):
    """Error during embedding operation."""
    pass


def embed_ideas(idea_ids: list[int], force: bool = False) -> int:
    """
    Embed ideas and store in database.

    Args:
        idea_ids: List of idea IDs to embed
        force: If True, re-embed even if already embedded

    Returns:
        Number of ideas embedded

    Raises:
        EmbeddingError: If embedding API fails
    """
    if not idea_ids:
        return 0

    db = get_db()

    # Get ideas that need embedding
    if force:
        # Get all ideas
        placeholders = ",".join("?" * len(idea_ids))
        cursor = db.execute(f"""
            SELECT id, content FROM ideas WHERE id IN ({placeholders})
        """, idea_ids)
    else:
        # Get ideas without embeddings
        placeholders = ",".join("?" * len(idea_ids))
        cursor = db.execute(f"""
            SELECT i.id, i.content FROM ideas i
            LEFT JOIN idea_embeddings e ON i.id = e.idea_id
            WHERE i.id IN ({placeholders}) AND e.idea_id IS NULL
        """, idea_ids)

    ideas_to_embed = [(row["id"], row["content"]) for row in cursor.fetchall()]

    if not ideas_to_embed:
        db.close()
        return 0

    total_embedded = 0

    try:
        # Process in batches
        for i in range(0, len(ideas_to_embed), BATCH_SIZE):
            batch = ideas_to_embed[i:i + BATCH_SIZE]
            texts = [content for _, content in batch]

            # Get embeddings from API
            embeddings = get_embeddings_batch(texts, use_cache=False)

            # Store in database
            for (idea_id, _), embedding in zip(batch, embeddings):
                serialized = serialize_embedding(embedding)

                if force:
                    # Delete existing and insert new
                    db.execute("DELETE FROM idea_embeddings WHERE idea_id = ?", (idea_id,))

                db.execute("""
                    INSERT INTO idea_embeddings (idea_id, embedding)
                    VALUES (?, ?)
                """, (idea_id, serialized))

                total_embedded += 1

            db.commit()

    except Exception as e:
        db.close()
        raise EmbeddingError(f"Failed to embed ideas: {e}") from e

    db.close()
    return total_embedded


def embed_span(span_id: int, force: bool = False) -> bool:
    """
    Embed span summary and store in database.

    Args:
        span_id: Span ID to embed
        force: If True, re-embed even if already embedded

    Returns:
        True if embedded, False if skipped

    Raises:
        EmbeddingError: If embedding API fails
    """
    db = get_db()

    # Check if already embedded
    if not force:
        cursor = db.execute("""
            SELECT span_id FROM span_embeddings WHERE span_id = ?
        """, (span_id,))
        if cursor.fetchone():
            db.close()
            return False

    # Get span summary
    cursor = db.execute("""
        SELECT name, summary FROM spans WHERE id = ?
    """, (span_id,))
    row = cursor.fetchone()

    if not row:
        db.close()
        return False

    # Combine name and summary for embedding
    text = f"{row['name']}: {row['summary'] or ''}"

    try:
        embeddings = get_embeddings_batch([text], use_cache=False)
        embedding = embeddings[0]
        serialized = serialize_embedding(embedding)

        if force:
            db.execute("DELETE FROM span_embeddings WHERE span_id = ?", (span_id,))

        db.execute("""
            INSERT INTO span_embeddings (span_id, embedding)
            VALUES (?, ?)
        """, (span_id, serialized))

        db.commit()

    except Exception as e:
        db.close()
        raise EmbeddingError(f"Failed to embed span: {e}") from e

    db.close()
    return True
