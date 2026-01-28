"""Async vector search operations for total-recall."""

from datetime import datetime
from typing import Optional

from db.async_connection import get_async_db
from embeddings.openai import get_embedding
from embeddings.cache import cache_source
from embeddings.serialize import serialize_embedding
from utils.async_retry import retry_with_backoff


async def _update_access_tracking(
    idea_ids: list[int],
    db=None,
    session: str = None
) -> None:
    """Update access_count and last_accessed for retrieved ideas.

    Also records activation in working memory if session is provided.

    Args:
        idea_ids: List of idea IDs that were accessed
        db: Optional database connection (will create one if not provided)
        session: Optional session for working memory activation
    """
    if not idea_ids:
        return

    async def do_update():
        nonlocal db
        close_db = False
        if db is None:
            db = await get_async_db()
            close_db = True

        try:
            now = datetime.utcnow().isoformat()
            placeholders = ','.join('?' * len(idea_ids))
            await db.execute(f"""
                UPDATE ideas
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id IN ({placeholders})
            """, [now] + idea_ids)

            # Also record in working memory if session provided
            if session:
                for idea_id in idea_ids:
                    await db.execute("""
                        INSERT INTO working_memory (session, idea_id, activation, last_access)
                        VALUES (?, ?, 1.0, ?)
                        ON CONFLICT(session, idea_id) DO UPDATE SET
                            activation = MIN(1.0, activation + 0.2),
                            last_access = excluded.last_access
                    """, (session, idea_id, now))

            await db.commit()
        finally:
            if close_db:
                await db.close()

    await retry_with_backoff(do_update)


async def search_ideas(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    intent: Optional[str] = None,
    recency_weight: float = 0.0,
    include_forgotten: bool = False,
    show_originals: bool = False,
    boost_active: bool = False
) -> list[dict]:
    """Search for similar ideas using vector similarity (async).

    Args:
        query: Search query text
        limit: Maximum results to return
        session: Optional session to filter by
        intent: Optional intent to filter by
        recency_weight: Weight for recency (not currently used)
        include_forgotten: Include forgotten ideas if True
        show_originals: Show consolidated original ideas if True
        boost_active: If True and session set, boost ideas active in working memory

    Returns:
        List of matching idea dicts
    """
    async with cache_source("search"):
        query_embedding = await get_embedding(query)

    async def do_search():
        db = await get_async_db()
        try:
            # Vector search with larger k for filtering
            k = limit * 3 if (session or intent) else limit

            # Build filters
            forgotten_filter = "" if include_forgotten else "AND (i.forgotten = FALSE OR i.forgotten IS NULL)"
            consolidated_filter = "" if show_originals else "AND (i.consolidated_into IS NULL)"
            cursor = await db.execute(f"""
                SELECT
                    i.id, i.content, i.intent, i.confidence,
                    i.source_file, i.source_line, i.created_at,
                    s.session, s.name as topic,
                    e.distance
                FROM idea_embeddings e
                JOIN ideas i ON i.id = e.idea_id
                LEFT JOIN spans s ON s.id = i.span_id
                WHERE e.embedding MATCH ? AND k = ?
                    {forgotten_filter}
                    {consolidated_filter}
                ORDER BY e.distance
            """, (serialize_embedding(query_embedding), k))

            results = []
            rows = await cursor.fetchall()
            for row in rows:
                # Apply filters
                if session and row['session'] != session:
                    continue
                if intent and row['intent'] != intent:
                    continue

                results.append(dict(row))
                if len(results) >= limit:
                    break

            # Boost active ideas in working memory
            if boost_active and session and results:
                cursor = await db.execute("""
                    SELECT idea_id, activation
                    FROM working_memory
                    WHERE session = ? AND activation > 0.1
                """, (session,))
                active = {row["idea_id"]: row["activation"] async for row in cursor}

                if active:
                    for r in results:
                        if r['id'] in active:
                            # Reduce distance proportional to activation (lower = better)
                            boost = active[r['id']] * 0.3  # Max 30% boost
                            r['distance'] = max(0, r['distance'] * (1 - boost))
                            r['_boosted'] = True
                    # Re-sort by boosted distance
                    results.sort(key=lambda r: r['distance'])

            # Update access tracking for returned results
            if results:
                await _update_access_tracking(
                    [r['id'] for r in results], db, session=session
                )

            return results
        finally:
            await db.close()

    return await retry_with_backoff(do_search)


async def find_similar_ideas(
    idea_id: int,
    limit: int = 5,
    exclude_related: bool = True,
    same_session: Optional[bool] = None,
    session: Optional[str] = None
) -> list[dict]:
    """Find ideas similar to a given idea (async).

    Uses the idea's embedding to find semantically similar ideas.

    Args:
        idea_id: ID of the source idea
        limit: Maximum similar ideas to return
        exclude_related: If True, exclude ideas already linked by relations
        same_session: If True, only same session; if False, only other sessions; if None, all
        session: Explicit session filter (overrides same_session)

    Returns:
        List of similar idea dicts with distance
    """
    async def do_find():
        db = await get_async_db()
        try:
            # Get the idea's embedding and session
            cursor = await db.execute("""
                SELECT e.embedding, s.session
                FROM idea_embeddings e
                JOIN ideas i ON i.id = e.idea_id
                LEFT JOIN spans s ON s.id = i.span_id
                WHERE e.idea_id = ?
            """, (idea_id,))
            row = await cursor.fetchone()

            if not row:
                return []

            embedding = row["embedding"]
            source_session = row["session"]

            # Find similar ideas (exclude forgotten and consolidated)
            cursor = await db.execute("""
                SELECT
                    i.id, i.content, i.intent, i.confidence,
                    i.source_file, i.source_line, i.created_at,
                    s.session, s.name as topic,
                    e.distance
                FROM idea_embeddings e
                JOIN ideas i ON i.id = e.idea_id
                LEFT JOIN spans s ON s.id = i.span_id
                WHERE e.embedding MATCH ? AND k = ?
                    AND (i.forgotten = FALSE OR i.forgotten IS NULL)
                    AND (i.consolidated_into IS NULL)
                ORDER BY e.distance
            """, (embedding, limit + 20))  # Get extra to filter

            results = []
            related_ids = set()

            if exclude_related:
                # Get related idea IDs
                rel_cursor = await db.execute("""
                    SELECT to_id FROM relations WHERE from_id = ?
                    UNION
                    SELECT from_id FROM relations WHERE to_id = ?
                """, (idea_id, idea_id))
                rel_rows = await rel_cursor.fetchall()
                related_ids = {
                    row["to_id"] if "to_id" in row.keys() else row["from_id"]
                    for row in rel_rows
                }

            rows = await cursor.fetchall()
            for row in rows:
                if row["id"] == idea_id:
                    continue  # Skip self
                if exclude_related and row["id"] in related_ids:
                    continue

                # Session filtering
                row_session = row["session"]
                if session is not None:
                    if row_session != session:
                        continue
                elif same_session is True:
                    if row_session != source_session:
                        continue
                elif same_session is False:
                    if row_session == source_session:
                        continue

                results.append(dict(row))
                if len(results) >= limit:
                    break

            # Update access tracking for returned results
            if results:
                # Use explicit session if provided, otherwise use source idea's session
                track_session = session if session else source_session
                await _update_access_tracking(
                    [r['id'] for r in results], db, session=track_session
                )

            return results
        finally:
            await db.close()

    return await retry_with_backoff(do_find)


async def enrich_with_relations(
    results: list[dict],
    max_related: int = 3
) -> list[dict]:
    """Enrich search results with related ideas (async).

    Adds a 'related' field to each result showing ideas that
    supersede, contradict, or answer it.

    Args:
        results: List of search result dicts (must have 'id' field)
        max_related: Maximum related ideas per result

    Returns:
        Results with 'related' field added
    """
    if not results:
        return results

    async def do_enrich():
        db = await get_async_db()
        try:
            for result in results:
                idea_id = result.get("id")
                if not idea_id:
                    continue

                related = []

                # Get outgoing relations
                cursor = await db.execute("""
                    SELECT r.relation_type, i.id, i.content, i.intent
                    FROM relations r
                    JOIN ideas i ON i.id = r.to_id
                    WHERE r.from_id = ?
                    LIMIT ?
                """, (idea_id, max_related))
                rows = await cursor.fetchall()
                for row in rows:
                    related.append({
                        "type": row["relation_type"],
                        "direction": "outgoing",
                        "id": row["id"],
                        "content": row["content"][:100],  # Truncate
                        "intent": row["intent"]
                    })

                # Get incoming relations (if room)
                if len(related) < max_related:
                    cursor = await db.execute("""
                        SELECT r.relation_type, i.id, i.content, i.intent
                        FROM relations r
                        JOIN ideas i ON i.id = r.from_id
                        WHERE r.to_id = ?
                        LIMIT ?
                    """, (idea_id, max_related - len(related)))
                    rows = await cursor.fetchall()
                    for row in rows:
                        related.append({
                            "type": row["relation_type"],
                            "direction": "incoming",
                            "id": row["id"],
                            "content": row["content"][:100],
                            "intent": row["intent"]
                        })

                result["related"] = related

            return results
        finally:
            await db.close()

    return await retry_with_backoff(do_enrich)


async def search_spans(query: str, limit: int = 5) -> list[dict]:
    """Search for similar topic spans (async).

    Args:
        query: Search query text
        limit: Maximum results to return

    Returns:
        List of matching span dicts
    """
    async with cache_source("search"):
        query_embedding = await get_embedding(query)

    async def do_search():
        db = await get_async_db()
        try:
            cursor = await db.execute("""
                SELECT
                    s.id, s.session, s.name, s.summary,
                    s.start_line, s.end_line, s.depth, s.created_at,
                    e.distance
                FROM span_embeddings e
                JOIN spans s ON s.id = e.span_id
                WHERE e.embedding MATCH ? AND k = ?
                ORDER BY e.distance
            """, (serialize_embedding(query_embedding), limit))

            rows = await cursor.fetchall()
            results = [dict(row) for row in rows]
            return results
        finally:
            await db.close()

    return await retry_with_backoff(do_search)
