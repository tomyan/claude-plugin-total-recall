"""Vector search operations for memgraph."""

from datetime import datetime
from typing import Optional

from db.connection import get_db
from embeddings.openai import get_embedding
from embeddings.serialize import serialize_embedding


def _update_access_tracking(idea_ids: list[int], db=None, session: str = None) -> None:
    """Update access_count and last_accessed for retrieved ideas.

    Also records activation in working memory if session is provided.

    Args:
        idea_ids: List of idea IDs that were accessed
        db: Optional database connection (will create one if not provided)
        session: Optional session for working memory activation
    """
    if not idea_ids:
        return

    close_db = False
    if db is None:
        db = get_db()
        close_db = True

    now = datetime.utcnow().isoformat()
    placeholders = ','.join('?' * len(idea_ids))
    db.execute(f"""
        UPDATE ideas
        SET access_count = access_count + 1,
            last_accessed = ?
        WHERE id IN ({placeholders})
    """, [now] + idea_ids)

    # Also record in working memory if session provided
    if session:
        for idea_id in idea_ids:
            db.execute("""
                INSERT INTO working_memory (session, idea_id, activation, last_access)
                VALUES (?, ?, 1.0, ?)
                ON CONFLICT(session, idea_id) DO UPDATE SET
                    activation = MIN(1.0, activation + 0.2),
                    last_access = excluded.last_access
            """, (session, idea_id, now))

    db.commit()

    if close_db:
        db.close()


def search_ideas(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    intent: Optional[str] = None,
    recency_weight: float = 0.0,
    include_forgotten: bool = False
) -> list[dict]:
    """Search for similar ideas using vector similarity."""
    query_embedding = get_embedding(query)
    db = get_db()

    # Vector search with larger k for filtering
    k = limit * 3 if (session or intent) else limit

    # Build query with optional forgotten filter
    forgotten_filter = "" if include_forgotten else "AND (i.forgotten = FALSE OR i.forgotten IS NULL)"
    cursor = db.execute(f"""
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
        ORDER BY e.distance
    """, (serialize_embedding(query_embedding), k))

    results = []
    for row in cursor:
        # Apply filters
        if session and row['session'] != session:
            continue
        if intent and row['intent'] != intent:
            continue

        results.append(dict(row))
        if len(results) >= limit:
            break

    # Update access tracking for returned results
    if results:
        _update_access_tracking([r['id'] for r in results], db, session=session)

    db.close()
    return results


def find_similar_ideas(
    idea_id: int,
    limit: int = 5,
    exclude_related: bool = True,
    same_session: Optional[bool] = None,
    session: Optional[str] = None
) -> list[dict]:
    """Find ideas similar to a given idea.

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
    db = get_db()

    # Get the idea's embedding and session
    cursor = db.execute("""
        SELECT e.embedding, s.session
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.idea_id = ?
    """, (idea_id,))
    row = cursor.fetchone()

    if not row:
        db.close()
        return []

    embedding = row["embedding"]
    source_session = row["session"]

    # Find similar ideas (exclude forgotten)
    cursor = db.execute("""
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
        ORDER BY e.distance
    """, (embedding, limit + 20))  # Get extra to filter

    results = []
    related_ids = set()

    if exclude_related:
        # Get related idea IDs
        rel_cursor = db.execute("""
            SELECT to_id FROM relations WHERE from_id = ?
            UNION
            SELECT from_id FROM relations WHERE to_id = ?
        """, (idea_id, idea_id))
        related_ids = {row["to_id"] if "to_id" in row.keys() else row["from_id"] for row in rel_cursor}

    for row in cursor:
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
        _update_access_tracking([r['id'] for r in results], db, session=track_session)

    db.close()
    return results


def enrich_with_relations(results: list[dict], max_related: int = 3) -> list[dict]:
    """Enrich search results with related ideas.

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

    db = get_db()

    for result in results:
        idea_id = result.get("id")
        if not idea_id:
            continue

        related = []

        # Get outgoing relations
        cursor = db.execute("""
            SELECT r.relation_type, i.id, i.content, i.intent
            FROM relations r
            JOIN ideas i ON i.id = r.to_id
            WHERE r.from_id = ?
            LIMIT ?
        """, (idea_id, max_related))
        for row in cursor:
            related.append({
                "type": row["relation_type"],
                "direction": "outgoing",
                "id": row["id"],
                "content": row["content"][:100],  # Truncate
                "intent": row["intent"]
            })

        # Get incoming relations (if room)
        if len(related) < max_related:
            cursor = db.execute("""
                SELECT r.relation_type, i.id, i.content, i.intent
                FROM relations r
                JOIN ideas i ON i.id = r.from_id
                WHERE r.to_id = ?
                LIMIT ?
            """, (idea_id, max_related - len(related)))
            for row in cursor:
                related.append({
                    "type": row["relation_type"],
                    "direction": "incoming",
                    "id": row["id"],
                    "content": row["content"][:100],
                    "intent": row["intent"]
                })

        result["related"] = related

    db.close()
    return results


def search_spans(query: str, limit: int = 5) -> list[dict]:
    """Search for similar topic spans."""
    db = get_db()
    query_embedding = get_embedding(query)

    cursor = db.execute("""
        SELECT
            s.id, s.session, s.name, s.summary,
            s.start_line, s.end_line, s.depth, s.created_at,
            e.distance
        FROM span_embeddings e
        JOIN spans s ON s.id = e.span_id
        WHERE e.embedding MATCH ? AND k = ?
        ORDER BY e.distance
    """, (serialize_embedding(query_embedding), limit))

    results = [dict(row) for row in cursor]
    db.close()
    return results
