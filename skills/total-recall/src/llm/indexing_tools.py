"""Indexing agent tool implementations - Slices 2.2-2.8.

These tools are exposed to the indexing agent for searching and querying
the knowledge base during transcript processing.
"""

import json
from typing import Any, Optional

from db.connection import get_db
from entities import find_golden_entity


async def search_ideas(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    intent: Optional[str] = None,
) -> list[dict]:
    """Search for ideas using vector similarity.

    This is the underlying search function used by tool_search_ideas.
    It wraps memory_db.search_ideas with appropriate parameters.
    """
    import memory_db

    return await memory_db.search_ideas(
        query=query,
        limit=limit,
        session=session,
        intent=intent,
    )


async def tool_search_ideas(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    intent: Optional[str] = None,
) -> list[dict]:
    """Search for ideas semantically similar to a query.

    Args:
        query: Search query text
        limit: Maximum number of results to return
        session: Optional session ID to filter by
        intent: Optional intent type to filter by

    Returns:
        List of matching ideas with id, content, intent, score
    """
    results = await search_ideas(
        query=query,
        limit=limit,
        session=session,
        intent=intent,
    )

    # Normalize result format
    return [
        {
            "id": r.get("id"),
            "content": r.get("content", ""),
            "intent": r.get("intent"),
            "score": r.get("score", 0.0),
            "source_line": r.get("source_line"),
        }
        for r in results
    ]


async def tool_get_open_questions(
    session: str,
    limit: int = 10,
) -> list[dict]:
    """Get unanswered questions for a session.

    Args:
        session: Session ID to filter by
        limit: Maximum number of results to return

    Returns:
        List of open questions with id, content, source_line
    """
    db = get_db()

    cursor = db.execute("""
        SELECT id, content, source_line, session
        FROM ideas
        WHERE intent = 'question'
          AND (answered IS NULL OR answered = FALSE)
          AND session = ?
        ORDER BY source_line DESC
        LIMIT ?
    """, (session, limit))

    results = [
        {
            "id": row["id"],
            "content": row["content"],
            "source_line": row["source_line"],
            "session": row["session"],
        }
        for row in cursor
    ]

    db.close()
    return results


async def tool_get_open_todos(
    session: str,
    limit: int = 10,
) -> list[dict]:
    """Get incomplete todos for a session.

    Args:
        session: Session ID to filter by
        limit: Maximum number of results to return

    Returns:
        List of open todos with id, content, source_line
    """
    db = get_db()

    cursor = db.execute("""
        SELECT id, content, source_line, session
        FROM ideas
        WHERE intent = 'todo'
          AND (completed IS NULL OR completed = FALSE)
          AND session = ?
        ORDER BY source_line DESC
        LIMIT ?
    """, (session, limit))

    results = [
        {
            "id": row["id"],
            "content": row["content"],
            "source_line": row["source_line"],
            "session": row["session"],
        }
        for row in cursor
    ]

    db.close()
    return results


async def tool_get_current_span(
    session: str,
) -> Optional[dict]:
    """Get the most recent span for a session.

    Args:
        session: Session ID

    Returns:
        Span dict with id, name, summary, start_line or None if no spans
    """
    db = get_db()

    cursor = db.execute("""
        SELECT id, name, summary, start_line, depth, parent_id
        FROM spans
        WHERE session = ?
        ORDER BY start_line DESC
        LIMIT 1
    """, (session,))

    row = cursor.fetchone()
    db.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "summary": row["summary"],
        "start_line": row["start_line"],
        "depth": row["depth"],
        "parent_id": row["parent_id"],
    }


async def tool_list_session_spans(
    session: str,
) -> list[dict]:
    """List all spans for a session, ordered by start_line.

    Args:
        session: Session ID

    Returns:
        List of spans with id, name, summary, start_line, depth, parent_id
    """
    db = get_db()

    cursor = db.execute("""
        SELECT id, name, summary, start_line, depth, parent_id
        FROM spans
        WHERE session = ?
        ORDER BY start_line ASC
    """, (session,))

    results = [
        {
            "id": row["id"],
            "name": row["name"],
            "summary": row["summary"],
            "start_line": row["start_line"],
            "depth": row["depth"],
            "parent_id": row["parent_id"],
        }
        for row in cursor
    ]

    db.close()
    return results


async def tool_search_entities(
    name: str,
    type: Optional[str] = None,
) -> list[dict]:
    """Search for golden entities by name with fuzzy matching.

    Args:
        name: Entity name to search for
        type: Optional entity type to filter by (from metadata)

    Returns:
        List of golden entities with id, canonical_name, type, mention_count
    """
    db = get_db()

    # First try exact match
    cursor = db.execute("""
        SELECT g.id, g.canonical_name, g.metadata,
               COUNT(m.id) as mention_count
        FROM golden_entities g
        LEFT JOIN entity_mentions m ON m.golden_id = g.id
        WHERE g.canonical_name = ?
        GROUP BY g.id
    """, (name,))

    exact_matches = list(cursor)

    # If no exact matches, try fuzzy via find_golden_entity
    if not exact_matches:
        fuzzy_result = find_golden_entity(name)  # sync function
        if fuzzy_result:
            cursor = db.execute("""
                SELECT g.id, g.canonical_name, g.metadata,
                       COUNT(m.id) as mention_count
                FROM golden_entities g
                LEFT JOIN entity_mentions m ON m.golden_id = g.id
                WHERE g.id = ?
                GROUP BY g.id
            """, (fuzzy_result["id"],))
            exact_matches = list(cursor)

    results = []
    for row in exact_matches:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        entity_type = metadata.get("type")

        # Filter by type if provided
        if type and entity_type != type:
            continue

        results.append({
            "id": row["id"],
            "canonical_name": row["canonical_name"],
            "type": entity_type,
            "mention_count": row["mention_count"],
        })

    db.close()
    return results


async def tool_get_recent_ideas(
    session: str,
    limit: int = 20,
    intent: Optional[str] = None,
) -> list[dict]:
    """Get recent ideas for a session, ordered by recency.

    Args:
        session: Session ID to filter by
        limit: Maximum number of results to return
        intent: Optional intent type to filter by

    Returns:
        List of ideas with id, content, intent, source_line
    """
    db = get_db()

    query = """
        SELECT id, content, intent, source_line, session
        FROM ideas
        WHERE session = ?
    """
    params: list[Any] = [session]

    if intent:
        query += " AND intent = ?"
        params.append(intent)

    query += " ORDER BY source_line DESC LIMIT ?"
    params.append(limit)

    cursor = db.execute(query, params)

    results = [
        {
            "id": row["id"],
            "content": row["content"],
            "intent": row["intent"],
            "source_line": row["source_line"],
            "session": row["session"],
        }
        for row in cursor
    ]

    db.close()
    return results
