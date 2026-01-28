"""Context builder - assembles hierarchy context for LLM prompts."""

from typing import Any

from db.connection import get_db


def build_context(session: str, span_id: int | None = None) -> dict[str, Any]:
    """
    Build hierarchy context for a given session and optional span.

    Args:
        session: The session identifier
        span_id: Optional current span ID

    Returns:
        Dict with hierarchy context:
        - project: {name, description} or None
        - topic: {name, summary} or None
        - parent_spans: list of {name, summary} from root to immediate parent
        - current_span: {name, summary} or None
        - recent_spans: list of recent spans from this session
    """
    db = get_db()

    result = {
        "project": None,
        "topic": None,
        "parent_spans": [],
        "current_span": None,
        "recent_spans": [],
    }

    # Get current span if specified
    if span_id is not None:
        cursor = db.execute("""
            SELECT id, name, summary, topic_id, parent_id
            FROM spans
            WHERE id = ?
        """, (span_id,))
        span_row = cursor.fetchone()

        if span_row:
            result["current_span"] = {
                "name": span_row["name"],
                "summary": span_row["summary"],
            }

            # Get parent chain
            parent_id = span_row["parent_id"]
            parents = []
            while parent_id is not None:
                cursor = db.execute("""
                    SELECT id, name, summary, parent_id
                    FROM spans
                    WHERE id = ?
                """, (parent_id,))
                parent_row = cursor.fetchone()
                if parent_row:
                    parents.append({
                        "name": parent_row["name"],
                        "summary": parent_row["summary"],
                    })
                    parent_id = parent_row["parent_id"]
                else:
                    break

            # Reverse to get root-to-parent order
            result["parent_spans"] = list(reversed(parents))

            # Get topic if linked
            topic_id = span_row["topic_id"]
            if topic_id:
                cursor = db.execute("""
                    SELECT id, name, summary, project_id
                    FROM topics
                    WHERE id = ?
                """, (topic_id,))
                topic_row = cursor.fetchone()

                if topic_row:
                    result["topic"] = {
                        "name": topic_row["name"],
                        "summary": topic_row["summary"],
                    }

                    # Get project if linked
                    project_id = topic_row["project_id"]
                    if project_id:
                        cursor = db.execute("""
                            SELECT name, description
                            FROM projects
                            WHERE id = ?
                        """, (project_id,))
                        project_row = cursor.fetchone()

                        if project_row:
                            result["project"] = {
                                "name": project_row["name"],
                                "description": project_row["description"],
                            }

    # Get recent spans from session
    cursor = db.execute("""
        SELECT id, name, summary
        FROM spans
        WHERE session = ?
        ORDER BY id ASC
    """, (session,))
    result["recent_spans"] = [
        {"name": row["name"], "summary": row["summary"]}
        for row in cursor.fetchall()
    ]

    db.close()
    return result


def get_session_topic_history(session: str) -> list[dict[str, Any]]:
    """
    Get the topic/span history for a session, with key ideas and line ranges.

    This helps the LLM detect when returning to a previous topic after a tangent.

    Args:
        session: The session identifier

    Returns:
        List of topic transitions in chronological order:
        - span_id: Span identifier
        - name: Topic/span name
        - summary: Topic summary (if available)
        - start_line: First line of this topic
        - end_line: Last line (None if still open)
        - idea_count: Number of ideas extracted from this topic
        - key_ideas: Up to 3 key ideas (decisions, conclusions) from this topic
        - is_return: True if this is returning to a previous topic name
    """
    db = get_db()

    # Get all spans for this session with their line ranges and idea counts
    cursor = db.execute("""
        SELECT s.id, s.name, s.summary, s.start_line, s.end_line,
               COUNT(i.id) as idea_count
        FROM spans s
        LEFT JOIN ideas i ON i.span_id = s.id
        WHERE s.session = ?
        GROUP BY s.id
        ORDER BY s.start_line ASC, s.id ASC
    """, (session,))

    spans = cursor.fetchall()

    # Track seen topic names to detect returns
    seen_names = set()
    history = []

    for row in spans:
        span_id = row["id"]
        name = row["name"]

        # Check if this is returning to a previous topic
        name_key = (name or "").lower().strip()[:30]
        is_return = name_key in seen_names if name_key else False
        if name_key:
            seen_names.add(name_key)

        # Get key ideas (decisions, conclusions) for this span
        cursor = db.execute("""
            SELECT content, intent
            FROM ideas
            WHERE span_id = ? AND intent IN ('decision', 'conclusion', 'solution')
            ORDER BY
                CASE intent
                    WHEN 'decision' THEN 1
                    WHEN 'conclusion' THEN 2
                    ELSE 3
                END,
                importance DESC
            LIMIT 3
        """, (span_id,))

        key_ideas = [
            {"content": r["content"][:100], "type": r["intent"]}
            for r in cursor.fetchall()
        ]

        history.append({
            "span_id": span_id,
            "name": name,
            "summary": row["summary"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "idea_count": row["idea_count"],
            "key_ideas": key_ideas,
            "is_return": is_return,
        })

    db.close()
    return history


def get_related_topics(topic_id: int | None, limit: int = 3) -> list[dict[str, Any]]:
    """
    Find related topics from other sessions.

    Uses topic_links table to find semantically related topics.

    Args:
        topic_id: Current topic ID
        limit: Maximum number of related topics to return

    Returns:
        List of related topics with name, summary, similarity
    """
    if topic_id is None:
        return []

    db = get_db()

    # Find related topics via topic_links
    cursor = db.execute("""
        SELECT t.id, t.name, t.summary, tl.similarity
        FROM topic_links tl
        JOIN topics t ON tl.related_topic_id = t.id
        WHERE tl.topic_id = ?
        ORDER BY tl.similarity DESC
        LIMIT ?
    """, (topic_id, limit))

    related = [
        {
            "id": row["id"],
            "name": row["name"],
            "summary": row["summary"],
            "similarity": row["similarity"],
        }
        for row in cursor.fetchall()
    ]

    db.close()
    return related
