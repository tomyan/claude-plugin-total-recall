"""Tools available to the indexing LLM."""

import json
from typing import Any

from db.connection import get_db


# Tool definitions for Claude
INDEXING_TOOLS = [
    {
        "name": "search_ideas",
        "description": "Search existing ideas by semantic similarity. Use this to find related ideas before creating new ones, to avoid duplicates and make connections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - describe what you're looking for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_idea",
        "description": "Get full details of a specific idea by ID, including its context and relations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "idea_id": {
                    "type": "integer",
                    "description": "The idea ID to retrieve"
                }
            },
            "required": ["idea_id"]
        }
    },
    {
        "name": "list_recent_topics",
        "description": "List recent topics/spans from this session to understand context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session identifier"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max topics to return (default 10)",
                    "default": 10
                }
            },
            "required": ["session"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict[str, Any], session: str = None) -> str:
    """Execute a tool and return the result as a string."""

    if tool_name == "search_ideas":
        return _search_ideas(tool_input.get("query", ""), tool_input.get("limit", 5))

    elif tool_name == "get_idea":
        return _get_idea(tool_input.get("idea_id"))

    elif tool_name == "list_recent_topics":
        sess = tool_input.get("session") or session
        return _list_recent_topics(sess, tool_input.get("limit", 10))

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _search_ideas(query: str, limit: int = 5) -> str:
    """Search ideas by content similarity."""
    from embeddings.openai import OpenAIEmbeddingProvider
    import os

    db = get_db()

    # Try vector search if embeddings available
    api_key = os.environ.get("OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS")
    if api_key:
        try:
            provider = OpenAIEmbeddingProvider(api_key=api_key)
            embedding = provider.embed(query)

            # Vector search
            cursor = db.execute("""
                SELECT i.id, i.content, i.intent, s.name as topic, s.session,
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
                    "content": row["content"],
                    "type": row["intent"],
                    "topic": row["topic"],
                    "session": row["session"],
                    "similarity": 1 - row["distance"]  # Convert distance to similarity
                })

            db.close()
            return json.dumps(results, indent=2)
        except Exception as e:
            db.close()
            return json.dumps({"error": f"Search failed: {e}"})

    # Fallback to text search
    cursor = db.execute("""
        SELECT i.id, i.content, i.intent, s.name as topic, s.session
        FROM ideas i
        LEFT JOIN spans s ON i.span_id = s.id
        WHERE i.content LIKE ?
        ORDER BY i.id DESC
        LIMIT ?
    """, (f"%{query}%", limit))

    results = [dict(row) for row in cursor.fetchall()]
    db.close()
    return json.dumps(results, indent=2)


def _get_idea(idea_id: int) -> str:
    """Get full details of an idea."""
    if not idea_id:
        return json.dumps({"error": "idea_id required"})

    db = get_db()

    cursor = db.execute("""
        SELECT i.id, i.content, i.intent, i.confidence, i.source_file, i.source_line,
               s.name as topic, s.summary as topic_summary, s.session
        FROM ideas i
        LEFT JOIN spans s ON i.span_id = s.id
        WHERE i.id = ?
    """, (idea_id,))

    row = cursor.fetchone()
    if not row:
        db.close()
        return json.dumps({"error": f"Idea {idea_id} not found"})

    result = dict(row)

    # Get related ideas
    cursor = db.execute("""
        SELECT r.relation_type, i.id, i.content, i.intent
        FROM relations r
        JOIN ideas i ON r.to_idea_id = i.id
        WHERE r.from_idea_id = ?
        UNION
        SELECT r.relation_type, i.id, i.content, i.intent
        FROM relations r
        JOIN ideas i ON r.from_idea_id = i.id
        WHERE r.to_idea_id = ?
    """, (idea_id, idea_id))

    result["relations"] = [dict(row) for row in cursor.fetchall()]

    db.close()
    return json.dumps(result, indent=2)


def _list_recent_topics(session: str, limit: int = 10) -> str:
    """List recent topics from a session."""
    if not session:
        return json.dumps({"error": "session required"})

    db = get_db()

    cursor = db.execute("""
        SELECT id, name, summary, start_line, end_line
        FROM spans
        WHERE session = ?
        ORDER BY id DESC
        LIMIT ?
    """, (session, limit))

    results = [dict(row) for row in cursor.fetchall()]
    db.close()
    return json.dumps(results, indent=2)
