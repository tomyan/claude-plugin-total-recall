"""HyDE (Hypothetical Document Embeddings) search for memgraph."""

from typing import Optional

import config
from db.connection import get_db
from embeddings.openai import get_embedding
from embeddings.serialize import serialize_embedding
from errors import MemgraphError
from llm.claude import claude_complete
from search.vector import _update_access_tracking


def generate_hypothetical_doc(query: str) -> str:
    """Generate a hypothetical document that would answer the query.

    Used for HyDE (Hypothetical Document Embeddings) to improve retrieval.
    Uses LLM to generate a plausible answer, which often matches stored
    content better than the original question.

    Args:
        query: The user's search query

    Returns:
        A hypothetical answer document
    """
    try:
        system_prompt = """You are helping with document retrieval. Given a question,
write a brief hypothetical answer that might appear in documentation or conversation logs.
Write in a direct, factual style as if you were recording a decision or explaining a solution.
Keep it to 1-3 sentences. Do not include phrases like "I think" or "probably"."""

        hypothetical = claude_complete(query, system=system_prompt).strip()
        config.logger.debug(f"HyDE generated: {hypothetical[:100]}...")
        return hypothetical

    except MemgraphError:
        raise  # Re-raise MemgraphError as-is
    except Exception as e:
        config.logger.error(f"LLM HyDE generation failed: {e}")
        raise MemgraphError(
            f"HyDE generation failed: {e}",
            "hyde_generation_error",
            {"query": query, "original_error": str(e)}
        ) from e


def hyde_search(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list[dict]:
    """HyDE search - generates hypothetical answer then searches.

    For vague queries, this often retrieves better matches than raw query.

    Args:
        query: Search query
        limit: Maximum results
        session: Optional session to filter by (project scope)
        since: Optional ISO datetime for temporal start
        until: Optional ISO datetime for temporal end

    Returns:
        List of matching idea dicts
    """
    # Generate hypothetical document
    hypothetical = generate_hypothetical_doc(query)

    # Embed the hypothetical document
    hypo_embedding = get_embedding(hypothetical)

    # Search with hypothetical embedding
    db = get_db()

    # Build query with filters
    sql = """
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            i.message_time,
            s.session, s.name as topic,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
    """
    params = [serialize_embedding(hypo_embedding), limit * 2]  # Get more for filtering

    if session:
        sql += " AND s.session = ?"
        params.append(session)
    if since:
        sql += " AND COALESCE(i.message_time, i.created_at) >= ?"
        params.append(since)
    if until:
        sql += " AND COALESCE(i.message_time, i.created_at) <= ?"
        params.append(until)

    sql += " ORDER BY e.distance LIMIT ?"
    params.append(limit)

    cursor = db.execute(sql, params)
    results = [dict(row) for row in cursor]

    # Update access tracking for returned results
    if results:
        _update_access_tracking([r['id'] for r in results], db)

    db.close()
    return results
