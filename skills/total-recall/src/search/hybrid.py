"""Async hybrid search combining vector similarity and BM25."""

from typing import Optional

from db.async_connection import get_async_db
from embeddings.openai import get_embedding_async
from embeddings.cache import cache_source
from embeddings.serialize import serialize_embedding
from search.vector import _update_access_tracking_async
from utils.async_retry import retry_with_backoff


async def hybrid_search_async(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list[dict]:
    """Hybrid search combining vector similarity and BM25 (async).

    Args:
        query: Search query
        limit: Maximum results
        session: Optional session to filter by (project scope)
        since: Optional ISO datetime for temporal start
        until: Optional ISO datetime for temporal end

    Returns:
        List of matching idea dicts
    """
    async with cache_source("search"):
        query_embedding = await get_embedding_async(query)

    async def do_search():
        db = await get_async_db()
        try:
            # Vector search
            vector_results = {}
            cursor = await db.execute("""
                SELECT idea_id, distance
                FROM idea_embeddings
                WHERE embedding MATCH ? AND k = ?
            """, (serialize_embedding(query_embedding), limit * 2))
            rows = await cursor.fetchall()
            for i, row in enumerate(rows):
                vector_results[row['idea_id']] = i + 1  # rank

            # BM25 search
            bm25_results = {}
            cursor = await db.execute("""
                SELECT rowid, bm25(ideas_fts) as score
                FROM ideas_fts
                WHERE ideas_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, limit * 2))
            rows = await cursor.fetchall()
            for i, row in enumerate(rows):
                bm25_results[row['rowid']] = i + 1  # rank

            # Reciprocal Rank Fusion
            k = 60  # RRF constant
            scores = {}
            all_ids = set(vector_results.keys()) | set(bm25_results.keys())
            for idea_id in all_ids:
                score = 0
                if idea_id in vector_results:
                    score += 1 / (k + vector_results[idea_id])
                if idea_id in bm25_results:
                    score += 1 / (k + bm25_results[idea_id])
                scores[idea_id] = score

            # Get top results
            top_ids = sorted(
                scores.keys(), key=lambda x: scores[x], reverse=True
            )[:limit * 2]  # Get more for filtering

            if not top_ids:
                return []

            # Fetch full records with filters (exclude forgotten and consolidated)
            placeholders = ','.join('?' * len(top_ids))
            sql = f"""
                SELECT
                    i.id, i.content, i.intent, i.confidence,
                    i.source_file, i.source_line, i.created_at,
                    i.message_time,
                    s.session, s.name as topic
                FROM ideas i
                LEFT JOIN spans s ON s.id = i.span_id
                WHERE i.id IN ({placeholders})
                    AND (i.forgotten = FALSE OR i.forgotten IS NULL)
                    AND (i.consolidated_into IS NULL)
            """
            params = list(top_ids)

            if session:
                sql += " AND s.session = ?"
                params.append(session)
            if since:
                sql += " AND COALESCE(i.message_time, i.created_at) >= ?"
                params.append(since)
            if until:
                sql += " AND COALESCE(i.message_time, i.created_at) <= ?"
                params.append(until)

            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            results_by_id = {row['id']: dict(row) for row in rows}

            # Get final result list
            final_results = [
                results_by_id[id] for id in top_ids if id in results_by_id
            ][:limit]

            # Update access tracking for returned results
            if final_results:
                await _update_access_tracking_async(
                    [r['id'] for r in final_results], db, session=session
                )

            return final_results
        finally:
            await db.close()

    return await retry_with_backoff(do_search)
