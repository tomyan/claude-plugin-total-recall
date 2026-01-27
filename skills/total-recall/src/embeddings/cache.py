"""Async embedding cache management for total-recall.

SQLite-backed cache with hit/miss statistics, write queue, and exponential backoff.
"""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Optional

from config import logger
from db.async_connection import get_async_db
from embeddings.serialize import serialize_embedding, deserialize_embedding
from utils.async_retry import retry_with_backoff
from utils.write_queue import WriteQueue

# Maximum entries before LRU eviction
_CACHE_MAX_SIZE = 50000

# Context variable for cache source (thread-safe)
_current_source: ContextVar[str] = ContextVar('cache_source', default='other')

# Global write queue (initialized lazily)
_write_queue: Optional[WriteQueue] = None
_queue_lock = asyncio.Lock()


async def _get_write_queue() -> WriteQueue:
    """Get or create the global write queue."""
    global _write_queue

    async with _queue_lock:
        if _write_queue is None:
            _write_queue = WriteQueue(_process_cache_write, max_retry_delay=5.0)
            await _write_queue.start()
            logger.debug("Async cache write queue started")
        return _write_queue


async def _process_cache_write(item: tuple):
    """Process a single cache write operation."""
    text_hash, text_preview, embedding_blob, created_at, source = item

    db = await get_async_db()
    try:
        # Check cache size and evict if necessary
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM embedding_cache")
        row = await cursor.fetchone()
        count = row['cnt']

        if count >= _CACHE_MAX_SIZE:
            # Evict oldest 10% by last_accessed
            evict_count = _CACHE_MAX_SIZE // 10
            await db.execute("""
                DELETE FROM embedding_cache
                WHERE text_hash IN (
                    SELECT text_hash FROM embedding_cache
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
            """, (evict_count,))
            logger.debug(f"Evicted {evict_count} old cache entries")

        # Insert or update
        await db.execute("""
            INSERT INTO embedding_cache (text_hash, text_preview, embedding, created_at, last_accessed, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(text_hash) DO UPDATE SET
                last_accessed = excluded.last_accessed
        """, (text_hash, text_preview, embedding_blob, created_at, created_at, source))

        await db.commit()
    finally:
        await db.close()


@asynccontextmanager
async def cache_source(source: str):
    """Async context manager to set the source for cache operations.

    Usage:
        async with cache_source("search"):
            embedding = await get_cached_embedding(text)
    """
    token = _current_source.set(source)
    try:
        yield
    finally:
        _current_source.reset(token)


def cache_source_sync(source: str):
    """Sync context manager for cache source (for compatibility).

    Usage:
        with cache_source_sync("search"):
            # operations that will eventually call async cache
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        token = _current_source.set(source)
        try:
            yield
        finally:
            _current_source.reset(token)

    return _ctx()


def _hash_text(text: str) -> str:
    """Create a hash of the text for use as cache key."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


async def _update_stats(hit: bool, source: str):
    """Update cache statistics asynchronously."""
    async def do_update():
        db = await get_async_db()
        try:
            if hit:
                source_col = f"hits_{source}" if source in ('search', 'indexing', 'backfill') else None
                if source_col:
                    await db.execute(f"""
                        UPDATE cache_stats
                        SET total_hits = total_hits + 1, {source_col} = {source_col} + 1
                        WHERE id = 1
                    """)
                else:
                    await db.execute("UPDATE cache_stats SET total_hits = total_hits + 1 WHERE id = 1")
            else:
                source_col = f"misses_{source}" if source in ('search', 'indexing', 'backfill') else None
                if source_col:
                    await db.execute(f"""
                        UPDATE cache_stats
                        SET total_misses = total_misses + 1, {source_col} = {source_col} + 1
                        WHERE id = 1
                    """)
                else:
                    await db.execute("UPDATE cache_stats SET total_misses = total_misses + 1 WHERE id = 1")
            await db.commit()
        finally:
            await db.close()

    await retry_with_backoff(do_update)


async def get_cached_embedding(text: str) -> Optional[list[float]]:
    """Get an embedding from cache, or None if not cached.

    Updates hit_count and last_accessed on hit.
    Updates cache_stats on hit or miss.

    Args:
        text: The text to look up

    Returns:
        Embedding vector or None if not cached
    """
    text_hash = _hash_text(text)
    source = _current_source.get()

    async def do_read():
        db = await get_async_db()
        try:
            cursor = await db.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                (text_hash,)
            )
            row = await cursor.fetchone()

            if row:
                # Update hit count and last_accessed
                now = datetime.utcnow().isoformat()
                await db.execute("""
                    UPDATE embedding_cache
                    SET hit_count = hit_count + 1, last_accessed = ?
                    WHERE text_hash = ?
                """, (now, text_hash))
                await db.commit()
                return row['embedding']
            return None
        finally:
            await db.close()

    result = await retry_with_backoff(do_read)

    if result is not None:
        # Fire and forget stats update
        asyncio.create_task(_update_stats(hit=True, source=source))
        return deserialize_embedding(result)
    else:
        asyncio.create_task(_update_stats(hit=False, source=source))
        return None


async def cache_embedding(text: str, embedding: list[float]):
    """Add an embedding to the cache.

    Writes are queued and processed by background worker with retry.

    Args:
        text: The text that was embedded
        embedding: The embedding vector
    """
    queue = await _get_write_queue()

    text_hash = _hash_text(text)
    text_preview = text[:200]
    embedding_blob = serialize_embedding(embedding)
    now = datetime.utcnow().isoformat()
    source = _current_source.get()

    # Queue the write (non-blocking)
    queue.put_nowait((text_hash, text_preview, embedding_blob, now, source))


async def get_embedding_cache_stats() -> dict:
    """Get comprehensive cache statistics.

    Returns:
        Dict with cache size, hit rate, breakdown by source and age, etc.
    """
    async def do_stats():
        db = await get_async_db()
        try:
            # Get cache size
            cursor = await db.execute("SELECT COUNT(*) as size FROM embedding_cache")
            row = await cursor.fetchone()
            size = row['size']

            # Get global stats
            cursor = await db.execute("SELECT * FROM cache_stats WHERE id = 1")
            stats_row = await cursor.fetchone()

            # Get per-source breakdown
            cursor = await db.execute("""
                SELECT source, COUNT(*) as count, SUM(hit_count) as total_hits
                FROM embedding_cache
                GROUP BY source
            """)
            rows = await cursor.fetchall()
            by_source = {row['source']: {'count': row['count'], 'hits': row['total_hits'] or 0}
                         for row in rows}

            # Get age distribution
            cursor = await db.execute("""
                SELECT
                    CASE
                        WHEN julianday('now') - julianday(created_at) < 1 THEN 'today'
                        WHEN julianday('now') - julianday(created_at) < 7 THEN 'this_week'
                        WHEN julianday('now') - julianday(created_at) < 30 THEN 'this_month'
                        ELSE 'older'
                    END as age_bucket,
                    COUNT(*) as count
                FROM embedding_cache
                GROUP BY age_bucket
            """)
            rows = await cursor.fetchall()
            by_age = {row['age_bucket']: row['count'] for row in rows}

            # Get top accessed entries
            cursor = await db.execute("""
                SELECT text_preview, hit_count, source
                FROM embedding_cache
                ORDER BY hit_count DESC
                LIMIT 10
            """)
            rows = await cursor.fetchall()
            top_hits = [{'text': row['text_preview'][:50], 'hits': row['hit_count'], 'source': row['source']}
                        for row in rows]

            # Get queue size
            queue = await _get_write_queue()
            queue_size = queue.size

            total_hits = stats_row['total_hits'] if stats_row else 0
            total_misses = stats_row['total_misses'] if stats_row else 0
            total_requests = total_hits + total_misses

            return {
                "size": size,
                "max_size": _CACHE_MAX_SIZE,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": round(total_hits / total_requests * 100, 1) if total_requests > 0 else 0,
                "write_queue_size": queue_size,
                "write_queue_stats": queue.stats,
                "by_source": {
                    "search": {
                        "hits": stats_row['hits_search'] if stats_row else 0,
                        "misses": stats_row['misses_search'] if stats_row else 0,
                    },
                    "indexing": {
                        "hits": stats_row['hits_indexing'] if stats_row else 0,
                        "misses": stats_row['misses_indexing'] if stats_row else 0,
                    },
                    "backfill": {
                        "hits": stats_row['hits_backfill'] if stats_row else 0,
                        "misses": stats_row['misses_backfill'] if stats_row else 0,
                    },
                },
                "entries_by_source": by_source,
                "entries_by_age": by_age,
                "top_hits": top_hits,
                "last_reset": stats_row['last_reset'] if stats_row else None,
            }
        finally:
            await db.close()

    return await retry_with_backoff(do_stats)


async def clear_embedding_cache():
    """Clear the embedding cache and reset stats."""
    async def do_clear():
        db = await get_async_db()
        try:
            await db.execute("DELETE FROM embedding_cache")
            await db.execute("""
                UPDATE cache_stats SET
                    total_hits = 0, total_misses = 0,
                    hits_search = 0, hits_indexing = 0, hits_backfill = 0,
                    misses_search = 0, misses_indexing = 0, misses_backfill = 0,
                    last_reset = CURRENT_TIMESTAMP
                WHERE id = 1
            """)
            await db.commit()
            logger.info("Cleared embedding cache")
        finally:
            await db.close()

    await retry_with_backoff(do_clear)


async def flush_write_queue(timeout: float = 30.0) -> bool:
    """Wait for write queue to drain.

    Args:
        timeout: Maximum time to wait

    Returns:
        True if queue drained, False if timeout
    """
    queue = await _get_write_queue()
    return await queue.flush(timeout)


async def shutdown():
    """Shutdown the cache module gracefully.

    Stops the write queue after draining remaining items.
    """
    global _write_queue

    async with _queue_lock:
        if _write_queue is not None:
            await _write_queue.stop()
            _write_queue = None
            logger.debug("Async cache write queue stopped")


def get_cache_max_size() -> int:
    """Get the maximum cache size."""
    return _CACHE_MAX_SIZE
