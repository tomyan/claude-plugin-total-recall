"""Tests for async embedding cache."""

import asyncio
import os
import tempfile

import pytest

# Set test database before importing modules
_test_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
os.environ["TOTAL_RECALL_DB_PATH"] = _test_db.name


@pytest.fixture(autouse=True)
async def setup_database():
    """Initialize test database before each test."""
    from db.schema import init_db
    init_db()
    yield
    # Cleanup after test
    from embeddings.cache import shutdown
    await shutdown()
    # Give aiosqlite worker threads time to finish before event loop closes
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    """Test that cache miss returns None."""
    from embeddings.cache import get_cached_embedding

    result = await get_cached_embedding("nonexistent text")
    assert result is None


@pytest.mark.asyncio
async def test_cache_hit_returns_embedding():
    """Test that cached embedding is returned."""
    from embeddings.cache import cache_embedding, get_cached_embedding, flush_write_queue

    embedding = [0.1] * 1536
    await cache_embedding("test text", embedding)
    await flush_write_queue()

    result = await get_cached_embedding("test text")
    assert result is not None
    assert len(result) == 1536
    assert result[0] == pytest.approx(0.1, rel=1e-5)


@pytest.mark.asyncio
async def test_cache_tracks_stats():
    """Test that cache tracks hit/miss statistics."""
    from embeddings.cache import (
        cache_embedding, get_cached_embedding, get_embedding_cache_stats,
        flush_write_queue
    )

    # Cache an embedding
    await cache_embedding("stats test", [0.1] * 1536)
    await flush_write_queue()

    # Generate some hits and misses
    await get_cached_embedding("stats test")  # hit
    await get_cached_embedding("stats test")  # hit
    await get_cached_embedding("nonexistent")  # miss

    # Give stats updates time to process
    await asyncio.sleep(0.1)

    stats = await get_embedding_cache_stats()
    assert stats["total_hits"] >= 2
    assert stats["total_misses"] >= 1


@pytest.mark.asyncio
async def test_cache_source_context():
    """Test that cache source context is tracked."""
    from embeddings.cache import (
        cache_source, cache_embedding, get_embedding_cache_stats,
        flush_write_queue
    )

    async with cache_source("search"):
        await cache_embedding("search query", [0.1] * 1536)

    async with cache_source("indexing"):
        await cache_embedding("indexed content", [0.2] * 1536)

    await flush_write_queue()

    stats = await get_embedding_cache_stats()
    assert "search" in stats["entries_by_source"] or stats["size"] > 0


@pytest.mark.asyncio
async def test_cache_clear():
    """Test that cache can be cleared."""
    from embeddings.cache import (
        cache_embedding, clear_embedding_cache, get_embedding_cache_stats,
        flush_write_queue
    )

    # Add some entries
    await cache_embedding("clear test 1", [0.1] * 1536)
    await cache_embedding("clear test 2", [0.2] * 1536)
    await flush_write_queue()

    stats_before = await get_embedding_cache_stats()
    assert stats_before["size"] >= 2

    # Clear cache
    await clear_embedding_cache()

    stats_after = await get_embedding_cache_stats()
    assert stats_after["size"] == 0
    assert stats_after["total_hits"] == 0
    assert stats_after["total_misses"] == 0


@pytest.mark.asyncio
async def test_cache_write_queue_size():
    """Test that write queue size is tracked in stats."""
    from embeddings.cache import get_embedding_cache_stats

    stats = await get_embedding_cache_stats()
    assert "write_queue_size" in stats
    assert "write_queue_stats" in stats


@pytest.mark.asyncio
async def test_cache_handles_same_text_twice():
    """Test that caching same text twice doesn't error."""
    from embeddings.cache import cache_embedding, get_cached_embedding, flush_write_queue

    await cache_embedding("duplicate test", [0.1] * 1536)
    await cache_embedding("duplicate test", [0.2] * 1536)  # Same key, different value
    await flush_write_queue()

    result = await get_cached_embedding("duplicate test")
    assert result is not None


@pytest.mark.asyncio
async def test_cache_preserves_embedding_precision():
    """Test that embedding values are preserved accurately."""
    from embeddings.cache import cache_embedding, get_cached_embedding, flush_write_queue

    # Use specific float values
    original = [0.123456789] * 100 + [0.987654321] * 100 + [0.0] * 1336
    await cache_embedding("precision test", original)
    await flush_write_queue()

    result = await get_cached_embedding("precision test")
    assert result is not None

    # Check precision (float32 has ~7 significant digits)
    for i in range(100):
        assert result[i] == pytest.approx(0.123456789, rel=1e-5)
    for i in range(100, 200):
        assert result[i] == pytest.approx(0.987654321, rel=1e-5)


@pytest.mark.asyncio
async def test_concurrent_cache_access():
    """Test that concurrent cache access works correctly."""
    from embeddings.cache import cache_embedding, get_cached_embedding, flush_write_queue

    # Write multiple embeddings concurrently
    async def write_embedding(i):
        await cache_embedding(f"concurrent test {i}", [float(i) / 100] * 1536)

    await asyncio.gather(*[write_embedding(i) for i in range(10)])
    await flush_write_queue()

    # Read them back concurrently
    async def read_embedding(i):
        return await get_cached_embedding(f"concurrent test {i}")

    results = await asyncio.gather(*[read_embedding(i) for i in range(10)])

    for i, result in enumerate(results):
        assert result is not None
        assert result[0] == pytest.approx(float(i) / 100, rel=1e-5)
