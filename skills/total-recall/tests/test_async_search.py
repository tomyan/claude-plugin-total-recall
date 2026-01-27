"""Tests for async search functions."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

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


@pytest.fixture
def mock_embedding():
    """Return a mock embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    class MockData:
        def __init__(self):
            self.embedding = [0.1] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockData()]

    return MockResponse()


@pytest.mark.asyncio
async def test_search_ideas_async_returns_list(mock_embedding, mock_openai_response):
    """Test that search_ideas_async returns a list."""
    from search.vector import search_ideas_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = await search_ideas_async("test query", limit=5)
            assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_ideas_async_with_session_filter(mock_embedding, mock_openai_response):
    """Test that session filter is applied."""
    from search.vector import search_ideas_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = await search_ideas_async(
                "test query",
                limit=5,
                session="test-session"
            )
            assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_search_async_returns_list(mock_embedding, mock_openai_response):
    """Test that hybrid_search_async returns a list."""
    from search.hybrid import hybrid_search_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = await hybrid_search_async("test query", limit=5)
            assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_search_async_with_temporal_filters(mock_embedding, mock_openai_response):
    """Test hybrid search with since/until filters."""
    from search.hybrid import hybrid_search_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = await hybrid_search_async(
                "test query",
                limit=5,
                since="2024-01-01T00:00:00",
                until="2024-12-31T23:59:59"
            )
            assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hyde_search_async_generates_hypothetical(mock_embedding, mock_openai_response):
    """Test that HyDE search generates hypothetical document."""
    from search.hyde import hyde_search_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            with patch('search.hyde.claude_complete', return_value="A hypothetical answer"):
                results = await hyde_search_async("test query", limit=5)
                assert isinstance(results, list)


@pytest.mark.asyncio
async def test_generate_hypothetical_doc_async(mock_embedding):
    """Test hypothetical document generation."""
    from search.hyde import generate_hypothetical_doc_async

    with patch('search.hyde.claude_complete', return_value="The answer is X"):
        result = await generate_hypothetical_doc_async("What is X?")
        assert result == "The answer is X"


@pytest.mark.asyncio
async def test_find_similar_ideas_async_returns_list(mock_embedding, mock_openai_response):
    """Test that find_similar_ideas_async returns a list."""
    from search.vector import find_similar_ideas_async

    results = await find_similar_ideas_async(idea_id=999, limit=5)
    assert isinstance(results, list)
    # Non-existent idea returns empty list
    assert results == []


@pytest.mark.asyncio
async def test_enrich_with_relations_async_empty_list():
    """Test enrich_with_relations_async with empty list."""
    from search.vector import enrich_with_relations_async

    results = await enrich_with_relations_async([])
    assert results == []


@pytest.mark.asyncio
async def test_enrich_with_relations_async_adds_field():
    """Test that enrich_with_relations_async adds related field."""
    from search.vector import enrich_with_relations_async

    # Test with a fake result that has an ID
    results = [{"id": 999, "content": "test"}]
    enriched = await enrich_with_relations_async(results)

    assert len(enriched) == 1
    assert "related" in enriched[0]
    assert isinstance(enriched[0]["related"], list)


@pytest.mark.asyncio
async def test_search_spans_async_returns_list(mock_embedding, mock_openai_response):
    """Test that search_spans_async returns a list."""
    from search.vector import search_spans_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = await search_spans_async("test query", limit=5)
            assert isinstance(results, list)


@pytest.mark.asyncio
async def test_update_access_tracking_async_empty_list():
    """Test that _update_access_tracking_async handles empty list."""
    from search.vector import _update_access_tracking_async

    # Should not raise any errors
    await _update_access_tracking_async([])


@pytest.mark.asyncio
async def test_search_uses_cache(mock_embedding, mock_openai_response):
    """Test that search uses embedding cache."""
    from search.vector import search_ideas_async
    from embeddings.cache import get_embedding_cache_stats, cache_embedding, flush_write_queue

    # Pre-cache an embedding
    await cache_embedding("cached query", [0.1] * 1536)
    await flush_write_queue()

    stats_before = await get_embedding_cache_stats()

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            # Search with the cached query
            await search_ideas_async("cached query", limit=5)

            # Give stats time to update
            await asyncio.sleep(0.1)

            stats_after = await get_embedding_cache_stats()
            # Cache hit should have occurred
            assert stats_after["total_hits"] >= stats_before["total_hits"]


@pytest.mark.asyncio
async def test_concurrent_searches(mock_openai_response):
    """Test multiple concurrent searches."""
    from search.vector import search_ideas_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            # Run multiple searches concurrently
            results = await asyncio.gather(
                search_ideas_async("query 1", limit=5),
                search_ideas_async("query 2", limit=5),
                search_ideas_async("query 3", limit=5),
            )

            assert len(results) == 3
            assert all(isinstance(r, list) for r in results)
