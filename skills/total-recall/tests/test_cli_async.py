"""Tests for async CLI adapter."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

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


def test_run_async_executes_coroutine():
    """Test that run_async executes a coroutine."""
    from cli_async import run_async

    async def simple_coro():
        return 42

    result = run_async(simple_coro())
    assert result == 42


def test_run_async_with_async_search():
    """Test run_async with actual async search function."""
    from cli_async import run_async, search_ideas

    class MockData:
        def __init__(self):
            self.embedding = [0.1] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockData()]

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MockResponse())
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = run_async(search_ideas("test query", limit=5))
            assert isinstance(results, list)


def test_run_async_with_hybrid_search():
    """Test run_async with hybrid search."""
    from cli_async import run_async, hybrid_search

    class MockData:
        def __init__(self):
            self.embedding = [0.1] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockData()]

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MockResponse())
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            results = run_async(hybrid_search("test query", limit=5))
            assert isinstance(results, list)


def test_run_async_with_hyde_search():
    """Test run_async with HyDE search."""
    from cli_async import run_async, hyde_search

    class MockData:
        def __init__(self):
            self.embedding = [0.1] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockData()]

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MockResponse())
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            with patch('search.hyde.claude_complete', return_value="A hypothetical"):
                results = run_async(hyde_search("test query", limit=5))
                assert isinstance(results, list)


def test_run_async_propagates_exceptions():
    """Test that run_async propagates exceptions from coroutines."""
    from cli_async import run_async

    async def failing_coro():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        run_async(failing_coro())


def test_shutdown_runs_without_error():
    """Test that shutdown completes without error."""
    from cli_async import shutdown

    # Should not raise
    shutdown()


def test_exports_are_available():
    """Test that all expected exports are available."""
    import cli_async

    # Core
    assert hasattr(cli_async, 'run_async')
    assert hasattr(cli_async, 'shutdown')

    # Search functions
    assert hasattr(cli_async, 'search_ideas')
    assert hasattr(cli_async, 'find_similar_ideas')
    assert hasattr(cli_async, 'hybrid_search')
    assert hasattr(cli_async, 'hyde_search')

    # Cache
    assert hasattr(cli_async, 'get_embedding_cache_stats')
    assert hasattr(cli_async, 'flush_write_queue')


@pytest.mark.asyncio
async def test_async_cache_stats():
    """Test getting cache stats through the adapter."""
    from cli_async import get_embedding_cache_stats

    stats = await get_embedding_cache_stats()
    assert isinstance(stats, dict)
    assert "size" in stats
    assert "total_hits" in stats


def test_sync_cache_stats_via_run_async():
    """Test getting cache stats synchronously via run_async."""
    from cli_async import run_async, get_embedding_cache_stats

    stats = run_async(get_embedding_cache_stats())
    assert isinstance(stats, dict)
    assert "size" in stats
