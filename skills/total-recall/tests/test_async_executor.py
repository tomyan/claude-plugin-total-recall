"""Tests for async executor."""

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
    # Give aiosqlite worker threads time to finish before event loop closes
    await asyncio.sleep(0.05)


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
async def test_store_idea_async(mock_openai_response):
    """Test storing an idea asynchronously."""
    from async_executor import store_idea_async

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            idea_id = await store_idea_async(
                content="Test idea content",
                source_file="/test/file.py",
                source_line=10,
                intent="decision"
            )

            assert idea_id is not None
            assert isinstance(idea_id, int)
            assert idea_id > 0


@pytest.mark.asyncio
async def test_store_idea_with_entities(mock_openai_response):
    """Test storing an idea with entities."""
    from async_executor import store_idea_async
    import uuid

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            idea_id = await store_idea_async(
                content="Using Python and PostgreSQL",
                source_file=f"/test/entities-{uuid.uuid4()}.py",
                source_line=10,
                entities=[("Python", "technology"), ("PostgreSQL", "technology")]
            )

            assert idea_id is not None


@pytest.mark.asyncio
async def test_store_ideas_batch_async():
    """Test storing multiple ideas in a batch."""
    from async_executor import store_ideas_batch_async
    from embeddings.openai import _reset_async_provider
    import uuid

    # Reset provider to ensure fresh mock
    _reset_async_provider()

    unique_id = uuid.uuid4()
    ideas = [
        {"content": f"Idea {i}", "source_file": f"/test/batch-{unique_id}.py", "source_line": i + 100}
        for i in range(3)
    ]

    # Create mock response with correct number of embeddings
    class MockData:
        def __init__(self, idx):
            self.embedding = [float(idx) / 100.0] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockData(i) for i in range(3)]

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=MockResponse())
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            idea_ids = await store_ideas_batch_async(ideas)

            assert len(idea_ids) == 3
            assert all(isinstance(id, int) for id in idea_ids)
            assert all(id > 0 for id in idea_ids)

    # Reset after test
    _reset_async_provider()


@pytest.mark.asyncio
async def test_store_ideas_batch_empty():
    """Test storing empty batch returns empty list."""
    from async_executor import store_ideas_batch_async

    idea_ids = await store_ideas_batch_async([])
    assert idea_ids == []


@pytest.mark.asyncio
async def test_create_span_async():
    """Test creating a span asynchronously."""
    from async_executor import create_span_async

    span_id = await create_span_async(
        session="test-session",
        name="Test Topic",
        start_line=1
    )

    assert span_id is not None
    assert isinstance(span_id, int)
    assert span_id > 0


@pytest.mark.asyncio
async def test_create_child_span_async():
    """Test creating a child span."""
    from async_executor import create_span_async

    parent_id = await create_span_async(
        session="test-session",
        name="Parent Topic",
        start_line=1
    )

    child_id = await create_span_async(
        session="test-session",
        name="Child Topic",
        start_line=5,
        parent_id=parent_id,
        depth=1
    )

    assert child_id is not None
    assert child_id != parent_id


@pytest.mark.asyncio
async def test_close_span_async():
    """Test closing a span."""
    from async_executor import create_span_async, close_span_async

    span_id = await create_span_async(
        session="test-session",
        name="Test Topic",
        start_line=1
    )

    await close_span_async(
        span_id=span_id,
        end_line=10,
        summary="Test summary"
    )

    # Verify it was closed by checking the database
    from db.async_connection import get_async_db
    db = await get_async_db()
    cursor = await db.execute("SELECT end_line, summary FROM spans WHERE id = ?", (span_id,))
    row = await cursor.fetchone()
    await db.close()

    assert row["end_line"] == 10
    assert row["summary"] == "Test summary"


@pytest.mark.asyncio
async def test_get_open_span_async():
    """Test getting open span for a session."""
    from async_executor import create_span_async, get_open_span_async

    # Initially no open span
    span = await get_open_span_async("new-session")
    assert span is None

    # Create a span
    await create_span_async(
        session="new-session",
        name="Open Topic",
        start_line=1
    )

    span = await get_open_span_async("new-session")
    assert span is not None
    assert span["name"] == "Open Topic"


@pytest.mark.asyncio
async def test_add_relation_async(mock_openai_response):
    """Test adding a relation between ideas."""
    from async_executor import store_idea_async, add_relation_async
    import uuid

    unique_id = uuid.uuid4()

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            idea1_id = await store_idea_async(
                content="First idea for relation test",
                source_file=f"/test/relation-{unique_id}.py",
                source_line=1
            )
            idea2_id = await store_idea_async(
                content="Second idea for relation test",
                source_file=f"/test/relation-{unique_id}.py",
                source_line=2
            )

            await add_relation_async(
                from_id=idea1_id,
                to_id=idea2_id,
                relation_type="builds_on"
            )

            # Verify relation exists
            from db.async_connection import get_async_db
            db = await get_async_db()
            cursor = await db.execute(
                "SELECT * FROM relations WHERE from_id = ? AND to_id = ?",
                (idea1_id, idea2_id)
            )
            row = await cursor.fetchone()
            await db.close()

            assert row is not None
            assert row["relation_type"] == "builds_on"


@pytest.mark.asyncio
async def test_update_span_embedding_async(mock_openai_response):
    """Test updating span embedding."""
    from async_executor import create_span_async, update_span_embedding_async
    import uuid

    with patch('embeddings.openai.AsyncOpenAI') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_class.return_value = mock_client

        with patch('embeddings.openai.get_openai_api_key', return_value='test-key'):
            span_id = await create_span_async(
                session=f"embedding-test-{uuid.uuid4()}",
                name="Test Topic for Embedding",
                start_line=1
            )

            await update_span_embedding_async(span_id, include_ideas=False)

            # Verify embedding exists
            from db.async_connection import get_async_db
            db = await get_async_db()
            cursor = await db.execute(
                "SELECT embedding FROM span_embeddings WHERE span_id = ?",
                (span_id,)
            )
            row = await cursor.fetchone()
            await db.close()

            assert row is not None
            assert row["embedding"] is not None


@pytest.mark.asyncio
async def test_flush_all_async():
    """Test flushing all pending writes."""
    from async_executor import flush_all_async

    # Should complete without error
    await flush_all_async()
