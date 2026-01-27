"""Tests for async database connection."""

import asyncio

import pytest
import pytest_asyncio

from db.async_connection import get_async_db


@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Give aiosqlite threads time to clean up after each test."""
    yield
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_get_async_db_returns_connection():
    """Test that get_async_db returns a valid connection."""
    db = await get_async_db()
    assert db is not None
    await db.close()


@pytest.mark.asyncio
async def test_async_db_can_execute_query():
    """Test that async db can execute queries."""
    db = await get_async_db()
    cursor = await db.execute("SELECT 1 as value")
    row = await cursor.fetchone()
    assert row['value'] == 1
    await db.close()


@pytest.mark.asyncio
async def test_async_db_has_wal_mode():
    """Test that async db is configured with WAL mode."""
    db = await get_async_db()
    cursor = await db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0].upper() == 'WAL'
    await db.close()


@pytest.mark.asyncio
async def test_async_db_has_busy_timeout():
    """Test that async db has busy timeout configured."""
    db = await get_async_db()
    cursor = await db.execute("PRAGMA busy_timeout")
    row = await cursor.fetchone()
    assert row[0] == 30000
    await db.close()
