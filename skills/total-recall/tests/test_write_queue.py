"""Tests for async write queue."""

import asyncio
import sqlite3

import pytest

from utils.write_queue import WriteQueue


@pytest.mark.asyncio
async def test_queue_processes_items():
    """Test that queue processes items in order."""
    processed = []

    async def processor(item):
        processed.append(item)

    queue = WriteQueue(processor)
    await queue.start()
    await queue.put("item1")
    await queue.put("item2")
    await queue.put("item3")
    await queue.flush()
    await queue.stop()

    assert processed == ["item1", "item2", "item3"]


@pytest.mark.asyncio
async def test_queue_retries_on_lock():
    """Test that queue retries on database lock errors."""
    attempts = []

    async def failing_processor(item):
        attempts.append(item)
        if len(attempts) < 3:
            raise sqlite3.OperationalError("database is locked")

    queue = WriteQueue(failing_processor, max_retry_delay=0.1)
    await queue.start()
    await queue.put("item")
    await queue.flush()
    await queue.stop()

    assert len(attempts) == 3
    assert queue.stats["processed"] == 1
    assert queue.stats["failed"] == 0


@pytest.mark.asyncio
async def test_queue_tracks_failures():
    """Test that queue tracks permanent failures."""
    async def always_fails(item):
        raise ValueError("permanent error")

    queue = WriteQueue(always_fails)
    await queue.start()
    await queue.put("item")
    await queue.flush()
    await queue.stop()

    assert queue.stats["processed"] == 0
    assert queue.stats["failed"] == 1


@pytest.mark.asyncio
async def test_queue_drains_on_shutdown():
    """Test that queue drains remaining items on shutdown."""
    processed = []

    async def slow_processor(item):
        await asyncio.sleep(0.01)
        processed.append(item)

    queue = WriteQueue(slow_processor)
    await queue.start()

    # Add items faster than they can be processed
    for i in range(5):
        await queue.put(f"item{i}")

    await queue.stop()  # Should drain all items

    assert len(processed) == 5


@pytest.mark.asyncio
async def test_queue_size():
    """Test queue size tracking."""
    processed_event = asyncio.Event()

    async def slow_processor(item):
        await processed_event.wait()

    queue = WriteQueue(slow_processor)
    await queue.start()

    await queue.put("item1")
    await queue.put("item2")
    await queue.put("item3")

    # Give worker time to pick up first item
    await asyncio.sleep(0.01)

    # Size should be 2 (item1 being processed, item2 and item3 waiting)
    assert queue.size >= 2

    processed_event.set()
    await queue.flush()
    await queue.stop()

    assert queue.size == 0


@pytest.mark.asyncio
async def test_queue_flush_timeout():
    """Test that flush respects timeout."""
    async def never_finish(item):
        await asyncio.sleep(10)  # Very slow

    queue = WriteQueue(never_finish)
    await queue.start()
    await queue.put("item")

    result = await queue.flush(timeout=0.1)
    assert result is False

    # Cleanup - cancel the task
    queue._running = False
    if queue._task:
        queue._task.cancel()
        try:
            await queue._task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_queue_put_nowait():
    """Test non-blocking put."""
    processed = []

    async def processor(item):
        processed.append(item)

    queue = WriteQueue(processor)
    await queue.start()

    queue.put_nowait("item1")
    queue.put_nowait("item2")

    await queue.flush()
    await queue.stop()

    assert processed == ["item1", "item2"]


@pytest.mark.asyncio
async def test_queue_stats():
    """Test queue statistics."""
    async def processor(item):
        pass

    queue = WriteQueue(processor)

    # Before start
    assert queue.stats["running"] is False

    await queue.start()
    assert queue.stats["running"] is True

    await queue.put("item1")
    await queue.put("item2")
    await queue.flush()

    assert queue.stats["processed"] == 2
    assert queue.stats["failed"] == 0

    await queue.stop()
    assert queue.stats["running"] is False
