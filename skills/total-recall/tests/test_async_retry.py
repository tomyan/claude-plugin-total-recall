"""Tests for async retry utility."""

import asyncio
import sqlite3
import time

import pytest

from utils.async_retry import retry_with_backoff


@pytest.mark.asyncio
async def test_retry_succeeds_first_try():
    """Test that successful function returns immediately."""
    call_count = 0

    async def succeed():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await retry_with_backoff(succeed)
    assert result == "success"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_lock_failures():
    """Test that function retries on database lock errors."""
    call_count = 0

    async def fail_twice_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    result = await retry_with_backoff(fail_twice_then_succeed, initial_delay=0.01)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_non_lock_errors():
    """Test that non-lock errors are raised immediately."""
    async def fail_with_other_error():
        raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        await retry_with_backoff(fail_with_other_error)


@pytest.mark.asyncio
async def test_retry_passes_arguments():
    """Test that arguments are passed through to function."""
    async def echo(a, b, c=None):
        return (a, b, c)

    result = await retry_with_backoff(echo, 1, 2, c=3)
    assert result == (1, 2, 3)


@pytest.mark.asyncio
async def test_retry_respects_max_delay():
    """Test that backoff is capped at max_delay."""
    call_count = 0
    delays = []
    last_time = time.time()

    async def track_delays():
        nonlocal call_count, last_time
        call_count += 1
        now = time.time()
        if call_count > 1:
            delays.append(now - last_time)
        last_time = now
        if call_count < 10:
            raise sqlite3.OperationalError("database is locked")
        return "done"

    await retry_with_backoff(track_delays, max_delay=0.2, initial_delay=0.05)

    # Later delays should be capped around max_delay (with jitter)
    # The last few delays should be close to 0.2s (within jitter range)
    assert max(delays[-3:]) < 0.4  # max_delay * 2 for jitter tolerance
