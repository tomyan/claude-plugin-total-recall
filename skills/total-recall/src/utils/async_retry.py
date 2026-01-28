"""Async retry utility with exponential backoff."""

import asyncio
import random
import sqlite3
from typing import TypeVar, Callable, Any

from config import logger

T = TypeVar('T')

# SQLite error messages that indicate transient conditions
_TRANSIENT_SQLITE_ERRORS = {"locked", "busy", "disk i/o error"}

# Permanent SQLite errors that should not be retried
_PERMANENT_SQLITE_ERRORS = {"no such table", "no such column", "syntax error"}


def _is_transient_sqlite(e: sqlite3.OperationalError) -> bool:
    """Check if a SQLite error is transient (worth retrying)."""
    msg = str(e).lower()
    for pattern in _TRANSIENT_SQLITE_ERRORS:
        if pattern in msg:
            return True
    return False


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_delay: float = 5.0,
    initial_delay: float = 0.1,
    **kwargs
) -> T:
    """Execute async function with exponential backoff on transient errors.

    Retries indefinitely on transient SQLite errors (locked, busy, disk I/O),
    with exponential backoff capped at max_delay. Permanent errors (schema,
    syntax) are raised immediately.

    Args:
        func: Async function to execute
        *args: Arguments to pass to func
        max_delay: Maximum delay between retries (default 5s)
        initial_delay: Initial delay (default 100ms)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func

    Raises:
        sqlite3.OperationalError: For permanent SQLite errors
        Any other exceptions from func
    """
    delay = initial_delay
    attempt = 0

    while True:
        try:
            return await func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            attempt += 1
            if _is_transient_sqlite(e):
                # Add jitter to prevent thundering herd
                jittered_delay = delay * (0.5 + random.random())
                logger.debug(f"Transient DB error ({e}), retry #{attempt} in {jittered_delay:.2f}s")
                await asyncio.sleep(jittered_delay)
                # Exponential backoff with cap
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"Permanent DB error (not retrying): {e}")
                raise
