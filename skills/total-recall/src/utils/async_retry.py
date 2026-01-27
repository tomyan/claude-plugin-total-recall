"""Async retry utility with exponential backoff."""

import asyncio
import random
import sqlite3
from typing import TypeVar, Callable, Any

from config import logger

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_delay: float = 5.0,
    initial_delay: float = 0.1,
    **kwargs
) -> T:
    """Execute async function with exponential backoff on database lock.

    Retries indefinitely until success, with exponential backoff capped at max_delay.

    Args:
        func: Async function to execute
        *args: Arguments to pass to func
        max_delay: Maximum delay between retries (default 5s)
        initial_delay: Initial delay (default 100ms)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func

    Raises:
        Any non-lock exceptions from func
    """
    delay = initial_delay

    while True:
        try:
            return await func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                # Add jitter to prevent thundering herd
                jittered_delay = delay * (0.5 + random.random())
                logger.debug(f"Database locked, retrying in {jittered_delay:.2f}s")
                await asyncio.sleep(jittered_delay)
                # Exponential backoff with cap
                delay = min(delay * 2, max_delay)
            else:
                raise
