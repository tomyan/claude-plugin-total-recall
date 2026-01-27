"""Async write queue for database operations."""

import asyncio
from typing import Callable, Any, Generic, TypeVar

from config import logger
from utils.async_retry import retry_with_backoff

T = TypeVar('T')


class WriteQueue(Generic[T]):
    """Async queue for processing database writes with retry.

    Items are processed sequentially by a background worker task.
    Each write is retried with exponential backoff on database lock.

    Usage:
        async def process_item(item):
            # Write item to database
            pass

        queue = WriteQueue(process_item)
        await queue.start()

        await queue.put(item1)
        await queue.put(item2)

        await queue.flush()  # Wait for all items to be processed
        await queue.stop()   # Stop the worker
    """

    def __init__(
        self,
        processor: Callable[[T], Any],
        max_retry_delay: float = 5.0
    ):
        """Initialize the write queue.

        Args:
            processor: Async function to process each item
            max_retry_delay: Maximum delay for retry backoff
        """
        self._queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._processor = processor
        self._max_retry_delay = max_retry_delay
        self._task: asyncio.Task | None = None
        self._running = False
        self._items_processed = 0
        self._items_failed = 0

    async def start(self):
        """Start the background worker task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._worker())
        logger.debug("Write queue worker started")

    async def stop(self):
        """Stop the background worker task.

        Drains remaining items before stopping.
        """
        if not self._running:
            return

        self._running = False
        await self._queue.put(None)  # Sentinel to stop worker

        if self._task:
            await self._task
            self._task = None

        logger.debug(
            f"Write queue worker stopped. "
            f"Processed: {self._items_processed}, Failed: {self._items_failed}"
        )

    async def put(self, item: T):
        """Add an item to the queue.

        Args:
            item: Item to be processed
        """
        await self._queue.put(item)

    def put_nowait(self, item: T):
        """Add an item to the queue without waiting.

        Args:
            item: Item to be processed
        """
        self._queue.put_nowait(item)

    async def flush(self, timeout: float | None = None):
        """Wait for all queued items to be processed.

        Args:
            timeout: Maximum time to wait (None for no limit)

        Returns:
            True if queue was drained, False if timeout
        """
        try:
            if timeout:
                await asyncio.wait_for(self._queue.join(), timeout)
            else:
                await self._queue.join()
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def size(self) -> int:
        """Current number of items in queue."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        """Queue statistics."""
        return {
            "size": self._queue.qsize(),
            "processed": self._items_processed,
            "failed": self._items_failed,
            "running": self._running,
        }

    async def _worker(self):
        """Background worker that processes queue items."""
        while self._running:
            try:
                item = await self._queue.get()

                if item is None:  # Sentinel - stop processing
                    self._queue.task_done()
                    break

                try:
                    await retry_with_backoff(
                        self._processor,
                        item,
                        max_delay=self._max_retry_delay
                    )
                    self._items_processed += 1
                except Exception as e:
                    logger.error(f"Write queue item failed: {e}")
                    self._items_failed += 1
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Write queue worker error: {e}")

        # Drain remaining items on shutdown
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    self._queue.task_done()
                    continue

                try:
                    await retry_with_backoff(
                        self._processor,
                        item,
                        max_delay=self._max_retry_delay
                    )
                    self._items_processed += 1
                except Exception as e:
                    logger.error(f"Write queue item failed during shutdown: {e}")
                    self._items_failed += 1
                finally:
                    self._queue.task_done()

            except asyncio.QueueEmpty:
                break
