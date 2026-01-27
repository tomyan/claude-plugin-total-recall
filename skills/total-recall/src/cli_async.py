"""Async CLI adapter for total-recall.

Provides utilities to bridge async functions to the synchronous CLI.
"""

import asyncio
from typing import Coroutine, TypeVar, Any

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous context.

    Uses asyncio.run() to execute the coroutine and return its result.
    This is the main bridge between sync CLI and async operations.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine

    Example:
        results = run_async(hybrid_search_async(query, limit=10))
    """
    return asyncio.run(coro)


async def shutdown_async_modules():
    """Shutdown all async modules gracefully.

    Should be called before program exit to ensure clean shutdown
    of write queues, connections, etc.
    """
    from embeddings.cache import shutdown as cache_shutdown
    await cache_shutdown()


def shutdown():
    """Synchronous wrapper to shutdown async modules."""
    run_async(shutdown_async_modules())


# Re-export async search functions for convenience
from search.vector import (
    search_ideas_async,
    find_similar_ideas_async,
    enrich_with_relations_async,
    search_spans_async,
)
from search.hybrid import hybrid_search_async
from search.hyde import (
    generate_hypothetical_doc_async,
    hyde_search_async,
)
from embeddings.cache import (
    get_embedding_cache_stats,
    clear_embedding_cache,
    flush_write_queue,
    cache_source,
)
from embeddings.openai import (
    get_embedding_async,
    get_embeddings_batch_async,
)

__all__ = [
    # Core async runner
    "run_async",
    "shutdown",
    "shutdown_async_modules",
    # Async search functions
    "search_ideas_async",
    "find_similar_ideas_async",
    "enrich_with_relations_async",
    "search_spans_async",
    "hybrid_search_async",
    "generate_hypothetical_doc_async",
    "hyde_search_async",
    # Cache operations
    "get_embedding_cache_stats",
    "clear_embedding_cache",
    "flush_write_queue",
    "cache_source",
    # Embedding operations
    "get_embedding_async",
    "get_embeddings_batch_async",
]
