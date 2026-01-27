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
        results = run_async(hybrid_search(query, limit=10))
    """
    return asyncio.run(coro)


async def shutdown_modules():
    """Shutdown all async modules gracefully.

    Should be called before program exit to ensure clean shutdown
    of write queues, connections, etc.
    """
    from embeddings.cache import shutdown as cache_shutdown
    await cache_shutdown()


def shutdown():
    """Synchronous wrapper to shutdown async modules."""
    run_async(shutdown_modules())


# Re-export async search functions for convenience
from search.vector import (
    search_ideas,
    find_similar_ideas,
    enrich_with_relations,
    search_spans,
)
from search.hybrid import hybrid_search
from search.hyde import (
    generate_hypothetical_doc,
    hyde_search,
)
from embeddings.cache import (
    get_embedding_cache_stats,
    clear_embedding_cache,
    flush_write_queue,
    cache_source,
)
from embeddings.openai import (
    get_embedding,
    get_embeddings_batch,
)

__all__ = [
    # Core async runner
    "run_async",
    "shutdown",
    "shutdown_modules",
    # Async search functions
    "search_ideas",
    "find_similar_ideas",
    "enrich_with_relations",
    "search_spans",
    "hybrid_search",
    "generate_hypothetical_doc",
    "hyde_search",
    # Cache operations
    "get_embedding_cache_stats",
    "clear_embedding_cache",
    "flush_write_queue",
    "cache_source",
    # Embedding operations
    "get_embedding",
    "get_embeddings_batch",
]
