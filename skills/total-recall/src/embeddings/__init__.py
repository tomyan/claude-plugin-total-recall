"""Embedding operations for total-recall.

All functions are async.
"""

from embeddings.cache import (
    cache_source,
    cache_source_sync,
    get_cached_embedding,
    cache_embedding,
    get_embedding_cache_stats,
    clear_embedding_cache,
    flush_write_queue,
    shutdown,
    get_cache_max_size,
)
from embeddings.openai import (
    get_embedding,
    get_embeddings_batch,
    AsyncOpenAIEmbeddings,
    _get_async_provider,
    _reset_async_provider,
)
from embeddings.serialize import serialize_embedding, deserialize_embedding

__all__ = [
    # Cache functions (async)
    "cache_source",
    "cache_source_sync",
    "get_cached_embedding",
    "cache_embedding",
    "get_embedding_cache_stats",
    "clear_embedding_cache",
    "flush_write_queue",
    "shutdown",
    "get_cache_max_size",
    # OpenAI functions (async)
    "get_embedding",
    "get_embeddings_batch",
    "AsyncOpenAIEmbeddings",
    "_get_async_provider",
    "_reset_async_provider",
    # Serialization (sync)
    "serialize_embedding",
    "deserialize_embedding",
]
