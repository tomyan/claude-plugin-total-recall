"""Embedding cache management for memgraph."""

import json
from pathlib import Path

from config import logger

# Cache configuration
CACHE_PATH = Path.home() / ".claude-plugin-memgraph" / "embedding_cache.json"
_CACHE_MAX_SIZE = 1000

# LRU cache for embeddings to reduce API calls
_embedding_cache: dict[str, list[float]] = {}


def get_cache() -> dict[str, list[float]]:
    """Get reference to the embedding cache."""
    return _embedding_cache


def get_cache_max_size() -> int:
    """Get the maximum cache size."""
    return _CACHE_MAX_SIZE


def clear_embedding_cache():
    """Clear the embedding cache."""
    _embedding_cache.clear()


def get_embedding_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_embedding_cache),
        "max_size": _CACHE_MAX_SIZE,
    }


def save_embedding_cache():
    """Save embedding cache to disk."""
    global _embedding_cache
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(_embedding_cache, f)
        logger.info(f"Saved {len(_embedding_cache)} embeddings to cache")
    except Exception as e:
        logger.warning(f"Failed to save embedding cache: {e}")


def load_embedding_cache():
    """Load embedding cache from disk."""
    global _embedding_cache
    if not CACHE_PATH.exists():
        return

    try:
        with open(CACHE_PATH, 'r') as f:
            loaded = json.load(f)
            _embedding_cache.update(loaded)
        logger.info(f"Loaded {len(loaded)} embeddings from cache")
    except Exception as e:
        logger.warning(f"Failed to load embedding cache: {e}")


def cache_embedding(text: str, embedding: list[float]):
    """Add an embedding to the cache with LRU eviction."""
    global _embedding_cache
    if len(_embedding_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (first key)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    _embedding_cache[text] = embedding


def get_cached_embedding(text: str) -> list[float] | None:
    """Get an embedding from cache, or None if not cached."""
    return _embedding_cache.get(text)


# Auto-load cache on module import
load_embedding_cache()
