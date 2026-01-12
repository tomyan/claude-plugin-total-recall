"""OpenAI embedding provider for memgraph."""

import os
from typing import Optional

from openai import OpenAI

from config import EMBEDDING_MODEL, logger
from embeddings.cache import cache_embedding, get_cached_embedding
from errors import MemgraphError


def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    """Get embedding from OpenAI with caching.

    Args:
        text: Text to embed
        use_cache: Whether to use cache (default True)

    Returns:
        Embedding vector (1536 floats)

    Raises:
        MemgraphError: If API key is missing or API call fails
    """
    # Check cache first
    if use_cache:
        cached = get_cached_embedding(text)
        if cached is not None:
            return cached

    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key:
        raise MemgraphError(
            "OPENAI_TOKEN_MEMORY_EMBEDDINGS environment variable not set. "
            "Set this to your OpenAI API key to enable memory search.",
            "missing_api_key"
        )

    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise MemgraphError(
            f"Failed to get embedding from OpenAI: {e}",
            "embedding_failed",
            {"model": EMBEDDING_MODEL, "original_error": str(e)}
        ) from e

    # Cache the result
    if use_cache:
        cache_embedding(text, embedding)

    return embedding


def get_embeddings_batch(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """Get embeddings for multiple texts in a single API call.

    More efficient than calling get_embedding() for each text separately.

    Args:
        texts: List of texts to embed
        use_cache: Whether to use cache (default True)

    Returns:
        List of embedding vectors, one per input text

    Raises:
        MemgraphError: If API key is missing or API call fails
    """
    if not texts:
        return []

    # Check cache and identify texts that need embedding
    results: list[Optional[list[float]]] = [None] * len(texts)
    texts_to_embed: list[tuple[int, str]] = []  # (index, text)

    if use_cache:
        for i, text in enumerate(texts):
            cached = get_cached_embedding(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append((i, text))
    else:
        texts_to_embed = list(enumerate(texts))

    # If everything was cached, return early
    if not texts_to_embed:
        return results  # type: ignore

    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key:
        raise MemgraphError(
            "OPENAI_TOKEN_MEMORY_EMBEDDINGS environment variable not set. "
            "Set this to your OpenAI API key to enable memory search.",
            "missing_api_key"
        )

    try:
        client = OpenAI(api_key=api_key)
        # Send all uncached texts in a single request
        batch_texts = [t for _, t in texts_to_embed]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )

        # Map results back to original positions and cache them
        for batch_idx, (orig_idx, text) in enumerate(texts_to_embed):
            embedding = response.data[batch_idx].embedding
            results[orig_idx] = embedding

            # Cache the result
            if use_cache:
                cache_embedding(text, embedding)

        logger.debug(f"Batch embedded {len(batch_texts)} texts ({len(texts) - len(batch_texts)} cached)")

    except Exception as e:
        logger.error(f"Batch embedding API call failed: {e}")
        raise MemgraphError(
            f"Failed to get embeddings from OpenAI: {e}",
            "embedding_failed",
            {"model": EMBEDDING_MODEL, "batch_size": len(texts_to_embed), "original_error": str(e)}
        ) from e

    return results  # type: ignore
