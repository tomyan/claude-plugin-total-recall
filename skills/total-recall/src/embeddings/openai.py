"""Async OpenAI embedding provider for total-recall."""

import asyncio
import random
from typing import Optional

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError

from config import EMBEDDING_MODEL, EMBEDDING_DIM, logger, get_openai_api_key, OPENAI_KEY_FILE
from embeddings.cache import (
    cache_embedding, get_cached_embedding, cache_source
)
from errors import TotalRecallError


def _is_transient_api_error(e: Exception) -> bool:
    """Check if an OpenAI API error is transient (worth retrying)."""
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(e, APIStatusError) and e.status_code >= 500:
        return True
    return False


class AsyncOpenAIEmbeddings:
    """Async OpenAI embedding provider using text-embedding-3-small."""

    def __init__(self, model: str = None, dimension: int = None):
        """Initialize async OpenAI embeddings.

        Args:
            model: Model name (defaults to config EMBEDDING_MODEL)
            dimension: Embedding dimension (defaults to config EMBEDDING_DIM)
        """
        self._model = model or EMBEDDING_MODEL
        self._dimension = dimension or EMBEDDING_DIM
        self._client: Optional[AsyncOpenAI] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            api_key = get_openai_api_key()
            if not api_key:
                raise TotalRecallError(
                    f"OpenAI API key not found. Create {OPENAI_KEY_FILE} with your key.",
                    "missing_api_key"
                )
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Get embedding from OpenAI with async caching and retry.

        Retries on transient errors (rate limit, timeout, server errors)
        with exponential backoff. Permanent errors (auth, bad request) fail fast.

        Args:
            text: Text to embed
            use_cache: Whether to use cache (default True)

        Returns:
            Embedding vector (1536 floats)

        Raises:
            TotalRecallError: If API key is missing or API call fails permanently
        """
        # Check cache first
        if use_cache:
            cached = await get_cached_embedding(text)
            if cached is not None:
                return cached

        embedding = await self._call_api_with_retry(text)

        # Cache the result
        if use_cache:
            await cache_embedding(text, embedding)

        return embedding

    async def _call_api_with_retry(
        self, text_or_texts, max_retries: int = 5, max_backoff: float = 60.0
    ):
        """Call embedding API with retry on transient errors.

        Args:
            text_or_texts: Single text string or list of texts
            max_retries: Maximum retry attempts for transient errors
            max_backoff: Maximum backoff delay in seconds

        Returns:
            API response

        Raises:
            TotalRecallError: On permanent errors or after max retries
        """
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()
                response = await client.embeddings.create(
                    model=self._model,
                    input=text_or_texts
                )
                if attempt > 0:
                    logger.info(f"Embedding API succeeded after {attempt + 1} attempts")
                if isinstance(text_or_texts, str):
                    return response.data[0].embedding
                return response
            except TotalRecallError:
                raise
            except Exception as e:
                if _is_transient_api_error(e) and attempt < max_retries:
                    jittered = delay * (0.5 + random.random())
                    logger.warning(
                        f"Transient embedding error ({type(e).__name__}), "
                        f"retry {attempt + 1}/{max_retries} in {jittered:.1f}s"
                    )
                    await asyncio.sleep(jittered)
                    delay = min(delay * 2, max_backoff)
                else:
                    kind = "transient (exhausted retries)" if _is_transient_api_error(e) else "permanent"
                    logger.error(f"Embedding API {kind} error: {e}")
                    raise TotalRecallError(
                        f"Failed to get embedding from OpenAI: {e}",
                        "embedding_failed",
                        {"model": self._model, "original_error": str(e), "attempts": attempt + 1}
                    ) from e

    async def get_embeddings_batch(
        self,
        texts: list[str],
        use_cache: bool = True
    ) -> list[list[float]]:
        """Get embeddings for multiple texts in a single API call.

        More efficient than calling get_embedding() for each text separately.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache (default True)

        Returns:
            List of embedding vectors, one per input text

        Raises:
            TotalRecallError: If API key is missing or API call fails
        """
        if not texts:
            return []

        # Check cache and identify texts that need embedding
        results: list[Optional[list[float]]] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)

        if use_cache:
            for i, text in enumerate(texts):
                cached = await get_cached_embedding(text)
                if cached is not None:
                    results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # If everything was cached, return early
        if not texts_to_embed:
            return results  # type: ignore

        # Call API with retry for transient errors
        batch_texts = [t for _, t in texts_to_embed]
        response = await self._call_api_with_retry(batch_texts)

        # Map results back to original positions and cache them
        for batch_idx, (orig_idx, text) in enumerate(texts_to_embed):
            embedding = response.data[batch_idx].embedding
            results[orig_idx] = embedding

            # Cache the result
            if use_cache:
                await cache_embedding(text, embedding)

        logger.debug(f"Async batch embedded {len(batch_texts)} texts ({len(texts) - len(batch_texts)} cached)")

        return results  # type: ignore


# Default provider instance
_default_provider: Optional[AsyncOpenAIEmbeddings] = None


def _get_async_provider() -> AsyncOpenAIEmbeddings:
    """Get or create the default async provider."""
    global _default_provider
    if _default_provider is None:
        _default_provider = AsyncOpenAIEmbeddings()
    return _default_provider


def _reset_async_provider() -> None:
    """Reset the default provider (for testing)."""
    global _default_provider
    _default_provider = None


# Convenience functions
async def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    """Get embedding from OpenAI with async caching.

    Args:
        text: Text to embed
        use_cache: Whether to use cache (default True)

    Returns:
        Embedding vector (1536 floats)

    Raises:
        TotalRecallError: If API key is missing or API call fails
    """
    return await _get_async_provider().get_embedding(text, use_cache)


async def get_embeddings_batch(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """Get embeddings for multiple texts in a single async API call.

    More efficient than calling get_embedding() for each text separately.

    Args:
        texts: List of texts to embed
        use_cache: Whether to use cache (default True)

    Returns:
        List of embedding vectors, one per input text

    Raises:
        TotalRecallError: If API key is missing or API call fails
    """
    return await _get_async_provider().get_embeddings_batch(texts, use_cache)
