"""Async OpenAI embedding provider for total-recall."""

from typing import Optional

from openai import AsyncOpenAI

from config import EMBEDDING_MODEL, EMBEDDING_DIM, logger, get_openai_api_key, OPENAI_KEY_FILE
from embeddings.cache import (
    cache_embedding, get_cached_embedding, cache_source
)
from errors import TotalRecallError


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
        """Get embedding from OpenAI with async caching.

        Args:
            text: Text to embed
            use_cache: Whether to use cache (default True)

        Returns:
            Embedding vector (1536 floats)

        Raises:
            TotalRecallError: If API key is missing or API call fails
        """
        # Check cache first
        if use_cache:
            cached = await get_cached_embedding(text)
            if cached is not None:
                return cached

        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self._model,
                input=text
            )
            embedding = response.data[0].embedding
        except TotalRecallError:
            raise
        except Exception as e:
            logger.error(f"Async embedding API call failed: {e}")
            raise TotalRecallError(
                f"Failed to get embedding from OpenAI: {e}",
                "embedding_failed",
                {"model": self._model, "original_error": str(e)}
            ) from e

        # Cache the result
        if use_cache:
            await cache_embedding(text, embedding)

        return embedding

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

        try:
            client = self._get_client()
            # Send all uncached texts in a single request
            batch_texts = [t for _, t in texts_to_embed]
            response = await client.embeddings.create(
                model=self._model,
                input=batch_texts
            )

            # Map results back to original positions and cache them
            for batch_idx, (orig_idx, text) in enumerate(texts_to_embed):
                embedding = response.data[batch_idx].embedding
                results[orig_idx] = embedding

                # Cache the result
                if use_cache:
                    await cache_embedding(text, embedding)

            logger.debug(f"Async batch embedded {len(batch_texts)} texts ({len(texts) - len(batch_texts)} cached)")

        except TotalRecallError:
            raise
        except Exception as e:
            logger.error(f"Async batch embedding API call failed: {e}")
            raise TotalRecallError(
                f"Failed to get embeddings from OpenAI: {e}",
                "embedding_failed",
                {"model": self._model, "batch_size": len(texts_to_embed), "original_error": str(e)}
            ) from e

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
