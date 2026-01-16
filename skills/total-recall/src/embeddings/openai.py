"""OpenAI embedding provider for total-recall."""

import os
from typing import Optional

from openai import OpenAI

from config import EMBEDDING_MODEL, EMBEDDING_DIM, logger
from embeddings.cache import cache_embedding, get_cached_embedding
from embeddings.provider import EmbeddingProvider
from errors import TotalRecallError


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    def __init__(self, model: str = None, dimension: int = None):
        """Initialize OpenAI embeddings.

        Args:
            model: Model name (defaults to config EMBEDDING_MODEL)
            dimension: Embedding dimension (defaults to config EMBEDDING_DIM)
        """
        self._model = model or EMBEDDING_MODEL
        self._dimension = dimension or EMBEDDING_DIM
        self._client: Optional[OpenAI] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            from config import get_openai_api_key, OPENAI_KEY_FILE
            api_key = get_openai_api_key()
            if not api_key:
                raise TotalRecallError(
                    f"OpenAI API key not found. Create {OPENAI_KEY_FILE} with your key, "
                    "or set OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS environment variable.",
                    "missing_api_key"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Get embedding from OpenAI with caching.

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
            cached = get_cached_embedding(text)
            if cached is not None:
                return cached

        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self._model,
                input=text
            )
            embedding = response.data[0].embedding
        except TotalRecallError:
            raise
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise TotalRecallError(
                f"Failed to get embedding from OpenAI: {e}",
                "embedding_failed",
                {"model": self._model, "original_error": str(e)}
            ) from e

        # Cache the result
        if use_cache:
            cache_embedding(text, embedding)

        return embedding

    def get_embeddings_batch(
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

        try:
            client = self._get_client()
            # Send all uncached texts in a single request
            batch_texts = [t for _, t in texts_to_embed]
            response = client.embeddings.create(
                model=self._model,
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

        except TotalRecallError:
            raise
        except Exception as e:
            logger.error(f"Batch embedding API call failed: {e}")
            raise TotalRecallError(
                f"Failed to get embeddings from OpenAI: {e}",
                "embedding_failed",
                {"model": self._model, "batch_size": len(texts_to_embed), "original_error": str(e)}
            ) from e

        return results  # type: ignore


# Default provider instance
_default_provider: Optional[OpenAIEmbeddings] = None


def _get_provider() -> OpenAIEmbeddings:
    """Get or create the default provider."""
    global _default_provider
    if _default_provider is None:
        _default_provider = OpenAIEmbeddings()
    return _default_provider


def _reset_provider() -> None:
    """Reset the default provider (for testing)."""
    global _default_provider
    _default_provider = None


# Backward-compatible functions
def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    """Get embedding from OpenAI with caching.

    Args:
        text: Text to embed
        use_cache: Whether to use cache (default True)

    Returns:
        Embedding vector (1536 floats)

    Raises:
        TotalRecallError: If API key is missing or API call fails
    """
    return _get_provider().get_embedding(text, use_cache)


def get_embeddings_batch(texts: list[str], use_cache: bool = True) -> list[list[float]]:
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
    return _get_provider().get_embeddings_batch(texts, use_cache)
