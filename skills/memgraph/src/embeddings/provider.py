"""Embedding provider abstraction for memgraph."""

from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Implementations must provide get_embedding() and optionally
    get_embeddings_batch() for efficiency.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openai', 'local')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 1536 for OpenAI)."""
        pass

    @abstractmethod
    def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector

        Raises:
            MemgraphError: If embedding fails
        """
        pass

    def get_embeddings_batch(
        self,
        texts: list[str],
        use_cache: bool = True
    ) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Default implementation calls get_embedding() for each text.
        Providers may override for efficiency.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors
        """
        return [self.get_embedding(text, use_cache) for text in texts]


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Not yet implemented - raises NotImplementedError.
    """

    @property
    def name(self) -> str:
        return "local"

    @property
    def dimension(self) -> int:
        return 384  # Default for all-MiniLM-L6-v2

    def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        raise NotImplementedError(
            "Local embeddings not yet implemented. "
            "Install sentence-transformers and set provider='local' in config."
        )
