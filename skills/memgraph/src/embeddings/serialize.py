"""Embedding serialization for sqlite-vec storage."""

import struct


def serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize embedding to bytes for sqlite-vec."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(data: bytes, dim: int = 1536) -> list[float]:
    """Deserialize embedding from bytes."""
    return list(struct.unpack(f'{dim}f', data))
