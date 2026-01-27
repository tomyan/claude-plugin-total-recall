"""Search operations for total-recall.

All functions are async.
"""

from search.vector import (
    search_ideas_async,
    find_similar_ideas_async,
    enrich_with_relations_async,
    search_spans_async,
    _update_access_tracking_async,
)
from search.hybrid import hybrid_search_async
from search.hyde import generate_hypothetical_doc_async, hyde_search_async

__all__ = [
    # Vector search (async)
    "search_ideas_async",
    "find_similar_ideas_async",
    "enrich_with_relations_async",
    "search_spans_async",
    "_update_access_tracking_async",
    # Hybrid search (async)
    "hybrid_search_async",
    # HyDE search (async)
    "generate_hypothetical_doc_async",
    "hyde_search_async",
]
