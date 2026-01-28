"""Search operations for total-recall.

All functions are async.
"""

from search.vector import (
    search_ideas,
    find_similar_ideas,
    enrich_with_relations,
    search_spans,
    _update_access_tracking,
)
from search.hybrid import hybrid_search
from search.hyde import generate_hypothetical_doc, hyde_search
from search.research import research

__all__ = [
    # Vector search (async)
    "search_ideas",
    "find_similar_ideas",
    "enrich_with_relations",
    "search_spans",
    "_update_access_tracking",
    # Hybrid search (async)
    "hybrid_search",
    # HyDE search (async)
    "generate_hypothetical_doc",
    "hyde_search",
    # Research bundle (async)
    "research",
]
