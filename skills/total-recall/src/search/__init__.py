"""Search operations for total-recall."""

from search.vector import (
    search_ideas,
    find_similar_ideas,
    enrich_with_relations,
    search_spans,
)
from search.hybrid import hybrid_search
from search.hyde import generate_hypothetical_doc, hyde_search

__all__ = [
    "search_ideas",
    "find_similar_ideas",
    "enrich_with_relations",
    "search_spans",
    "hybrid_search",
    "generate_hypothetical_doc",
    "hyde_search",
]
