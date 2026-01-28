"""Research bundle - comprehensive retrieval in a single call.

Consolidates multiple search strategies into one efficient operation,
returning a rich bundle of information to minimize round-trips.
"""

import asyncio
from typing import Optional
from datetime import datetime, timedelta

from config import logger
from search.vector import search_ideas, find_similar_ideas, enrich_with_relations, search_spans
from search.hybrid import hybrid_search


async def research(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    deep: bool = False
) -> dict:
    """Comprehensive research on a topic in a single call.

    Runs multiple search strategies in parallel and consolidates results
    into a rich bundle. This is designed to replace multiple fine-grained
    search calls with one efficient operation.

    Args:
        query: Research query
        limit: Maximum results per search strategy
        session: Optional session/project to scope to
        deep: If True, also find similar ideas and enrich with relations

    Returns:
        Research bundle dict with:
            - query: Original query
            - summary: Brief summary of findings
            - ideas: Deduplicated list of relevant ideas (ranked by relevance)
            - topics: Related topic spans
            - timeline: Ideas organized by time
            - stats: Search statistics
    """
    start_time = datetime.now()
    logger.info(f"Research: '{query[:50]}...' deep={deep}")

    # Run multiple search strategies in parallel
    search_tasks = [
        search_ideas(query, limit=limit, session=session),
        hybrid_search(query, limit=limit, session=session),
        search_spans(query, limit=5),
    ]

    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    vector_results = results[0] if not isinstance(results[0], Exception) else []
    hybrid_results = results[1] if not isinstance(results[1], Exception) else []
    span_results = results[2] if not isinstance(results[2], Exception) else []

    # Log any errors
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(f"Research search {i} failed: {r}")

    # Deduplicate and rank ideas
    seen_ids = set()
    all_ideas = []
    idea_scores = {}  # id -> score (lower is better)

    # Process vector results (rank 0-N)
    for i, idea in enumerate(vector_results):
        idea_id = idea.get('id')
        if idea_id and idea_id not in seen_ids:
            seen_ids.add(idea_id)
            all_ideas.append(idea)
            idea_scores[idea_id] = i

    # Process hybrid results (boost if also in vector)
    for i, idea in enumerate(hybrid_results):
        idea_id = idea.get('id')
        if idea_id:
            if idea_id in seen_ids:
                # Boost score for appearing in both
                idea_scores[idea_id] = min(idea_scores.get(idea_id, i), i) - 0.5
            else:
                seen_ids.add(idea_id)
                all_ideas.append(idea)
                idea_scores[idea_id] = i + len(vector_results)

    # Sort by combined score
    all_ideas.sort(key=lambda x: idea_scores.get(x.get('id'), 999))

    # Limit final results
    top_ideas = all_ideas[:limit * 2]

    # Deep mode: enrich with relations and find similar
    if deep and top_ideas:
        try:
            top_ideas = await enrich_with_relations(top_ideas)
        except Exception as e:
            logger.warning(f"Failed to enrich with relations: {e}")

        # Find similar ideas for top 3 results
        if len(top_ideas) >= 1:
            similar_tasks = [
                find_similar_ideas(idea['id'], limit=3, same_session=False)
                for idea in top_ideas[:3]
                if idea.get('id')
            ]
            if similar_tasks:
                try:
                    similar_results = await asyncio.gather(*similar_tasks, return_exceptions=True)
                    for i, similar in enumerate(similar_results):
                        if isinstance(similar, list) and similar:
                            # Add similar ideas that aren't already in results
                            for s in similar:
                                if s.get('id') not in seen_ids:
                                    s['found_via'] = 'similar'
                                    s['similar_to'] = top_ideas[i].get('id')
                                    top_ideas.append(s)
                                    seen_ids.add(s.get('id'))
                except Exception as e:
                    logger.warning(f"Failed to find similar ideas: {e}")

    # Build timeline view (group by date)
    timeline = {}
    for idea in top_ideas:
        # Use message_time or created_at
        ts = idea.get('message_time') or idea.get('created_at')
        if ts:
            try:
                # Parse and group by date
                if isinstance(ts, str):
                    date_key = ts[:10]  # YYYY-MM-DD
                else:
                    date_key = str(ts)[:10]
                if date_key not in timeline:
                    timeline[date_key] = []
                timeline[date_key].append({
                    'id': idea.get('id'),
                    'content': idea.get('content', '')[:200],
                    'intent': idea.get('intent'),
                    'topic': idea.get('topic')
                })
            except Exception:
                pass

    # Sort timeline by date descending
    sorted_timeline = dict(sorted(timeline.items(), reverse=True))

    # Calculate stats
    elapsed = (datetime.now() - start_time).total_seconds()
    stats = {
        'total_ideas': len(top_ideas),
        'unique_sessions': len(set(i.get('session') for i in top_ideas if i.get('session'))),
        'unique_topics': len(set(i.get('topic') for i in top_ideas if i.get('topic'))),
        'search_time_ms': int(elapsed * 1000),
        'searches_combined': 3 if not deep else 4
    }

    # Build summary
    intents = {}
    for idea in top_ideas[:10]:
        intent = idea.get('intent', 'unknown')
        intents[intent] = intents.get(intent, 0) + 1

    intent_summary = ', '.join(f"{k}: {v}" for k, v in sorted(intents.items(), key=lambda x: -x[1]))
    summary = f"Found {len(top_ideas)} relevant ideas across {stats['unique_sessions']} sessions. "
    if intent_summary:
        summary += f"Types: {intent_summary}. "
    if span_results:
        summary += f"Related topics: {', '.join(s.get('name', 'unknown')[:30] for s in span_results[:3])}."

    return {
        'query': query,
        'summary': summary,
        'ideas': top_ideas,
        'topics': [dict(s) for s in span_results],
        'timeline': sorted_timeline,
        'stats': stats
    }
