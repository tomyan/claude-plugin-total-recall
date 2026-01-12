#!/usr/bin/env python3
"""
Memory database for Claude memory graph skill.
Uses SQLite + sqlite-vec for vector similarity search.
Uses OpenAI for embeddings, Claude CLI for LLM tasks.
"""

import json
import os
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional

import sqlite_vec
from openai import OpenAI

# Import config - constants and logger
from config import DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIM, LOG_PATH, logger  # noqa: E402


# =============================================================================
# Custom Exception
# =============================================================================

from errors import MemgraphError  # noqa: E402 - imported here for backward compatibility


# =============================================================================
# Claude CLI Integration
# =============================================================================

from llm.claude import claude_complete  # noqa: E402 - imported here for backward compatibility

# Import database operations from db module
from db.connection import get_db  # noqa: E402
from db.schema import init_db  # noqa: E402
from db.migrations import migrate_timestamps_from_transcripts  # noqa: E402


# Import serialization from embeddings module
from embeddings.serialize import serialize_embedding, deserialize_embedding  # noqa: E402


# Import cache functions from embeddings module
from embeddings.cache import (  # noqa: E402
    CACHE_PATH,
    clear_embedding_cache,
    get_embedding_cache_stats,
    save_embedding_cache,
    load_embedding_cache,
    get_cache,
)
# Re-export cache for backward compatibility (tests access this directly)
_embedding_cache = get_cache()

# Import embedding functions from embeddings module
from embeddings.openai import get_embedding, get_embeddings_batch  # noqa: E402

# Import search functions from search module
from search.vector import (  # noqa: E402
    search_ideas,
    find_similar_ideas,
    enrich_with_relations,
    search_spans,
)
from search.hybrid import hybrid_search  # noqa: E402
from search.hyde import generate_hypothetical_doc, hyde_search  # noqa: E402


# =============================================================================
# Project Operations
# =============================================================================

def create_project(name: str, description: Optional[str] = None) -> int:
    """Create a new project.

    Args:
        name: Project name (must be unique)
        description: Optional description

    Returns:
        Project ID
    """
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO projects (name, description) VALUES (?, ?)",
            (name, description)
        )
        project_id = cursor.lastrowid
        db.commit()
    except sqlite3.IntegrityError:
        db.close()
        raise MemgraphError(
            f"Project '{name}' already exists",
            "duplicate_project",
            {"name": name}
        )
    db.close()
    return project_id


def get_project(project_id: int) -> Optional[dict]:
    """Get project by ID."""
    db = get_db()
    cursor = db.execute(
        "SELECT id, name, description, created_at FROM projects WHERE id = ?",
        (project_id,)
    )
    row = cursor.fetchone()
    db.close()
    return dict(row) if row else None


def get_project_by_name(name: str) -> Optional[dict]:
    """Get project by name."""
    db = get_db()
    cursor = db.execute(
        "SELECT id, name, description, created_at FROM projects WHERE name = ?",
        (name,)
    )
    row = cursor.fetchone()
    db.close()
    return dict(row) if row else None


def list_projects() -> list[dict]:
    """List all projects with topic counts."""
    db = get_db()
    cursor = db.execute("""
        SELECT p.id, p.name, p.description, p.created_at,
               COUNT(DISTINCT t.id) as topic_count,
               (SELECT COUNT(*) FROM ideas i
                JOIN spans s ON s.id = i.span_id
                JOIN topics t2 ON t2.id = s.topic_id
                WHERE t2.project_id = p.id) as idea_count
        FROM projects p
        LEFT JOIN topics t ON t.project_id = p.id
        GROUP BY p.id
        ORDER BY p.name
    """)
    projects = [dict(row) for row in cursor]
    db.close()
    return projects


def assign_topic_to_project(topic_id: int, project_id: Optional[int]) -> bool:
    """Assign a topic to a project (or remove from project if None).

    Args:
        topic_id: Topic to assign
        project_id: Project ID or None to unassign

    Returns:
        True if successful
    """
    db = get_db()
    db.execute(
        "UPDATE topics SET project_id = ? WHERE id = ?",
        (project_id, topic_id)
    )
    db.commit()
    db.close()
    return True


def unparent_topic(topic_id: int) -> bool:
    """Remove a topic from its parent, making it a top-level topic.

    Args:
        topic_id: Topic to unparent

    Returns:
        True if updated
    """
    db = get_db()
    db.execute(
        "UPDATE topics SET parent_id = NULL WHERE id = ?",
        (topic_id,)
    )
    db.commit()
    db.close()
    return True


def reparent_topic(topic_id: int, parent_id: int) -> bool:
    """Set a topic's parent, creating a hierarchy.

    Args:
        topic_id: Topic to reparent
        parent_id: New parent topic ID

    Returns:
        True if updated

    Raises:
        MemgraphError: If topic or parent not found, or would create cycle
    """
    db = get_db()

    # Verify both topics exist
    topic = db.execute("SELECT id FROM topics WHERE id = ?", (topic_id,)).fetchone()
    parent = db.execute("SELECT id, parent_id FROM topics WHERE id = ?", (parent_id,)).fetchone()

    if not topic:
        db.close()
        raise MemgraphError(f"Topic {topic_id} not found", "topic_not_found")
    if not parent:
        db.close()
        raise MemgraphError(f"Parent topic {parent_id} not found", "topic_not_found")

    # Check for cycles - walk up the parent chain
    current = parent_id
    visited = {topic_id}
    while current:
        if current in visited:
            db.close()
            raise MemgraphError(
                f"Cannot reparent: would create cycle",
                "cycle_detected",
                {"topic_id": topic_id, "parent_id": parent_id}
            )
        visited.add(current)
        row = db.execute("SELECT parent_id FROM topics WHERE id = ?", (current,)).fetchone()
        current = row["parent_id"] if row else None

    db.execute(
        "UPDATE topics SET parent_id = ? WHERE id = ?",
        (parent_id, topic_id)
    )
    db.commit()
    db.close()
    return True


def delete_topic(topic_id: int, delete_ideas: bool = False) -> dict:
    """Delete a topic and optionally its ideas.

    Args:
        topic_id: Topic to delete
        delete_ideas: If True, delete all ideas in this topic's spans

    Returns:
        Dict with counts of deleted items
    """
    db = get_db()

    ideas_deleted = 0
    spans_deleted = 0

    if delete_ideas:
        # Get all idea IDs in this topic
        cursor = db.execute("""
            SELECT i.id FROM ideas i
            JOIN spans s ON s.id = i.span_id
            WHERE s.topic_id = ?
        """, (topic_id,))
        idea_ids = [row["id"] for row in cursor]

        if idea_ids:
            placeholders = ",".join("?" * len(idea_ids))
            db.execute(f"DELETE FROM idea_embeddings WHERE idea_id IN ({placeholders})", idea_ids)
            db.execute(f"DELETE FROM relations WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                      idea_ids + idea_ids)
            db.execute(f"DELETE FROM idea_entities WHERE idea_id IN ({placeholders})", idea_ids)
            try:
                db.execute(f"DELETE FROM ideas_fts WHERE rowid IN ({placeholders})", idea_ids)
            except Exception:
                pass  # FTS might not have these rows
            db.execute(f"DELETE FROM ideas WHERE id IN ({placeholders})", idea_ids)
            ideas_deleted = len(idea_ids)

    # Delete spans
    cursor = db.execute("DELETE FROM spans WHERE topic_id = ?", (topic_id,))
    spans_deleted = cursor.rowcount

    # Update any child topics to have no parent
    db.execute("UPDATE topics SET parent_id = NULL WHERE parent_id = ?", (topic_id,))

    # Delete topic
    db.execute("DELETE FROM topics WHERE id = ?", (topic_id,))

    db.commit()
    db.close()

    return {
        "ideas_deleted": ideas_deleted,
        "spans_deleted": spans_deleted,
    }


def get_project_tree() -> list[dict]:
    """Get hierarchical view of projects -> topics (with nested children).

    Returns:
        List of projects with nested topics (topics can have children)
    """
    db = get_db()

    # Get all projects
    cursor = db.execute("""
        SELECT id, name, description FROM projects ORDER BY name
    """)
    projects = [dict(row) for row in cursor]

    # Get topics with parent_id for hierarchy
    cursor = db.execute("""
        SELECT t.id, t.name, t.project_id, t.parent_id,
               COUNT(DISTINCT s.id) as span_count,
               COUNT(DISTINCT i.id) as idea_count
        FROM topics t
        LEFT JOIN spans s ON s.topic_id = t.id
        LEFT JOIN ideas i ON i.span_id = s.id
        GROUP BY t.id
        ORDER BY t.name
    """)
    topics = [dict(row) for row in cursor]
    db.close()

    # Build topic hierarchy
    topic_map = {t["id"]: {**t, "children": []} for t in topics}

    # Link children to parents
    root_topics = []
    for topic in topics:
        if topic["parent_id"] and topic["parent_id"] in topic_map:
            topic_map[topic["parent_id"]]["children"].append(topic_map[topic["id"]])
        else:
            root_topics.append(topic_map[topic["id"]])

    # Build project tree with hierarchical topics
    project_map = {p["id"]: {**p, "topics": []} for p in projects}
    unassigned = {"id": None, "name": "(Unassigned)", "description": None, "topics": []}

    for topic in root_topics:
        if topic["project_id"] and topic["project_id"] in project_map:
            project_map[topic["project_id"]]["topics"].append(topic)
        else:
            unassigned["topics"].append(topic)

    result = list(project_map.values())
    if unassigned["topics"]:
        result.append(unassigned)

    return result


def auto_categorize_topics(dry_run: bool = True) -> dict:
    """Automatically categorize unassigned topics and fix mismatched ones.

    Uses LLM to analyze topic content and match to existing projects.

    Args:
        dry_run: If True, only report what would change

    Returns:
        Dict with categorization results
    """
    db = get_db()

    # Get all projects
    cursor = db.execute("SELECT id, name, description FROM projects")
    projects = {row["id"]: dict(row) for row in cursor}

    if not projects:
        db.close()
        return {"error": "No projects defined. Create projects first."}

    # Get unassigned topics with sample content
    cursor = db.execute("""
        SELECT t.id, t.name, t.summary,
               (SELECT GROUP_CONCAT(i.content, ' | ')
                FROM ideas i
                JOIN spans s ON s.id = i.span_id
                WHERE s.topic_id = t.id
                LIMIT 10) as sample_content
        FROM topics t
        WHERE t.project_id IS NULL
    """)
    unassigned = [dict(row) for row in cursor]
    db.close()

    if not unassigned:
        return {"message": "No unassigned topics", "changes": []}

    # Use Claude CLI to categorize each topic
    project_list = "\n".join(f"- {p['name']}: {p['description'] or 'No description'}"
                             for p in projects.values())

    changes = []
    for topic in unassigned:
        # Build context for LLM
        content_sample = (topic["sample_content"] or "")[:1000]
        prompt = f"""Given these projects:
{project_list}

Which project best matches this topic?

Topic: {topic['name']}
Summary: {topic['summary'] or 'None'}
Sample content: {content_sample}

Reply with ONLY the project name that best matches, or "NONE" if no good match."""

        try:
            suggested = claude_complete(prompt).strip()

            # Find matching project
            matched_project = None
            for pid, p in projects.items():
                if p["name"].lower() == suggested.lower():
                    matched_project = p
                    break

            if matched_project:
                changes.append({
                    "topic_id": topic["id"],
                    "topic_name": topic["name"],
                    "suggested_project": matched_project["name"],
                    "project_id": matched_project["id"],
                })

                if not dry_run:
                    assign_topic_to_project(topic["id"], matched_project["id"])

        except MemgraphError:
            raise  # Re-raise MemgraphError as-is
        except Exception as e:
            logger.error(f"Failed to categorize topic {topic['id']}: {e}")
            raise MemgraphError(
                f"Failed to categorize topic: {e}",
                "auto_categorize_error",
                {"topic_id": topic["id"], "original_error": str(e)}
            ) from e

    return {
        "reviewed": len(unassigned),
        "changes": changes,
        "dry_run": dry_run,
    }


def improve_categorization() -> dict:
    """Run all categorization improvements.

    1. Auto-assign unassigned topics to projects
    2. Review topic names for quality
    3. Detect and flag catch-all topics

    Returns:
        Summary of all improvements made
    """
    results = {}

    # Auto-categorize unassigned topics
    cat_result = auto_categorize_topics(dry_run=False)
    results["categorization"] = cat_result

    # Review topic names
    review_result = review_topics()
    if review_result.get("bad_names"):
        # Auto-rename bad topics
        renamed = []
        for bad in review_result["bad_names"][:10]:  # Limit to 10
            suggested = suggest_topic_name(bad["topic_id"])
            if suggested:
                rename_topic(bad["topic_id"], suggested)
                renamed.append({
                    "topic_id": bad["topic_id"],
                    "old_name": bad["name"],
                    "new_name": suggested,
                })
        results["renamed_topics"] = renamed

    results["remaining_issues"] = {
        "catch_all_topics": len(review_result.get("catch_all", [])),
        "empty_topics": len(review_result.get("empty", [])),
    }

    return results


# =============================================================================
# Topic Operations
# =============================================================================

def canonicalize_topic_name(name: str) -> str:
    """Convert topic name to canonical form for deduplication."""
    import re
    # Lowercase, strip, remove common prefixes
    canonical = name.lower().strip()
    # Remove "session start", "let's", "now let's" prefixes
    canonical = re.sub(r"^(session start|let'?s|now let'?s|okay,?\s*let'?s)\s*", "", canonical)
    # Remove trailing punctuation
    canonical = re.sub(r"[.:,;!?]+$", "", canonical)
    # Collapse whitespace
    canonical = re.sub(r"\s+", " ", canonical)
    # Truncate for matching (first 50 chars)
    return canonical[:50]


def find_or_create_topic(name: str, summary: Optional[str] = None) -> int:
    """Find existing topic by canonical name or create new one.

    Args:
        name: Topic name (will be canonicalized for matching)
        summary: Optional summary for new topics

    Returns:
        Topic ID
    """
    db = get_db()
    canonical = canonicalize_topic_name(name)

    # Try to find existing topic
    cursor = db.execute(
        "SELECT id FROM topics WHERE canonical_name = ?",
        (canonical,)
    )
    row = cursor.fetchone()

    if row:
        topic_id = row["id"]
        # Update summary if provided and current is empty
        if summary:
            db.execute(
                "UPDATE topics SET summary = COALESCE(summary, ?) WHERE id = ?",
                (summary, topic_id)
            )
            db.commit()
    else:
        # Create new topic
        cursor = db.execute(
            "INSERT INTO topics (name, canonical_name, summary) VALUES (?, ?, ?)",
            (name[:100], canonical, summary)
        )
        topic_id = cursor.lastrowid
        db.commit()

    db.close()
    return topic_id


def get_topic(topic_id: int) -> Optional[dict]:
    """Get topic by ID."""
    db = get_db()
    cursor = db.execute(
        "SELECT id, name, canonical_name, summary, created_at FROM topics WHERE id = ?",
        (topic_id,)
    )
    row = cursor.fetchone()
    db.close()
    return dict(row) if row else None


def list_topics() -> list[dict]:
    """List all topics with span counts."""
    db = get_db()
    cursor = db.execute("""
        SELECT t.id, t.name, t.summary, t.created_at,
               COUNT(DISTINCT s.id) as span_count,
               COUNT(DISTINCT i.id) as idea_count
        FROM topics t
        LEFT JOIN spans s ON s.topic_id = t.id
        LEFT JOIN ideas i ON i.span_id = s.id
        GROUP BY t.id
        ORDER BY t.created_at DESC
    """)
    topics = [dict(row) for row in cursor]
    db.close()
    return topics


def merge_topics(source_topic_id: int, target_topic_id: int) -> dict:
    """Merge one topic into another, moving all spans.

    Args:
        source_topic_id: Topic to merge from (will be deleted)
        target_topic_id: Topic to merge into

    Returns:
        Dict with merge stats
    """
    db = get_db()

    # Move spans to target topic
    cursor = db.execute(
        "UPDATE spans SET topic_id = ? WHERE topic_id = ?",
        (target_topic_id, source_topic_id)
    )
    spans_moved = cursor.rowcount

    # Delete source topic
    db.execute("DELETE FROM topics WHERE id = ?", (source_topic_id,))
    db.commit()
    db.close()

    return {"spans_moved": spans_moved, "source_deleted": source_topic_id}


# =============================================================================
# Topic Linking (Cross-Session)
# =============================================================================

def find_related_topics(
    topic_id: int,
    exclude_sessions: list[str] = None,
    min_similarity: float = None,  # Uses config.topic_similarity_threshold if None
    limit: int = 5
) -> list[dict]:
    """Find topics semantically similar to the given topic.

    Args:
        topic_id: The topic to find relations for
        exclude_sessions: Sessions to exclude (typically current session)
        min_similarity: Minimum similarity threshold (0-1, higher = more similar)
        limit: Maximum results

    Returns:
        List of dicts with topic info and similarity score
    """
    # Use config default if not specified
    if min_similarity is None:
        from config import get_config
        min_similarity = get_config().topic_similarity_threshold

    db = get_db()

    # Get the topic's embedding (average of its span embeddings)
    cursor = db.execute("""
        SELECT se.embedding
        FROM span_embeddings se
        JOIN spans s ON s.id = se.span_id
        WHERE s.topic_id = ?
        LIMIT 1
    """, (topic_id,))
    row = cursor.fetchone()

    if not row:
        db.close()
        return []

    topic_embedding = deserialize_embedding(row['embedding'])

    # Convert similarity threshold to distance (lower distance = more similar)
    # For normalized embeddings, distance ≈ 2*(1-similarity)
    max_distance = 2 * (1 - min_similarity)

    # Find similar topics via their span embeddings
    sql = """
        SELECT DISTINCT
            t.id, t.name, t.summary, t.first_seen, t.last_seen,
            s.session,
            MIN(se.distance) as distance
        FROM span_embeddings se
        JOIN spans s ON s.id = se.span_id
        JOIN topics t ON t.id = s.topic_id
        WHERE se.embedding MATCH ? AND k = ?
          AND t.id != ?
    """
    params = [serialize_embedding(topic_embedding), limit * 3, topic_id]

    if exclude_sessions:
        placeholders = ','.join('?' * len(exclude_sessions))
        sql += f" AND s.session NOT IN ({placeholders})"
        params.extend(exclude_sessions)

    sql += """
        GROUP BY t.id
        HAVING MIN(se.distance) <= ?
        ORDER BY MIN(se.distance)
        LIMIT ?
    """
    params.extend([max_distance, limit])

    cursor = db.execute(sql, params)
    results = []
    for row in cursor:
        r = dict(row)
        # Convert distance to similarity
        r['similarity'] = 1 - (r['distance'] / 2)
        results.append(r)

    db.close()
    return results


def link_topics(
    topic_id: int,
    related_topic_id: int,
    similarity: float,
    time_overlap: bool = False,
    link_type: str = 'semantic'
) -> int:
    """Create a link between two topics.

    Args:
        topic_id: First topic ID
        related_topic_id: Second topic ID
        similarity: Similarity score (0-1)
        time_overlap: Whether the topics' time ranges overlap
        link_type: Type of link ('semantic', 'manual', 'merged')

    Returns:
        Link ID
    """
    db = get_db()
    cursor = db.execute("""
        INSERT OR REPLACE INTO topic_links
            (topic_id, related_topic_id, similarity, time_overlap, link_type)
        VALUES (?, ?, ?, ?, ?)
    """, (topic_id, related_topic_id, similarity, time_overlap, link_type))
    link_id = cursor.lastrowid
    db.commit()
    db.close()
    return link_id


def get_topic_links(topic_id: int, include_reverse: bool = True) -> list[dict]:
    """Get all links for a topic.

    Args:
        topic_id: Topic ID
        include_reverse: Also include links where this topic is the related_topic

    Returns:
        List of link dicts with related topic info
    """
    db = get_db()

    if include_reverse:
        cursor = db.execute("""
            SELECT
                tl.id as link_id,
                tl.similarity,
                tl.time_overlap,
                tl.link_type,
                tl.created_at as linked_at,
                CASE WHEN tl.topic_id = ? THEN tl.related_topic_id ELSE tl.topic_id END as other_topic_id,
                t.name as other_topic_name,
                t.summary as other_topic_summary,
                t.first_seen,
                t.last_seen,
                (SELECT DISTINCT session FROM spans WHERE topic_id = t.id LIMIT 1) as session
            FROM topic_links tl
            JOIN topics t ON t.id = CASE WHEN tl.topic_id = ? THEN tl.related_topic_id ELSE tl.topic_id END
            WHERE tl.topic_id = ? OR tl.related_topic_id = ?
            ORDER BY tl.similarity DESC
        """, (topic_id, topic_id, topic_id, topic_id))
    else:
        cursor = db.execute("""
            SELECT
                tl.id as link_id,
                tl.related_topic_id as other_topic_id,
                tl.similarity,
                tl.time_overlap,
                tl.link_type,
                tl.created_at as linked_at,
                t.name as other_topic_name,
                t.summary as other_topic_summary,
                t.first_seen,
                t.last_seen,
                (SELECT DISTINCT session FROM spans WHERE topic_id = t.id LIMIT 1) as session
            FROM topic_links tl
            JOIN topics t ON t.id = tl.related_topic_id
            WHERE tl.topic_id = ?
            ORDER BY tl.similarity DESC
        """, (topic_id,))

    results = [dict(row) for row in cursor]
    db.close()
    return results


def check_time_overlap(
    start1: Optional[str],
    end1: Optional[str],
    start2: Optional[str],
    end2: Optional[str]
) -> bool:
    """Check if two time ranges overlap.

    Args:
        start1, end1: First time range (ISO strings)
        start2, end2: Second time range (ISO strings)

    Returns:
        True if ranges overlap
    """
    if not all([start1, end1, start2, end2]):
        return False

    # Two ranges overlap if one starts before the other ends
    return start1 <= end2 and start2 <= end1


def auto_link_topic(topic_id: int, current_session: str, min_similarity: float = None) -> list[dict]:
    """Automatically link a topic to similar topics in other sessions.

    Args:
        topic_id: Topic to link
        current_session: Current session to exclude
        min_similarity: Minimum similarity for linking (uses config default if None)

    Returns:
        List of created links
    """
    # Use config default if not specified
    if min_similarity is None:
        from config import get_config
        min_similarity = get_config().topic_similarity_threshold

    # Find related topics
    related = find_related_topics(
        topic_id,
        exclude_sessions=[current_session],
        min_similarity=min_similarity
    )

    if not related:
        return []

    # Get current topic's time range
    db = get_db()
    cursor = db.execute("""
        SELECT MIN(start_time) as start_time, MAX(end_time) as end_time
        FROM spans WHERE topic_id = ?
    """, (topic_id,))
    current_times = cursor.fetchone()
    db.close()

    created_links = []
    for rel in related:
        # Check time overlap
        overlap = check_time_overlap(
            current_times['start_time'], current_times['end_time'],
            rel.get('first_seen'), rel.get('last_seen')
        )

        # Create bidirectional link
        link_id = link_topics(
            topic_id, rel['id'],
            similarity=rel['similarity'],
            time_overlap=overlap,
            link_type='semantic'
        )

        created_links.append({
            'link_id': link_id,
            'related_topic_id': rel['id'],
            'related_topic_name': rel['name'],
            'similarity': rel['similarity'],
            'time_overlap': overlap,
            'session': rel.get('session')
        })

    return created_links


# =============================================================================
# Timeline Functions
# =============================================================================

def get_topic_timeline(topic_id: int = None, topic_name: str = None) -> dict:
    """Get timeline for a topic showing activity across sessions.

    Args:
        topic_id: Topic ID (preferred)
        topic_name: Topic name (used if topic_id not provided)

    Returns:
        Dict with topic info and timeline entries grouped by date
    """
    db = get_db()

    # Find topic
    if topic_id:
        cursor = db.execute(
            "SELECT id, name, summary, first_seen, last_seen FROM topics WHERE id = ?",
            (topic_id,)
        )
    else:
        cursor = db.execute(
            "SELECT id, name, summary, first_seen, last_seen FROM topics WHERE name LIKE ?",
            (f"%{topic_name}%",)
        )
    topic = cursor.fetchone()
    if not topic:
        db.close()
        return {"error": "Topic not found"}

    topic_id = topic['id']

    # Get all spans for this topic with their time ranges and idea counts
    cursor = db.execute("""
        SELECT
            s.id as span_id,
            s.session,
            s.name as span_name,
            s.start_time,
            s.end_time,
            COUNT(i.id) as idea_count
        FROM spans s
        LEFT JOIN ideas i ON i.span_id = s.id
        WHERE s.topic_id = ?
        GROUP BY s.id
        ORDER BY COALESCE(s.start_time, s.created_at)
    """, (topic_id,))
    spans = [dict(r) for r in cursor]

    # Get key ideas (decisions, conclusions) for each span
    for span in spans:
        cursor = db.execute("""
            SELECT content, intent, message_time
            FROM ideas
            WHERE span_id = ? AND intent IN ('decision', 'conclusion')
            ORDER BY COALESCE(message_time, created_at)
            LIMIT 3
        """, (span['span_id'],))
        span['key_ideas'] = [dict(r) for r in cursor]

    db.close()

    # Group by date
    from datetime import datetime
    timeline = {}
    for span in spans:
        if span['start_time']:
            try:
                dt = datetime.fromisoformat(span['start_time'].replace('Z', '+00:00'))
                date_key = dt.strftime('%Y-%m-%d')
            except:
                date_key = 'unknown'
        else:
            date_key = 'unknown'

        if date_key not in timeline:
            timeline[date_key] = []
        timeline[date_key].append(span)

    return {
        "topic": dict(topic),
        "timeline": timeline,
        "total_spans": len(spans),
        "total_ideas": sum(s['idea_count'] for s in spans)
    }


def get_project_timeline(session: str, days: int = 7) -> dict:
    """Get activity timeline for a project/session.

    Args:
        session: Session name (project)
        days: Number of days to look back

    Returns:
        Dict with daily activity grouped by date
    """
    from datetime import datetime, timedelta

    db = get_db()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    start_iso = start_date.isoformat() + 'Z'

    # Get spans with their topics and idea counts
    cursor = db.execute("""
        SELECT
            s.id as span_id,
            s.name as span_name,
            s.start_time,
            s.end_time,
            t.id as topic_id,
            t.name as topic_name,
            COUNT(i.id) as idea_count
        FROM spans s
        LEFT JOIN topics t ON t.id = s.topic_id
        LEFT JOIN ideas i ON i.span_id = s.id
        WHERE s.session = ?
          AND COALESCE(s.start_time, s.created_at) >= ?
        GROUP BY s.id
        ORDER BY COALESCE(s.start_time, s.created_at)
    """, (session, start_iso))
    spans = [dict(r) for r in cursor]
    db.close()

    # Group by date
    timeline = {}
    for span in spans:
        if span['start_time']:
            try:
                dt = datetime.fromisoformat(span['start_time'].replace('Z', '+00:00'))
                date_key = dt.strftime('%Y-%m-%d')
            except:
                date_key = 'unknown'
        else:
            date_key = 'unknown'

        if date_key not in timeline:
            timeline[date_key] = {
                "topics": set(),
                "idea_count": 0,
                "spans": []
            }
        timeline[date_key]["topics"].add(span['topic_name'] or span['span_name'][:30])
        timeline[date_key]["idea_count"] += span['idea_count']
        timeline[date_key]["spans"].append(span)

    # Convert sets to lists for JSON
    for date_key in timeline:
        timeline[date_key]["topics"] = list(timeline[date_key]["topics"])

    return {
        "session": session,
        "days": days,
        "timeline": timeline,
        "total_spans": len(spans),
        "total_ideas": sum(s['idea_count'] for s in spans)
    }


def get_activity_by_period(
    period: str = "day",
    days: int = 7,
    session: str = None
) -> dict:
    """Get idea activity aggregated by time period.

    Args:
        period: Aggregation period - 'day', 'week', or 'month'
        days: Number of days to look back
        session: Optional session to filter by

    Returns:
        Dict with period counts and metadata
    """
    from datetime import datetime, timedelta

    db = get_db()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    start_iso = start_date.isoformat() + 'Z'

    # Build query
    sql = """
        SELECT
            i.id,
            COALESCE(i.message_time, i.created_at) as idea_time,
            i.intent,
            s.session
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        WHERE COALESCE(i.message_time, i.created_at) >= ?
    """
    params = [start_iso]

    if session:
        sql += " AND s.session = ?"
        params.append(session)

    sql += " ORDER BY idea_time"

    cursor = db.execute(sql, params)
    ideas = list(cursor)
    db.close()

    # Aggregate by period
    period_counts = {}
    intent_counts = {}

    for idea in ideas:
        idea_time = idea['idea_time']
        intent = idea['intent']

        # Parse time and determine period key
        try:
            dt = datetime.fromisoformat(idea_time.replace('Z', '+00:00'))
            if period == "day":
                period_key = dt.strftime('%Y-%m-%d')
            elif period == "week":
                # ISO week
                period_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
            elif period == "month":
                period_key = dt.strftime('%Y-%m')
            else:
                period_key = dt.strftime('%Y-%m-%d')
        except:
            period_key = 'unknown'

        # Count by period
        if period_key not in period_counts:
            period_counts[period_key] = {
                "total": 0,
                "decisions": 0,
                "questions": 0,
                "conclusions": 0,
                "sessions": set()
            }
        period_counts[period_key]["total"] += 1
        period_counts[period_key]["sessions"].add(idea['session'])
        if intent == 'decision':
            period_counts[period_key]["decisions"] += 1
        elif intent == 'question':
            period_counts[period_key]["questions"] += 1
        elif intent == 'conclusion':
            period_counts[period_key]["conclusions"] += 1

        # Count by intent
        if intent not in intent_counts:
            intent_counts[intent] = 0
        intent_counts[intent] += 1

    # Convert sets to counts for JSON
    for pk in period_counts:
        period_counts[pk]["session_count"] = len(period_counts[pk]["sessions"])
        del period_counts[pk]["sessions"]

    return {
        "period": period,
        "days": days,
        "session": session,
        "total_ideas": len(ideas),
        "by_period": period_counts,
        "by_intent": intent_counts
    }


def get_topic_activity(
    topic_id: int,
    period: str = "week",
    days: int = 90
) -> dict:
    """Get activity for a specific topic over time.

    Args:
        topic_id: Topic ID to analyze
        period: Aggregation period - 'day', 'week', or 'month'
        days: Number of days to look back

    Returns:
        Dict with topic info and activity by period
    """
    from datetime import datetime, timedelta

    db = get_db()

    # Get topic info
    cursor = db.execute(
        "SELECT id, name, summary, first_seen, last_seen FROM topics WHERE id = ?",
        (topic_id,)
    )
    topic = cursor.fetchone()
    if not topic:
        db.close()
        return {"error": "Topic not found"}

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    start_iso = start_date.isoformat() + 'Z'

    # Get ideas for this topic
    cursor = db.execute("""
        SELECT
            i.id,
            COALESCE(i.message_time, i.created_at) as idea_time,
            i.intent,
            i.content,
            s.session
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        WHERE s.topic_id = ?
          AND COALESCE(i.message_time, i.created_at) >= ?
        ORDER BY idea_time
    """, (topic_id, start_iso))
    ideas = list(cursor)
    db.close()

    # Aggregate by period
    period_activity = {}

    for idea in ideas:
        idea_time = idea['idea_time']

        try:
            dt = datetime.fromisoformat(idea_time.replace('Z', '+00:00'))
            if period == "day":
                period_key = dt.strftime('%Y-%m-%d')
            elif period == "week":
                period_key = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
            elif period == "month":
                period_key = dt.strftime('%Y-%m')
            else:
                period_key = dt.strftime('%Y-%m-%d')
        except:
            period_key = 'unknown'

        if period_key not in period_activity:
            period_activity[period_key] = {
                "total": 0,
                "sessions": set(),
                "key_ideas": []
            }
        period_activity[period_key]["total"] += 1
        period_activity[period_key]["sessions"].add(idea['session'])

        # Track key ideas (decisions/conclusions)
        if idea['intent'] in ('decision', 'conclusion') and len(period_activity[period_key]["key_ideas"]) < 3:
            period_activity[period_key]["key_ideas"].append({
                "content": idea['content'][:100],
                "intent": idea['intent']
            })

    # Convert sets to lists for JSON
    for pk in period_activity:
        period_activity[pk]["sessions"] = list(period_activity[pk]["sessions"])

    return {
        "topic": dict(topic),
        "period": period,
        "days": days,
        "total_ideas": len(ideas),
        "by_period": period_activity
    }


# Bad topic name patterns (matched case-insensitively)
_BAD_NAME_PATTERNS = [
    r"^this session is being continued",
    r"^session\s*(start|begin|init)",
    r"^let'?s\s",
    r"now let'?s\s",  # Anywhere in string
    r"^okay,?\s*(let'?s|brilliant|so)",
    r"^alright,?\s*let'?s",
    r"^got it",  # Acknowledgement
    r"^seems to",  # Observation
    r"^working with",  # Action in progress
    r"^looking at",  # Action in progress
    r"^\*\*",  # Starts with markdown bold
    r"^#+\s",  # Starts with markdown heading
    r"\.\.\.$",  # Ends with ellipsis (truncated)
]

# Case-sensitive patterns (matched against original name)
_BAD_NAME_PATTERNS_CASE_SENSITIVE = [
    r"^[a-z]",  # Starts with lowercase (message fragment, not a title)
    r"^.{80,}",  # Very long (message fragment)
]


def review_topics() -> dict:
    """Review topics for quality issues.

    Returns:
        Dict with issues categorized by type
    """
    import re

    db = get_db()
    issues = {
        "catch_all": [],      # Topics with too many ideas
        "bad_names": [],      # Poorly named topics
        "duplicates": [],     # Potential duplicate topics
        "empty": [],          # Topics with no ideas
    }

    # Get all topics with counts
    cursor = db.execute("""
        SELECT t.id, t.name, t.summary, t.canonical_name,
               COUNT(DISTINCT s.id) as span_count,
               COUNT(DISTINCT i.id) as idea_count
        FROM topics t
        LEFT JOIN spans s ON s.topic_id = t.id
        LEFT JOIN ideas i ON i.span_id = s.id
        GROUP BY t.id
    """)
    topics = [dict(row) for row in cursor]
    db.close()

    # Check each topic
    for topic in topics:
        topic_id = topic["id"]
        name = topic["name"]
        idea_count = topic["idea_count"]

        # Catch-all detection (>100 ideas or >20% of total)
        total_ideas = sum(t["idea_count"] for t in topics)
        if idea_count > 100 or (total_ideas > 0 and idea_count / total_ideas > 0.2):
            issues["catch_all"].append({
                "topic_id": topic_id,
                "name": name,
                "idea_count": idea_count,
                "percentage": round(100 * idea_count / total_ideas, 1) if total_ideas > 0 else 0,
                "suggestion": "Split into more specific topics"
            })

        # Bad name detection
        name_lower = name.lower()
        bad_reason = None
        # Check case-insensitive patterns
        for pattern in _BAD_NAME_PATTERNS:
            if re.search(pattern, name_lower):
                bad_reason = f"Matches pattern: {pattern}"
                break
        # Check case-sensitive patterns against original name
        if not bad_reason:
            for pattern in _BAD_NAME_PATTERNS_CASE_SENSITIVE:
                if re.search(pattern, name):
                    bad_reason = f"Matches pattern: {pattern}"
                    break
        if bad_reason:
            issues["bad_names"].append({
                "topic_id": topic_id,
                "name": name,
                "reason": bad_reason,
                "suggestion": "Rename to describe the actual topic"
            })

        # Empty topics
        if idea_count == 0:
            issues["empty"].append({
                "topic_id": topic_id,
                "name": name,
                "suggestion": "Delete or merge with related topic"
            })

    # Duplicate detection using embedding similarity
    duplicates = find_duplicate_topics(threshold=0.85)
    issues["duplicates"] = duplicates

    # Summary
    issues["summary"] = {
        "total_topics": len(topics),
        "catch_all_count": len(issues["catch_all"]),
        "bad_names_count": len(issues["bad_names"]),
        "duplicate_pairs": len(issues["duplicates"]),
        "empty_count": len(issues["empty"]),
        "has_issues": any([
            issues["catch_all"],
            issues["bad_names"],
            issues["duplicates"],
            issues["empty"]
        ])
    }

    return issues


def find_duplicate_topics(threshold: float = None) -> list[dict]:
    """Find topics that are semantically similar.

    Args:
        threshold: Similarity threshold (0-1, higher = more similar).
                   Uses config.duplicate_topic_threshold if None.

    Returns:
        List of duplicate pairs with similarity scores
    """
    # Use config default if not specified
    if threshold is None:
        from config import get_config
        threshold = get_config().duplicate_topic_threshold

    db = get_db()

    # Get topics with their embeddings (from their name + summary)
    cursor = db.execute("""
        SELECT id, name, summary FROM topics
    """)
    topics = list(cursor)
    db.close()

    if len(topics) < 2:
        return []

    # Get embeddings for each topic
    topic_embeddings = {}
    for topic in topics:
        text = f"{topic['name']}: {topic['summary'] or ''}"
        try:
            embedding = get_embedding(text)
            topic_embeddings[topic["id"]] = {
                "name": topic["name"],
                "embedding": embedding
            }
        except Exception:
            continue

    # Compare all pairs
    duplicates = []
    topic_ids = list(topic_embeddings.keys())

    for i, id1 in enumerate(topic_ids):
        for id2 in topic_ids[i+1:]:
            emb1 = topic_embeddings[id1]["embedding"]
            emb2 = topic_embeddings[id2]["embedding"]

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

            if similarity >= threshold:
                duplicates.append({
                    "topic1_id": id1,
                    "topic1_name": topic_embeddings[id1]["name"],
                    "topic2_id": id2,
                    "topic2_name": topic_embeddings[id2]["name"],
                    "similarity": round(similarity, 3),
                    "suggestion": f"merge-topics {id1} {id2}"
                })

    # Sort by similarity descending
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)
    return duplicates


def cluster_topics(min_cluster_size: int = 5) -> dict:
    """Analyze idea embeddings to suggest topic reorganization.

    Uses agglomerative clustering on idea embeddings to find natural groupings,
    then compares against current topic assignments to suggest:
    - Topics that should be merged (ideas cluster together)
    - Topics that should be split (ideas in multiple clusters)
    - Misplaced ideas (closer to another topic's centroid)

    Args:
        min_cluster_size: Minimum ideas for a cluster to be significant

    Returns:
        Dict with clustering analysis and suggestions
    """
    db = get_db()

    # Get all ideas with embeddings and topic info
    cursor = db.execute("""
        SELECT i.id, i.content, i.span_id, s.topic_id, t.name as topic_name,
               e.embedding
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        JOIN topics t ON t.id = s.topic_id
        JOIN idea_embeddings e ON e.idea_id = i.id
    """)
    ideas = []
    for row in cursor:
        emb_bytes = row["embedding"]
        embedding = list(struct.unpack(f'{EMBEDDING_DIM}f', emb_bytes))
        ideas.append({
            "id": row["id"],
            "content": row["content"][:100],
            "topic_id": row["topic_id"],
            "topic_name": row["topic_name"],
            "embedding": embedding,
        })
    db.close()

    if len(ideas) < 10:
        return {"error": "Not enough ideas for clustering", "idea_count": len(ideas)}

    # Compute topic centroids
    topic_ideas = {}
    for idea in ideas:
        tid = idea["topic_id"]
        if tid not in topic_ideas:
            topic_ideas[tid] = {"name": idea["topic_name"], "embeddings": [], "ideas": []}
        topic_ideas[tid]["embeddings"].append(idea["embedding"])
        topic_ideas[tid]["ideas"].append(idea)

    topic_centroids = {}
    for tid, data in topic_ideas.items():
        embs = data["embeddings"]
        centroid = [sum(e[i] for e in embs) / len(embs) for i in range(EMBEDDING_DIM)]
        # Normalize
        norm = sum(x*x for x in centroid) ** 0.5
        if norm > 0:
            centroid = [x / norm for x in centroid]
        topic_centroids[tid] = {
            "name": data["name"],
            "centroid": centroid,
            "idea_count": len(embs),
        }

    # Find misplaced ideas (closer to another topic's centroid)
    misplaced = []
    for idea in ideas:
        own_tid = idea["topic_id"]
        own_centroid = topic_centroids[own_tid]["centroid"]
        emb = idea["embedding"]

        # Distance to own centroid
        own_dist = 1 - sum(a*b for a, b in zip(emb, own_centroid))

        # Find closest other centroid
        best_other_tid = None
        best_other_dist = float('inf')
        for tid, data in topic_centroids.items():
            if tid == own_tid:
                continue
            dist = 1 - sum(a*b for a, b in zip(emb, data["centroid"]))
            if dist < best_other_dist:
                best_other_dist = dist
                best_other_tid = tid

        # If significantly closer to another topic (>20% closer)
        if best_other_tid and best_other_dist < own_dist * 0.8:
            misplaced.append({
                "idea_id": idea["id"],
                "content": idea["content"],
                "current_topic_id": own_tid,
                "current_topic": topic_centroids[own_tid]["name"],
                "suggested_topic_id": best_other_tid,
                "suggested_topic": topic_centroids[best_other_tid]["name"],
                "distance_improvement": round((own_dist - best_other_dist) / own_dist * 100, 1),
            })

    # Sort by improvement
    misplaced.sort(key=lambda x: x["distance_improvement"], reverse=True)

    # Compute internal coherence for each topic (average distance to centroid)
    topic_coherence = {}
    for tid, data in topic_ideas.items():
        centroid = topic_centroids[tid]["centroid"]
        distances = []
        for emb in data["embeddings"]:
            dist = 1 - sum(a*b for a, b in zip(emb, centroid))
            distances.append(dist)
        avg_dist = sum(distances) / len(distances) if distances else 0
        max_dist = max(distances) if distances else 0
        topic_coherence[tid] = {
            "name": data["name"],
            "idea_count": len(distances),
            "avg_distance": round(avg_dist, 4),
            "max_distance": round(max_dist, 4),
            "coherence_score": round(1 - avg_dist, 3),  # Higher = more coherent
        }

    # Find topics that might need splitting (low coherence, high count)
    split_candidates = []
    for tid, coh in topic_coherence.items():
        if coh["idea_count"] >= min_cluster_size and coh["coherence_score"] < 0.7:
            split_candidates.append({
                "topic_id": tid,
                "name": coh["name"],
                "idea_count": coh["idea_count"],
                "coherence_score": coh["coherence_score"],
                "suggestion": "Topic has diverse content - consider splitting",
            })
    split_candidates.sort(key=lambda x: x["coherence_score"])

    # Find topics that might merge (centroids very close)
    merge_candidates = []
    topic_ids = list(topic_centroids.keys())
    for i, tid1 in enumerate(topic_ids):
        for tid2 in topic_ids[i+1:]:
            c1 = topic_centroids[tid1]["centroid"]
            c2 = topic_centroids[tid2]["centroid"]
            similarity = sum(a*b for a, b in zip(c1, c2))
            if similarity > 0.85:  # Very similar centroids
                merge_candidates.append({
                    "topic1_id": tid1,
                    "topic1_name": topic_centroids[tid1]["name"],
                    "topic1_count": topic_centroids[tid1]["idea_count"],
                    "topic2_id": tid2,
                    "topic2_name": topic_centroids[tid2]["name"],
                    "topic2_count": topic_centroids[tid2]["idea_count"],
                    "similarity": round(similarity, 3),
                    "suggestion": f"merge-topics {tid1} {tid2}",
                })
    merge_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "total_ideas": len(ideas),
        "total_topics": len(topic_centroids),
        "topic_coherence": sorted(topic_coherence.values(), key=lambda x: x["coherence_score"]),
        "misplaced_ideas": misplaced[:30],  # Top 30
        "split_candidates": split_candidates,
        "merge_candidates": merge_candidates,
        "summary": {
            "misplaced_count": len(misplaced),
            "low_coherence_topics": len(split_candidates),
            "merge_pairs": len(merge_candidates),
        }
    }


def recluster_topic(topic_id: int, num_clusters: int = None) -> dict:
    """Analyze a single topic and suggest how to split it.

    Uses k-means style clustering on the topic's ideas to find natural sub-groups.

    Args:
        topic_id: Topic to analyze
        num_clusters: Number of clusters (auto-detected if None)

    Returns:
        Dict with cluster analysis and suggested splits
    """
    db = get_db()

    # Get topic info
    cursor = db.execute("SELECT name, summary FROM topics WHERE id = ?", (topic_id,))
    topic_row = cursor.fetchone()
    if not topic_row:
        db.close()
        return {"error": f"Topic {topic_id} not found"}

    # Get ideas with embeddings
    cursor = db.execute("""
        SELECT i.id, i.content, i.intent, e.embedding
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        JOIN idea_embeddings e ON e.idea_id = i.id
        WHERE s.topic_id = ?
    """, (topic_id,))

    ideas = []
    for row in cursor:
        emb_bytes = row["embedding"]
        embedding = list(struct.unpack(f'{EMBEDDING_DIM}f', emb_bytes))
        ideas.append({
            "id": row["id"],
            "content": row["content"],
            "intent": row["intent"],
            "embedding": embedding,
        })
    db.close()

    if len(ideas) < 4:
        return {"error": "Not enough ideas to cluster", "idea_count": len(ideas)}

    # Auto-detect number of clusters if not specified
    if num_clusters is None:
        # Heuristic: sqrt(n) clusters, min 2, max 8
        num_clusters = max(2, min(8, int(len(ideas) ** 0.5)))

    # Simple k-means clustering
    import random
    random.seed(42)

    # Initialize centroids randomly
    centroid_indices = random.sample(range(len(ideas)), min(num_clusters, len(ideas)))
    centroids = [ideas[i]["embedding"][:] for i in centroid_indices]

    # Iterate k-means
    for _ in range(20):
        # Assign ideas to nearest centroid
        assignments = []
        for idea in ideas:
            emb = idea["embedding"]
            best_cluster = 0
            best_dist = float('inf')
            for ci, centroid in enumerate(centroids):
                dist = sum((a-b)**2 for a, b in zip(emb, centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = ci
            assignments.append(best_cluster)

        # Update centroids
        new_centroids = []
        for ci in range(len(centroids)):
            cluster_embs = [ideas[i]["embedding"] for i, a in enumerate(assignments) if a == ci]
            if cluster_embs:
                new_centroid = [sum(e[d] for e in cluster_embs) / len(cluster_embs)
                               for d in range(EMBEDDING_DIM)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[ci])
        centroids = new_centroids

    # Build cluster info
    clusters = {i: [] for i in range(len(centroids))}
    for i, cluster_id in enumerate(assignments):
        clusters[cluster_id].append(ideas[i])

    # Use Claude CLI to suggest names for each cluster
    cluster_info = []

    for cluster_id, cluster_ideas in clusters.items():
        if not cluster_ideas:
            continue

        # Sample content for naming
        sample_content = "\n".join(f"- {idea['content'][:150]}" for idea in cluster_ideas[:10])

        try:
            prompt = f"""These ideas were clustered together. Suggest a concise topic name (3-6 words) that captures their common theme:

{sample_content}

Reply with ONLY the topic name, nothing else."""
            suggested_name = claude_complete(prompt).strip()
        except MemgraphError:
            raise  # Re-raise MemgraphError as-is
        except Exception as e:
            logger.error(f"Failed to generate cluster name: {e}")
            raise MemgraphError(
                f"Failed to generate cluster name: {e}",
                "cluster_naming_error",
                {"cluster_id": cluster_id, "original_error": str(e)}
            ) from e

        cluster_info.append({
            "cluster_id": cluster_id,
            "suggested_name": suggested_name,
            "idea_count": len(cluster_ideas),
            "sample_ideas": [{"id": i["id"], "content": i["content"][:100]} for i in cluster_ideas[:5]],
        })

    # Sort by size
    cluster_info.sort(key=lambda x: x["idea_count"], reverse=True)

    return {
        "topic_id": topic_id,
        "topic_name": topic_row["name"],
        "total_ideas": len(ideas),
        "num_clusters": len([c for c in cluster_info if c["idea_count"] > 0]),
        "clusters": cluster_info,
    }


def split_topic(topic_id: int, num_clusters: int = None, min_cluster_size: int = 3,
                delete_original: bool = False, delete_junk: bool = True) -> dict:
    """Split a topic into sub-topics based on clustering.

    Creates child topics from natural clusters in the idea embeddings.

    Args:
        topic_id: Topic to split
        num_clusters: Number of clusters (auto-detected if None)
        min_cluster_size: Minimum ideas for a cluster to become a topic
        delete_original: If True, delete original topic after split
        delete_junk: If True, delete ideas in tiny clusters (<min_cluster_size)

    Returns:
        Dict with split results
    """
    # First run clustering
    cluster_result = recluster_topic(topic_id, num_clusters)
    if "error" in cluster_result:
        return cluster_result

    db = get_db()

    # Get original topic info
    cursor = db.execute(
        "SELECT id, name, project_id FROM topics WHERE id = ?",
        (topic_id,)
    )
    original = cursor.fetchone()
    if not original:
        db.close()
        return {"error": f"Topic {topic_id} not found"}

    project_id = original["project_id"]
    original_name = original["name"]

    created_topics = []
    moved_ideas = 0
    deleted_ideas = 0
    junk_ideas = []

    for cluster in cluster_result["clusters"]:
        if cluster["idea_count"] == 0:
            continue

        idea_ids = [i["id"] for i in cluster["sample_ideas"]]
        # Get ALL idea IDs for this cluster (sample_ideas only has 5)
        # We need to re-run clustering to get full assignments
        # For now, use the sample - we'll fix this properly

        if cluster["idea_count"] < min_cluster_size:
            # Too small - mark as junk
            junk_ideas.extend(idea_ids)
            continue

        # Create new child topic
        new_name = cluster["suggested_name"] or f"{original_name} - Cluster {cluster['cluster_id']+1}"
        canonical = canonicalize_topic_name(new_name)

        cursor = db.execute("""
            INSERT INTO topics (name, canonical_name, parent_id, project_id)
            VALUES (?, ?, ?, ?)
        """, (new_name, canonical, topic_id, project_id))
        new_topic_id = cursor.lastrowid

        created_topics.append({
            "id": new_topic_id,
            "name": new_name,
            "idea_count": cluster["idea_count"],
        })

    db.commit()
    db.close()

    # Now we need to actually move the ideas - re-run clustering with full assignments
    result = _execute_topic_split(topic_id, created_topics, min_cluster_size, delete_junk)

    return {
        "original_topic_id": topic_id,
        "original_name": original_name,
        "created_topics": result["created_topics"],
        "moved_ideas": result["moved_ideas"],
        "deleted_ideas": result["deleted_ideas"],
        "kept_in_original": result["kept_in_original"],
    }


def _execute_topic_split(topic_id: int, created_topics: list, min_cluster_size: int,
                         delete_junk: bool) -> dict:
    """Execute the actual idea movement for a topic split."""
    import random
    random.seed(42)

    db = get_db()

    # Get all ideas with embeddings for this topic
    cursor = db.execute("""
        SELECT i.id, i.span_id, e.embedding
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        JOIN idea_embeddings e ON e.idea_id = i.id
        WHERE s.topic_id = ?
    """, (topic_id,))

    ideas = []
    for row in cursor:
        emb_bytes = row["embedding"]
        embedding = list(struct.unpack(f'{EMBEDDING_DIM}f', emb_bytes))
        ideas.append({
            "id": row["id"],
            "span_id": row["span_id"],
            "embedding": embedding,
        })

    if not ideas or not created_topics:
        db.close()
        return {"created_topics": created_topics, "moved_ideas": 0, "deleted_ideas": 0, "kept_in_original": len(ideas)}

    num_clusters = len(created_topics)

    # K-means clustering
    centroid_indices = random.sample(range(len(ideas)), min(num_clusters, len(ideas)))
    centroids = [ideas[i]["embedding"][:] for i in centroid_indices]

    for _ in range(20):
        assignments = []
        for idea in ideas:
            emb = idea["embedding"]
            best_cluster = 0
            best_dist = float('inf')
            for ci, centroid in enumerate(centroids):
                dist = sum((a-b)**2 for a, b in zip(emb, centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = ci
            assignments.append(best_cluster)

        new_centroids = []
        for ci in range(len(centroids)):
            cluster_embs = [ideas[i]["embedding"] for i, a in enumerate(assignments) if a == ci]
            if cluster_embs:
                new_centroid = [sum(e[d] for e in cluster_embs) / len(cluster_embs)
                               for d in range(EMBEDDING_DIM)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[ci])
        centroids = new_centroids

    # Group ideas by cluster
    clusters = {i: [] for i in range(num_clusters)}
    for i, cluster_id in enumerate(assignments):
        clusters[cluster_id].append(ideas[i])

    # Match clusters to created topics by size (largest cluster -> first topic)
    cluster_sizes = [(ci, len(ideas_list)) for ci, ideas_list in clusters.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    moved_ideas = 0
    deleted_ideas = 0

    for idx, (cluster_id, size) in enumerate(cluster_sizes):
        if idx >= len(created_topics):
            break

        new_topic = created_topics[idx]
        cluster_ideas = clusters[cluster_id]

        if size < min_cluster_size:
            # Junk cluster
            if delete_junk:
                idea_ids = [i["id"] for i in cluster_ideas]
                if idea_ids:
                    placeholders = ",".join("?" * len(idea_ids))
                    db.execute(f"DELETE FROM idea_embeddings WHERE idea_id IN ({placeholders})", idea_ids)
                    db.execute(f"DELETE FROM relations WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                              idea_ids + idea_ids)
                    db.execute(f"DELETE FROM idea_entities WHERE idea_id IN ({placeholders})", idea_ids)
                    db.execute(f"DELETE FROM ideas_fts WHERE rowid IN ({placeholders})", idea_ids)
                    db.execute(f"DELETE FROM ideas WHERE id IN ({placeholders})", idea_ids)
                    deleted_ideas += len(idea_ids)
            continue

        # Create a new span for this topic
        # Get a representative span from the ideas
        first_idea = cluster_ideas[0]
        cursor = db.execute("SELECT session, source_file FROM spans s JOIN ideas i ON i.span_id = s.id WHERE i.id = ?",
                           (first_idea["id"],))
        span_info = cursor.fetchone()

        cursor = db.execute("""
            INSERT INTO spans (session, topic_id, name, start_line, end_line, depth)
            VALUES (?, ?, ?, 0, 0, 0)
        """, (span_info["session"] if span_info else "unknown", new_topic["id"], new_topic["name"]))
        new_span_id = cursor.lastrowid

        # Move ideas to new span
        idea_ids = [i["id"] for i in cluster_ideas]
        placeholders = ",".join("?" * len(idea_ids))
        db.execute(f"UPDATE ideas SET span_id = ? WHERE id IN ({placeholders})",
                  [new_span_id] + idea_ids)
        moved_ideas += len(idea_ids)
        new_topic["idea_count"] = len(idea_ids)

    db.commit()
    db.close()

    return {
        "created_topics": created_topics,
        "moved_ideas": moved_ideas,
        "deleted_ideas": deleted_ideas,
        "kept_in_original": len(ideas) - moved_ideas - deleted_ideas,
    }


def rename_topic(topic_id: int, new_name: str) -> bool:
    """Rename a topic.

    Args:
        topic_id: Topic to rename
        new_name: New name

    Returns:
        True if renamed, False if topic not found
    """
    db = get_db()
    cursor = db.execute(
        "UPDATE topics SET name = ?, canonical_name = ? WHERE id = ?",
        (new_name, canonicalize_topic_name(new_name), topic_id)
    )
    db.commit()
    updated = cursor.rowcount > 0
    db.close()
    return updated


def suggest_topic_name(topic_id: int) -> Optional[str]:
    """Use LLM to suggest a better name for a topic.

    Args:
        topic_id: Topic to rename

    Returns:
        Suggested name or None if failed
    """
    db = get_db()

    # Get topic and sample ideas
    cursor = db.execute("""
        SELECT t.name, t.summary FROM topics t WHERE t.id = ?
    """, (topic_id,))
    topic = cursor.fetchone()

    if not topic:
        db.close()
        return None

    # Get sample ideas from this topic
    cursor = db.execute("""
        SELECT i.content FROM ideas i
        JOIN spans s ON s.id = i.span_id
        WHERE s.topic_id = ?
        ORDER BY i.created_at
        LIMIT 10
    """, (topic_id,))
    ideas = [row["content"] for row in cursor]
    db.close()

    if not ideas:
        return None

    # Use Claude CLI to generate name
    try:
        prompt = f"""Based on these conversation excerpts, suggest a concise topic name (2-5 words):

Current name: {topic['name']}
Summary: {topic['summary'] or 'None'}

Sample content:
{chr(10).join(f'- {idea[:200]}' for idea in ideas[:5])}

Reply with ONLY the suggested topic name, nothing else."""

        name = claude_complete(prompt).strip()
        # Clean the output - strip markdown and quotes
        import re
        name = re.sub(r'^\*\*(.+)\*\*$', r'\1', name)  # Remove bold
        name = re.sub(r'^\*(.+)\*$', r'\1', name)  # Remove italic
        name = name.strip('"\'')  # Remove quotes
        return name
    except MemgraphError:
        raise  # Re-raise MemgraphError as-is
    except Exception as e:
        logger.error(f"Failed to suggest topic name: {e}")
        raise MemgraphError(
            f"Failed to suggest topic name: {e}",
            "topic_name_suggestion_error",
            {"topic_id": topic_id, "original_error": str(e)}
        ) from e


# =============================================================================
# Span Operations
# =============================================================================

def create_span(
    session: str,
    name: str,
    start_line: int,
    parent_id: Optional[int] = None,
    depth: int = 0,
    topic_id: Optional[int] = None,
    start_time: Optional[str] = None
) -> int:
    """Create a new span, linking to existing or new topic.

    Args:
        session: Session identifier
        name: Span name (used to find/create topic if topic_id not provided)
        start_line: Starting line number
        parent_id: Parent span ID for hierarchy
        depth: Nesting depth
        topic_id: Explicit topic ID (if None, will find/create from name)
        start_time: ISO timestamp when span began (from first message)

    Returns:
        Span ID
    """
    # Find or create topic if not provided
    if topic_id is None:
        topic_id = find_or_create_topic(name)

    db = get_db()
    cursor = db.execute("""
        INSERT INTO spans (topic_id, session, parent_id, name, start_line, depth, start_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (topic_id, session, parent_id, name, start_line, depth, start_time))
    span_id = cursor.lastrowid

    # Update topic first_seen if this is earlier
    if start_time and topic_id:
        db.execute("""
            UPDATE topics
            SET first_seen = ?
            WHERE id = ? AND (first_seen IS NULL OR first_seen > ?)
        """, (start_time, topic_id, start_time))

    db.commit()
    db.close()
    return span_id


def close_span(span_id: int, end_line: int, summary: str, end_time: Optional[str] = None):
    """Close a span with summary and embed it.

    Args:
        span_id: Span to close
        end_line: Ending line number
        summary: Summary of the span content
        end_time: ISO timestamp when span ended (from last message)
    """
    db = get_db()
    topic_id = None
    session = None

    try:
        # Update span with end_line, summary, and end_time
        db.execute("""
            UPDATE spans SET end_line = ?, summary = ?, end_time = ? WHERE id = ?
        """, (end_line, summary, end_time, span_id))

        # Get span for embedding and topic update
        cursor = db.execute("SELECT name, summary, topic_id, session FROM spans WHERE id = ?", (span_id,))
        row = cursor.fetchone()
        topic_id = row['topic_id']
        session = row['session']

        # Update topic last_seen if this is later
        if end_time and topic_id:
            db.execute("""
                UPDATE topics
                SET last_seen = ?
                WHERE id = ? AND (last_seen IS NULL OR last_seen < ?)
            """, (end_time, topic_id, end_time))

        # Embed and store - delete first for sqlite-vec compatibility
        embed_text = f"{row['name']}: {row['summary']}"
        embedding = get_embedding(embed_text)
        db.execute("DELETE FROM span_embeddings WHERE span_id = ?", (span_id,))
        db.execute("""
            INSERT INTO span_embeddings (span_id, embedding)
            VALUES (?, ?)
        """, (span_id, serialize_embedding(embedding)))

        # Update FTS - use INSERT OR REPLACE for safety
        db.execute("""
            INSERT OR REPLACE INTO spans_fts (rowid, name, summary)
            VALUES (?, ?, ?)
        """, (span_id, row['name'], row['summary']))

        db.commit()
    finally:
        db.close()

    # Auto-link topic to similar topics in other sessions (after db is closed)
    if topic_id:
        try:
            links = auto_link_topic(topic_id, session, min_similarity=0.8)
            if links:
                logger.info(f"Auto-linked topic {topic_id} to {len(links)} related topics")
        except Exception as e:
            logger.warning(f"Auto-link failed for topic {topic_id}: {e}")


def update_span_embedding(span_id: int, include_ideas: bool = True) -> bool:
    """Update span embedding incrementally.

    Can be called during indexing to keep embeddings fresh without waiting
    for span close. Uses span name + summary (if available) + sample of ideas.

    Args:
        span_id: Span to update
        include_ideas: Whether to include idea content in embedding

    Returns:
        True if embedding was updated
    """
    db = get_db()
    span = None

    try:
        # Get span info
        cursor = db.execute(
            "SELECT name, summary, topic_id, session FROM spans WHERE id = ?",
            (span_id,)
        )
        span = cursor.fetchone()
        if not span:
            return False

        # Build embedding text from span name + summary + sample ideas
        parts = [span['name']]
        if span['summary']:
            parts.append(span['summary'])

        if include_ideas:
            # Get a sample of ideas (first few + most recent, exclude forgotten)
            cursor = db.execute("""
                SELECT content FROM ideas
                WHERE span_id = ?
                    AND (forgotten = FALSE OR forgotten IS NULL)
                ORDER BY id ASC
                LIMIT 3
            """, (span_id,))
            first_ideas = [r['content'][:200] for r in cursor]

            cursor = db.execute("""
                SELECT content FROM ideas
                WHERE span_id = ?
                    AND (forgotten = FALSE OR forgotten IS NULL)
                ORDER BY id DESC
                LIMIT 3
            """, (span_id,))
            recent_ideas = [r['content'][:200] for r in cursor]

            # Combine unique ideas
            all_ideas = first_ideas + [i for i in recent_ideas if i not in first_ideas]
            if all_ideas:
                parts.append("Key points: " + "; ".join(all_ideas[:5]))

        embed_text = " | ".join(parts)[:2000]  # Limit length
        embedding = get_embedding(embed_text)

        # Delete first for sqlite-vec compatibility
        db.execute("DELETE FROM span_embeddings WHERE span_id = ?", (span_id,))
        db.execute("""
            INSERT INTO span_embeddings (span_id, embedding)
            VALUES (?, ?)
        """, (span_id, serialize_embedding(embedding)))
        db.commit()
    finally:
        db.close()

    # Try to auto-link if we have a topic (after db is closed)
    if span and span['topic_id']:
        try:
            links = auto_link_topic(span['topic_id'], span['session'], min_similarity=0.8)
            if links:
                logger.info(f"Auto-linked topic {span['topic_id']} to {len(links)} related topics")
        except Exception as e:
            logger.debug(f"Auto-link check: {e}")

    return True


def get_open_span(session: str) -> Optional[dict]:
    """Get the current open span for a session."""
    db = get_db()
    cursor = db.execute("""
        SELECT * FROM spans
        WHERE session = ? AND end_line IS NULL
        ORDER BY depth DESC, id DESC
        LIMIT 1
    """, (session,))
    row = cursor.fetchone()
    db.close()
    return dict(row) if row else None


# =============================================================================
# Semantic Topic Shift Detection
# =============================================================================

def check_topic_similarity(span_id: int, message: str) -> Optional[float]:
    """Check how similar a message is to the current span's topic.

    Args:
        span_id: Current span ID
        message: New message content

    Returns:
        Similarity score (0-1), or None if span has no embedding
    """
    db = get_db()

    # Get span embedding
    cursor = db.execute(
        "SELECT embedding FROM span_embeddings WHERE span_id = ?",
        (span_id,)
    )
    row = cursor.fetchone()
    db.close()

    if not row:
        return None

    span_embedding = deserialize_embedding(row['embedding'])
    message_embedding = get_embedding(message[:1000])  # Limit message length

    # Compute cosine similarity directly for accuracy
    import math
    dot_product = sum(a * b for a, b in zip(span_embedding, message_embedding))
    norm_a = math.sqrt(sum(a * a for a in span_embedding))
    norm_b = math.sqrt(sum(b * b for b in message_embedding))

    if norm_a == 0 or norm_b == 0:
        return None

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def detect_semantic_topic_shift(
    span_id: int,
    message: str,
    threshold: float = None,  # Uses config.topic_shift_threshold if None
    divergence_history: list[float] = None
) -> tuple[bool, float, list[float]]:
    """Detect if a message represents a topic shift using semantic similarity.

    Uses hysteresis: requires multiple consecutive divergent messages to
    trigger a shift, avoiding false positives from brief tangents.

    Args:
        span_id: Current span ID
        message: New message content
        threshold: Similarity threshold (below = divergent)
        divergence_history: List of recent similarity scores (modified in place)

    Returns:
        Tuple of (is_shift, similarity, updated_history)
    """
    if divergence_history is None:
        divergence_history = []

    # Use config default if threshold not specified
    if threshold is None:
        from config import get_config
        threshold = get_config().topic_shift_threshold

    similarity = check_topic_similarity(span_id, message)

    if similarity is None:
        # No embedding yet, can't detect shift semantically
        return False, 0.0, divergence_history

    # Track recent similarities
    divergence_history.append(similarity)

    # Keep only last 3 messages for hysteresis
    if len(divergence_history) > 3:
        divergence_history.pop(0)

    # Require 2+ consecutive divergent messages for a shift
    # This avoids triggering on brief tangents or clarifying questions
    divergent_count = sum(1 for s in divergence_history if s < threshold)

    # Strong shift: very low similarity on current message
    strong_shift = similarity < (threshold - 0.15)

    # Gradual shift: multiple consecutive divergent messages
    gradual_shift = divergent_count >= 2 and similarity < threshold

    is_shift = strong_shift or gradual_shift

    return is_shift, similarity, divergence_history


# =============================================================================
# Idea Operations
# =============================================================================

def store_idea(
    content: str,
    source_file: str,
    source_line: int,
    span_id: Optional[int] = None,
    intent: Optional[str] = None,
    confidence: float = None,  # Uses config.default_confidence if None
    entities: Optional[list[tuple[str, str]]] = None,  # [(name, type), ...]
    message_time: Optional[str] = None  # ISO timestamp from transcript
) -> int:
    """Store an idea with its embedding.

    Args:
        content: The idea content
        source_file: Path to transcript file
        source_line: Line number in transcript
        span_id: Optional span this idea belongs to
        intent: Type of idea (decision, question, etc.)
        confidence: Confidence score 0-1 (uses config default if None)
        entities: Optional list of (name, type) entity tuples
        message_time: ISO timestamp when message occurred in conversation
    """
    # Use config default if not specified
    if confidence is None:
        from config import get_config
        confidence = get_config().default_confidence

    db = get_db()
    cursor = db.cursor()

    # Insert idea with message_time
    cursor.execute("""
        INSERT INTO ideas (span_id, content, intent, confidence, source_file, source_line, message_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (span_id, content, intent, confidence, source_file, source_line, message_time))
    idea_id = cursor.lastrowid

    # Get and store embedding
    embedding = get_embedding(content)
    cursor.execute("""
        INSERT INTO idea_embeddings (idea_id, embedding)
        VALUES (?, ?)
    """, (idea_id, serialize_embedding(embedding)))

    # Update FTS
    cursor.execute("""
        INSERT INTO ideas_fts (rowid, content)
        VALUES (?, ?)
    """, (idea_id, content))

    # Store entities
    if entities:
        for name, etype in entities:
            # Get or create entity
            cursor.execute("""
                INSERT OR IGNORE INTO entities (name, type) VALUES (?, ?)
            """, (name, etype))
            cursor.execute("""
                SELECT id FROM entities WHERE name = ? AND type = ?
            """, (name, etype))
            entity_id = cursor.fetchone()[0]

            # Link to idea
            cursor.execute("""
                INSERT OR IGNORE INTO idea_entities (idea_id, entity_id)
                VALUES (?, ?)
            """, (idea_id, entity_id))

    db.commit()
    db.close()
    return idea_id


def add_relation(from_id: int, to_id: int, relation_type: str):
    """Add a relation between ideas."""
    db = get_db()
    db.execute("""
        INSERT OR IGNORE INTO relations (from_id, to_id, relation_type)
        VALUES (?, ?, ?)
    """, (from_id, to_id, relation_type))
    db.commit()
    db.close()


def mark_question_answered(idea_id: int):
    """Mark a question idea as answered.

    Args:
        idea_id: ID of the question to mark
    """
    db = get_db()
    db.execute("""
        UPDATE ideas SET answered = 1 WHERE id = ? AND intent = 'question'
    """, (idea_id,))
    db.commit()
    db.close()


def get_unanswered_questions(session: Optional[str] = None) -> list[dict]:
    """Get list of unanswered questions.

    Args:
        session: Optional session filter

    Returns:
        List of question idea dicts
    """
    db = get_db()

    sql = """
        SELECT i.id, i.content, i.source_file, i.source_line, i.created_at,
               s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.intent = 'question' AND (i.answered IS NULL OR i.answered = 0)
    """
    params = []

    if session:
        sql += " AND s.session = ?"
        params.append(session)

    sql += " ORDER BY i.created_at DESC"

    cursor = db.execute(sql, params)
    results = [dict(row) for row in cursor]
    db.close()

    return results


def search_with_topic_expansion(
    query: str,
    limit: int = 10,
    session: str = None,
    expand_limit: int = 5
) -> dict:
    """Search with automatic expansion to linked topics in other sessions.

    This search first finds relevant ideas, then identifies their topics
    and follows topic links to find related ideas in other sessions.

    Args:
        query: Search query
        limit: Maximum results per session
        session: Primary session to search (None = all)
        expand_limit: Max linked topics to expand to

    Returns:
        Dict with:
            - primary_results: Ideas from primary session
            - linked_results: Ideas from linked topics (grouped by session)
            - topic_links: The topic links that were followed
    """
    # First do regular search
    if session:
        primary_results = search_ideas(query, limit=limit, session=session)
    else:
        primary_results = search_ideas(query, limit=limit)

    if not primary_results:
        return {
            "primary_results": [],
            "linked_results": {},
            "topic_links": []
        }

    # Find topics from results
    topic_ids = set()
    for r in primary_results:
        if r.get('span_id'):
            # Get topic for this span
            db = get_db()
            cursor = db.execute(
                "SELECT topic_id FROM spans WHERE id = ?",
                (r['span_id'],)
            )
            row = cursor.fetchone()
            db.close()
            if row and row['topic_id']:
                topic_ids.add(row['topic_id'])

    if not topic_ids:
        return {
            "primary_results": primary_results,
            "linked_results": {},
            "topic_links": []
        }

    # Get linked topics
    all_links = []
    linked_topic_ids = set()
    for topic_id in topic_ids:
        links = get_topic_links(topic_id)
        for link in links:
            if link['other_topic_id'] not in topic_ids:
                all_links.append(link)
                linked_topic_ids.add(link['other_topic_id'])
                if len(linked_topic_ids) >= expand_limit:
                    break
        if len(linked_topic_ids) >= expand_limit:
            break

    if not linked_topic_ids:
        return {
            "primary_results": primary_results,
            "linked_results": {},
            "topic_links": []
        }

    # Search within linked topics
    db = get_db()
    query_embedding = get_embedding(query)

    # Get ideas from linked topics
    placeholders = ','.join('?' * len(linked_topic_ids))
    cursor = db.execute(f"""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            i.message_time,
            s.session, s.name as topic,
            s.topic_id,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
          AND s.topic_id IN ({placeholders})
        ORDER BY e.distance
        LIMIT ?
    """, [serialize_embedding(query_embedding), limit * 2] + list(linked_topic_ids) + [limit])

    linked_results_raw = [dict(row) for row in cursor]
    db.close()

    # Group by session
    linked_results = {}
    for r in linked_results_raw:
        sess = r.get('session', 'unknown')
        if sess not in linked_results:
            linked_results[sess] = []
        linked_results[sess].append(r)

    return {
        "primary_results": primary_results,
        "linked_results": linked_results,
        "topic_links": all_links
    }


def expand_with_relations(idea_ids: list[int]) -> list[int]:
    """Expand a set of idea IDs by following relations.

    Args:
        idea_ids: Initial set of idea IDs

    Returns:
        Expanded set including related ideas
    """
    if not idea_ids:
        return []

    db = get_db()
    expanded = set(idea_ids)

    # Follow relations in both directions
    placeholders = ','.join('?' * len(idea_ids))

    # Ideas that the input ideas relate to
    cursor = db.execute(f"""
        SELECT to_id FROM relations WHERE from_id IN ({placeholders})
    """, idea_ids)
    for row in cursor:
        expanded.add(row['to_id'])

    # Ideas that relate to the input ideas
    cursor = db.execute(f"""
        SELECT from_id FROM relations WHERE to_id IN ({placeholders})
    """, idea_ids)
    for row in cursor:
        expanded.add(row['from_id'])

    db.close()
    return list(expanded)


def get_idea_context(idea_id: int) -> dict:
    """Get context for an idea including its span.

    Args:
        idea_id: ID of the idea

    Returns:
        Dict with idea and span context
    """
    db = get_db()
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.id as span_id, s.name as span_name, s.summary as span_summary,
            s.session
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.id = ?
    """, (idea_id,))
    row = cursor.fetchone()
    db.close()

    if row:
        return dict(row)
    return {}


def get_idea_with_relations(idea_id: int) -> dict:
    """Get an idea with all its relations.

    Args:
        idea_id: ID of the idea

    Returns:
        Dict with idea content and lists of related ideas by type
    """
    db = get_db()

    # Get the idea itself
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.id = ?
    """, (idea_id,))
    row = cursor.fetchone()

    if not row:
        db.close()
        return {}

    result = dict(row)
    result["relations"] = {}

    # Get outgoing relations (this idea -> others)
    cursor = db.execute("""
        SELECT r.relation_type, r.to_id, i.content
        FROM relations r
        JOIN ideas i ON i.id = r.to_id
        WHERE r.from_id = ?
    """, (idea_id,))
    for rel_row in cursor:
        rel_type = rel_row["relation_type"]
        if rel_type not in result["relations"]:
            result["relations"][rel_type] = []
        result["relations"][rel_type].append({
            "id": rel_row["to_id"],
            "content": rel_row["content"],
            "direction": "outgoing"
        })

    # Get incoming relations (others -> this idea)
    cursor = db.execute("""
        SELECT r.relation_type, r.from_id, i.content
        FROM relations r
        JOIN ideas i ON i.id = r.from_id
        WHERE r.to_id = ?
    """, (idea_id,))
    for rel_row in cursor:
        rel_type = rel_row["relation_type"]
        if rel_type not in result["relations"]:
            result["relations"][rel_type] = []
        result["relations"][rel_type].append({
            "id": rel_row["from_id"],
            "content": rel_row["content"],
            "direction": "incoming"
        })

    db.close()
    return result


def search_ideas_temporal(
    query: str,
    limit: int = 10,
    since: Optional[str] = None,
    until: Optional[str] = None,
    relative: Optional[str] = None,
    session: Optional[str] = None,
    include_forgotten: bool = False
) -> list[dict]:
    """Search ideas with temporal filtering.

    Args:
        query: Search query
        limit: Maximum results
        since: ISO datetime string for start of range
        until: ISO datetime string for end of range
        relative: Duration string like "1d", "1w", "1m" (overrides since/until)
        session: Filter to specific session
        include_forgotten: If True, include forgotten ideas

    Returns:
        List of matching idea dicts with message_time
    """
    # Resolve relative time if provided
    if relative:
        since, until = resolve_temporal_qualifier(relative)

    db = get_db()
    query_embedding = get_embedding(query)

    # Build query with temporal filter - use message_time with fallback to created_at
    forgotten_filter = "" if include_forgotten else "AND (i.forgotten = FALSE OR i.forgotten IS NULL)"
    sql = f"""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            i.message_time,
            s.session, s.name as topic,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
            {forgotten_filter}
    """
    params = [serialize_embedding(query_embedding), limit * 2]

    if since:
        sql += " AND COALESCE(i.message_time, i.created_at) >= ?"
        params.append(since)
    if until:
        sql += " AND COALESCE(i.message_time, i.created_at) <= ?"
        params.append(until)
    if session:
        sql += " AND s.session = ?"
        params.append(session)

    sql += " ORDER BY e.distance LIMIT ?"
    params.append(limit)

    cursor = db.execute(sql, params)
    results = [dict(row) for row in cursor]
    db.close()
    return results


# Synonym mappings for query expansion
_SYNONYMS = {
    "auth": ["authentication", "login", "signin", "sign-in"],
    "authentication": ["auth", "login", "signin", "credentials"],
    "login": ["auth", "authentication", "signin", "sign-in"],
    "db": ["database", "storage", "persistence"],
    "database": ["db", "storage", "data store", "persistence"],
    "api": ["endpoint", "rest", "interface", "service"],
    "endpoint": ["api", "route", "path"],
    "config": ["configuration", "settings", "options"],
    "configuration": ["config", "settings", "setup"],
    "error": ["exception", "failure", "bug", "issue"],
    "bug": ["error", "issue", "problem", "defect"],
    "test": ["testing", "spec", "unit test", "integration"],
    "cache": ["caching", "memoization", "redis"],
    "perf": ["performance", "speed", "optimization"],
    "performance": ["perf", "speed", "latency", "throughput"],
    "deploy": ["deployment", "release", "ship"],
    "deployment": ["deploy", "release", "production"],
    "ui": ["frontend", "interface", "user interface"],
    "frontend": ["ui", "client", "browser"],
    "backend": ["server", "api", "service"],
}


def expand_query(query: str) -> list[str]:
    """Expand a query with synonyms for better recall.

    Args:
        query: Original search query

    Returns:
        List of expanded query terms (including original)
    """
    words = query.lower().split()
    expanded = set(words)

    for word in words:
        if word in _SYNONYMS:
            expanded.update(_SYNONYMS[word])

    return list(expanded)


def resolve_temporal_qualifier(qualifier: str) -> tuple[str, str]:
    """Convert a temporal qualifier to absolute date range.

    Args:
        qualifier: Duration string like "1d", "1w", "1m", "3m", "1y"
                   or natural language like "last week", "yesterday",
                   "since tuesday", "since jan 5"

    Returns:
        Tuple of (since, until) ISO datetime strings
    """
    from datetime import datetime, timedelta
    import re

    now = datetime.utcnow()
    until = now.isoformat() + "Z"
    qualifier = qualifier.lower().strip()

    # Day names for "since tuesday" etc.
    day_names = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    # Month names
    month_names = {
        "january": 1, "jan": 1, "february": 2, "feb": 2,
        "march": 3, "mar": 3, "april": 4, "apr": 4,
        "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    # Handle "since <day>" (e.g., "since tuesday", "since monday")
    since_day_match = re.match(r"since\s+(\w+)", qualifier)
    if since_day_match:
        day_word = since_day_match.group(1)
        if day_word in day_names:
            target_weekday = day_names[day_word]
            current_weekday = now.weekday()
            days_ago = (current_weekday - target_weekday) % 7
            if days_ago == 0:
                days_ago = 7  # If same day, go back a week
            target_date = now - timedelta(days=days_ago)
            since = target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
            return (since, until)
        # Check if it's a month reference (e.g., "since january")
        elif day_word in month_names:
            target_month = month_names[day_word]
            year = now.year if target_month <= now.month else now.year - 1
            target_date = datetime(year, target_month, 1)
            since = target_date.isoformat() + "Z"
            return (since, until)

    # Handle "since <month> <day>" (e.g., "since jan 5", "since january 5")
    since_date_match = re.match(r"since\s+(\w+)\s+(\d{1,2})", qualifier)
    if since_date_match:
        month_word = since_date_match.group(1)
        day_num = int(since_date_match.group(2))
        if month_word in month_names:
            target_month = month_names[month_word]
            year = now.year
            target_date = datetime(year, target_month, day_num)
            if target_date > now:
                year -= 1
                target_date = datetime(year, target_month, day_num)
            since = target_date.isoformat() + "Z"
            return (since, until)

    # Simple natural language mappings
    if qualifier == "today":
        since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
        return (since, until)
    elif qualifier == "yesterday":
        yesterday = now - timedelta(days=1)
        since = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
        yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
        return (since, yesterday_end)
    elif qualifier in ("last week", "this week"):
        since = (now - timedelta(weeks=1)).isoformat() + "Z"
        return (since, until)
    elif qualifier in ("last month", "this month"):
        since = (now - timedelta(days=30)).isoformat() + "Z"
        return (since, until)
    elif qualifier in ("recently", "recent"):
        since = (now - timedelta(weeks=1)).isoformat() + "Z"
        return (since, until)

    # Parse duration like "1d", "2w", "1m", "1y"
    match = re.match(r"(\d+)([dwmy])", qualifier)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)

        if unit == "d":
            delta = timedelta(days=amount)
        elif unit == "w":
            delta = timedelta(weeks=amount)
        elif unit == "m":
            delta = timedelta(days=amount * 30)  # Approximate
        elif unit == "y":
            delta = timedelta(days=amount * 365)  # Approximate
        else:
            delta = timedelta(weeks=1)  # Default

        since = (now - delta).isoformat() + "Z"
        return (since, until)

    # Default: last week
    since = (now - timedelta(weeks=1)).isoformat() + "Z"
    return (since, until)


def analyze_query(query: str) -> dict:
    """Analyze a query to extract filters and entities.

    Args:
        query: Search query

    Returns:
        Dict with temporal, intent_filter, entities, expanded_terms
    """
    import re
    query_lower = query.lower()
    result = {}

    # Add expanded terms
    result["expanded_terms"] = expand_query(query)

    # Temporal qualifiers
    temporal_patterns = [
        ("last week", "1w"),
        ("yesterday", "1d"),
        ("last month", "1m"),
        ("recently", "1w"),
        ("recent", "1w"),
        ("today", "1d"),
        ("this week", "1w"),
    ]
    for pattern, duration in temporal_patterns:
        if pattern in query_lower:
            result["temporal"] = duration
            break

    # Intent filters
    intent_patterns = [
        (r"\bdecisions?\b", "decision"),
        (r"\bproblems?\b", "problem"),
        (r"\bquestions?\b", "question"),
        (r"\bsolutions?\b", "solution"),
        (r"\btodos?\b", "todo"),
        (r"\bconclusions?\b", "conclusion"),
    ]
    for pattern, intent in intent_patterns:
        if re.search(pattern, query_lower):
            result["intent_filter"] = intent
            break

    # Extract entities (technologies)
    technologies = [
        "postgresql", "mysql", "redis", "mongodb", "elasticsearch",
        "python", "javascript", "typescript", "rust", "go",
        "react", "vue", "angular", "django", "flask", "fastapi",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "jwt", "oauth", "graphql", "rest", "grpc",
    ]
    found_entities = []
    for tech in technologies:
        if tech in query_lower:
            # Find original case
            pattern = re.compile(re.escape(tech), re.IGNORECASE)
            match = pattern.search(query)
            if match:
                found_entities.append(match.group())

    if found_entities:
        result["entities"] = found_entities

    return result


def decompose_query(query: str) -> dict:
    """Decompose a complex query into sub-queries.

    Detects patterns like:
    - "X and Y" -> two separate searches merged
    - "decisions about X" -> intent filter + search term
    - "how X relates to Y" -> find connecting ideas

    Args:
        query: The search query

    Returns:
        Dict with decomposition info:
        - type: "simple", "conjunction", "intent_scoped", "relation"
        - sub_queries: List of simpler queries
        - intent: Optional intent filter
        - entities: Related entities
    """
    import re
    query_lower = query.lower().strip()

    result = {
        "type": "simple",
        "original": query,
        "sub_queries": [query],
        "intent": None,
        "entities": [],
    }

    # Detect "decisions/questions/todos about X" pattern
    intent_about_match = re.match(
        r"(decisions?|questions?|todos?|conclusions?|problems?|solutions?)\s+(about|regarding|on|for)\s+(.+)",
        query_lower
    )
    if intent_about_match:
        intent_word = intent_about_match.group(1).rstrip("s")  # Singularize
        topic = intent_about_match.group(3)
        result["type"] = "intent_scoped"
        result["intent"] = intent_word
        result["sub_queries"] = [topic]
        result["entities"] = [topic]
        return result

    # Detect "how X relates to Y" or "relationship between X and Y" pattern
    relation_match = re.match(
        r"(?:how\s+)?(.+?)\s+(?:relates?\s+to|(?:and|,)\s*its?\s+(?:relationship|connection)\s+(?:to|with))\s+(.+)",
        query_lower
    )
    if not relation_match:
        relation_match = re.match(
            r"(?:relationship|connection)\s+between\s+(.+?)\s+and\s+(.+)",
            query_lower
        )
    if relation_match:
        entity1 = relation_match.group(1).strip()
        entity2 = relation_match.group(2).strip()
        result["type"] = "relation"
        result["sub_queries"] = [entity1, entity2]
        result["entities"] = [entity1, entity2]
        return result

    # Detect "X and Y" pattern (but not common phrases)
    # Skip if it's "X and Y" where Y is less than 3 words (likely a phrase)
    and_match = re.match(r"(.+?)\s+and\s+(.+)", query_lower)
    if and_match:
        left = and_match.group(1).strip()
        right = and_match.group(2).strip()
        # Only split if both parts are substantial (3+ words or entities)
        left_words = len(left.split())
        right_words = len(right.split())
        if left_words >= 2 and right_words >= 2:
            result["type"] = "conjunction"
            result["sub_queries"] = [left, right]
            result["entities"] = [left, right]
            return result

    return result


def decomposed_search(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    show_decomposition: bool = False
) -> dict:
    """Search with automatic query decomposition.

    Handles complex queries by decomposing them and merging results.

    Args:
        query: Search query
        limit: Max results per sub-query
        session: Optional session filter
        show_decomposition: If True, include decomposition details

    Returns:
        Dict with results and optional decomposition info
    """
    decomp = decompose_query(query)

    result = {
        "results": [],
        "decomposition": decomp if show_decomposition else None,
    }

    if decomp["type"] == "simple":
        # Standard search
        results = search_ideas(query, limit=limit, session=session)
        result["results"] = results

    elif decomp["type"] == "intent_scoped":
        # Search with intent filter
        topic = decomp["sub_queries"][0]
        intent = decomp["intent"]
        results = search_ideas(topic, limit=limit, session=session, intent=intent)
        result["results"] = results

    elif decomp["type"] == "conjunction":
        # Run searches for each part and merge with RRF
        all_results = {}
        for i, sub_query in enumerate(decomp["sub_queries"]):
            sub_results = search_ideas(sub_query, limit=limit, session=session)
            for j, r in enumerate(sub_results):
                idea_id = r["id"]
                if idea_id not in all_results:
                    all_results[idea_id] = {"idea": r, "ranks": []}
                all_results[idea_id]["ranks"].append((i, j + 1))

        # RRF scoring
        k = 60
        scored = []
        for idea_id, data in all_results.items():
            score = sum(1 / (k + rank) for _, rank in data["ranks"])
            # Bonus for appearing in multiple sub-queries
            if len(data["ranks"]) > 1:
                score *= 1.5
            scored.append((score, data["idea"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        result["results"] = [item[1] for item in scored[:limit]]

    elif decomp["type"] == "relation":
        # Find ideas that connect the two entities
        entity1, entity2 = decomp["sub_queries"]

        # Search for both entities
        results1 = search_ideas(entity1, limit=limit * 2, session=session)
        results2 = search_ideas(entity2, limit=limit * 2, session=session)

        # Find overlapping results (appear in both)
        ids1 = {r["id"] for r in results1}
        ids2 = {r["id"] for r in results2}
        shared_ids = ids1 & ids2

        # Prioritize shared results
        shared = [r for r in results1 if r["id"] in shared_ids]
        unique = [r for r in results1 + results2 if r["id"] not in shared_ids]

        # Deduplicate unique
        seen = set(r["id"] for r in shared)
        for r in unique:
            if r["id"] not in seen:
                shared.append(r)
                seen.add(r["id"])
            if len(shared) >= limit:
                break

        result["results"] = shared[:limit]

    return result


# =============================================================================
# Index State
# =============================================================================

def get_last_indexed_line(file_path: str) -> int:
    """Get the last indexed line number for a file."""
    db = get_db()
    cursor = db.execute(
        "SELECT last_line FROM index_state WHERE file_path = ?",
        (file_path,)
    )
    row = cursor.fetchone()
    db.close()
    return row['last_line'] if row else 0


def update_index_state(file_path: str, last_line: int):
    """Update the index state for a file."""
    db = get_db()
    db.execute("""
        INSERT INTO index_state (file_path, last_line, last_indexed)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(file_path) DO UPDATE SET
            last_line = excluded.last_line,
            last_indexed = excluded.last_indexed
    """, (file_path, last_line))
    db.commit()
    db.close()


# =============================================================================
# Utilities
# =============================================================================

def get_stats() -> dict:
    """Get database statistics."""
    db = get_db()
    stats = {}

    cursor = db.execute("SELECT COUNT(*) as count FROM ideas")
    stats["total_ideas"] = cursor.fetchone()['count']

    cursor = db.execute("SELECT COUNT(*) as count FROM spans")
    stats["total_spans"] = cursor.fetchone()['count']

    cursor = db.execute("SELECT COUNT(*) as count FROM entities")
    stats["total_entities"] = cursor.fetchone()['count']

    cursor = db.execute("SELECT COUNT(*) as count FROM relations")
    stats["total_relations"] = cursor.fetchone()['count']

    cursor = db.execute("SELECT COUNT(DISTINCT session) as count FROM spans")
    stats["sessions_indexed"] = cursor.fetchone()['count']

    cursor = db.execute("""
        SELECT intent, COUNT(*) as count
        FROM ideas
        WHERE intent IS NOT NULL
        GROUP BY intent
    """)
    stats["by_intent"] = {row['intent']: row['count'] for row in cursor}

    cursor = db.execute("""
        SELECT type, COUNT(*) as count
        FROM entities
        GROUP BY type
    """)
    stats["entities_by_type"] = {row['type']: row['count'] for row in cursor}

    # Count unanswered questions
    cursor = db.execute("""
        SELECT COUNT(*) as count FROM ideas
        WHERE intent = 'question' AND (answered IS NULL OR answered = 0)
    """)
    stats["unanswered_questions"] = cursor.fetchone()['count']

    # Access tracking statistics
    cursor = db.execute("""
        SELECT
            SUM(COALESCE(access_count, 0)) as total_accesses,
            AVG(COALESCE(access_count, 0)) as avg_accesses,
            COUNT(CASE WHEN COALESCE(access_count, 0) = 0 THEN 1 END) as never_accessed,
            COUNT(CASE WHEN COALESCE(access_count, 0) > 0 THEN 1 END) as accessed_at_least_once
        FROM ideas
    """)
    row = cursor.fetchone()
    stats["access_tracking"] = {
        "total_accesses": row['total_accesses'] or 0,
        "avg_accesses_per_idea": round(row['avg_accesses'] or 0, 2),
        "never_accessed": row['never_accessed'] or 0,
        "accessed_at_least_once": row['accessed_at_least_once'] or 0
    }

    # Most accessed ideas (top 5)
    cursor = db.execute("""
        SELECT id, content, access_count, last_accessed
        FROM ideas
        WHERE access_count > 0
        ORDER BY access_count DESC
        LIMIT 5
    """)
    stats["most_accessed_ideas"] = [
        {"id": row['id'], "content": row['content'][:100], "access_count": row['access_count']}
        for row in cursor
    ]

    db.close()

    # Add cache stats
    stats["embedding_cache"] = get_embedding_cache_stats()

    # Add database file size
    if DB_PATH.exists():
        size_bytes = DB_PATH.stat().st_size
        stats["db_size_mb"] = round(size_bytes / (1024 * 1024), 2)

    return stats


# =============================================================================
# Working Memory Operations
# =============================================================================

def activate_idea(session: str, idea_id: int, activation: float = 1.0) -> None:
    """Add or update an idea in working memory.

    Args:
        session: Current session
        idea_id: ID of the idea to activate
        activation: Activation level (0-1, default 1.0)
    """
    from datetime import datetime

    db = get_db()
    now = datetime.utcnow().isoformat()

    # Upsert into working_memory
    db.execute("""
        INSERT INTO working_memory (session, idea_id, activation, last_access)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session, idea_id) DO UPDATE SET
            activation = MAX(activation, excluded.activation),
            last_access = excluded.last_access
    """, (session, idea_id, activation, now))
    db.commit()
    db.close()


def get_active_ideas(session: str, min_activation: float = 0.1, limit: int = 20) -> list[dict]:
    """Get currently active ideas in working memory.

    Args:
        session: Current session
        min_activation: Minimum activation threshold
        limit: Maximum ideas to return

    Returns:
        List of idea dicts with activation levels
    """
    db = get_db()
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic,
            wm.activation, wm.last_access
        FROM working_memory wm
        JOIN ideas i ON i.id = wm.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE wm.session = ? AND wm.activation >= ?
        ORDER BY wm.activation DESC
        LIMIT ?
    """, (session, min_activation, limit))

    results = [dict(row) for row in cursor]
    db.close()
    return results


def decay_working_memory(session: str, decay_rate: float = 0.9) -> int:
    """Apply time decay to working memory activations.

    Call this at session start or periodically to fade old activations.

    Args:
        session: Session to decay
        decay_rate: Multiplier for decay (0-1, lower = faster decay)

    Returns:
        Number of records updated
    """
    db = get_db()

    # Apply decay
    cursor = db.execute("""
        UPDATE working_memory
        SET activation = activation * ?
        WHERE session = ?
    """, (decay_rate, session))
    updated = cursor.rowcount

    # Clean up very low activations
    db.execute("""
        DELETE FROM working_memory
        WHERE session = ? AND activation < 0.01
    """, (session,))

    db.commit()
    db.close()

    return updated


def boost_results_by_activation(
    results: list[dict],
    session: str,
    boost_weight: float = 0.3
) -> list[dict]:
    """Re-rank search results by working memory activation.

    Blends vector similarity ranking with working memory activation.

    Args:
        results: Search results (must have 'id' field)
        session: Current session
        boost_weight: How much to weight activation (0-1, 0=no boost, 1=activation only)

    Returns:
        Re-ranked results
    """
    if not results or boost_weight == 0:
        return results

    db = get_db()

    # Get activations for result IDs
    idea_ids = [r['id'] for r in results]
    placeholders = ','.join('?' * len(idea_ids))
    cursor = db.execute(f"""
        SELECT idea_id, activation
        FROM working_memory
        WHERE session = ? AND idea_id IN ({placeholders})
    """, [session] + idea_ids)

    activations = {row['idea_id']: row['activation'] for row in cursor}
    db.close()

    # Calculate blended scores
    # Original order is best similarity, so rank 0 = highest score
    for i, r in enumerate(results):
        original_score = 1.0 - (i / len(results))  # 1.0 to 0.0
        activation = activations.get(r['id'], 0)
        r['_boost_score'] = (1 - boost_weight) * original_score + boost_weight * activation

    # Sort by blended score
    results.sort(key=lambda r: r.get('_boost_score', 0), reverse=True)

    # Clean up
    for r in results:
        r.pop('_boost_score', None)

    return results


# =============================================================================
# Soft Forgetting Operations
# =============================================================================

def forget_idea(idea_id: int) -> bool:
    """Mark an idea as forgotten (soft delete).

    Forgotten ideas are excluded from search results but not deleted.

    Args:
        idea_id: ID of the idea to forget

    Returns:
        True if idea was found and updated
    """
    db = get_db()
    cursor = db.execute("""
        UPDATE ideas SET forgotten = TRUE WHERE id = ?
    """, (idea_id,))
    updated = cursor.rowcount > 0
    db.commit()
    db.close()
    return updated


def unforget_idea(idea_id: int) -> bool:
    """Restore a forgotten idea.

    Args:
        idea_id: ID of the idea to restore

    Returns:
        True if idea was found and updated
    """
    db = get_db()
    cursor = db.execute("""
        UPDATE ideas SET forgotten = FALSE WHERE id = ?
    """, (idea_id,))
    updated = cursor.rowcount > 0
    db.commit()
    db.close()
    return updated


def get_forgotten_ideas(limit: int = 50) -> list[dict]:
    """List all forgotten ideas.

    Args:
        limit: Maximum ideas to return

    Returns:
        List of forgotten idea dicts
    """
    db = get_db()
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            i.access_count, i.last_accessed,
            s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.forgotten = TRUE
        ORDER BY i.created_at DESC
        LIMIT ?
    """, (limit,))
    results = [dict(row) for row in cursor]
    db.close()
    return results


def retention_score(idea: dict, now: str = None) -> float:
    """Calculate retention score for an idea (0-1, higher = more worth keeping).

    Score is based on:
    - Recency: How recently the idea was created (decays over time)
    - Access frequency: How often the idea has been accessed
    - Importance: Intent-based importance (decisions, conclusions are important)

    Args:
        idea: Dict with id, created_at, access_count, last_accessed, intent
        now: Current ISO datetime (defaults to utcnow)

    Returns:
        Float 0-1, where 1 = definitely keep, 0 = safe to forget
    """
    from datetime import datetime

    if now is None:
        now = datetime.utcnow().isoformat()

    # Parse dates
    created_at = idea.get("created_at") or idea.get("message_time") or now
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except:
        created = datetime.utcnow()

    try:
        current = datetime.fromisoformat(now.replace("Z", "+00:00"))
    except:
        current = datetime.utcnow()

    # Age in days (capped at 365)
    age_days = min((current - created).days, 365)

    # Recency score: decays from 1.0 to 0.2 over 90 days
    recency = max(0.2, 1.0 - (age_days / 90) * 0.8)

    # Access score: 0 accesses = 0.0, 5+ = 1.0
    access_count = idea.get("access_count") or 0
    access_score = min(1.0, access_count / 5)

    # Importance by intent (decisions and conclusions are protected)
    intent = idea.get("intent") or "context"
    importance = {
        "decision": 1.0,
        "conclusion": 1.0,
        "solution": 0.8,
        "problem": 0.6,
        "question": 0.5,
        "todo": 0.7,
        "context": 0.3,
    }.get(intent, 0.3)

    # Weighted combination
    score = (recency * 0.3) + (access_score * 0.3) + (importance * 0.4)

    return round(score, 3)


def get_forgettable_ideas(
    threshold: float = 0.3,
    limit: int = 100,
    session: Optional[str] = None
) -> list[dict]:
    """Find ideas with low retention scores that are candidates for forgetting.

    Never returns decisions or conclusions (protected by importance).

    Args:
        threshold: Maximum retention score to include (0-1)
        limit: Maximum ideas to return
        session: Optional session filter

    Returns:
        List of idea dicts with retention_score field, sorted by score ascending
    """
    db = get_db()

    # Get non-forgotten ideas with low importance intents
    sql = """
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            i.message_time, i.access_count, i.last_accessed,
            s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE (i.forgotten = FALSE OR i.forgotten IS NULL)
            AND i.intent NOT IN ('decision', 'conclusion')
    """
    params = []

    if session:
        sql += " AND s.session = ?"
        params.append(session)

    cursor = db.execute(sql, params)
    ideas = [dict(row) for row in cursor]
    db.close()

    # Calculate retention scores
    from datetime import datetime
    now = datetime.utcnow().isoformat()

    scored = []
    for idea in ideas:
        score = retention_score(idea, now)
        if score <= threshold:
            idea["retention_score"] = score
            scored.append(idea)

    # Sort by score ascending (lowest = most forgettable)
    scored.sort(key=lambda x: x["retention_score"])

    return scored[:limit]


def auto_forget_ideas(
    threshold: float = 0.3,
    limit: int = 100,
    session: Optional[str] = None,
    dry_run: bool = True
) -> dict:
    """Automatically forget ideas with low retention scores.

    Protected intents (decision, conclusion) are never forgotten.

    Args:
        threshold: Maximum retention score to forget (0-1)
        limit: Maximum ideas to forget in one run
        session: Optional session filter
        dry_run: If True, return candidates without forgetting

    Returns:
        Dict with candidates, count, and samples
    """
    candidates = get_forgettable_ideas(threshold=threshold, limit=limit, session=session)

    result = {
        "threshold": threshold,
        "candidates": len(candidates),
        "samples": [
            {
                "id": c["id"],
                "content": c["content"][:100],
                "intent": c["intent"],
                "retention_score": c["retention_score"],
            }
            for c in candidates[:10]
        ],
        "dry_run": dry_run,
        "forgotten": 0,
    }

    if not dry_run and candidates:
        db = get_db()
        for idea in candidates:
            db.execute("UPDATE ideas SET forgotten = TRUE WHERE id = ?", (idea["id"],))
        db.commit()
        db.close()
        result["forgotten"] = len(candidates)

    return result


# =============================================================================
# Consolidation Operations
# =============================================================================

def should_preserve(idea: dict) -> bool:
    """Check if an idea should be preserved (not consolidated).

    Decisions, conclusions, and high-confidence ideas are protected.

    Args:
        idea: Dict with intent and confidence fields

    Returns:
        True if idea should be preserved individually
    """
    intent = idea.get("intent") or "context"
    confidence = idea.get("confidence") or 0.5

    # Protected intents
    if intent in ("decision", "conclusion"):
        return True

    # High-confidence ideas are important
    if confidence >= 0.9:
        return True

    return False


def get_consolidatable_ideas(
    topic_id: int,
    min_ideas: int = 5,
    max_age_days: int = 30
) -> list[dict]:
    """Find ideas in a topic that are candidates for consolidation.

    Returns context ideas that are older than max_age_days and not already
    consolidated. Protected ideas (decisions, conclusions) are excluded.

    Args:
        topic_id: Topic to check
        min_ideas: Minimum ideas needed for consolidation
        max_age_days: Only consider ideas older than this

    Returns:
        List of consolidatable ideas
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

    db = get_db()
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.created_at, i.message_time,
            s.session, s.name as topic, t.id as topic_id
        FROM ideas i
        JOIN spans s ON s.id = i.span_id
        JOIN topics t ON t.id = s.topic_id
        WHERE t.id = ?
            AND (i.forgotten = FALSE OR i.forgotten IS NULL)
            AND (i.consolidated_into IS NULL)
            AND (i.is_consolidated = FALSE OR i.is_consolidated IS NULL)
            AND i.intent NOT IN ('decision', 'conclusion')
            AND COALESCE(i.message_time, i.created_at) < ?
        ORDER BY COALESCE(i.message_time, i.created_at)
    """, (topic_id, cutoff))
    ideas = [dict(row) for row in cursor]
    db.close()

    # Filter out high-confidence ideas
    ideas = [i for i in ideas if not should_preserve(i)]

    if len(ideas) < min_ideas:
        return []

    return ideas


def consolidate_topic(
    topic_id: int,
    min_ideas: int = 5,
    max_age_days: int = 30,
    dry_run: bool = True
) -> dict:
    """Consolidate old context ideas in a topic into a summary.

    Creates a new summary idea and links original ideas to it.
    Protected ideas (decisions, conclusions) are preserved.

    Args:
        topic_id: Topic to consolidate
        min_ideas: Minimum ideas for consolidation
        max_age_days: Only consolidate ideas older than this
        dry_run: If True, return preview without changes

    Returns:
        Dict with consolidation summary
    """
    candidates = get_consolidatable_ideas(topic_id, min_ideas, max_age_days)

    if not candidates:
        return {
            "topic_id": topic_id,
            "candidates": 0,
            "message": "Not enough consolidatable ideas",
            "dry_run": dry_run,
        }

    # Get topic name
    db = get_db()
    cursor = db.execute("SELECT name, summary FROM topics WHERE id = ?", (topic_id,))
    topic_row = cursor.fetchone()
    topic_name = topic_row["name"] if topic_row else f"Topic {topic_id}"

    result = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "candidates": len(candidates),
        "samples": [
            {"id": c["id"], "content": c["content"][:80], "intent": c["intent"]}
            for c in candidates[:5]
        ],
        "dry_run": dry_run,
        "consolidated": 0,
        "summary_id": None,
    }

    if dry_run:
        db.close()
        return result

    # Generate summary using LLM
    content_sample = "\n".join([
        f"- [{c['intent']}] {c['content'][:200]}"
        for c in candidates[:20]  # Use first 20 for summary
    ])

    summary_prompt = f"""Summarize these conversation context points from the topic "{topic_name}" into a single concise paragraph.
Focus on key facts, decisions context, and important details. Skip trivial acknowledgments.

Context points:
{content_sample}

Write a 2-4 sentence summary that captures the essential information:"""

    try:
        summary_text = claude_complete(summary_prompt)
    except Exception as e:
        db.close()
        return {**result, "error": f"LLM summary failed: {e}"}

    # Get a span for the summary (use first candidate's span)
    span_id = None
    if candidates:
        cursor = db.execute("SELECT span_id FROM ideas WHERE id = ?", (candidates[0]["id"],))
        span_row = cursor.fetchone()
        if span_row:
            span_id = span_row["span_id"]

    # Create summary idea
    cursor = db.execute("""
        INSERT INTO ideas (span_id, content, intent, confidence, is_consolidated)
        VALUES (?, ?, 'context', 0.8, TRUE)
    """, (span_id, f"[Consolidated summary] {summary_text}"))
    summary_id = cursor.lastrowid

    # Link original ideas to summary
    for idea in candidates:
        db.execute(
            "UPDATE ideas SET consolidated_into = ? WHERE id = ?",
            (summary_id, idea["id"])
        )

    db.commit()
    db.close()

    result["consolidated"] = len(candidates)
    result["summary_id"] = summary_id
    result["summary"] = summary_text[:200]

    return result


def get_consolidation_candidates(
    session: Optional[str] = None,
    min_ideas: int = 5,
    max_age_days: int = 30
) -> list[dict]:
    """Find topics that have ideas ready for consolidation.

    Args:
        session: Optional session filter
        min_ideas: Minimum ideas for consolidation
        max_age_days: Only consider ideas older than this

    Returns:
        List of topics with consolidation candidates
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

    db = get_db()

    # Get topics with old, unconsolidated context ideas
    sql = """
        SELECT
            t.id as topic_id, t.name as topic_name,
            COUNT(*) as candidate_count
        FROM topics t
        JOIN spans s ON s.topic_id = t.id
        JOIN ideas i ON i.span_id = s.id
        WHERE (i.forgotten = FALSE OR i.forgotten IS NULL)
            AND (i.consolidated_into IS NULL)
            AND (i.is_consolidated = FALSE OR i.is_consolidated IS NULL)
            AND i.intent NOT IN ('decision', 'conclusion')
            AND i.confidence < 0.9
            AND COALESCE(i.message_time, i.created_at) < ?
    """
    params = [cutoff]

    if session:
        sql += " AND s.session = ?"
        params.append(session)

    sql += f"""
        GROUP BY t.id
        HAVING COUNT(*) >= ?
        ORDER BY COUNT(*) DESC
    """
    params.append(min_ideas)

    cursor = db.execute(sql, params)
    results = [dict(row) for row in cursor]
    db.close()

    return results


def get_session_time_range(session: str) -> dict | None:
    """Get the time range (first/last idea) for a session.

    Args:
        session: Session name

    Returns:
        Dict with start_time, end_time (ISO format) or None if not found
    """
    db = get_db()
    cursor = db.execute("""
        SELECT
            MIN(COALESCE(i.message_time, i.created_at)) as start_time,
            MAX(COALESCE(i.message_time, i.created_at)) as end_time
        FROM spans s
        JOIN ideas i ON i.span_id = s.id
        WHERE s.session = ?
    """, (session,))
    row = cursor.fetchone()
    db.close()

    if row and row['start_time']:
        return {
            "session": session,
            "start_time": row['start_time'],
            "end_time": row['end_time']
        }
    return None


def list_sessions() -> list[dict]:
    """List all indexed sessions with their stats.

    Returns:
        List of session dicts with name, idea count, topic count, latest date
    """
    db = get_db()

    cursor = db.execute("""
        SELECT
            s.session,
            COUNT(DISTINCT s.id) as topic_count,
            COUNT(DISTINCT i.id) as idea_count,
            MAX(i.created_at) as latest_idea,
            MIN(i.created_at) as first_idea
        FROM spans s
        LEFT JOIN ideas i ON i.span_id = s.id
        GROUP BY s.session
        ORDER BY MAX(i.created_at) DESC
    """)

    sessions = []
    for row in cursor:
        sessions.append({
            "session": row["session"],
            "topic_count": row["topic_count"],
            "idea_count": row["idea_count"],
            "first_idea": row["first_idea"],
            "latest_idea": row["latest_idea"],
        })

    db.close()
    return sessions


# =============================================================================
# Graph Revision Operations
# =============================================================================

def update_idea_intent(idea_id: int, new_intent: str) -> bool:
    """Update an idea's intent classification.

    Args:
        idea_id: ID of the idea
        new_intent: New intent value

    Returns:
        True if updated, False if idea not found
    """
    db = get_db()
    cursor = db.execute(
        "UPDATE ideas SET intent = ? WHERE id = ?",
        (new_intent, idea_id)
    )
    db.commit()
    updated = cursor.rowcount > 0
    db.close()
    return updated


def move_idea_to_span(idea_id: int, span_id: int) -> bool:
    """Move an idea to a different topic span.

    Args:
        idea_id: ID of the idea
        span_id: ID of the target span

    Returns:
        True if updated, False if idea not found
    """
    db = get_db()
    cursor = db.execute(
        "UPDATE ideas SET span_id = ? WHERE id = ?",
        (span_id, idea_id)
    )
    db.commit()
    updated = cursor.rowcount > 0
    db.close()
    return updated


def merge_spans(source_span_id: int, target_span_id: int) -> dict:
    """Merge one span into another.

    All ideas from source span are moved to target span,
    and the source span is deleted.

    Args:
        source_span_id: Span to merge from (will be deleted)
        target_span_id: Span to merge into

    Returns:
        Dict with merge stats
    """
    db = get_db()

    # Move all ideas
    cursor = db.execute(
        "UPDATE ideas SET span_id = ? WHERE span_id = ?",
        (target_span_id, source_span_id)
    )
    ideas_moved = cursor.rowcount

    # Move child spans
    cursor = db.execute(
        "UPDATE spans SET parent_id = ? WHERE parent_id = ?",
        (target_span_id, source_span_id)
    )
    children_moved = cursor.rowcount

    # Delete source span embedding
    db.execute(
        "DELETE FROM span_embeddings WHERE span_id = ?",
        (source_span_id,)
    )

    # Delete source span
    db.execute(
        "DELETE FROM spans WHERE id = ?",
        (source_span_id,)
    )

    db.commit()
    db.close()

    return {
        "source_span_id": source_span_id,
        "target_span_id": target_span_id,
        "ideas_moved": ideas_moved,
        "children_moved": children_moved
    }


def supersede_idea(old_idea_id: int, new_idea_id: int, reason: Optional[str] = None) -> bool:
    """Mark an idea as superseded by another.

    Creates a 'supersedes' relation and optionally stores the reason.

    Args:
        old_idea_id: The old/wrong idea
        new_idea_id: The new/corrected idea
        reason: Optional explanation

    Returns:
        True if relation created
    """
    add_relation(new_idea_id, old_idea_id, "supersedes")
    return True


def update_span_name(span_id: int, new_name: str) -> bool:
    """Update a span's name.

    Args:
        span_id: ID of the span
        new_name: New name

    Returns:
        True if updated
    """
    db = get_db()
    cursor = db.execute(
        "UPDATE spans SET name = ? WHERE id = ?",
        (new_name, span_id)
    )
    db.commit()
    updated = cursor.rowcount > 0
    db.close()
    return updated


def reparent_span(span_id: int, new_parent_id: Optional[int], new_depth: Optional[int] = None) -> bool:
    """Change a span's parent in the hierarchy.

    Args:
        span_id: ID of the span to move
        new_parent_id: New parent span ID (None for top-level)
        new_depth: New depth (auto-calculated if None)

    Returns:
        True if updated
    """
    db = get_db()

    # Calculate depth if not provided
    if new_depth is None:
        if new_parent_id is None:
            new_depth = 0
        else:
            cursor = db.execute(
                "SELECT depth FROM spans WHERE id = ?",
                (new_parent_id,)
            )
            row = cursor.fetchone()
            new_depth = (row["depth"] + 1) if row else 0

    cursor = db.execute(
        "UPDATE spans SET parent_id = ?, depth = ? WHERE id = ?",
        (new_parent_id, new_depth, span_id)
    )
    db.commit()
    updated = cursor.rowcount > 0
    db.close()
    return updated


def prune_old_ideas(
    older_than_days: int = 90,
    session: Optional[str] = None,
    dry_run: bool = True
) -> dict:
    """Prune old ideas from the database.

    Args:
        older_than_days: Remove ideas older than this many days
        session: Only prune from specific session (None = all)
        dry_run: If True, only count what would be removed

    Returns:
        Dict with counts of pruned/pruneable items
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    db = get_db()

    # Build query
    if session:
        count_sql = """
            SELECT COUNT(*) as count FROM ideas i
            JOIN spans s ON s.id = i.span_id
            WHERE i.created_at < ? AND s.session = ?
        """
        params = (cutoff, session)
    else:
        count_sql = """
            SELECT COUNT(*) as count FROM ideas
            WHERE created_at < ?
        """
        params = (cutoff,)

    cursor = db.execute(count_sql, params)
    count = cursor.fetchone()["count"]

    result = {
        "would_remove": count,
        "older_than_days": older_than_days,
        "cutoff_date": cutoff,
        "session": session,
        "dry_run": dry_run,
    }

    if not dry_run and count > 0:
        # Get IDs to delete
        if session:
            id_sql = """
                SELECT i.id FROM ideas i
                JOIN spans s ON s.id = i.span_id
                WHERE i.created_at < ? AND s.session = ?
            """
        else:
            id_sql = """
                SELECT id FROM ideas WHERE created_at < ?
            """

        cursor = db.execute(id_sql, params)
        idea_ids = [row["id"] for row in cursor]

        if idea_ids:
            placeholders = ",".join("?" * len(idea_ids))

            # Delete embeddings
            db.execute(f"DELETE FROM idea_embeddings WHERE idea_id IN ({placeholders})", idea_ids)

            # Delete relations
            db.execute(f"DELETE FROM relations WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                      idea_ids + idea_ids)

            # Delete entity links
            db.execute(f"DELETE FROM idea_entities WHERE idea_id IN ({placeholders})", idea_ids)

            # Delete FTS entries
            db.execute(f"DELETE FROM ideas_fts WHERE rowid IN ({placeholders})", idea_ids)

            # Delete ideas
            db.execute(f"DELETE FROM ideas WHERE id IN ({placeholders})", idea_ids)

            db.commit()
            result["removed"] = len(idea_ids)
            result["dry_run"] = False

    db.close()
    return result


def review_ideas_against_filters(
    topic_id: Optional[int] = None,
    dry_run: bool = True
) -> dict:
    """Review existing ideas against current transcript filters.

    Checks which ideas would not pass the current is_indexable() filters
    and optionally removes them.

    Args:
        topic_id: Only review ideas from this topic (None = all)
        dry_run: If True, only report what would be removed

    Returns:
        Dict with review results and counts by filter reason
    """
    from transcript import get_filter_reason

    db = get_db()

    # Get ideas to review
    if topic_id:
        cursor = db.execute("""
            SELECT i.id, i.content, i.intent, i.source_line, s.session, t.name as topic_name
            FROM ideas i
            JOIN spans s ON s.id = i.span_id
            LEFT JOIN topics t ON t.id = s.topic_id
            WHERE s.topic_id = ?
            ORDER BY i.source_line
        """, (topic_id,))
    else:
        cursor = db.execute("""
            SELECT i.id, i.content, i.intent, i.source_line, s.session, t.name as topic_name
            FROM ideas i
            JOIN spans s ON s.id = i.span_id
            LEFT JOIN topics t ON t.id = s.topic_id
            ORDER BY s.topic_id, i.source_line
        """)

    ideas = [dict(row) for row in cursor]

    # Check each idea against filters
    would_filter = []
    filter_reasons = {}

    for idea in ideas:
        # Simulate the message structure expected by is_indexable
        message = {
            "content": idea["content"],
            "type": "assistant",  # Assume assistant for stricter filtering
            "has_tool_use": False,
        }

        reason = get_filter_reason(message)
        if reason:
            would_filter.append({
                "id": idea["id"],
                "content": idea["content"][:100],
                "intent": idea["intent"],
                "topic": idea["topic_name"],
                "reason": reason,
            })
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

    result = {
        "total_reviewed": len(ideas),
        "would_filter": len(would_filter),
        "would_keep": len(ideas) - len(would_filter),
        "filter_reasons": filter_reasons,
        "dry_run": dry_run,
        "topic_id": topic_id,
    }

    # Add sample of what would be filtered
    result["samples"] = would_filter[:20]

    if not dry_run and would_filter:
        # Delete the filtered ideas
        idea_ids = [item["id"] for item in would_filter]
        placeholders = ",".join("?" * len(idea_ids))

        # Delete embeddings
        db.execute(f"DELETE FROM idea_embeddings WHERE idea_id IN ({placeholders})", idea_ids)

        # Delete relations
        db.execute(f"DELETE FROM relations WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                  idea_ids + idea_ids)

        # Delete entity links
        db.execute(f"DELETE FROM idea_entities WHERE idea_id IN ({placeholders})", idea_ids)

        # Delete FTS entries
        db.execute(f"DELETE FROM ideas_fts WHERE rowid IN ({placeholders})", idea_ids)

        # Delete ideas
        db.execute(f"DELETE FROM ideas WHERE id IN ({placeholders})", idea_ids)

        db.commit()
        result["removed"] = len(idea_ids)
        result["dry_run"] = False

    db.close()
    return result


def llm_filter_ideas(
    topic_id: Optional[int] = None,
    batch_size: int = 20,
    dry_run: bool = True
) -> dict:
    """Use LLM to identify low-value ideas that regex filters can't catch.

    Catches things like:
    - Generic statements without specific information
    - Redundant or repetitive content
    - Overly verbose preambles that passed regex
    - Content that's too context-dependent to be useful standalone

    Args:
        topic_id: Only filter ideas from this topic (None = all)
        batch_size: Number of ideas to evaluate per LLM call
        dry_run: If True, only report what would be removed

    Returns:
        Dict with filtering results
    """
    db = get_db()

    # Get ideas to review
    if topic_id:
        cursor = db.execute("""
            SELECT i.id, i.content, i.intent, i.source_line, s.session, t.name as topic_name
            FROM ideas i
            JOIN spans s ON s.id = i.span_id
            LEFT JOIN topics t ON t.id = s.topic_id
            WHERE s.topic_id = ?
            ORDER BY i.source_line
        """, (topic_id,))
    else:
        cursor = db.execute("""
            SELECT i.id, i.content, i.intent, i.source_line, s.session, t.name as topic_name
            FROM ideas i
            JOIN spans s ON s.id = i.span_id
            LEFT JOIN topics t ON t.id = s.topic_id
            ORDER BY s.topic_id, i.source_line
        """)

    ideas = [dict(row) for row in cursor]

    flagged = []
    total_reviewed = 0

    # Process in batches
    for i in range(0, len(ideas), batch_size):
        batch = ideas[i:i + batch_size]
        total_reviewed += len(batch)

        # Format batch for LLM
        ideas_text = "\n".join(
            f"[{idx}] ({idea['intent']}) {idea['content'][:200]}"
            for idx, idea in enumerate(batch)
        )

        prompt = f"""You are filtering a conversation memory database. Review these extracted "ideas" and identify which ones are LOW VALUE and should be removed.

Remove ideas that are:
- Generic statements without specific/actionable information
- Preambles like "Let me help you with that" or "I'll work on this"
- Status updates without substance ("Working on it", "Almost done")
- Redundant/repetitive - says same thing as likely neighbors
- Too context-dependent to be useful standalone (references "this" or "that" without specifics)
- Procedural noise ("I'll use tool X", "Running the command")

Keep ideas that are:
- Specific decisions with reasoning
- Concrete conclusions or findings
- Actionable questions or problems
- Technical explanations with detail
- Solutions with implementation specifics

Ideas to review:
{ideas_text}

Reply with ONLY a JSON array of indices that should be REMOVED, e.g. [0, 3, 5]
If none should be removed, reply: []"""

        try:
            response_text = claude_complete(prompt).strip()

            # Parse response - handle various formats
            import re
            # Find array in response
            match = re.search(r'\[[\d,\s]*\]', response_text)
            if match:
                indices = json.loads(match.group())
                for idx in indices:
                    if 0 <= idx < len(batch):
                        flagged.append({
                            "id": batch[idx]["id"],
                            "content": batch[idx]["content"][:100],
                            "intent": batch[idx]["intent"],
                            "topic": batch[idx]["topic_name"],
                            "reason": "llm_low_value",
                        })

        except MemgraphError:
            db.close()
            raise  # Re-raise MemgraphError as-is
        except Exception as e:
            db.close()
            logger.error(f"LLM filter batch failed: {e}")
            raise MemgraphError(
                f"LLM filter failed: {e}",
                "llm_filter_error",
                {"batch_start": i, "original_error": str(e)}
            ) from e

    result = {
        "total_reviewed": total_reviewed,
        "flagged": len(flagged),
        "kept": total_reviewed - len(flagged),
        "dry_run": dry_run,
        "topic_id": topic_id,
        "samples": flagged[:20],
    }

    if not dry_run and flagged:
        # Delete the flagged ideas
        idea_ids = [item["id"] for item in flagged]
        placeholders = ",".join("?" * len(idea_ids))

        # Delete embeddings
        db.execute(f"DELETE FROM idea_embeddings WHERE idea_id IN ({placeholders})", idea_ids)

        # Delete relations
        db.execute(f"DELETE FROM relations WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                  idea_ids + idea_ids)

        # Delete entity links
        db.execute(f"DELETE FROM idea_entities WHERE idea_id IN ({placeholders})", idea_ids)

        # Delete FTS entries
        db.execute(f"DELETE FROM ideas_fts WHERE rowid IN ({placeholders})", idea_ids)

        # Delete ideas
        db.execute(f"DELETE FROM ideas WHERE id IN ({placeholders})", idea_ids)

        db.commit()
        result["removed"] = len(idea_ids)

    db.close()
    return result


def get_context(idea_id: int, lines_before: int = 5, lines_after: int = 5) -> dict:
    """Get the source transcript context around an idea.

    Args:
        idea_id: The idea to get context for
        lines_before: Number of lines before the idea
        lines_after: Number of lines after the idea

    Returns:
        Dict with idea info, context lines, and source info
    """
    db = get_db()
    cursor = db.execute("""
        SELECT i.id, i.content, i.intent, i.source_file, i.source_line,
               s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.id = ?
    """, (idea_id,))
    row = cursor.fetchone()
    db.close()

    if not row:
        raise MemgraphError(
            f"Idea {idea_id} not found",
            "not_found",
            {"idea_id": idea_id}
        )

    idea = dict(row)
    source_file = idea.get("source_file")
    source_line = idea.get("source_line")

    if not source_file or not source_line:
        return {
            "idea": idea,
            "context": None,
            "error": "No source location recorded"
        }

    # Try to read the source file
    source_path = Path(source_file)
    if not source_path.exists():
        return {
            "idea": idea,
            "context": None,
            "error": f"Source file not found: {source_file}"
        }

    # Read relevant lines
    context_lines = []
    start_line = max(1, source_line - lines_before)
    end_line = source_line + lines_after

    try:
        with open(source_path, 'r') as f:
            for i, line in enumerate(f, 1):
                if i < start_line:
                    continue
                if i > end_line:
                    break

                # Parse JSONL line
                try:
                    entry = json.loads(line)
                    message_type = entry.get("type", "unknown")
                    if message_type == "user" and "message" in entry:
                        content = entry["message"].get("content", "")
                    elif message_type == "assistant" and "message" in entry:
                        content = entry["message"].get("content", "")
                        if isinstance(content, list):
                            # Handle content blocks
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            content = "\n".join(text_parts)
                    else:
                        content = str(entry)[:200]  # Truncate other types

                    context_lines.append({
                        "line": i,
                        "type": message_type,
                        "content": content[:500] if isinstance(content, str) else str(content)[:500],
                        "is_source": i == source_line
                    })
                except json.JSONDecodeError:
                    context_lines.append({
                        "line": i,
                        "type": "parse_error",
                        "content": line[:200],
                        "is_source": i == source_line
                    })
    except Exception as e:
        return {
            "idea": idea,
            "context": None,
            "error": f"Failed to read source: {e}"
        }

    return {
        "idea": idea,
        "context": context_lines,
        "source_file": source_file,
        "source_line": source_line,
        "lines_shown": f"{start_line}-{end_line}"
    }


def export_data(session: Optional[str] = None, include_embeddings: bool = False) -> dict:
    """Export ideas and spans as JSON for backup.

    Args:
        session: Only export from specific session (None = all)
        include_embeddings: Include embedding vectors (makes export much larger)

    Returns:
        Dict with ideas, spans, relations, entities
    """
    db = get_db()

    # Export spans
    if session:
        cursor = db.execute("""
            SELECT id, session, parent_id, name, summary, start_line, end_line, depth, created_at
            FROM spans WHERE session = ? ORDER BY id
        """, (session,))
    else:
        cursor = db.execute("""
            SELECT id, session, parent_id, name, summary, start_line, end_line, depth, created_at
            FROM spans ORDER BY id
        """)
    spans = [dict(row) for row in cursor]

    # Export ideas
    if session:
        cursor = db.execute("""
            SELECT i.id, i.span_id, i.content, i.intent, i.confidence, i.answered,
                   i.source_file, i.source_line, i.created_at
            FROM ideas i
            JOIN spans s ON s.id = i.span_id
            WHERE s.session = ? ORDER BY i.id
        """, (session,))
    else:
        cursor = db.execute("""
            SELECT id, span_id, content, intent, confidence, answered,
                   source_file, source_line, created_at
            FROM ideas ORDER BY id
        """)
    ideas = [dict(row) for row in cursor]

    # Get idea IDs for filtering relations and entities
    idea_ids = {i["id"] for i in ideas}

    # Export relations
    cursor = db.execute("SELECT from_id, to_id, relation_type FROM relations ORDER BY id")
    relations = [
        dict(row) for row in cursor
        if row["from_id"] in idea_ids or row["to_id"] in idea_ids
    ]

    # Export entities and their links
    cursor = db.execute("SELECT id, name, type FROM entities ORDER BY id")
    all_entities = {row["id"]: dict(row) for row in cursor}

    cursor = db.execute("SELECT idea_id, entity_id FROM idea_entities")
    entity_links = [
        dict(row) for row in cursor
        if row["idea_id"] in idea_ids
    ]

    # Filter to only entities that are used
    used_entity_ids = {link["entity_id"] for link in entity_links}
    entities = [e for eid, e in all_entities.items() if eid in used_entity_ids]

    db.close()

    return {
        "version": 1,
        "session_filter": session,
        "spans": spans,
        "ideas": ideas,
        "relations": relations,
        "entities": entities,
        "entity_links": entity_links,
        "stats": {
            "spans_count": len(spans),
            "ideas_count": len(ideas),
            "relations_count": len(relations),
            "entities_count": len(entities),
        }
    }


def import_data(data: dict, replace: bool = False) -> dict:
    """Import data from an export JSON.

    Args:
        data: Export data dict (from export_data)
        replace: If True, clear existing data first. If False, merge.

    Returns:
        Dict with import stats and ID mappings
    """
    version = data.get("version", 1)
    if version != 1:
        raise MemgraphError(
            f"Unsupported export version: {version}",
            "import_error",
            {"version": version}
        )

    db = get_db()

    if replace:
        # Clear existing data
        db.execute("DELETE FROM idea_entities")
        db.execute("DELETE FROM relations")
        db.execute("DELETE FROM ideas")
        db.execute("DELETE FROM idea_embeddings")
        db.execute("DELETE FROM spans")
        db.execute("DELETE FROM entities")
        db.commit()

    # ID mapping: old_id -> new_id
    span_id_map = {}
    idea_id_map = {}
    entity_id_map = {}

    # Import spans (maintaining parent relationships)
    # Sort by depth so parents are created first
    spans_by_depth = sorted(data.get("spans", []), key=lambda s: s.get("depth", 0))
    for span in spans_by_depth:
        old_id = span["id"]
        old_parent_id = span.get("parent_id")
        new_parent_id = span_id_map.get(old_parent_id) if old_parent_id else None

        cursor = db.execute("""
            INSERT INTO spans (session, parent_id, name, summary, start_line, end_line, depth, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            span["session"],
            new_parent_id,
            span.get("name"),
            span.get("summary"),
            span.get("start_line"),
            span.get("end_line"),
            span.get("depth", 0),
            span.get("created_at")
        ))
        span_id_map[old_id] = cursor.lastrowid

    # Import entities
    for entity in data.get("entities", []):
        old_id = entity["id"]
        cursor = db.execute("""
            INSERT INTO entities (name, type)
            VALUES (?, ?)
        """, (entity["name"], entity.get("type")))
        entity_id_map[old_id] = cursor.lastrowid

    # Import ideas and generate embeddings
    ideas_imported = 0
    for idea in data.get("ideas", []):
        old_id = idea["id"]
        old_span_id = idea["span_id"]
        new_span_id = span_id_map.get(old_span_id)

        if not new_span_id:
            logger.warning(f"Skipping idea {old_id}: span {old_span_id} not found")
            continue

        # Generate embedding for the idea
        try:
            embedding = get_embedding(idea["content"])
        except Exception as e:
            logger.warning(f"Failed to embed idea {old_id}: {e}")
            continue

        cursor = db.execute("""
            INSERT INTO ideas (span_id, content, intent, confidence, answered, source_file, source_line, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_span_id,
            idea["content"],
            idea.get("intent"),
            idea.get("confidence"),
            idea.get("answered"),
            idea.get("source_file"),
            idea.get("source_line"),
            idea.get("created_at")
        ))
        new_idea_id = cursor.lastrowid
        idea_id_map[old_id] = new_idea_id

        # Store embedding
        db.execute(
            "INSERT INTO idea_embeddings (idea_id, embedding) VALUES (?, ?)",
            (new_idea_id, serialize_embedding(embedding))
        )
        ideas_imported += 1

    # Import relations
    relations_imported = 0
    for rel in data.get("relations", []):
        new_from_id = idea_id_map.get(rel["from_id"])
        new_to_id = idea_id_map.get(rel["to_id"])
        if new_from_id and new_to_id:
            try:
                db.execute("""
                    INSERT INTO relations (from_id, to_id, relation_type)
                    VALUES (?, ?, ?)
                """, (new_from_id, new_to_id, rel["relation_type"]))
                relations_imported += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate relation

    # Import entity links
    entity_links_imported = 0
    for link in data.get("entity_links", []):
        new_idea_id = idea_id_map.get(link["idea_id"])
        new_entity_id = entity_id_map.get(link["entity_id"])
        if new_idea_id and new_entity_id:
            try:
                db.execute("""
                    INSERT INTO idea_entities (idea_id, entity_id)
                    VALUES (?, ?)
                """, (new_idea_id, new_entity_id))
                entity_links_imported += 1
            except sqlite3.IntegrityError:
                pass  # Duplicate link

    db.commit()
    db.close()

    return {
        "success": True,
        "stats": {
            "spans_imported": len(span_id_map),
            "ideas_imported": ideas_imported,
            "relations_imported": relations_imported,
            "entities_imported": len(entity_id_map),
            "entity_links_imported": entity_links_imported,
        },
        "id_mappings": {
            "spans": span_id_map,
            "ideas": idea_id_map,
            "entities": entity_id_map,
        }
    }


def extract_session_from_path(file_path: str) -> str:
    """Extract session name from Claude transcript path.

    Paths look like: ~/.claude/projects/-Users-alice-my-project/xyz.jsonl
    The project dir encodes the actual path with dashes: /Users/alice/my-project
    Returns: 'my-project' (the project folder name)
    """
    import re
    match = re.search(r'/\.claude/projects/([^/]+)/', file_path)
    if match:
        project_dir = match.group(1)
        # The format is: -<path>-<segments>-<joined>-<by>-<dashes>
        # e.g., -Users-alice-my-project or -home-bob-code-app
        # We want to skip the home directory prefix

        # Common patterns: -Users-<username>- or -home-<username>-
        # Skip first 3 parts (empty, Users/home, username)
        patterns = [
            r'^-Users-[^-]+-(.+)$',  # macOS: -Users-alice-project
            r'^-home-[^-]+-(.+)$',   # Linux: -home-bob-project
        ]
        for pattern in patterns:
            m = re.match(pattern, project_dir)
            if m:
                return m.group(1)

        # Fallback: remove leading dash and return as-is
        return project_dir.lstrip('-')

    return Path(file_path).stem


def cwd_to_session(cwd: str) -> str | None:
    """Convert a working directory path to a session name.

    The session name matches how Claude encodes project paths:
    /Users/tom/rad/control-v1.1 -> rad-control-v1-1

    Args:
        cwd: Current working directory path

    Returns:
        Session name if it can be derived, None otherwise
    """
    import re
    from pathlib import Path

    # Normalize and expand the path
    cwd = str(Path(cwd).expanduser().resolve())

    # Strip home directory prefix
    home = str(Path.home())
    if cwd.startswith(home):
        # Get the relative path from home
        relative = cwd[len(home):].lstrip('/')
    else:
        # Not under home directory - use full path
        relative = cwd.lstrip('/')

    if not relative:
        return None

    # Convert path separators and dots to dashes (matching Claude's encoding)
    # /Users/tom/rad/control-v1.1 -> rad-control-v1-1
    # Note: Claude replaces / with - and . with -
    session = relative.replace('/', '-').replace('.', '-')

    # Clean up multiple dashes
    session = re.sub(r'-+', '-', session)
    session = session.strip('-')

    return session if session else None


def get_session_for_cwd(cwd: str) -> str | None:
    """Get the session name for a working directory, matching against known sessions.

    First tries exact match, then tries fuzzy matching for slight variations
    in how paths might be encoded.

    Args:
        cwd: Current working directory path

    Returns:
        Matched session name if found, None otherwise
    """
    derived = cwd_to_session(cwd)
    if not derived:
        return None

    # Get all known sessions
    sessions = list_sessions()
    session_names = [s["session"] for s in sessions]

    # Try exact match first
    if derived in session_names:
        return derived

    # Try case-insensitive match
    derived_lower = derived.lower()
    for name in session_names:
        if name.lower() == derived_lower:
            return name

    # Try suffix match (in case home dir encoding differs)
    # e.g., derived might be "projects-myapp" but session is "myapp"
    for name in session_names:
        if derived.endswith(name) or name.endswith(derived):
            return name

    # Try matching the last N segments
    derived_parts = derived.split('-')
    for name in session_names:
        name_parts = name.split('-')
        # Check if last 2+ parts match
        if len(derived_parts) >= 2 and len(name_parts) >= 2:
            if derived_parts[-2:] == name_parts[-2:]:
                return name
            if derived_parts[-1] == name_parts[-1] and len(name_parts) == 1:
                return name

    return None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: memory_db.py <command> [args]")
        print("Commands: init, store, search, search-spans, hybrid, hyde, stats")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "init":
        init_db()
        print("Database initialized")

    elif cmd == "store":
        if len(sys.argv) < 5:
            print("Usage: memory_db.py store <content> <source_file> <source_line> [span_id] [intent] [confidence]")
            sys.exit(1)
        content = sys.argv[2]
        source_file = sys.argv[3]
        source_line = int(sys.argv[4])
        span_id = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "null" else None
        intent = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "null" else None
        confidence = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
        idea_id = store_idea(content, source_file, source_line, span_id, intent, confidence)
        print(f"Stored idea {idea_id}")

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py search <query> [limit] [session] [intent]")
            sys.exit(1)
        query = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        session = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "null" else None
        intent = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "null" else None
        results = search_ideas(query, limit, session, intent)
        print(json.dumps(results, indent=2, default=str))

    elif cmd == "search-spans":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py search-spans <query> [limit]")
            sys.exit(1)
        query = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        results = search_spans(query, limit)
        print(json.dumps(results, indent=2, default=str))

    elif cmd == "hybrid":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py hybrid <query> [limit]")
            sys.exit(1)
        query = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        results = hybrid_search(query, limit)
        print(json.dumps(results, indent=2, default=str))

    elif cmd == "hyde":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py hyde <query> [limit]")
            sys.exit(1)
        query = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        results = hyde_search(query, limit)
        print(json.dumps(results, indent=2, default=str))

    elif cmd == "stats":
        stats = get_stats()
        print(json.dumps(stats, indent=2))

    elif cmd == "session-from-path":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py session-from-path <file_path>")
            sys.exit(1)
        print(extract_session_from_path(sys.argv[2]))

    elif cmd == "create-span":
        if len(sys.argv) < 5:
            print("Usage: memory_db.py create-span <session> <name> <start_line> [parent_id] [depth]")
            sys.exit(1)
        session = sys.argv[2]
        name = sys.argv[3]
        start_line = int(sys.argv[4])
        parent_id = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "null" else None
        depth = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        span_id = create_span(session, name, start_line, parent_id, depth)
        print(f"Created span {span_id}")

    elif cmd == "close-span":
        if len(sys.argv) < 5:
            print("Usage: memory_db.py close-span <span_id> <end_line> <summary>")
            sys.exit(1)
        span_id = int(sys.argv[2])
        end_line = int(sys.argv[3])
        summary = sys.argv[4]
        close_span(span_id, end_line, summary)
        print(f"Closed span {span_id}")

    elif cmd == "open-span":
        if len(sys.argv) < 3:
            print("Usage: memory_db.py open-span <session>")
            sys.exit(1)
        session = sys.argv[2]
        span = get_open_span(session)
        if span:
            print(json.dumps(span, indent=2, default=str))
        else:
            print("null")

    elif cmd == "list-spans":
        if len(sys.argv) < 2:
            print("Usage: memory_db.py list-spans [session]")
            sys.exit(1)
        session = sys.argv[2] if len(sys.argv) > 2 else None
        db = get_db()
        if session:
            cursor = db.execute("""
                SELECT id, session, parent_id, name, summary, start_line, end_line, depth, created_at
                FROM spans WHERE session = ? ORDER BY start_line
            """, (session,))
        else:
            cursor = db.execute("""
                SELECT id, session, parent_id, name, summary, start_line, end_line, depth, created_at
                FROM spans ORDER BY session, start_line
            """)
        results = [dict(row) for row in cursor]
        db.close()
        print(json.dumps(results, indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
