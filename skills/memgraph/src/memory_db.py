#!/usr/bin/env python3
"""
Memory database for Claude memory graph skill.
Uses SQLite + sqlite-vec for vector similarity search.
Uses OpenAI for embeddings.
"""

import json
import os
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Optional

import sqlite_vec
from openai import OpenAI


# Database location
DB_PATH = Path.home() / ".claude-plugin-memgraph" / "memory.db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_db() -> sqlite3.Connection:
    """Get database connection with sqlite-vec loaded."""
    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    """Initialize the database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = get_db()

    db.executescript("""
        -- Hierarchical spans (topics and sub-topics)
        CREATE TABLE IF NOT EXISTS spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session TEXT NOT NULL,
            parent_id INTEGER REFERENCES spans(id),
            name TEXT NOT NULL,
            summary TEXT,
            start_line INTEGER NOT NULL,
            end_line INTEGER,
            depth INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session);
        CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_id);

        -- Atomic ideas/insights
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            span_id INTEGER REFERENCES spans(id),
            content TEXT NOT NULL,
            intent TEXT CHECK(intent IN (
                'decision', 'conclusion', 'question', 'problem',
                'solution', 'todo', 'context'
            )),
            confidence REAL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
            answered BOOLEAN,
            source_file TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ideas_span ON ideas(span_id);
        CREATE INDEX IF NOT EXISTS idx_ideas_intent ON ideas(intent);

        -- Named entities
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT CHECK(type IN (
                'project', 'technology', 'concept', 'person', 'file'
            )),
            UNIQUE(name, type)
        );

        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

        -- Idea-entity links
        CREATE TABLE IF NOT EXISTS idea_entities (
            idea_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (idea_id, entity_id)
        );

        -- Relations between ideas
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
            to_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
            relation_type TEXT CHECK(relation_type IN (
                'supersedes', 'builds_on', 'contradicts', 'answers', 'relates_to'
            )),
            UNIQUE(from_id, to_id, relation_type)
        );

        CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id);
        CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_id);

        -- Index state tracking
        CREATE TABLE IF NOT EXISTS index_state (
            file_path TEXT PRIMARY KEY,
            last_line INTEGER DEFAULT 0,
            last_indexed TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Vector embeddings for spans
        CREATE VIRTUAL TABLE IF NOT EXISTS span_embeddings USING vec0(
            span_id INTEGER PRIMARY KEY,
            embedding FLOAT[1536]
        );

        -- Vector embeddings for ideas
        CREATE VIRTUAL TABLE IF NOT EXISTS idea_embeddings USING vec0(
            idea_id INTEGER PRIMARY KEY,
            embedding FLOAT[1536]
        );
    """)

    # Create FTS tables separately (can't be in executescript with IF NOT EXISTS issues)
    try:
        db.execute("""
            CREATE VIRTUAL TABLE ideas_fts USING fts5(
                content,
                content='ideas',
                content_rowid='id'
            )
        """)
    except sqlite3.OperationalError:
        pass  # Already exists

    try:
        db.execute("""
            CREATE VIRTUAL TABLE spans_fts USING fts5(
                name,
                summary,
                content='spans',
                content_rowid='id'
            )
        """)
    except sqlite3.OperationalError:
        pass  # Already exists

    db.commit()
    db.close()


def serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize embedding to bytes for sqlite-vec."""
    return struct.pack(f'{len(embedding)}f', *embedding)


# LRU cache for embeddings to reduce API calls
_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX_SIZE = 1000


def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    """Get embedding from OpenAI with caching.

    Args:
        text: Text to embed
        use_cache: Whether to use cache (default True)

    Returns:
        Embedding vector (1536 floats)
    """
    # Check cache first
    if use_cache and text in _embedding_cache:
        return _embedding_cache[text]

    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key:
        raise ValueError("OPENAI_TOKEN_MEMORY_EMBEDDINGS environment variable not set")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embedding = response.data[0].embedding

    # Cache the result (with simple LRU eviction)
    if use_cache:
        if len(_embedding_cache) >= _CACHE_MAX_SIZE:
            # Remove oldest entry (first key)
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]
        _embedding_cache[text] = embedding

    return embedding


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}


def get_embedding_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_embedding_cache),
        "max_size": _CACHE_MAX_SIZE,
    }


# =============================================================================
# Span Operations
# =============================================================================

def create_span(
    session: str,
    name: str,
    start_line: int,
    parent_id: Optional[int] = None,
    depth: int = 0
) -> int:
    """Create a new span."""
    db = get_db()
    cursor = db.execute("""
        INSERT INTO spans (session, parent_id, name, start_line, depth)
        VALUES (?, ?, ?, ?, ?)
    """, (session, parent_id, name, start_line, depth))
    span_id = cursor.lastrowid
    db.commit()
    db.close()
    return span_id


def close_span(span_id: int, end_line: int, summary: str):
    """Close a span with summary and embed it."""
    db = get_db()

    # Update span
    db.execute("""
        UPDATE spans SET end_line = ?, summary = ? WHERE id = ?
    """, (end_line, summary, span_id))

    # Get span for embedding
    cursor = db.execute("SELECT name, summary FROM spans WHERE id = ?", (span_id,))
    row = cursor.fetchone()

    # Embed and store
    embed_text = f"{row['name']}: {row['summary']}"
    embedding = get_embedding(embed_text)
    db.execute("""
        INSERT OR REPLACE INTO span_embeddings (span_id, embedding)
        VALUES (?, ?)
    """, (span_id, serialize_embedding(embedding)))

    # Update FTS
    db.execute("""
        INSERT INTO spans_fts (rowid, name, summary)
        VALUES (?, ?, ?)
    """, (span_id, row['name'], row['summary']))

    db.commit()
    db.close()


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
# Idea Operations
# =============================================================================

def store_idea(
    content: str,
    source_file: str,
    source_line: int,
    span_id: Optional[int] = None,
    intent: Optional[str] = None,
    confidence: float = 0.5,
    entities: Optional[list[tuple[str, str]]] = None  # [(name, type), ...]
) -> int:
    """Store an idea with its embedding."""
    db = get_db()
    cursor = db.cursor()

    # Insert idea
    cursor.execute("""
        INSERT INTO ideas (span_id, content, intent, confidence, source_file, source_line)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (span_id, content, intent, confidence, source_file, source_line))
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


# =============================================================================
# Search Operations
# =============================================================================

def search_ideas(
    query: str,
    limit: int = 10,
    session: Optional[str] = None,
    intent: Optional[str] = None,
    recency_weight: float = 0.0
) -> list[dict]:
    """Search for similar ideas using vector similarity."""
    db = get_db()
    query_embedding = get_embedding(query)

    # Vector search with larger k for filtering
    k = limit * 3 if (session or intent) else limit
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
        ORDER BY e.distance
    """, (serialize_embedding(query_embedding), k))

    results = []
    for row in cursor:
        # Apply filters
        if session and row['session'] != session:
            continue
        if intent and row['intent'] != intent:
            continue

        results.append(dict(row))
        if len(results) >= limit:
            break

    db.close()
    return results


def search_spans(query: str, limit: int = 5) -> list[dict]:
    """Search for similar topic spans."""
    db = get_db()
    query_embedding = get_embedding(query)

    cursor = db.execute("""
        SELECT
            s.id, s.session, s.name, s.summary,
            s.start_line, s.end_line, s.depth, s.created_at,
            e.distance
        FROM span_embeddings e
        JOIN spans s ON s.id = e.span_id
        WHERE e.embedding MATCH ? AND k = ?
        ORDER BY e.distance
    """, (serialize_embedding(query_embedding), limit))

    results = [dict(row) for row in cursor]
    db.close()
    return results


def hybrid_search(query: str, limit: int = 10) -> list[dict]:
    """Hybrid search combining vector similarity and BM25."""
    db = get_db()
    query_embedding = get_embedding(query)

    # Vector search
    vector_results = {}
    cursor = db.execute("""
        SELECT idea_id, distance
        FROM idea_embeddings
        WHERE embedding MATCH ? AND k = ?
    """, (serialize_embedding(query_embedding), limit * 2))
    for i, row in enumerate(cursor):
        vector_results[row['idea_id']] = i + 1  # rank

    # BM25 search
    bm25_results = {}
    cursor = db.execute("""
        SELECT rowid, bm25(ideas_fts) as score
        FROM ideas_fts
        WHERE ideas_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, limit * 2))
    for i, row in enumerate(cursor):
        bm25_results[row['rowid']] = i + 1  # rank

    # Reciprocal Rank Fusion
    k = 60  # RRF constant
    scores = {}
    all_ids = set(vector_results.keys()) | set(bm25_results.keys())
    for idea_id in all_ids:
        score = 0
        if idea_id in vector_results:
            score += 1 / (k + vector_results[idea_id])
        if idea_id in bm25_results:
            score += 1 / (k + bm25_results[idea_id])
        scores[idea_id] = score

    # Get top results
    top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

    if not top_ids:
        db.close()
        return []

    # Fetch full records
    placeholders = ','.join('?' * len(top_ids))
    cursor = db.execute(f"""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic
        FROM ideas i
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE i.id IN ({placeholders})
    """, top_ids)

    results = {row['id']: dict(row) for row in cursor}
    db.close()

    # Return in ranked order
    return [results[id] for id in top_ids if id in results]


# =============================================================================
# Advanced Retrieval
# =============================================================================

def generate_hypothetical_doc(query: str) -> str:
    """Generate a hypothetical document that would answer the query.

    Used for HyDE (Hypothetical Document Embeddings) to improve retrieval.

    Args:
        query: The user's search query

    Returns:
        A hypothetical answer document
    """
    # TODO: Use LLM to generate better hypothetical docs
    # For now, expand the query into a statement
    query_lower = query.lower().strip()

    # Remove question words and rephrase as statement
    for prefix in ["how ", "what ", "why ", "when ", "where ", "which ", "who "]:
        if query_lower.startswith(prefix):
            query_lower = query_lower[len(prefix):]
            break

    # Remove trailing question mark
    query_lower = query_lower.rstrip("?")

    # Create a hypothetical answer
    return f"The answer involves {query_lower}. This was decided based on careful consideration of the requirements and constraints."


def hyde_search(query: str, limit: int = 10) -> list[dict]:
    """HyDE search - generates hypothetical answer then searches.

    For vague queries, this often retrieves better matches than raw query.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of matching idea dicts
    """
    # Generate hypothetical document
    hypothetical = generate_hypothetical_doc(query)

    # Embed the hypothetical document
    hypo_embedding = get_embedding(hypothetical)

    # Search with hypothetical embedding
    db = get_db()
    cursor = db.execute("""
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
        ORDER BY e.distance
    """, (serialize_embedding(hypo_embedding), limit))

    results = [dict(row) for row in cursor]
    db.close()
    return results


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


def search_ideas_temporal(
    query: str,
    limit: int = 10,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list[dict]:
    """Search ideas with temporal filtering.

    Args:
        query: Search query
        limit: Maximum results
        since: ISO datetime string for start of range
        until: ISO datetime string for end of range

    Returns:
        List of matching idea dicts
    """
    db = get_db()
    query_embedding = get_embedding(query)

    # Build query with temporal filter
    sql = """
        SELECT
            i.id, i.content, i.intent, i.confidence,
            i.source_file, i.source_line, i.created_at,
            s.session, s.name as topic,
            e.distance
        FROM idea_embeddings e
        JOIN ideas i ON i.id = e.idea_id
        LEFT JOIN spans s ON s.id = i.span_id
        WHERE e.embedding MATCH ? AND k = ?
    """
    params = [serialize_embedding(query_embedding), limit * 2]

    if since:
        sql += " AND i.created_at >= ?"
        params.append(since)
    if until:
        sql += " AND i.created_at <= ?"
        params.append(until)

    sql += " ORDER BY e.distance LIMIT ?"
    params.append(limit)

    cursor = db.execute(sql, params)
    results = [dict(row) for row in cursor]
    db.close()
    return results


def analyze_query(query: str) -> dict:
    """Analyze a query to extract filters and entities.

    Args:
        query: Search query

    Returns:
        Dict with temporal, intent_filter, entities
    """
    import re
    query_lower = query.lower()
    result = {}

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

    db.close()
    return stats


def extract_session_from_path(file_path: str) -> str:
    """Extract session name from Claude transcript path.

    Paths look like: ~/.claude/projects/-Users-tom-rad-control-v1-1/xyz.jsonl
    The project dir encodes the actual path with dashes: /Users/tom/rad-control-v1-1
    Returns: 'rad-control-v1-1' (the project folder name)
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
