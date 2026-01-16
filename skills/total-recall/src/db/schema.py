"""Database schema and initialization for total-recall."""

import sqlite3

import config
from db.connection import get_db
from errors import TotalRecallError


# Schema SQL for all tables
SCHEMA_SQL = """
    -- Projects (high-level groupings of related topics)
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Topics (conceptual subjects that can span multiple conversation sections)
    -- Note: project_id may be added by migration for existing databases
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        canonical_name TEXT UNIQUE,
        summary TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Spans (physical sections of transcript, linked to topics)
    -- Note: topic_id may be added by migration for existing databases
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
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ideas_source_unique ON ideas(source_file, source_line);

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

    -- Topic links for cross-session topic relationships
    CREATE TABLE IF NOT EXISTS topic_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
        related_topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
        similarity REAL NOT NULL,
        time_overlap BOOLEAN DEFAULT FALSE,
        link_type TEXT DEFAULT 'semantic' CHECK(link_type IN (
            'semantic', 'manual', 'merged'
        )),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(topic_id, related_topic_id)
    );

    CREATE INDEX IF NOT EXISTS idx_topic_links_topic ON topic_links(topic_id);
    CREATE INDEX IF NOT EXISTS idx_topic_links_related ON topic_links(related_topic_id);

    -- Raw messages from transcripts (full content for FTS and RAG)
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT NOT NULL,
        line_num INTEGER NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        timestamp TEXT,
        source_file TEXT NOT NULL,
        UNIQUE(source_file, line_num)
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session);
    CREATE INDEX IF NOT EXISTS idx_messages_source ON messages(source_file);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

    -- Index state tracking (byte position for efficient seeking)
    CREATE TABLE IF NOT EXISTS index_state (
        file_path TEXT PRIMARY KEY,
        byte_position INTEGER DEFAULT 0,
        last_indexed TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Work queue for daemon processing
    CREATE TABLE IF NOT EXISTS work_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        queued_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_work_queue_file ON work_queue(file_path);

    -- Working memory (recently activated ideas with decay)
    CREATE TABLE IF NOT EXISTS working_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT NOT NULL,
        idea_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
        activation REAL DEFAULT 1.0 CHECK(activation >= 0 AND activation <= 1),
        last_access TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(session, idea_id)
    );

    CREATE INDEX IF NOT EXISTS idx_working_memory_session ON working_memory(session);
    CREATE INDEX IF NOT EXISTS idx_working_memory_activation ON working_memory(activation);

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

    -- Vector embeddings for messages
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id INTEGER PRIMARY KEY,
        embedding FLOAT[1536]
    );
"""


def init_db():
    """Initialize the database schema."""
    try:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        config.logger.error(f"Failed to create database directory: {e}")
        raise TotalRecallError(
            f"Cannot create database directory: {e}",
            "database_error",
            {"path": str(config.DB_PATH.parent), "original_error": str(e)}
        ) from e

    db = get_db()

    # Create tables
    db.executescript(SCHEMA_SQL)

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

    try:
        db.execute("""
            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content,
                content='messages',
                content_rowid='id'
            )
        """)
    except sqlite3.OperationalError:
        pass  # Already exists

    db.commit()

    # Run migrations - import here to avoid circular dependency
    from db.migrations import migrate_schema, migrate_spans_to_topics

    migrate_schema(db)
    migrate_spans_to_topics(db)

    db.close()
