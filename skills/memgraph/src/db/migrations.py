"""Database migrations for memgraph."""

import json
import sqlite3
from pathlib import Path

import config
from db.connection import get_db


def migrate_schema(db):
    """Add new columns and indexes to existing tables."""
    # Check if topic_id column exists in spans
    cursor = db.execute("PRAGMA table_info(spans)")
    span_columns = [row[1] for row in cursor]

    if "topic_id" not in span_columns:
        config.logger.info("Adding topic_id column to spans table")
        db.execute("ALTER TABLE spans ADD COLUMN topic_id INTEGER REFERENCES topics(id)")
        db.commit()

    # Add start_time/end_time to spans for temporal tracking
    if "start_time" not in span_columns:
        config.logger.info("Adding start_time column to spans table")
        db.execute("ALTER TABLE spans ADD COLUMN start_time TEXT")
        db.commit()

    if "end_time" not in span_columns:
        config.logger.info("Adding end_time column to spans table")
        db.execute("ALTER TABLE spans ADD COLUMN end_time TEXT")
        db.commit()

    # Check if project_id column exists in topics
    cursor = db.execute("PRAGMA table_info(topics)")
    topic_columns = [row[1] for row in cursor]

    if "project_id" not in topic_columns:
        config.logger.info("Adding project_id column to topics table")
        db.execute("ALTER TABLE topics ADD COLUMN project_id INTEGER REFERENCES projects(id)")
        db.commit()

    if "parent_id" not in topic_columns:
        config.logger.info("Adding parent_id column to topics table for hierarchy")
        db.execute("ALTER TABLE topics ADD COLUMN parent_id INTEGER REFERENCES topics(id)")
        db.commit()

    # Add first_seen/last_seen to topics for temporal tracking
    if "first_seen" not in topic_columns:
        config.logger.info("Adding first_seen column to topics table")
        db.execute("ALTER TABLE topics ADD COLUMN first_seen TEXT")
        db.commit()

    if "last_seen" not in topic_columns:
        config.logger.info("Adding last_seen column to topics table")
        db.execute("ALTER TABLE topics ADD COLUMN last_seen TEXT")
        db.commit()

    # Check if message_time column exists in ideas
    cursor = db.execute("PRAGMA table_info(ideas)")
    idea_columns = [row[1] for row in cursor]

    if "message_time" not in idea_columns:
        config.logger.info("Adding message_time column to ideas table")
        db.execute("ALTER TABLE ideas ADD COLUMN message_time TEXT")
        db.commit()

    # Add access tracking columns for working memory / forgetting
    if "access_count" not in idea_columns:
        config.logger.info("Adding access_count column to ideas table")
        db.execute("ALTER TABLE ideas ADD COLUMN access_count INTEGER DEFAULT 0")
        db.commit()

    if "last_accessed" not in idea_columns:
        config.logger.info("Adding last_accessed column to ideas table")
        db.execute("ALTER TABLE ideas ADD COLUMN last_accessed TEXT")
        db.commit()

    # Create indexes that depend on migrated columns
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_spans_topic ON spans(topic_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_topics_canonical ON topics(canonical_name)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_topics_project ON topics(project_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_ideas_message_time ON ideas(message_time)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time)")
        db.commit()
    except sqlite3.OperationalError:
        pass  # Index already exists


def migrate_spans_to_topics(db):
    """Migrate existing spans without topic_id to topics."""
    # Check if there are spans without topic_id
    cursor = db.execute("""
        SELECT id, name, summary FROM spans
        WHERE topic_id IS NULL
    """)
    orphan_spans = list(cursor)

    if not orphan_spans:
        return

    config.logger.info(f"Migrating {len(orphan_spans)} spans to topics")

    for span in orphan_spans:
        span_id = span["id"]
        name = span["name"]
        summary = span["summary"]

        # Canonicalize and find/create topic
        canonical = name.lower().strip()[:50]

        # Check for existing topic
        cursor = db.execute(
            "SELECT id FROM topics WHERE canonical_name = ?",
            (canonical,)
        )
        row = cursor.fetchone()

        if row:
            topic_id = row["id"]
        else:
            # Create new topic
            cursor = db.execute(
                "INSERT INTO topics (name, canonical_name, summary) VALUES (?, ?, ?)",
                (name[:100], canonical, summary)
            )
            topic_id = cursor.lastrowid

        # Link span to topic
        db.execute("UPDATE spans SET topic_id = ? WHERE id = ?", (topic_id, span_id))

    db.commit()
    config.logger.info(f"Migration complete: {len(orphan_spans)} spans linked to topics")


def migrate_timestamps_from_transcripts() -> dict:
    """Migrate existing ideas/spans to use real timestamps from transcripts.

    Reads transcript files and updates:
    - ideas.message_time from transcript message timestamps
    - spans.start_time/end_time from first/last message timestamps

    Returns:
        Dict with migration stats
    """
    db = get_db()

    # Find all ideas without message_time that have a source_file
    cursor = db.execute("""
        SELECT DISTINCT source_file FROM ideas
        WHERE message_time IS NULL AND source_file IS NOT NULL
    """)
    files_to_process = [row["source_file"] for row in cursor]

    if not files_to_process:
        db.close()
        return {"files_processed": 0, "ideas_updated": 0, "spans_updated": 0}

    ideas_updated = 0
    spans_updated = 0

    for file_path in files_to_process:
        if not Path(file_path).exists():
            config.logger.warning(f"Transcript file not found: {file_path}")
            continue

        # Read transcript and build line -> timestamp map
        line_timestamps = {}
        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        timestamp = data.get("timestamp", "")
                        if timestamp:
                            line_timestamps[line_num] = timestamp
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            config.logger.warning(f"Error reading transcript {file_path}: {e}")
            continue

        if not line_timestamps:
            continue

        # Update ideas with timestamps
        cursor = db.execute("""
            SELECT id, source_line FROM ideas
            WHERE source_file = ? AND message_time IS NULL
        """, (file_path,))

        for row in cursor:
            idea_id = row["id"]
            source_line = row["source_line"]
            if source_line in line_timestamps:
                db.execute(
                    "UPDATE ideas SET message_time = ? WHERE id = ?",
                    (line_timestamps[source_line], idea_id)
                )
                ideas_updated += 1

        db.commit()

    # Update spans with start_time/end_time from their ideas
    cursor = db.execute("""
        SELECT s.id, MIN(i.message_time) as start_t, MAX(i.message_time) as end_t
        FROM spans s
        JOIN ideas i ON i.span_id = s.id
        WHERE s.start_time IS NULL AND i.message_time IS NOT NULL
        GROUP BY s.id
    """)

    for row in cursor:
        span_id = row["id"]
        start_t = row["start_t"]
        end_t = row["end_t"]
        if start_t or end_t:
            db.execute(
                "UPDATE spans SET start_time = ?, end_time = ? WHERE id = ?",
                (start_t, end_t, span_id)
            )
            spans_updated += 1

    # Update topics with first_seen/last_seen from their spans
    db.execute("""
        UPDATE topics SET
            first_seen = (
                SELECT MIN(s.start_time) FROM spans s
                WHERE s.topic_id = topics.id AND s.start_time IS NOT NULL
            ),
            last_seen = (
                SELECT MAX(s.end_time) FROM spans s
                WHERE s.topic_id = topics.id AND s.end_time IS NOT NULL
            )
        WHERE first_seen IS NULL OR last_seen IS NULL
    """)

    db.commit()
    db.close()

    config.logger.info(f"Timestamp migration: {len(files_to_process)} files, {ideas_updated} ideas, {spans_updated} spans")
    return {
        "files_processed": len(files_to_process),
        "ideas_updated": ideas_updated,
        "spans_updated": spans_updated
    }
