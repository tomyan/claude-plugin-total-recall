"""Action executor - stores ideas, updates topics, creates relations."""

from typing import Any

from db.connection import get_db


def execute_ideas(
    items: list[dict[str, Any]],
    span_id: int | None,
    source_file: str
) -> list[int]:
    """
    Store ideas from parsed LLM response.

    Args:
        items: List of idea items from LLM response
        span_id: Current span ID to link ideas to
        source_file: Source transcript file path

    Returns:
        List of created/updated idea IDs
    """
    db = get_db()
    idea_ids = []

    for item in items:
        content = item.get("content")
        intent = item.get("type", "context")
        source_line = item.get("source_line")
        confidence = item.get("confidence", 0.5)
        entities = item.get("entities", [])

        # Skip malformed items
        if not content or not source_line:
            continue

        # Upsert idea (update if exists, insert if not)
        cursor = db.execute("""
            INSERT INTO ideas (span_id, content, intent, confidence, source_file, source_line)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file, source_line) DO UPDATE SET
                content = excluded.content,
                intent = excluded.intent,
                confidence = excluded.confidence,
                span_id = COALESCE(excluded.span_id, span_id)
        """, (span_id, content, intent, confidence, source_file, source_line))

        # Get the idea ID
        cursor = db.execute("""
            SELECT id FROM ideas WHERE source_file = ? AND source_line = ?
        """, (source_file, source_line))
        idea_id = cursor.fetchone()["id"]
        idea_ids.append(idea_id)

        # Store entities
        for entity_name in entities:
            # Get or create entity
            cursor = db.execute("""
                INSERT INTO entities (name, type)
                VALUES (?, 'concept')
                ON CONFLICT(name, type) DO UPDATE SET name = name
            """, (entity_name,))

            cursor = db.execute("""
                SELECT id FROM entities WHERE name = ? AND type = 'concept'
            """, (entity_name,))
            entity_id = cursor.fetchone()["id"]

            # Link idea to entity
            db.execute("""
                INSERT OR IGNORE INTO idea_entities (idea_id, entity_id)
                VALUES (?, ?)
            """, (idea_id, entity_id))

    db.commit()
    db.close()
    return idea_ids


def execute_topic_update(
    topic_update: dict[str, str],
    span_id: int
) -> None:
    """
    Update span name/summary and link to topic.

    Args:
        topic_update: Dict with 'name' and 'summary'
        span_id: Span ID to update
    """
    db = get_db()

    name = topic_update.get("name")
    summary = topic_update.get("summary")

    # Update span
    if name or summary:
        if name and summary:
            db.execute("""
                UPDATE spans SET name = ?, summary = ? WHERE id = ?
            """, (name, summary, span_id))
        elif name:
            db.execute("UPDATE spans SET name = ? WHERE id = ?", (name, span_id))
        else:
            db.execute("UPDATE spans SET summary = ? WHERE id = ?", (summary, span_id))

    # Find or create topic
    if name:
        canonical = name.lower().strip()[:50]

        cursor = db.execute("""
            SELECT id FROM topics WHERE canonical_name = ?
        """, (canonical,))
        row = cursor.fetchone()

        if row:
            topic_id = row["id"]
        else:
            cursor = db.execute("""
                INSERT INTO topics (name, canonical_name, summary)
                VALUES (?, ?, ?)
            """, (name[:100], canonical, summary))
            topic_id = cursor.lastrowid

        # Link span to topic
        db.execute("UPDATE spans SET topic_id = ? WHERE id = ?", (topic_id, span_id))

    db.commit()
    db.close()


def execute_new_span(
    new_span: dict[str, str],
    session: str,
    parent_id: int | None,
    start_line: int
) -> int:
    """
    Create a new child span for topic shift.

    Args:
        new_span: Dict with 'name' and 'reason'
        session: Session identifier
        parent_id: Parent span ID
        start_line: Starting line number

    Returns:
        New span ID
    """
    db = get_db()

    name = new_span.get("name", "New Span")
    reason = new_span.get("reason", "")

    # Get parent depth
    depth = 0
    if parent_id:
        cursor = db.execute("SELECT depth FROM spans WHERE id = ?", (parent_id,))
        row = cursor.fetchone()
        if row:
            depth = row["depth"] + 1

    cursor = db.execute("""
        INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session, name, reason, start_line, depth, parent_id))

    span_id = cursor.lastrowid
    db.commit()
    db.close()

    return span_id


def execute_relations(
    relations: list[dict[str, Any]],
    source_file: str
) -> int:
    """
    Create relations between ideas.

    Args:
        relations: List of relation dicts with from_line, to_idea_id, type
        source_file: Source file to find ideas by line

    Returns:
        Number of relations created
    """
    db = get_db()
    created = 0

    for rel in relations:
        from_line = rel.get("from_line")
        to_idea_id = rel.get("to_idea_id")
        rel_type = rel.get("type", "related")

        # Skip malformed relations
        if not from_line or not to_idea_id:
            continue

        # Find source idea
        cursor = db.execute("""
            SELECT id FROM ideas WHERE source_file = ? AND source_line = ?
        """, (source_file, from_line))
        row = cursor.fetchone()

        if not row:
            continue

        from_id = row["id"]

        # Verify target exists
        cursor = db.execute("SELECT id FROM ideas WHERE id = ?", (to_idea_id,))
        if not cursor.fetchone():
            continue

        # Create relation
        try:
            db.execute("""
                INSERT INTO relations (from_id, to_id, relation_type)
                VALUES (?, ?, ?)
            """, (from_id, to_idea_id, rel_type))
            created += 1
        except Exception:
            # Duplicate or other error
            pass

    db.commit()
    db.close()

    return created


def embed_ideas(idea_ids: list[int]) -> int:
    """
    Generate embeddings for ideas.

    Args:
        idea_ids: List of idea IDs to embed

    Returns:
        Number of ideas embedded
    """
    import os
    if not idea_ids:
        return 0

    api_key = os.environ.get("OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS")
    if not api_key:
        # Silently skip embedding if no API key - will error on search
        return 0

    try:
        from embeddings.openai import OpenAIEmbeddings
        provider = OpenAIEmbeddings()
    except Exception:
        return 0

    db = get_db()
    embedded = 0

    for idea_id in idea_ids:
        # Get idea content
        cursor = db.execute("SELECT content FROM ideas WHERE id = ?", (idea_id,))
        row = cursor.fetchone()
        if not row:
            continue

        # Check if already embedded
        cursor = db.execute("SELECT 1 FROM idea_embeddings WHERE idea_id = ?", (idea_id,))
        if cursor.fetchone():
            continue

        # Generate embedding
        try:
            embedding = provider.get_embedding(row["content"])

            # Store embedding
            import json
            db.execute("""
                INSERT INTO idea_embeddings (idea_id, embedding)
                VALUES (?, ?)
            """, (idea_id, json.dumps(embedding)))
            embedded += 1
        except Exception:
            continue

    db.commit()
    db.close()

    return embedded


def embed_messages(message_ids: list[int]) -> int:
    """
    Generate embeddings for messages.

    Args:
        message_ids: List of message IDs to embed

    Returns:
        Number of messages embedded
    """
    import os
    if not message_ids:
        return 0

    api_key = os.environ.get("OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS")
    if not api_key:
        return 0

    try:
        from embeddings.openai import OpenAIEmbeddings
        provider = OpenAIEmbeddings()
    except Exception:
        return 0

    db = get_db()
    embedded = 0

    for message_id in message_ids:
        # Get message content
        cursor = db.execute("SELECT content FROM messages WHERE id = ?", (message_id,))
        row = cursor.fetchone()
        if not row:
            continue

        # Check if already embedded
        cursor = db.execute("SELECT 1 FROM message_embeddings WHERE message_id = ?", (message_id,))
        if cursor.fetchone():
            continue

        # Generate embedding
        try:
            embedding = provider.get_embedding(row["content"])

            # Store embedding
            import json
            db.execute("""
                INSERT INTO message_embeddings (message_id, embedding)
                VALUES (?, ?)
            """, (message_id, json.dumps(embedding)))
            embedded += 1
        except Exception:
            continue

    db.commit()
    db.close()

    return embedded
