"""Indexer executor - Slices 4.2-4.9.

Executes parsed agent output by storing ideas, updating spans,
creating relations, and managing entity links.
"""

import sqlite3
from typing import Any, Optional

from db.connection import get_db
from indexer.output_parser import (
    AgentOutput,
    IdeaOutput,
    TopicUpdate,
    TopicChange,
    AnsweredQuestion,
    CompletedTodo,
    RelationOutput,
    EntityLink,
)


async def execute_ideas(
    ideas: list[IdeaOutput],
    session: str,
    source_file: str,
    span_id: int,
) -> list[int]:
    """Store ideas from agent output.

    Args:
        ideas: List of parsed idea outputs
        session: Session ID
        source_file: Source transcript file
        span_id: Span ID to associate ideas with

    Returns:
        List of created idea IDs
    """
    db = get_db()
    created_ids = []

    for idea in ideas:
        try:
            cursor = db.execute("""
                INSERT INTO ideas (
                    content, intent, source_file, source_line,
                    span_id, session, confidence, importance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idea.content,
                idea.intent,
                source_file,
                idea.source_line,
                span_id,
                session,
                idea.confidence,
                idea.importance,
            ))
            created_ids.append(cursor.lastrowid)
        except sqlite3.IntegrityError:
            # Duplicate source_file/source_line - skip
            pass

    db.commit()
    db.close()

    return created_ids


async def execute_topic_updates(updates: list[TopicUpdate]) -> None:
    """Update spans from agent output.

    Args:
        updates: List of topic updates
    """
    db = get_db()

    for update in updates:
        if update.name is not None and update.summary is not None:
            db.execute("""
                UPDATE spans SET name = ?, summary = ?
                WHERE id = ?
            """, (update.name, update.summary, update.span_id))
        elif update.name is not None:
            db.execute("""
                UPDATE spans SET name = ?
                WHERE id = ?
            """, (update.name, update.span_id))
        elif update.summary is not None:
            db.execute("""
                UPDATE spans SET summary = ?
                WHERE id = ?
            """, (update.summary, update.span_id))

    db.commit()
    db.close()


async def execute_topic_changes(
    changes: list[TopicChange],
    session: str,
) -> list[int]:
    """Create new spans for topic shifts.

    Args:
        changes: List of topic changes
        session: Session ID

    Returns:
        List of created span IDs
    """
    db = get_db()
    created_ids = []

    for change in changes:
        # Get parent span's depth
        cursor = db.execute(
            "SELECT depth FROM spans WHERE id = ?",
            (change.from_span_id,)
        )
        parent = cursor.fetchone()
        depth = (parent["depth"] + 1) if parent else 0

        cursor = db.execute("""
            INSERT INTO spans (session, name, summary, start_line, depth, parent_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session,
            change.new_name,
            change.reason,
            change.at_line,
            depth,
            change.from_span_id,
        ))
        created_ids.append(cursor.lastrowid)

    db.commit()
    db.close()

    return created_ids


async def execute_answered_questions(answers: list[AnsweredQuestion]) -> None:
    """Mark questions as answered.

    Args:
        answers: List of answered questions
    """
    db = get_db()

    for answer in answers:
        db.execute("""
            UPDATE ideas SET answered = TRUE
            WHERE id = ? AND intent = 'question'
        """, (answer.question_id,))

    db.commit()
    db.close()


async def execute_completed_todos(todos: list[CompletedTodo]) -> None:
    """Mark todos as completed.

    Args:
        todos: List of completed todos
    """
    db = get_db()

    for todo in todos:
        db.execute("""
            UPDATE ideas SET completed = TRUE
            WHERE id = ? AND intent = 'todo'
        """, (todo.todo_id,))

    db.commit()
    db.close()


async def execute_relations(
    relations: list[RelationOutput],
    source_file: str,
    idea_line_map: dict[int, int],
) -> int:
    """Create relations between ideas.

    Args:
        relations: List of relations to create
        source_file: Source file path
        idea_line_map: Map from source_line to idea_id

    Returns:
        Count of relations created
    """
    db = get_db()
    count = 0

    for rel in relations:
        from_id = idea_line_map.get(rel.from_line)
        if from_id is None:
            continue

        try:
            db.execute("""
                INSERT INTO relations (from_id, to_id, relation_type)
                VALUES (?, ?, ?)
            """, (from_id, rel.to_idea_id, rel.relation_type))
            count += 1
        except sqlite3.IntegrityError:
            # Duplicate relation - skip
            pass

    db.commit()
    db.close()

    return count


async def execute_entity_links(
    links: list[EntityLink],
    source_file: str,
    idea_line_map: dict[int, int],
) -> None:
    """Link entity mentions to golden entities.

    Args:
        links: List of entity links
        source_file: Source file path
        idea_line_map: Map from line to idea_id
    """
    from entities import link_mention_to_golden

    for link in links:
        # Find mention by source location
        # This is a simplified version - full implementation would track mentions
        pass


async def execute_activated_ideas(
    idea_ids: list[int],
    session: str,
) -> None:
    """Update working memory activations.

    Args:
        idea_ids: List of idea IDs to activate
        session: Session ID
    """
    db = get_db()

    for idea_id in idea_ids:
        # Upsert into working memory
        db.execute("""
            INSERT INTO working_memory (session, idea_id, activation, last_access)
            VALUES (?, ?, 1.0, datetime('now'))
            ON CONFLICT(session, idea_id) DO UPDATE SET
                activation = MIN(1.0, activation + 0.2),
                last_access = datetime('now')
        """, (session, idea_id))

    db.commit()
    db.close()


async def generate_embeddings(idea_ids: list[int]) -> None:
    """Generate embeddings for new ideas.

    Args:
        idea_ids: List of idea IDs to generate embeddings for
    """
    # Import here to avoid circular imports
    from embeddings import get_embeddings_batch
    from db.connection import get_db

    if not idea_ids:
        return

    db = get_db()

    # Get idea content
    placeholders = ",".join("?" * len(idea_ids))
    cursor = db.execute(f"""
        SELECT id, content FROM ideas WHERE id IN ({placeholders})
    """, idea_ids)

    ideas = list(cursor)
    db.close()

    if not ideas:
        return

    # Generate embeddings
    texts = [idea["content"] for idea in ideas]
    try:
        embeddings = await get_embeddings_batch(texts)

        # Store embeddings
        db = get_db()
        for idea, embedding in zip(ideas, embeddings):
            db.execute("""
                INSERT OR REPLACE INTO idea_embeddings (idea_id, embedding)
                VALUES (?, ?)
            """, (idea["id"], embedding))
        db.commit()
        db.close()
    except Exception:
        # Embedding generation failed - continue without embeddings
        pass


async def execute_agent_output(
    output: AgentOutput,
    session: str,
    source_file: str,
    span_id: int,
) -> dict[str, Any]:
    """Execute complete agent output.

    Args:
        output: Parsed agent output
        session: Session ID
        source_file: Source transcript file
        span_id: Current span ID

    Returns:
        Dict with execution stats
    """
    stats = {
        "ideas_created": 0,
        "topic_updates": 0,
        "topic_changes": 0,
        "questions_answered": 0,
        "todos_completed": 0,
        "relations_created": 0,
    }

    # Execute ideas
    idea_ids = await execute_ideas(output.ideas, session, source_file, span_id)
    stats["ideas_created"] = len(idea_ids)

    # Build line -> idea_id map for relations
    idea_line_map = {}
    for idea, idea_id in zip(output.ideas, idea_ids):
        idea_line_map[idea.source_line] = idea_id

    # Execute topic updates
    await execute_topic_updates(output.topic_updates)
    stats["topic_updates"] = len(output.topic_updates)

    # Execute topic changes
    new_span_ids = await execute_topic_changes(output.topic_changes, session)
    stats["topic_changes"] = len(new_span_ids)

    # Execute answered questions
    await execute_answered_questions(output.answered_questions)
    stats["questions_answered"] = len(output.answered_questions)

    # Execute completed todos
    await execute_completed_todos(output.completed_todos)
    stats["todos_completed"] = len(output.completed_todos)

    # Execute relations
    relation_count = await execute_relations(output.relations, source_file, idea_line_map)
    stats["relations_created"] = relation_count

    # Execute entity links
    await execute_entity_links(output.entity_links, source_file, idea_line_map)

    # Execute activated ideas
    await execute_activated_ideas(output.activated_ideas, session)

    # Generate embeddings for new ideas
    await generate_embeddings(idea_ids)

    return stats
