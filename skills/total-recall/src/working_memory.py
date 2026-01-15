"""Working memory - activation-based idea retrieval for session context."""

from typing import Any

from db.connection import get_db


def activate_ideas(session: str, content: str, idea_ids: list[int]) -> None:
    """
    Activate ideas based on being mentioned in content.

    Args:
        session: The session identifier
        content: The content that triggered activation
        idea_ids: List of idea IDs to activate
    """
    if not idea_ids:
        return

    db = get_db()

    for idea_id in idea_ids:
        # Insert or update activation
        db.execute("""
            INSERT INTO working_memory (session, idea_id, activation, last_access)
            VALUES (?, ?, 1.0, CURRENT_TIMESTAMP)
            ON CONFLICT(session, idea_id) DO UPDATE SET
                activation = MIN(activation + 0.5, 1.0),
                last_access = CURRENT_TIMESTAMP
        """, (session, idea_id))

    db.commit()
    db.close()


def decay_activations(session: str, factor: float = 0.9) -> None:
    """
    Decay all activations for a session.

    Args:
        session: The session identifier
        factor: Decay multiplier (0-1), lower = faster decay
    """
    db = get_db()

    db.execute("""
        UPDATE working_memory
        SET activation = activation * ?
        WHERE session = ?
    """, (factor, session))

    db.commit()
    db.close()


def prune_activations(session: str, threshold: float = 0.01) -> None:
    """
    Remove activations below threshold.

    Args:
        session: The session identifier
        threshold: Minimum activation to keep
    """
    db = get_db()

    db.execute("""
        DELETE FROM working_memory
        WHERE session = ? AND activation < ?
    """, (session, threshold))

    db.commit()
    db.close()


def get_active_ideas(session: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get currently active ideas for a session.

    Args:
        session: The session identifier
        limit: Maximum number of ideas to return

    Returns:
        List of idea dicts with id, content, intent, activation
    """
    db = get_db()

    cursor = db.execute("""
        SELECT i.id, i.content, i.intent, wm.activation
        FROM working_memory wm
        JOIN ideas i ON wm.idea_id = i.id
        WHERE wm.session = ?
        ORDER BY wm.activation DESC
        LIMIT ?
    """, (session, limit))

    ideas = [
        {
            "id": row["id"],
            "content": row["content"],
            "intent": row["intent"],
            "activation": row["activation"],
        }
        for row in cursor.fetchall()
    ]

    db.close()
    return ideas
