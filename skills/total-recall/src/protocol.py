"""LLM protocol - input formatting and output parsing."""

from typing import Any

from batcher import Batch


def format_llm_input(
    batch: Batch,
    context: dict[str, Any],
    recent_messages: list[dict[str, str]],
    max_recent: int = 10
) -> dict[str, Any]:
    """
    Format a batch and context into LLM input structure.

    Args:
        batch: Batch of new messages to process
        context: Hierarchy context from build_context()
        recent_messages: Recent messages for context (will be truncated)
        max_recent: Maximum number of recent messages to include

    Returns:
        Dict structured for LLM consumption with:
        - hierarchy: project, topic, span info
        - recent_messages: context from earlier in session
        - new_messages: batch messages with line numbers
    """
    # Build hierarchy from context
    hierarchy = {
        "project": context.get("project"),
        "topic": context.get("topic"),
        "span": context.get("current_span"),
        "parent_spans": context.get("parent_spans", []),
    }

    # Truncate recent messages to keep most recent
    truncated_recent = recent_messages[-max_recent:] if recent_messages else []

    # Format new messages with line numbers
    new_messages = []
    for msg in batch.messages:
        new_messages.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "line": msg.line_num,
        })

    return {
        "hierarchy": hierarchy,
        "recent_messages": truncated_recent,
        "new_messages": new_messages,
    }
