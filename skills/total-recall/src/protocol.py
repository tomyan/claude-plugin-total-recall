"""LLM protocol - input formatting and output parsing."""

import json
from dataclasses import dataclass, field
from typing import Any

from batcher import Batch


# Valid intent types for ideas
VALID_INTENTS = {
    'decision', 'conclusion', 'question', 'problem',
    'solution', 'todo', 'context'
}


class ProtocolError(Exception):
    """Error in LLM protocol parsing/validation."""
    pass


@dataclass
class LLMOutput:
    """Parsed and validated LLM output."""
    topic_update: dict[str, str] | None = None
    new_span: dict[str, str] | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    skip_lines: list[int] = field(default_factory=list)


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


def parse_llm_output(response: dict[str, Any]) -> LLMOutput:
    """
    Parse and validate LLM response.

    Args:
        response: Parsed JSON response from LLM

    Returns:
        Validated LLMOutput dataclass

    Raises:
        ProtocolError: If response contains invalid data
    """
    result = LLMOutput()

    # Parse topic_update (optional)
    if "topic_update" in response:
        result.topic_update = response["topic_update"]

    # Parse new_span (optional)
    if "new_span" in response:
        result.new_span = response["new_span"]

    # Parse items (optional, but validated if present)
    if "items" in response:
        items = response["items"]
        validated_items = []

        for item in items:
            # Validate required fields
            if "type" not in item:
                raise ProtocolError("Item missing required field: type")
            if "content" not in item:
                raise ProtocolError("Item missing required field: content")
            if "source_line" not in item:
                raise ProtocolError("Item missing required field: source_line")

            # Validate intent type
            intent = item["type"]
            if intent not in VALID_INTENTS:
                raise ProtocolError(
                    f"Invalid intent type: {intent}. "
                    f"Must be one of: {', '.join(sorted(VALID_INTENTS))}"
                )

            # Apply defaults
            validated_item = dict(item)
            if "confidence" not in validated_item:
                validated_item["confidence"] = 0.5

            validated_items.append(validated_item)

        result.items = validated_items

    # Parse relations (optional)
    if "relations" in response:
        result.relations = response["relations"]

    # Parse skip_lines (optional)
    if "skip_lines" in response:
        result.skip_lines = response["skip_lines"]

    return result


def parse_llm_output_str(response_str: str) -> LLMOutput:
    """
    Parse LLM response from raw JSON string.

    Args:
        response_str: Raw JSON string from LLM

    Returns:
        Validated LLMOutput dataclass

    Raises:
        ProtocolError: If JSON is malformed or response is invalid
    """
    try:
        response = json.loads(response_str)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"Failed to parse JSON: {e}")

    return parse_llm_output(response)
