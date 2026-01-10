"""Transcript parsing for Claude conversation logs."""

import json
from typing import Optional


def extract_message_content(message: dict) -> str:
    """Extract text content from a message structure.

    Handles both string content and list content with text blocks.
    Filters out tool_use blocks.
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)

    return ""


def parse_transcript_line(line: str) -> Optional[dict]:
    """Parse a single JSON line from a Claude transcript.

    Returns a normalized dict with:
        - type: "user" or "assistant"
        - content: extracted text content
        - timestamp: ISO timestamp
        - has_tool_use: bool (for assistant messages)

    Returns None for:
        - Invalid JSON
        - Tool result messages
        - Non-indexable content
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    msg_type = data.get("type")
    if msg_type not in ("user", "assistant"):
        return None

    message = data.get("message", {})

    # Skip tool results
    if "toolUseResult" in data:
        return None

    content_raw = message.get("content", "")
    if isinstance(content_raw, list):
        # Check if it's just tool results
        has_text = any(
            isinstance(b, dict) and b.get("type") == "text"
            for b in content_raw
        )
        if not has_text:
            return None

    content = extract_message_content(message)

    # Check for tool use in assistant messages
    has_tool_use = False
    if msg_type == "assistant" and isinstance(content_raw, list):
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content_raw
        )

    return {
        "type": msg_type,
        "content": content,
        "timestamp": data.get("timestamp", ""),
        "has_tool_use": has_tool_use,
    }
