"""Transcript parsing for Claude conversation logs."""

import json
from typing import Iterator, Optional


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


def read_transcript(
    file_path: str,
    start_line: int = 1
) -> Iterator[tuple[int, dict]]:
    """Read a transcript file and yield parsed messages.

    Args:
        file_path: Path to the JSONL transcript file
        start_line: Line number to start from (1-indexed)

    Yields:
        Tuples of (line_number, parsed_message_dict)
        Skips lines that can't be parsed or aren't indexable.
    """
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if line_num < start_line:
                continue

            line = line.strip()
            if not line:
                continue

            parsed = parse_transcript_line(line)
            if parsed is not None:
                yield (line_num, parsed)


# Low-value patterns to filter out
_GREETING_PATTERNS = frozenset([
    "hello", "hi", "hey", "hi there", "hello there",
    "good morning", "good afternoon", "good evening",
])

_ACKNOWLEDGMENT_PATTERNS = frozenset([
    "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "thanks", "thank you", "got it", "understood",
    "sure", "right", "correct", "exactly",
    "sounds good", "sounds great", "looks good", "looks great",
    "perfect", "great", "awesome", "nice", "cool",
])

_TOOL_PREAMBLE_PATTERNS = frozenset([
    "let me", "i'll", "i will", "let's",
])

# Assistant greeting patterns (prefix match)
_ASSISTANT_GREETING_PREFIXES = [
    "hello!", "hi!", "hey!",
    "hello,", "hi,", "hey,",
    "hello.", "hi.", "hey.",
]

MIN_INDEXABLE_LENGTH = 20


def is_indexable(message: dict) -> bool:
    """Determine if a message contains indexable content.

    Filters out:
    - Greetings ("hello", "hi there")
    - Simple acknowledgments ("ok", "yes", "thanks")
    - Very short content (< 20 chars)
    - Tool use preambles without substance

    Args:
        message: Parsed message dict with 'type', 'content', 'has_tool_use'

    Returns:
        True if message should be indexed, False otherwise
    """
    content = message.get("content", "").strip()

    # Empty or whitespace-only
    if not content:
        return False

    content_lower = content.lower()

    # Too short
    if len(content) < MIN_INDEXABLE_LENGTH:
        return False

    # Check for greeting patterns
    if content_lower in _GREETING_PATTERNS:
        return False

    # Check for acknowledgment patterns
    if content_lower in _ACKNOWLEDGMENT_PATTERNS:
        return False

    # Check for assistant greeting responses (short messages starting with greeting)
    if len(content) < 50:
        for prefix in _ASSISTANT_GREETING_PREFIXES:
            if content_lower.startswith(prefix):
                return False

    # For assistant messages with tool use, check if it's just preamble
    if message.get("has_tool_use"):
        # If short and starts with tool preamble, skip
        if len(content) < 50:
            for pattern in _TOOL_PREAMBLE_PATTERNS:
                if content_lower.startswith(pattern):
                    return False

    return True


def get_indexable_messages(
    file_path: str,
    start_line: int = 1
) -> list[dict]:
    """Get all indexable messages from a transcript file.

    Combines read_transcript and is_indexable filtering.
    Returns a list of dicts with line_num added to each message.

    Args:
        file_path: Path to the JSONL transcript file
        start_line: Line number to start from (1-indexed)

    Returns:
        List of message dicts, each with:
            - line_num: Source line number
            - type: "user" or "assistant"
            - content: Text content
            - timestamp: ISO timestamp
            - has_tool_use: bool (for assistant messages)
    """
    messages = []
    for line_num, msg in read_transcript(file_path, start_line):
        if is_indexable(msg):
            messages.append({
                "line_num": line_num,
                **msg
            })
    return messages
