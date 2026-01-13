"""Transcript parsing for Claude conversation logs."""

import json
import re
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

# Preamble patterns - messages that just introduce tool use
# These are regex patterns matched against the full content
_PREAMBLE_PATTERNS = [
    # "Let me X" / "I'll X" / "Now let me X" patterns
    r"^(now\s+)?let\s+me\s+(try|check|fetch|read|run|search|look|see|find|get|create|update|fix|test|verify|examine|review|inspect|analyze)",
    r"^i'?ll\s+(try|check|fetch|read|run|search|look|see|find|get|create|update|fix|test|verify|help|start|begin)",
    r"^(now\s+)?i'?ll\s+",
    r"^let'?s\s+(try|check|see|look|start|begin|run|test)",

    # Status/transition messages
    r"^(good|great|perfect|excellent)[,.]?\s*(now|let)",
    r"^(now|next|first|then)[,:]?\s*(let me|i'?ll|let'?s)",
    r"^(okay|ok|alright)[,.]?\s*(now|let|i'?ll)",

    # Result announcements (short)
    r"^(here'?s?|this is)\s+(the|what)",
    r"^the (output|result|error|file|code) (is|shows|says)",

    # Continuation markers
    r"^(and\s+)?(now|next)\s+",
    r"^moving on",
    r"^continuing with",
]

_PREAMBLE_RE = re.compile("|".join(_PREAMBLE_PATTERNS), re.IGNORECASE)

# Patterns that indicate substantive content (override preamble detection)
_SUBSTANTIVE_PATTERNS = [
    r"\bbecause\b",
    r"\bthe (reason|issue|problem|solution|key|important|main)\b",
    r"\bdecided\b",
    r"\bshould\b.*\bbecause\b",
    r"\binstead of\b",
    r"\brather than\b",
    r"\bthe approach\b",
    r"\bwe need to\b.*\bto\b",  # "we need to X to Y" is substantive
]

_SUBSTANTIVE_RE = re.compile("|".join(_SUBSTANTIVE_PATTERNS), re.IGNORECASE)

# Patterns for messages that are just tool output narration
_TOOL_NARRATION_PATTERNS = [
    r"^(the command|it|this) (completed|succeeded|failed|ran|finished|returned|shows|outputs)",
    r"^(running|executing|checking|reading|fetching|creating)",
    r"^(done|complete|success|finished)[.!]?$",
    r"^(no errors?|all (tests? )?pass(ed|ing)?)[.!]?$",
]

_TOOL_NARRATION_RE = re.compile("|".join(_TOOL_NARRATION_PATTERNS), re.IGNORECASE)

# System/meta messages to filter
_SYSTEM_PATTERNS = [
    r"^\[request interrupted",
    r"^\[user cancelled",
    r"^\[tool (error|timeout|failed)",
    r"^<system",
]

_SYSTEM_RE = re.compile("|".join(_SYSTEM_PATTERNS), re.IGNORECASE)

# Short user instructions that aren't valuable on their own
# Must be careful not to match questions ("do we need...?")
_SHORT_INSTRUCTION_PATTERNS = [
    r"^(please\s+)?(run|try|check|fix|update|create|delete|show|list|find|search)\s+it\b",
    r"^(please\s+)?(run|try|check|fix|update|create|delete|show|list|find|search)\s+this\b",
    r"^(please\s+)?(run|try|check|fix|update|create|delete|show|list|find|search)\s+that\b",
    r"^go\s*(ahead)?[.!]?$",
    r"^(do it|proceed|continue)[.!]?$",
    r"^yes[,.]?\s*please[.!]?$",
    r"^okay[,.]?\s*do it[.!]?$",
]

_SHORT_INSTRUCTION_RE = re.compile("|".join(_SHORT_INSTRUCTION_PATTERNS), re.IGNORECASE)

# Assistant greeting patterns (prefix match)
_ASSISTANT_GREETING_PREFIXES = [
    "hello!", "hi!", "hey!",
    "hello,", "hi,", "hey,",
    "hello.", "hi.", "hey.",
]

# Minimum length thresholds
MIN_INDEXABLE_LENGTH = 20
MIN_SUBSTANTIVE_LENGTH = 80  # Messages under this get extra scrutiny


def is_indexable(message: dict) -> bool:
    """Determine if a message contains indexable content.

    Filters out:
    - Greetings ("hello", "hi there")
    - Simple acknowledgments ("ok", "yes", "thanks")
    - Very short content (< 20 chars)
    - Tool use preambles ("Let me try...", "I'll check...")
    - Tool narration ("Running the command...", "Done!")
    - Status updates without substance

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

    # Check for greeting patterns (exact match)
    if content_lower in _GREETING_PATTERNS:
        return False

    # Check for acknowledgment patterns (exact match)
    if content_lower in _ACKNOWLEDGMENT_PATTERNS:
        return False

    # Check for assistant greeting responses (short messages starting with greeting)
    if len(content) < 50:
        for prefix in _ASSISTANT_GREETING_PREFIXES:
            if content_lower.startswith(prefix):
                return False

    # Check for tool narration (these are almost never valuable)
    if _TOOL_NARRATION_RE.match(content_lower):
        return False

    # Check for system/meta messages
    if _SYSTEM_RE.match(content_lower):
        return False

    # Check for short user instructions (< 50 chars and matches pattern)
    if len(content) < 50 and _SHORT_INSTRUCTION_RE.match(content_lower):
        return False

    # For shorter messages, check for preamble patterns
    if len(content) < MIN_SUBSTANTIVE_LENGTH:
        # Check if it's a preamble
        if _PREAMBLE_RE.match(content_lower):
            # But allow if it contains substantive content
            if not _SUBSTANTIVE_RE.search(content_lower):
                return False

    # For assistant messages with tool use, be stricter
    if message.get("has_tool_use"):
        # Short messages with tool use are almost always preambles
        if len(content) < 100:
            # Check for any preamble pattern
            if _PREAMBLE_RE.match(content_lower):
                return False
            # Also filter very short tool introductions
            if len(content) < 60:
                return False

    return True


def get_filter_reason(message: dict) -> Optional[str]:
    """Get the reason a message would be filtered (for debugging/review).

    Args:
        message: Parsed message dict

    Returns:
        Reason string if filtered, None if indexable
    """
    content = message.get("content", "").strip()

    if not content:
        return "empty"

    content_lower = content.lower()

    if len(content) < MIN_INDEXABLE_LENGTH:
        return f"too_short ({len(content)} < {MIN_INDEXABLE_LENGTH})"

    if content_lower in _GREETING_PATTERNS:
        return "greeting"

    if content_lower in _ACKNOWLEDGMENT_PATTERNS:
        return "acknowledgment"

    if len(content) < 50:
        for prefix in _ASSISTANT_GREETING_PREFIXES:
            if content_lower.startswith(prefix):
                return "greeting_response"

    if _TOOL_NARRATION_RE.match(content_lower):
        return "tool_narration"

    if _SYSTEM_RE.match(content_lower):
        return "system_message"

    if len(content) < 50 and _SHORT_INSTRUCTION_RE.match(content_lower):
        return "short_instruction"

    if len(content) < MIN_SUBSTANTIVE_LENGTH:
        if _PREAMBLE_RE.match(content_lower):
            if not _SUBSTANTIVE_RE.search(content_lower):
                return "preamble"

    if message.get("has_tool_use"):
        if len(content) < 100:
            if _PREAMBLE_RE.match(content_lower):
                return "tool_preamble"
            if len(content) < 60:
                return "short_tool_intro"

    return None  # Indexable


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


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcript parsing utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get-indexable command
    get_cmd = subparsers.add_parser(
        "get-indexable",
        help="Get indexable messages from a transcript"
    )
    get_cmd.add_argument("transcript", help="Path to JSONL transcript file")
    get_cmd.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="Line number to start from (1-indexed)"
    )

    args = parser.parse_args()

    if args.command == "get-indexable":
        messages = get_indexable_messages(args.transcript, args.start_line)
        print(json.dumps(messages))


if __name__ == "__main__":
    main()
