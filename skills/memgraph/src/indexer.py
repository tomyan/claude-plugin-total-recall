"""Indexer module for topic tracking and idea extraction.

Processes transcripts to:
- Detect topic shifts and create hierarchical spans
- Extract ideas with intent classification
- Extract entities (technologies, files, concepts)
- Assess confidence levels
"""

import json
import re
from typing import Optional

import memory_db
from memory_db import DB_PATH, get_embedding
from transcript import get_indexable_messages


# Topic shift patterns
_TRANSITION_PATTERNS = [
    r"let'?s move on to",
    r"let'?s switch to",
    r"let'?s work on",
    r"let'?s discuss",
    r"back to the",
    r"now let'?s",
    r"switching to",
    r"moving on to",
    r"turning to",
    r"next,? let'?s",
    r"okay,? let'?s",
    r"alright,? let'?s",
]

_TRANSITION_RE = re.compile("|".join(_TRANSITION_PATTERNS), re.IGNORECASE)


def detect_topic_shift(content: str, context: dict) -> bool:
    """Detect if content represents a topic shift.

    Args:
        content: Message content
        context: Current context with last_embedding, threshold, etc.

    Returns:
        True if topic shift detected
    """
    content_lower = content.lower()

    # Check for explicit transition patterns
    if _TRANSITION_RE.search(content_lower):
        return True

    # TODO: Add embedding distance check for semantic shifts
    # if "last_embedding" in context:
    #     current_embedding = get_embedding(content)
    #     distance = cosine_distance(context["last_embedding"], current_embedding)
    #     if distance > context.get("threshold", 0.5):
    #         return True

    return False


# Intent classification patterns
_DECISION_PATTERNS = [
    r"^we decided",
    r"^i decided",
    r"^decided to",
    r"^going with",
    r"^i'?ll use",
    r"^we'?ll use",
    r"^using\b",
    r"^chose\b",
    r"^choosing\b",
    r"the decision is",
    r"final decision",
]

_QUESTION_PATTERNS = [
    r"\?$",
    r"^how (should|can|do|would)",
    r"^what (should|is|are|would)",
    r"^should (we|i)",
    r"^which\b",
    r"^where\b",
    r"^when\b",
    r"^why\b",
    r"^can (we|you|i)",
    r"^is (it|there|this)",
]

_PROBLEM_PATTERNS = [
    r"^the (issue|problem) (is|with)",
    r"^issue:",
    r"^problem:",
    r"running into",
    r"struggling with",
    r"(is|are) (failing|broken|not working)",
    r"error:",
    r"bug:",
    r"doesn'?t work",
]

_SOLUTION_PATTERNS = [
    r"^fixed (by|it|this)",
    r"^the (solution|fix) is",
    r"^resolved (by|this)",
    r"^solved (by|it)",
    r"^to fix this",
    r"^the answer is",
    r"works now",
]

_CONCLUSION_PATTERNS = [
    r"^the key (insight|takeaway)",
    r"^in conclusion",
    r"^learned that",
    r"^the (main|key) (point|thing)",
    r"^takeaway:",
    r"^insight:",
    r"turns out",
    r"realized that",
]

_TODO_PATTERNS = [
    r"^need to",
    r"^should (implement|add|create|fix)",
    r"^todo:",
    r"^TODO:",
    r"^must (implement|add|create|fix)",
    r"^next steps?:",
    r"remaining work",
]


def classify_intent(content: str) -> str:
    """Classify the intent of a message.

    Args:
        content: Message content

    Returns:
        Intent string: decision, question, problem, solution, conclusion, todo, or context
    """
    content_lower = content.lower().strip()

    # Check patterns in priority order
    for pattern in _DECISION_PATTERNS:
        if re.search(pattern, content_lower):
            return "decision"

    for pattern in _SOLUTION_PATTERNS:
        if re.search(pattern, content_lower):
            return "solution"

    for pattern in _PROBLEM_PATTERNS:
        if re.search(pattern, content_lower):
            return "problem"

    for pattern in _CONCLUSION_PATTERNS:
        if re.search(pattern, content_lower):
            return "conclusion"

    for pattern in _TODO_PATTERNS:
        if re.search(pattern, content_lower):
            return "todo"

    for pattern in _QUESTION_PATTERNS:
        if re.search(pattern, content_lower):
            return "question"

    # Default to context
    return "context"


# Known technologies for entity extraction
_TECHNOLOGIES = {
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "kafka", "rabbitmq", "dynamodb", "cassandra",
    "python", "javascript", "typescript", "rust", "go", "golang", "java",
    "react", "vue", "angular", "svelte", "nextjs", "next.js",
    "node", "nodejs", "node.js", "deno", "bun",
    "django", "flask", "fastapi", "express", "koa", "nestjs",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "aws", "gcp", "azure", "vercel", "netlify", "cloudflare",
    "graphql", "rest", "grpc", "websocket", "websockets", "sse",
    "jwt", "oauth", "oauth2", "openid",
    "esp32", "arduino", "raspberry pi", "stm32", "lora", "sx1262",
    "git", "github", "gitlab", "bitbucket",
}

# File path pattern
_FILE_PATTERN = re.compile(r'[\w./\-]+\.(py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|md|json|yaml|yml|toml|sql|sh|bash)')


def extract_entities(content: str) -> list[tuple[str, str]]:
    """Extract entities from content.

    Args:
        content: Message content

    Returns:
        List of (name, type) tuples
    """
    entities = []
    content_lower = content.lower()

    # Extract technologies
    for tech in _TECHNOLOGIES:
        if tech in content_lower:
            # Find original case in content
            pattern = re.compile(re.escape(tech), re.IGNORECASE)
            match = pattern.search(content)
            if match:
                entities.append((match.group(), "technology"))

    # Extract file paths
    for match in _FILE_PATTERN.finditer(content):
        entities.append((match.group(), "file"))

    # Extract concepts (noun phrases with specific patterns)
    concept_patterns = [
        r"(rate limiting)",
        r"(sliding window)",
        r"(connection pool\w*)",
        r"(load balanc\w+)",
        r"(caching)",
        r"(authentication)",
        r"(authorization)",
        r"(encryption)",
        r"(hashing)",
        r"(indexing)",
    ]
    for pattern in concept_patterns:
        match = re.search(pattern, content_lower)
        if match:
            entities.append((match.group(1), "concept"))

    return list(set(entities))  # Dedupe


def assess_confidence(content: str, intent: str) -> float:
    """Assess confidence level of a statement.

    Args:
        content: Message content
        intent: Classified intent

    Returns:
        Confidence score 0.0 to 1.0
    """
    content_lower = content.lower()

    # Tentative markers reduce confidence
    tentative_markers = [
        "maybe", "might", "could", "perhaps", "possibly",
        "i think", "i guess", "not sure", "uncertain",
        "probably", "likely", "seems like",
    ]

    # Firm markers increase confidence
    firm_markers = [
        "definitely", "certainly", "absolutely", "clearly",
        "decided", "confirmed", "verified", "final",
        "must", "will", "shall",
    ]

    # Base confidence by intent
    base_confidence = {
        "decision": 0.8,
        "solution": 0.8,
        "conclusion": 0.7,
        "problem": 0.7,
        "context": 0.6,
        "todo": 0.5,
        "question": 0.5,
    }.get(intent, 0.5)

    confidence = base_confidence

    # Adjust based on markers
    for marker in tentative_markers:
        if marker in content_lower:
            confidence -= 0.15
            break

    for marker in firm_markers:
        if marker in content_lower:
            confidence += 0.1
            break

    # Clamp to valid range
    return max(0.1, min(1.0, confidence))


def summarize_span(messages: list[dict]) -> str:
    """Generate a summary of messages in a span.

    Args:
        messages: List of message dicts with content

    Returns:
        Summary string
    """
    # TODO: Use LLM for better summarization
    # For now, extract key sentences
    if not messages:
        return ""

    # Get first substantive message as topic indicator
    contents = [m.get("content", "") for m in messages if len(m.get("content", "")) > 30]
    if not contents:
        return "Brief discussion"

    # Return first sentence of first message as simple summary
    first = contents[0]
    sentences = re.split(r'[.!?]', first)
    if sentences:
        return sentences[0].strip()[:200]

    return first[:200]


def index_transcript(
    file_path: str,
    start_line: Optional[int] = None
) -> dict:
    """Index a transcript file with full topic tracking and intent classification.

    Args:
        file_path: Path to JSONL transcript file
        start_line: Optional line to start from (default: last indexed + 1)

    Returns:
        Dict with indexing stats
    """
    # Get start line from index state if not provided
    if start_line is None:
        last_indexed = memory_db.get_last_indexed_line(file_path)
        start_line = last_indexed + 1

    # Get indexable messages
    messages = get_indexable_messages(file_path, start_line)

    if not messages:
        return {
            "file_path": file_path,
            "messages_indexed": 0,
            "ideas_created": 0,
            "spans_created": 0,
        }

    # Extract session name
    session = memory_db.extract_session_from_path(file_path)

    # Get or create current span
    current_span = memory_db.get_open_span(session)
    current_span_id = current_span["id"] if current_span else None
    span_messages = []

    ideas_created = 0
    spans_created = 0
    last_line = start_line - 1

    for msg in messages:
        content = msg["content"]
        line_num = msg["line_num"]

        # Check for topic shift
        if detect_topic_shift(content, {}):
            # Close current span if exists
            if current_span_id and span_messages:
                summary = summarize_span(span_messages)
                try:
                    memory_db.close_span(current_span_id, line_num - 1, summary)
                except Exception:
                    pass  # May fail if no embedding API key

            # Create new span
            span_name = content[:100]  # Use transition message as name
            current_span_id = memory_db.create_span(
                session=session,
                name=span_name,
                start_line=line_num,
                depth=0
            )
            spans_created += 1
            span_messages = []

        # If no span exists, create initial one
        if current_span_id is None:
            current_span_id = memory_db.create_span(
                session=session,
                name=f"Session start",
                start_line=line_num,
                depth=0
            )
            spans_created += 1

        # Classify intent
        intent = classify_intent(content)

        # Extract entities
        entities = extract_entities(content)

        # Assess confidence
        confidence = assess_confidence(content, intent)

        # Store idea
        try:
            memory_db.store_idea(
                content=content,
                source_file=file_path,
                source_line=line_num,
                span_id=current_span_id,
                intent=intent,
                confidence=confidence,
                entities=entities if entities else None
            )
            ideas_created += 1
        except Exception:
            pass  # May fail if no embedding API key

        span_messages.append(msg)
        last_line = max(last_line, line_num)

    # Count total lines
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    # Update index state
    if total_lines > 0:
        memory_db.update_index_state(file_path, total_lines)

    return {
        "file_path": file_path,
        "messages_indexed": len(messages),
        "ideas_created": ideas_created,
        "spans_created": spans_created,
        "start_line": start_line,
        "end_line": last_line,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Index transcripts with topic tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    idx_cmd = subparsers.add_parser("index", help="Index a transcript file")
    idx_cmd.add_argument("file", help="Path to JSONL transcript file")
    idx_cmd.add_argument("--start-line", type=int, help="Line number to start from")

    args = parser.parse_args()

    if args.command == "index":
        result = index_transcript(args.file, args.start_line)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
