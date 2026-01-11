"""Indexer module for topic tracking and idea extraction.

Processes transcripts to:
- Detect topic shifts and create hierarchical spans
- Extract ideas with intent classification
- Extract entities (technologies, files, concepts)
- Assess confidence levels
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import memory_db
from memory_db import DB_PATH, MemgraphError, get_embedding, logger
from transcript import get_indexable_messages

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Valid intent types
VALID_INTENTS = {"decision", "conclusion", "question", "problem", "solution", "todo", "context"}

# Bad topic name patterns - names matching these should be improved
_BAD_TOPIC_PATTERNS = [
    r"^session\s*(start|begin|init)",  # Generic session start
    r"^(okay|alright|now|so|well),?\s*(let'?s|we)",  # Action phrases
    r"^let'?s\s+(build|start|do|work|move|switch)",  # Let's do X
    r"^(hi|hello|hey)\b",  # Greetings
    r"^(introduction|overview|summary)$",  # Too generic
    r"^\*\*.*\*\*$",  # Just markdown bold
    r"^(the|a|an)\s+\w+$",  # Just "the X" or "a X"
]
_BAD_TOPIC_RE = re.compile("|".join(_BAD_TOPIC_PATTERNS), re.IGNORECASE)


def clean_topic_name(name: str) -> str:
    """Clean up a topic name, removing markdown and action phrases.

    Args:
        name: Raw topic name

    Returns:
        Cleaned topic name
    """
    # Strip whitespace
    name = name.strip()

    # Remove markdown formatting
    name = re.sub(r'\*\*([^*]+)\*\*', r'\1', name)  # Bold
    name = re.sub(r'\*([^*]+)\*', r'\1', name)  # Italic
    name = re.sub(r'`([^`]+)`', r'\1', name)  # Code
    name = re.sub(r'^#+\s*', '', name)  # Headers

    # Remove action prefixes
    action_prefixes = [
        r"^(okay|alright|now|so|well|right),?\s*",
        r"^let'?s\s+(now\s+)?",
        r"^(we should|we need to|i'?ll|we'?ll)\s+",
        r"^(first|next|then|finally),?\s*",
    ]
    for pattern in action_prefixes:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # Capitalize first letter
    if name and name[0].islower():
        name = name[0].upper() + name[1:]

    # Truncate to reasonable length
    if len(name) > 80:
        # Try to cut at word boundary
        cut_point = name.rfind(' ', 0, 80)
        if cut_point > 40:
            name = name[:cut_point]
        else:
            name = name[:80]

    return name.strip()


def is_bad_topic_name(name: str) -> bool:
    """Check if a topic name is too generic or poorly formatted.

    Args:
        name: Topic name to check

    Returns:
        True if the name is bad and should be improved
    """
    if not name or len(name) < 5:
        return True

    if _BAD_TOPIC_RE.search(name):
        return True

    return False


def generate_topic_name_from_content(messages: list[dict], fallback: str = "Discussion") -> str:
    """Generate a topic name from message content using LLM.

    Args:
        messages: List of message dicts with content
        fallback: Fallback name if LLM fails

    Returns:
        Generated topic name
    """
    if not messages:
        return fallback

    # Get API key
    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key or OpenAI is None:
        # No LLM available - extract key terms from first message
        first_content = messages[0].get("content", "")[:200]
        # Try to find a meaningful phrase
        cleaned = clean_topic_name(first_content)
        if not is_bad_topic_name(cleaned):
            return cleaned
        return fallback

    # Build prompt with first few messages
    contents = [m.get("content", "")[:300] for m in messages[:5] if m.get("content")]
    if not contents:
        return fallback

    combined = "\n---\n".join(contents)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Generate a concise topic name (3-8 words) for this conversation excerpt.
The name should describe WHAT is being discussed, not actions being taken.
Good: "ESP32 I2C Sensor Integration", "Rate Limiting Design", "User Authentication Flow"
Bad: "Let's build the project", "Session start", "Working on code"
Return just the topic name, nothing else."""
                },
                {"role": "user", "content": combined}
            ],
            max_tokens=30,
            temperature=0.3,
        )
        name = response.choices[0].message.content.strip()
        # Clean up any quotes
        name = name.strip('"\'')
        logger.info(f"Generated topic name: {name}")
        return name
    except Exception as e:
        logger.warning(f"Topic name generation failed: {e}")
        return fallback


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


# Sub-topic patterns (drilling down into a topic)
_SUBTOPIC_PATTERNS = [
    r"specifically,",
    r"diving into",
    r"let'?s dig into",
    r"let'?s look at.+in detail",
    r"more specifically",
    r"in particular,",
    r"focusing on",
    r"to be more specific",
    r"regarding the.+part",
    r"as for the",
    r"on the topic of",
    r"breaking down",
    r"within that,",
]

_SUBTOPIC_RE = re.compile("|".join(_SUBTOPIC_PATTERNS), re.IGNORECASE)


# Return-to-parent patterns (going back up the hierarchy)
_RETURN_PATTERNS = [
    r"back to the bigger picture",
    r"stepping back,",
    r"zooming out,",
    r"returning to",
    r"more broadly,",
    r"on a higher level",
    r"overall,",
    r"in summary,",
    r"to summarize,",
    r"wrapping up",
]

_RETURN_RE = re.compile("|".join(_RETURN_PATTERNS), re.IGNORECASE)


def detect_subtopic(content: str) -> bool:
    """Detect if content is diving into a sub-topic.

    Args:
        content: Message content

    Returns:
        True if sub-topic dive detected
    """
    return bool(_SUBTOPIC_RE.search(content.lower()))


def detect_return_to_parent(content: str) -> bool:
    """Detect if content is returning to parent topic.

    Args:
        content: Message content

    Returns:
        True if return to parent detected
    """
    return bool(_RETURN_RE.search(content.lower()))


def detect_topic_shift(content: str, context: dict) -> bool:
    """Detect if content represents a topic shift.

    Uses both keyword patterns and semantic embedding distance.

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

    # Check for semantic shift using embeddings
    if detect_topic_shift_semantic(content, context):
        return True

    return False


def detect_topic_shift_semantic(content: str, context: dict) -> bool:
    """Detect topic shift using embedding similarity.

    Args:
        content: Message content
        context: Context with last_embedding and threshold

    Returns:
        True if semantic topic shift detected
    """
    if "last_embedding" not in context:
        return False

    threshold = context.get("threshold", 0.5)

    try:
        current_embedding = get_embedding(content)
        similarity = cosine_similarity(context["last_embedding"], current_embedding)

        # Low similarity = topic shift
        if similarity < threshold:
            logger.info(f"Semantic topic shift detected (similarity: {similarity:.2f})")
            return True

    except MemgraphError:
        # Can't get embedding, skip semantic check
        pass

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
    """Classify the intent of a message using regex patterns.

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


def classify_intent_with_llm(content: str) -> str:
    """Classify the intent using LLM for better accuracy on ambiguous content.

    Falls back to regex classification if LLM unavailable.

    Args:
        content: Message content

    Returns:
        Intent string: decision, question, problem, solution, conclusion, todo, or context
    """
    # First try regex - if it finds a clear pattern, use it
    regex_intent = classify_intent(content)

    # Get API key
    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key or OpenAI is None:
        return regex_intent

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Classify this message into exactly one category:
- decision: A choice or decision that was made
- conclusion: An insight or conclusion reached
- question: A question being asked
- problem: A problem or issue being described
- solution: A solution to a problem
- todo: A task that needs to be done
- context: General context or information

Respond with just the category name, nothing else."""
                },
                {"role": "user", "content": content[:500]}
            ],
            max_tokens=10,
            temperature=0,
        )
        llm_intent = response.choices[0].message.content.strip().lower()

        # Validate the response
        if llm_intent in VALID_INTENTS:
            return llm_intent
        else:
            logger.warning(f"LLM returned invalid intent '{llm_intent}', using regex fallback")
            return regex_intent

    except Exception as e:
        logger.warning(f"LLM intent classification failed, using regex: {e}")
        return regex_intent


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
    "pg", "js", "ts",
}

# Entity resolution: maps variants to canonical names
_ENTITY_RESOLUTION = {
    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "pg": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "redis": "Redis",
    "sqlite": "SQLite",
    "dynamodb": "DynamoDB",
    "elasticsearch": "Elasticsearch",
    "elastic": "Elasticsearch",
    "cassandra": "Cassandra",

    # Languages
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "python": "Python",
    "rust": "Rust",
    "golang": "Go",
    "go": "Go",
    "java": "Java",

    # Runtimes
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "deno": "Deno",
    "bun": "Bun",

    # Frameworks
    "react": "React",
    "reactjs": "React",
    "vue": "Vue",
    "vuejs": "Vue",
    "angular": "Angular",
    "svelte": "Svelte",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "express": "Express",
    "nestjs": "NestJS",

    # Infrastructure
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",

    # Cloud
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",

    # Protocols
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "websocket": "WebSocket",
    "websockets": "WebSocket",

    # Auth
    "jwt": "JWT",
    "oauth": "OAuth",
    "oauth2": "OAuth",
    "openid": "OpenID",
}


def resolve_entity(name: str, entity_type: str) -> str:
    """Resolve an entity name to its canonical form.

    Args:
        name: Entity name to resolve
        entity_type: Type of entity (technology, file, etc.)

    Returns:
        Canonical entity name
    """
    # Only resolve technology entities
    if entity_type != "technology":
        return name

    # Look up in resolution map (case-insensitive)
    canonical = _ENTITY_RESOLUTION.get(name.lower())
    if canonical:
        return canonical

    # Return as-is if not found
    return name

# File path pattern
_FILE_PATTERN = re.compile(r'[\w./\-]+\.(py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|md|json|yaml|yml|toml|sql|sh|bash)')


def extract_entities(content: str) -> list[tuple[str, str]]:
    """Extract entities from content with resolution to canonical names.

    Args:
        content: Message content

    Returns:
        List of (name, type) tuples with resolved names
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
                # Resolve to canonical name
                resolved = resolve_entity(match.group(), "technology")
                entities.append((resolved, "technology"))

    # Extract file paths (no resolution needed)
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


def detect_relations(content: str, intent: str, recent_ideas: list[dict]) -> list[tuple[int, str]]:
    """Detect relations between new content and recent ideas.

    Args:
        content: New message content
        intent: Intent of new message
        recent_ideas: List of recent idea dicts with id, content, intent

    Returns:
        List of (idea_id, relation_type) tuples
    """
    relations = []
    content_lower = content.lower()

    # Supersession patterns - new idea replaces old
    supersession_markers = [
        "instead", "rather than", "changed to", "switching to",
        "no longer", "not anymore", "actually", "correction",
        "updated", "revised", "new approach",
    ]

    # Build-on patterns - new idea extends old
    buildon_markers = [
        "additionally", "also", "furthermore", "moreover",
        "building on", "extending", "adding to", "on top of",
        "in addition", "as well",
    ]

    # Check for supersession
    for marker in supersession_markers:
        if marker in content_lower:
            # Find related ideas to supersede (same topic/intent)
            for idea in recent_ideas:
                if idea["intent"] == intent or _content_overlap(content, idea["content"]):
                    relations.append((idea["id"], "supersedes"))
                    break
            break

    # Check for build-on
    for marker in buildon_markers:
        if marker in content_lower:
            # Find related ideas to build on
            for idea in recent_ideas:
                if _content_overlap(content, idea["content"]):
                    relations.append((idea["id"], "builds_on"))
                    break
            break

    # Solution answers question/problem
    if intent == "solution":
        for idea in recent_ideas:
            if idea["intent"] in ("question", "problem"):
                if _content_overlap(content, idea["content"]):
                    relations.append((idea["id"], "answers"))

    return relations


def _content_overlap(content1: str, content2: str) -> bool:
    """Check if two content strings have significant word overlap."""
    # Extract significant words (> 4 chars, not common)
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "will", "would", "could", "should"}

    words1 = set(w.lower() for w in re.findall(r'\b\w{5,}\b', content1)) - stopwords
    words2 = set(w.lower() for w in re.findall(r'\b\w{5,}\b', content2)) - stopwords

    if not words1 or not words2:
        return False

    overlap = len(words1 & words2)
    return overlap >= 2  # At least 2 significant words in common


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_ideas(query: str, limit: int = 5, threshold: float = 0.7) -> list[dict]:
    """Find semantically similar ideas using embedding search.

    Args:
        query: Text to find similar ideas for
        limit: Maximum number of results
        threshold: Minimum similarity score (0-1)

    Returns:
        List of similar idea dicts with similarity scores
    """
    try:
        results = memory_db.search_ideas(query, limit=limit)
        # Add similarity scores (distance is already in results from vec search)
        for r in results:
            if "distance" in r:
                # sqlite-vec distance is L2, convert to similarity
                # Lower distance = higher similarity
                r["similarity"] = 1 / (1 + r["distance"])
        return results
    except MemgraphError:
        return []


def detect_relations_with_embeddings(
    content: str,
    intent: str,
    candidate_ids: list[int],
    similarity_threshold: float = 0.75
) -> list[tuple[int, str]]:
    """Detect relations using embedding similarity.

    Args:
        content: New content to find relations for
        intent: Intent of the new content
        candidate_ids: IDs of candidate ideas to check
        similarity_threshold: Minimum similarity to consider related

    Returns:
        List of (idea_id, relation_type) tuples
    """
    if not candidate_ids:
        return []

    try:
        content_embedding = get_embedding(content)
    except MemgraphError:
        # Fallback to keyword-based detection
        return []

    relations = []
    db = memory_db.get_db()

    for idea_id in candidate_ids:
        # Get idea embedding
        cursor = db.execute("""
            SELECT i.content, i.intent, e.embedding
            FROM ideas i
            JOIN idea_embeddings e ON e.idea_id = i.id
            WHERE i.id = ?
        """, (idea_id,))
        row = cursor.fetchone()
        if not row:
            continue

        # Deserialize embedding
        import struct
        embedding_bytes = row["embedding"]
        idea_embedding = list(struct.unpack(f'{1536}f', embedding_bytes))

        # Calculate similarity
        similarity = cosine_similarity(content_embedding, idea_embedding)

        if similarity < similarity_threshold:
            continue

        # Determine relation type based on intent and content
        idea_intent = row["intent"]
        idea_content = row["content"]

        if intent == "solution" and idea_intent in ("question", "problem"):
            relations.append((idea_id, "answers"))
        elif _has_supersession_markers(content):
            relations.append((idea_id, "supersedes"))
        elif _has_buildon_markers(content):
            relations.append((idea_id, "builds_on"))
        elif similarity > 0.85:
            # High similarity but no specific markers = related
            relations.append((idea_id, "relates_to"))

    db.close()
    return relations


def _has_supersession_markers(content: str) -> bool:
    """Check if content has markers indicating supersession."""
    markers = [
        "instead", "rather than", "changed to", "switching to",
        "no longer", "not anymore", "actually", "correction",
        "updated", "revised", "new approach",
    ]
    content_lower = content.lower()
    return any(m in content_lower for m in markers)


def _has_buildon_markers(content: str) -> bool:
    """Check if content has markers indicating building on previous work."""
    markers = [
        "additionally", "also", "furthermore", "moreover",
        "building on", "extending", "adding to", "on top of",
        "in addition", "as well",
    ]
    content_lower = content.lower()
    return any(m in content_lower for m in markers)


def summarize_span(messages: list[dict]) -> str:
    """Generate a basic summary of messages in a span (fallback).

    Args:
        messages: List of message dicts with content

    Returns:
        Summary string
    """
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


def summarize_span_with_llm(messages: list[dict]) -> str:
    """Generate an LLM-powered summary of messages in a span.

    Falls back to basic summarization if LLM is unavailable.

    Args:
        messages: List of message dicts with content

    Returns:
        Summary string
    """
    if not messages:
        return ""

    # Get API key
    api_key = os.environ.get("OPENAI_TOKEN_MEMORY_EMBEDDINGS")
    if not api_key or OpenAI is None:
        return summarize_span(messages)

    # Build prompt with message content
    contents = [m.get("content", "")[:500] for m in messages if m.get("content")]
    if not contents:
        return "Brief discussion"

    combined = "\n---\n".join(contents[:10])  # Limit to first 10 messages

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following conversation excerpt in 1-2 sentences. "
                               "Focus on key decisions, conclusions, or topics discussed. "
                               "Be concise and factual."
                },
                {"role": "user", "content": combined}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"LLM summary generated: {summary[:50]}...")
        return summary
    except Exception as e:
        logger.warning(f"LLM summarization failed, using fallback: {e}")
        return summarize_span(messages)


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

    Raises:
        MemgraphError: If file not found or other critical errors
    """
    # Check file exists
    if not Path(file_path).exists():
        raise MemgraphError(
            f"Transcript file not found: {file_path}",
            "file_not_found",
            {"path": file_path}
        )

    # Get start line from index state if not provided
    if start_line is None:
        last_indexed = memory_db.get_last_indexed_line(file_path)
        start_line = last_indexed + 1

    # Get indexable messages
    try:
        messages = get_indexable_messages(file_path, start_line)
    except Exception as e:
        logger.warning(f"Error reading transcript: {e}")
        # Continue with empty messages - file might be corrupted or malformed
        messages = []

    if not messages:
        return {
            "file_path": file_path,
            "messages_indexed": 0,
            "ideas_created": 0,
            "spans_created": 0,
        }

    # Progress tracking
    import sys
    total_messages = len(messages)
    print(f"Indexing {total_messages} messages from {Path(file_path).name}...", file=sys.stderr)

    # Extract session name
    session = memory_db.extract_session_from_path(file_path)

    # Get or create current span
    current_span = memory_db.get_open_span(session)
    current_span_id = current_span["id"] if current_span else None
    current_depth = current_span["depth"] if current_span else 0
    span_messages = []

    # Stack of spans for hierarchy (each entry: {"id": int, "depth": int, "messages": list, "start_time": str, "last_time": str})
    span_stack: list[dict] = []
    if current_span_id:
        span_stack.append({"id": current_span_id, "depth": current_depth, "messages": [], "start_time": None, "last_time": None})

    ideas_created = 0
    spans_created = 0
    relations_created = 0
    last_line = start_line - 1

    # Track recent ideas for relation detection
    recent_ideas: list[dict] = []
    MAX_RECENT = 10  # Only check last N ideas for relations

    def close_span_with_summary(span_id: int, end_line: int, messages: list, end_time: Optional[str] = None):
        """Helper to close a span with LLM summary and end time."""
        summary = summarize_span_with_llm(messages) if messages else ""
        try:
            memory_db.close_span(span_id, end_line, summary, end_time=end_time)
        except MemgraphError as e:
            logger.warning(f"Failed to close span {span_id}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error closing span: {e}")

    for msg_idx, msg in enumerate(messages):
        content = msg["content"]
        line_num = msg["line_num"]
        msg_time = msg.get("timestamp", "")  # ISO timestamp from transcript

        # Progress update every 10 messages
        if msg_idx > 0 and msg_idx % 10 == 0:
            pct = int(100 * msg_idx / total_messages)
            print(f"  [{pct:3d}%] {msg_idx}/{total_messages} - {ideas_created} ideas, {spans_created} topics", file=sys.stderr)

        # Update last_time for current span
        if span_stack and msg_time:
            span_stack[-1]["last_time"] = msg_time

        # Check for return to parent topic
        if detect_return_to_parent(content) and len(span_stack) > 1:
            # Pop current span and return to parent
            current = span_stack.pop()
            close_span_with_summary(current["id"], line_num - 1, current["messages"], end_time=current.get("last_time"))
            current_span_id = span_stack[-1]["id"] if span_stack else None
            current_depth = span_stack[-1]["depth"] if span_stack else 0
            span_messages = span_stack[-1]["messages"] if span_stack else []

        # Check for sub-topic dive
        elif detect_subtopic(content) and span_stack:
            # Create a new child span
            parent_id = span_stack[-1]["id"]
            new_depth = span_stack[-1]["depth"] + 1
            # Clean and validate the topic name
            span_name = clean_topic_name(content[:100])
            if is_bad_topic_name(span_name):
                span_name = generate_topic_name_from_content([msg], fallback=span_name)
            new_span_id = memory_db.create_span(
                session=session,
                name=span_name,
                start_line=line_num,
                parent_id=parent_id,
                depth=new_depth,
                start_time=msg_time or None
            )
            spans_created += 1
            span_stack.append({"id": new_span_id, "depth": new_depth, "messages": [], "start_time": msg_time, "last_time": msg_time})
            current_span_id = new_span_id
            current_depth = new_depth
            span_messages = span_stack[-1]["messages"]

        # Check for topic shift (new top-level topic)
        elif detect_topic_shift(content, {}):
            # Close all open spans with their end times
            while span_stack:
                current = span_stack.pop()
                close_span_with_summary(current["id"], line_num - 1, current["messages"], end_time=current.get("last_time"))

            # Create new top-level span with cleaned name
            span_name = clean_topic_name(content[:100])
            if is_bad_topic_name(span_name):
                span_name = generate_topic_name_from_content([msg], fallback=span_name)
            current_span_id = memory_db.create_span(
                session=session,
                name=span_name,
                start_line=line_num,
                depth=0,
                start_time=msg_time or None
            )
            spans_created += 1
            current_depth = 0
            span_stack = [{"id": current_span_id, "depth": 0, "messages": [], "start_time": msg_time, "last_time": msg_time}]
            span_messages = span_stack[-1]["messages"]
            # Clear recent ideas on topic shift
            recent_ideas = []

        # If no span exists, create initial one
        # Use first message content to generate a name instead of generic "Session start"
        if current_span_id is None:
            span_name = clean_topic_name(content[:100])
            if is_bad_topic_name(span_name):
                span_name = generate_topic_name_from_content([msg], fallback="Initial Discussion")
            current_span_id = memory_db.create_span(
                session=session,
                name=span_name,
                start_line=line_num,
                depth=0,
                start_time=msg_time or None
            )
            spans_created += 1
            current_depth = 0
            span_stack = [{"id": current_span_id, "depth": 0, "messages": [], "start_time": msg_time, "last_time": msg_time}]
            span_messages = span_stack[-1]["messages"]

        # Classify intent
        intent = classify_intent(content)

        # Extract entities
        entities = extract_entities(content)

        # Assess confidence
        confidence = assess_confidence(content, intent)

        # Store idea with message timestamp
        idea_id = None
        try:
            idea_id = memory_db.store_idea(
                content=content,
                source_file=file_path,
                source_line=line_num,
                span_id=current_span_id,
                intent=intent,
                confidence=confidence,
                entities=entities if entities else None,
                message_time=msg_time or None
            )
            ideas_created += 1

            # Detect and store relations to recent ideas
            if recent_ideas:
                recent_ids = [r["id"] for r in recent_ideas]
                try:
                    relations = detect_relations_with_embeddings(
                        content, intent, recent_ids, similarity_threshold=0.7
                    )
                    for to_id, relation_type in relations:
                        memory_db.add_relation(idea_id, to_id, relation_type)
                        relations_created += 1

                        # Mark questions as answered when solution found
                        if relation_type == "answers":
                            memory_db.mark_question_answered(to_id)

                except Exception as e:
                    logger.debug(f"Relation detection failed: {e}")

            # Add to recent ideas
            recent_ideas.append({"id": idea_id, "content": content, "intent": intent})
            if len(recent_ideas) > MAX_RECENT:
                recent_ideas.pop(0)

        except MemgraphError as e:
            # Log embedding failures but continue processing
            if e.error_code == "missing_api_key":
                logger.warning("Skipping idea storage: API key not configured")
            else:
                logger.warning(f"Failed to store idea at line {line_num}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error storing idea at line {line_num}: {e}")

        span_messages.append(msg)
        last_line = max(last_line, line_num)

    # Count total lines
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    # Update index state
    if total_lines > 0:
        memory_db.update_index_state(file_path, total_lines)

    # Save embedding cache to disk for future runs
    memory_db.save_embedding_cache()

    print(f"  [100%] Done! {ideas_created} ideas, {spans_created} topics, {relations_created} relations", file=sys.stderr)

    return {
        "file_path": file_path,
        "messages_indexed": len(messages),
        "ideas_created": ideas_created,
        "spans_created": spans_created,
        "relations_created": relations_created,
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
