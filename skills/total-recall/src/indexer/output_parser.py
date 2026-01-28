"""Agent output parser - Slice 4.1.

Parses and validates the JSON output from the indexing agent,
converting it into structured data objects for execution.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


class ParseError(Exception):
    """Raised when parsing agent output fails."""
    pass


# Valid intent types (must match schema)
VALID_INTENTS = frozenset([
    "decision", "conclusion", "question", "problem",
    "solution", "todo", "context", "observation"
])

# Valid relation types (must match schema)
VALID_RELATIONS = frozenset([
    "supersedes", "builds_on", "contradicts", "answers", "relates_to"
])


@dataclass
class IdeaOutput:
    """Parsed idea from agent output.

    Attributes:
        intent: Idea type (decision, question, etc.)
        content: Idea content text
        source_line: Line number in source transcript
        confidence: Confidence score 0-1
        importance: Importance score 0-1
        entities: List of entity names mentioned
    """
    intent: str
    content: str
    source_line: int
    confidence: float = 0.5
    importance: float = 0.5
    entities: list[str] = field(default_factory=list)


@dataclass
class TopicUpdate:
    """Update to an existing span/topic.

    Attributes:
        span_id: ID of span to update
        name: New name (optional)
        summary: New summary (optional)
    """
    span_id: int
    name: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class TopicChange:
    """Topic/span change (new topic detected).

    Attributes:
        from_span_id: ID of previous span
        new_name: Name for new span
        reason: Reason for topic change
        at_line: Line number where change occurs
    """
    from_span_id: int
    new_name: str
    reason: str
    at_line: int


@dataclass
class AnsweredQuestion:
    """A question that was answered.

    Attributes:
        question_id: ID of the question idea
        answer_line: Line where answer appears
    """
    question_id: int
    answer_line: int


@dataclass
class CompletedTodo:
    """A todo that was completed.

    Attributes:
        todo_id: ID of the todo idea
    """
    todo_id: int


@dataclass
class RelationOutput:
    """Relation between ideas.

    Attributes:
        from_line: Source line of the new idea
        to_idea_id: ID of existing idea being related to
        relation_type: Type of relation
    """
    from_line: int
    to_idea_id: int
    relation_type: str


@dataclass
class EntityLink:
    """Link from a mention to a golden entity.

    Attributes:
        mention_line: Line where entity was mentioned
        golden_id: ID of golden entity to link to
    """
    mention_line: int
    golden_id: str


@dataclass
class AgentOutput:
    """Complete parsed output from indexing agent.

    Contains all extracted information organized by type.
    """
    ideas: list[IdeaOutput] = field(default_factory=list)
    topic_updates: list[TopicUpdate] = field(default_factory=list)
    topic_changes: list[TopicChange] = field(default_factory=list)
    answered_questions: list[AnsweredQuestion] = field(default_factory=list)
    completed_todos: list[CompletedTodo] = field(default_factory=list)
    relations: list[RelationOutput] = field(default_factory=list)
    entity_links: list[EntityLink] = field(default_factory=list)
    skip_lines: list[int] = field(default_factory=list)
    activated_ideas: list[int] = field(default_factory=list)


def parse_agent_output(raw: dict[str, Any]) -> AgentOutput:
    """Parse raw agent output dict into structured AgentOutput.

    Args:
        raw: Raw dict from agent JSON response

    Returns:
        Parsed AgentOutput with all fields populated

    Raises:
        ParseError: If validation fails
    """
    output = AgentOutput()

    # Parse ideas
    for item in raw.get("ideas", []):
        intent = item.get("type", "")
        if intent not in VALID_INTENTS:
            raise ParseError(f"Invalid intent type: {intent}")

        output.ideas.append(IdeaOutput(
            intent=intent,
            content=item.get("content", ""),
            source_line=item.get("source_line", 0),
            confidence=item.get("confidence", 0.5),
            importance=item.get("importance", 0.5),
            entities=item.get("entities", []),
        ))

    # Parse topic updates
    for item in raw.get("topic_updates", []):
        output.topic_updates.append(TopicUpdate(
            span_id=item.get("span_id", 0),
            name=item.get("name"),
            summary=item.get("summary"),
        ))

    # Parse topic changes
    for item in raw.get("topic_changes", []):
        output.topic_changes.append(TopicChange(
            from_span_id=item.get("from_span_id", 0),
            new_name=item.get("new_name", ""),
            reason=item.get("reason", ""),
            at_line=item.get("at_line", 0),
        ))

    # Parse answered questions
    for item in raw.get("answered_questions", []):
        output.answered_questions.append(AnsweredQuestion(
            question_id=item.get("question_id", 0),
            answer_line=item.get("answer_line", 0),
        ))

    # Parse completed todos
    for item in raw.get("completed_todos", []):
        output.completed_todos.append(CompletedTodo(
            todo_id=item.get("todo_id", 0),
        ))

    # Parse relations
    for item in raw.get("relations", []):
        rel_type = item.get("type", "")
        if rel_type not in VALID_RELATIONS:
            raise ParseError(f"Invalid relation type: {rel_type}")

        output.relations.append(RelationOutput(
            from_line=item.get("from_line", 0),
            to_idea_id=item.get("to_idea_id", 0),
            relation_type=rel_type,
        ))

    # Parse entity links
    for item in raw.get("entity_links", []):
        output.entity_links.append(EntityLink(
            mention_line=item.get("mention_line", 0),
            golden_id=item.get("golden_id", ""),
        ))

    # Parse simple lists
    output.skip_lines = raw.get("skip_lines", [])
    output.activated_ideas = raw.get("activated_ideas", [])

    return output
