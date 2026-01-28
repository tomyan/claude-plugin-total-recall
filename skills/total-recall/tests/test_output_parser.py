"""Tests for agent output parser - Slice 4.1."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestParseAgentOutput:
    """Tests for parse_agent_output function."""

    def test_parses_all_fields_from_valid_output(self):
        """Should parse all fields from valid output."""
        from indexer.output_parser import parse_agent_output

        raw = {
            "ideas": [
                {"type": "decision", "content": "Use JWT", "source_line": 10, "confidence": 0.9}
            ],
            "topic_updates": [
                {"span_id": 1, "name": "Authentication", "summary": "Auth work"}
            ],
            "topic_changes": [
                {"from_span_id": 1, "new_name": "OAuth", "reason": "Switching to OAuth", "at_line": 50}
            ],
            "answered_questions": [
                {"question_id": 5, "answer_line": 20}
            ],
            "completed_todos": [
                {"todo_id": 7}
            ],
            "relations": [
                {"from_line": 10, "to_idea_id": 3, "type": "supersedes"}
            ],
            "entity_links": [
                {"mention_line": 15, "golden_id": "abc123"}
            ],
            "skip_lines": [1, 2, 3],
            "activated_ideas": [10, 20, 30],
        }

        output = parse_agent_output(raw)

        assert len(output.ideas) == 1
        assert len(output.topic_updates) == 1
        assert len(output.topic_changes) == 1
        assert len(output.answered_questions) == 1
        assert len(output.completed_todos) == 1
        assert len(output.relations) == 1
        assert len(output.entity_links) == 1
        assert output.skip_lines == [1, 2, 3]
        assert output.activated_ideas == [10, 20, 30]

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields."""
        from indexer.output_parser import parse_agent_output

        raw = {
            "ideas": [
                {"type": "context", "content": "Info", "source_line": 5}
            ]
        }

        output = parse_agent_output(raw)

        assert len(output.ideas) == 1
        assert output.topic_updates == []
        assert output.topic_changes == []
        assert output.answered_questions == []
        assert output.completed_todos == []
        assert output.relations == []
        assert output.entity_links == []
        assert output.skip_lines == []
        assert output.activated_ideas == []

    def test_validates_intent_types(self):
        """Should validate intent types on ideas."""
        from indexer.output_parser import parse_agent_output, ParseError

        raw = {
            "ideas": [
                {"type": "invalid_type", "content": "Test", "source_line": 1}
            ]
        }

        with pytest.raises(ParseError, match="Invalid intent type"):
            parse_agent_output(raw)

    def test_validates_relation_types(self):
        """Should validate relation types."""
        from indexer.output_parser import parse_agent_output, ParseError

        raw = {
            "ideas": [],
            "relations": [
                {"from_line": 1, "to_idea_id": 2, "type": "invalid_relation"}
            ]
        }

        with pytest.raises(ParseError, match="Invalid relation type"):
            parse_agent_output(raw)

    def test_returns_empty_lists_for_missing_arrays(self):
        """Should return empty lists for missing array fields."""
        from indexer.output_parser import parse_agent_output

        raw = {}

        output = parse_agent_output(raw)

        assert output.ideas == []
        assert output.topic_updates == []
        assert output.relations == []

    def test_parses_idea_with_all_fields(self):
        """Should parse idea with all optional fields."""
        from indexer.output_parser import parse_agent_output

        raw = {
            "ideas": [
                {
                    "type": "decision",
                    "content": "Use PostgreSQL",
                    "source_line": 25,
                    "confidence": 0.85,
                    "importance": 0.9,
                    "entities": ["PostgreSQL", "database"],
                }
            ]
        }

        output = parse_agent_output(raw)

        idea = output.ideas[0]
        assert idea.intent == "decision"
        assert idea.content == "Use PostgreSQL"
        assert idea.source_line == 25
        assert idea.confidence == 0.85
        assert idea.importance == 0.9
        assert idea.entities == ["PostgreSQL", "database"]

    def test_default_confidence_is_half(self):
        """Should default confidence to 0.5."""
        from indexer.output_parser import parse_agent_output

        raw = {
            "ideas": [
                {"type": "context", "content": "Info", "source_line": 1}
            ]
        }

        output = parse_agent_output(raw)

        assert output.ideas[0].confidence == 0.5


class TestOutputDataClasses:
    """Tests for output data classes."""

    def test_idea_output_creation(self):
        """Can create IdeaOutput."""
        from indexer.output_parser import IdeaOutput

        idea = IdeaOutput(
            intent="decision",
            content="Test decision",
            source_line=10,
            confidence=0.8
        )

        assert idea.intent == "decision"
        assert idea.content == "Test decision"

    def test_topic_update_creation(self):
        """Can create TopicUpdate."""
        from indexer.output_parser import TopicUpdate

        update = TopicUpdate(
            span_id=1,
            name="New Name",
            summary="New summary"
        )

        assert update.span_id == 1
        assert update.name == "New Name"

    def test_topic_change_creation(self):
        """Can create TopicChange."""
        from indexer.output_parser import TopicChange

        change = TopicChange(
            from_span_id=1,
            new_name="Different Topic",
            reason="Topic shifted",
            at_line=100
        )

        assert change.from_span_id == 1
        assert change.at_line == 100

    def test_relation_output_creation(self):
        """Can create RelationOutput."""
        from indexer.output_parser import RelationOutput

        relation = RelationOutput(
            from_line=10,
            to_idea_id=5,
            relation_type="supersedes"
        )

        assert relation.from_line == 10
        assert relation.relation_type == "supersedes"


class TestValidIntentTypes:
    """Tests for valid intent type checking."""

    def test_all_valid_intents_accepted(self):
        """All valid intent types should be accepted."""
        from indexer.output_parser import parse_agent_output

        valid_intents = [
            "decision", "conclusion", "question", "problem",
            "solution", "todo", "context", "observation"
        ]

        for intent in valid_intents:
            raw = {
                "ideas": [{"type": intent, "content": "Test", "source_line": 1}]
            }
            output = parse_agent_output(raw)
            assert output.ideas[0].intent == intent


class TestValidRelationTypes:
    """Tests for valid relation type checking."""

    def test_all_valid_relations_accepted(self):
        """All valid relation types should be accepted."""
        from indexer.output_parser import parse_agent_output

        valid_relations = [
            "supersedes", "builds_on", "contradicts", "answers", "relates_to"
        ]

        for rel_type in valid_relations:
            raw = {
                "ideas": [],
                "relations": [{"from_line": 1, "to_idea_id": 2, "type": rel_type}]
            }
            output = parse_agent_output(raw)
            assert output.relations[0].relation_type == rel_type
