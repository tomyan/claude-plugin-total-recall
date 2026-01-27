"""Tests for agent message types - Slice 3.1."""

import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_create_system_message(self):
        """Can create a system message."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="system", content="You are an indexing agent.")

        assert msg.role == "system"
        assert msg.content == "You are an indexing agent."
        assert msg.tool_calls is None

    def test_create_user_message(self):
        """Can create a user message."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="user", content="Process these messages.")

        assert msg.role == "user"
        assert msg.content == "Process these messages."

    def test_create_assistant_message(self):
        """Can create an assistant message."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="assistant", content="I'll analyze the transcript.")

        assert msg.role == "assistant"
        assert msg.content == "I'll analyze the transcript."

    def test_create_message_with_tool_calls(self):
        """Can create a message with tool calls."""
        from llm.agent_types import AgentMessage, ToolCall

        tool_calls = [
            ToolCall(id="call_1", name="search_ideas", arguments={"query": "auth"})
        ]

        msg = AgentMessage(
            role="assistant",
            content="Let me search for related ideas.",
            tool_calls=tool_calls
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search_ideas"

    def test_create_tool_result_message(self):
        """Can create a tool result message."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(
            role="tool_result",
            content='[{"id": 1, "content": "Use JWT tokens"}]',
            tool_call_id="call_1"
        )

        assert msg.role == "tool_result"
        assert msg.tool_call_id == "call_1"

    def test_serialize_to_api_format(self):
        """Can serialize message to Claude API format."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="user", content="Hello")
        api_format = msg.to_api_format()

        assert api_format["role"] == "user"
        assert api_format["content"] == "Hello"

    def test_serialize_system_to_api_format(self):
        """System messages serialize correctly."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="system", content="You are helpful.")
        api_format = msg.to_api_format()

        # System messages use 'system' role in API
        assert api_format["role"] == "system"
        assert api_format["content"] == "You are helpful."

    def test_serialize_assistant_with_tool_calls(self):
        """Assistant messages with tool calls serialize correctly."""
        from llm.agent_types import AgentMessage, ToolCall

        msg = AgentMessage(
            role="assistant",
            content="Searching...",
            tool_calls=[
                ToolCall(id="call_1", name="search_ideas", arguments={"query": "test"})
            ]
        )

        api_format = msg.to_api_format()

        assert api_format["role"] == "assistant"
        assert "content" in api_format
        # Tool calls should be in content blocks
        content_blocks = api_format["content"]
        assert isinstance(content_blocks, list)
        tool_use_block = next(b for b in content_blocks if b.get("type") == "tool_use")
        assert tool_use_block["name"] == "search_ideas"
        assert tool_use_block["id"] == "call_1"

    def test_serialize_tool_result_to_api_format(self):
        """Tool result messages serialize correctly."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(
            role="tool_result",
            content='{"results": []}',
            tool_call_id="call_1"
        )

        api_format = msg.to_api_format()

        assert api_format["role"] == "user"  # Tool results are sent as user messages
        content_blocks = api_format["content"]
        assert isinstance(content_blocks, list)
        result_block = content_blocks[0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "call_1"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_tool_call(self):
        """Can create a tool call."""
        from llm.agent_types import ToolCall

        call = ToolCall(
            id="call_123",
            name="search_ideas",
            arguments={"query": "authentication"}
        )

        assert call.id == "call_123"
        assert call.name == "search_ideas"
        assert call.arguments["query"] == "authentication"

    def test_tool_call_with_multiple_arguments(self):
        """Tool call can have multiple arguments."""
        from llm.agent_types import ToolCall

        call = ToolCall(
            id="call_456",
            name="search_ideas",
            arguments={
                "query": "test",
                "limit": 5,
                "session": "session-1"
            }
        )

        assert len(call.arguments) == 3

    def test_tool_call_from_api_response(self):
        """Can create ToolCall from API response format."""
        from llm.agent_types import ToolCall

        api_block = {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "search_ideas",
            "input": {"query": "test"}
        }

        call = ToolCall.from_api_response(api_block)

        assert call.id == "toolu_abc123"
        assert call.name == "search_ideas"
        assert call.arguments["query"] == "test"


class TestMessageSerialization:
    """Tests for message serialization edge cases."""

    def test_messages_are_json_serializable(self):
        """Messages should be JSON serializable via to_api_format."""
        from llm.agent_types import AgentMessage, ToolCall

        messages = [
            AgentMessage(role="system", content="You are helpful."),
            AgentMessage(role="user", content="Hello"),
            AgentMessage(
                role="assistant",
                content="Searching",
                tool_calls=[ToolCall(id="c1", name="search", arguments={})]
            ),
            AgentMessage(role="tool_result", content="{}", tool_call_id="c1"),
        ]

        for msg in messages:
            api_format = msg.to_api_format()
            # Should not raise
            json.dumps(api_format)

    def test_empty_content_handling(self):
        """Empty content is handled correctly."""
        from llm.agent_types import AgentMessage

        msg = AgentMessage(role="assistant", content="")
        api_format = msg.to_api_format()

        assert api_format["content"] == ""

    def test_special_characters_in_content(self):
        """Special characters in content are preserved."""
        from llm.agent_types import AgentMessage

        content = 'Code: `print("hello")`\nNewline\ttab'
        msg = AgentMessage(role="user", content=content)
        api_format = msg.to_api_format()

        assert api_format["content"] == content
