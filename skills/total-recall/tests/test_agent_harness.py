"""Tests for agent harness - Slices 3.2-3.4."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAgentTurn:
    """Tests for single agent turn - Slice 3.2."""

    @pytest.mark.asyncio
    async def test_sends_messages_to_llm(self):
        """Should send messages to LLM."""
        from llm.agent_harness import agent_turn
        from llm.agent_types import AgentMessage
        from llm.tool_registry import ToolRegistry

        messages = [
            AgentMessage(role="system", content="You are helpful."),
            AgentMessage(role="user", content="Hello"),
        ]

        registry = ToolRegistry()

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": "Hi there!"}],
                "stop_reason": "end_turn"
            }

            result = await agent_turn(messages, registry)

        mock_call.assert_called_once()
        # Check messages were passed
        call_args = mock_call.call_args
        assert len(call_args[0][0]) == 2  # Two messages

    @pytest.mark.asyncio
    async def test_includes_tool_definitions(self):
        """Should include tool definitions in LLM call."""
        from llm.agent_harness import agent_turn
        from llm.agent_types import AgentMessage
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        messages = [AgentMessage(role="user", content="Search")]
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="search",
                description="Search function",
                parameters={"q": ToolParameter(type="string", description="Query")}
            ),
            AsyncMock()
        )

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "end_turn"
            }

            await agent_turn(messages, registry)

        call_args = mock_call.call_args
        tools = call_args[0][1]  # Second positional arg is tools
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_returns_assistant_message(self):
        """Should return assistant message from response."""
        from llm.agent_harness import agent_turn
        from llm.agent_types import AgentMessage
        from llm.tool_registry import ToolRegistry

        messages = [AgentMessage(role="user", content="Hello")]
        registry = ToolRegistry()

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": "Hello back!"}],
                "stop_reason": "end_turn"
            }

            result = await agent_turn(messages, registry)

        assert result.role == "assistant"
        assert result.content == "Hello back!"

    @pytest.mark.asyncio
    async def test_parses_tool_calls_from_response(self):
        """Should parse tool calls from LLM response."""
        from llm.agent_harness import agent_turn
        from llm.agent_types import AgentMessage
        from llm.tool_registry import ToolRegistry

        messages = [AgentMessage(role="user", content="Search for auth")]
        registry = ToolRegistry()

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "search_ideas",
                        "input": {"query": "authentication"}
                    }
                ],
                "stop_reason": "tool_use"
            }

            result = await agent_turn(messages, registry)

        assert result.role == "assistant"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_ideas"
        assert result.tool_calls[0].id == "call_123"


class TestToolExecution:
    """Tests for tool execution - Slice 3.3."""

    @pytest.mark.asyncio
    async def test_executes_each_tool_call(self):
        """Should execute each tool call."""
        from llm.agent_harness import execute_tool_calls
        from llm.agent_types import ToolCall
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()
        mock_handler = AsyncMock(return_value=[{"id": 1}])

        registry.register(
            ToolDefinition(
                name="search",
                description="Search",
                parameters={"q": ToolParameter(type="string", description="Query")}
            ),
            mock_handler
        )

        tool_calls = [
            ToolCall(id="call_1", name="search", arguments={"q": "test"})
        ]

        results = await execute_tool_calls(tool_calls, registry)

        mock_handler.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_creates_result_message_per_call(self):
        """Should create a tool_result message for each call."""
        from llm.agent_harness import execute_tool_calls
        from llm.agent_types import ToolCall
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="tool1",
                description="Tool 1",
                parameters={}
            ),
            AsyncMock(return_value="result1")
        )
        registry.register(
            ToolDefinition(
                name="tool2",
                description="Tool 2",
                parameters={}
            ),
            AsyncMock(return_value="result2")
        )

        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={}),
            ToolCall(id="call_2", name="tool2", arguments={}),
        ]

        results = await execute_tool_calls(tool_calls, registry)

        assert len(results) == 2
        assert results[0].role == "tool_result"
        assert results[0].tool_call_id == "call_1"
        assert results[1].tool_call_id == "call_2"

    @pytest.mark.asyncio
    async def test_handles_tool_errors_gracefully(self):
        """Should handle tool errors and include error in result."""
        from llm.agent_harness import execute_tool_calls
        from llm.agent_types import ToolCall
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="failing_tool",
                description="A tool that fails",
                parameters={}
            ),
            AsyncMock(side_effect=Exception("Tool error"))
        )

        tool_calls = [
            ToolCall(id="call_1", name="failing_tool", arguments={})
        ]

        results = await execute_tool_calls(tool_calls, registry)

        assert len(results) == 1
        assert results[0].role == "tool_result"
        assert "error" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_includes_tool_call_id_in_result(self):
        """Should include tool_call_id in result message."""
        from llm.agent_harness import execute_tool_calls
        from llm.agent_types import ToolCall
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="test", description="Test", parameters={}),
            AsyncMock(return_value={})
        )

        tool_calls = [ToolCall(id="unique_id_123", name="test", arguments={})]

        results = await execute_tool_calls(tool_calls, registry)

        assert results[0].tool_call_id == "unique_id_123"


class TestAgentLoopMultiTurn:
    """Tests for multi-turn agent loop - Slice 3.4."""

    @pytest.mark.asyncio
    async def test_runs_until_no_tool_calls(self):
        """Should run until assistant responds without tool calls."""
        from llm.agent_harness import run_agent
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="search", description="Search", parameters={}),
            AsyncMock(return_value=[])
        )

        responses = [
            # First response: tool call
            {
                "content": [
                    {"type": "text", "text": "Searching..."},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {}}
                ],
                "stop_reason": "tool_use"
            },
            # Second response: final answer
            {
                "content": [{"type": "text", "text": '{"items": []}'}],
                "stop_reason": "end_turn"
            }
        ]

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.side_effect = responses

            result = await run_agent(
                system_prompt="You are helpful",
                user_input="Search for ideas",
                tools=registry
            )

        assert mock_call.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_max_turns_limit(self):
        """Should stop at max_turns even if still making tool calls."""
        from llm.agent_harness import run_agent
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="search", description="Search", parameters={}),
            AsyncMock(return_value=[])
        )

        # Always returns tool calls
        endless_response = {
            "content": [
                {"type": "tool_use", "id": "c1", "name": "search", "input": {}}
            ],
            "stop_reason": "tool_use"
        }

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = endless_response

            result = await run_agent(
                system_prompt="You are helpful",
                user_input="Search forever",
                tools=registry,
                max_turns=3
            )

        # Should stop at max_turns
        assert mock_call.call_count <= 3

    @pytest.mark.asyncio
    async def test_accumulates_conversation_history(self):
        """Should accumulate messages through the conversation."""
        from llm.agent_harness import run_agent
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="search", description="Search", parameters={}),
            AsyncMock(return_value=[{"id": 1}])
        )

        responses = [
            {
                "content": [
                    {"type": "text", "text": "Searching..."},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {}}
                ],
                "stop_reason": "tool_use"
            },
            {
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "end_turn"
            }
        ]

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.side_effect = responses

            await run_agent(
                system_prompt="You are helpful",
                user_input="Search",
                tools=registry
            )

        # Second call should have more messages than first
        first_call_msgs = mock_call.call_args_list[0][0][0]
        second_call_msgs = mock_call.call_args_list[1][0][0]
        assert len(second_call_msgs) > len(first_call_msgs)

    @pytest.mark.asyncio
    async def test_parses_final_response_as_json(self):
        """Should parse final response as JSON."""
        from llm.agent_harness import run_agent
        from llm.tool_registry import ToolRegistry

        registry = ToolRegistry()

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": '{"items": [{"type": "decision"}]}'}],
                "stop_reason": "end_turn"
            }

            result = await run_agent(
                system_prompt="You are helpful",
                user_input="Process",
                tools=registry
            )

        assert "items" in result
        assert len(result["items"]) == 1

    @pytest.mark.asyncio
    async def test_handles_malformed_json_gracefully(self):
        """Should handle malformed JSON in final response."""
        from llm.agent_harness import run_agent
        from llm.tool_registry import ToolRegistry

        registry = ToolRegistry()

        with patch('llm.agent_harness.call_claude_with_tools') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": "Not valid JSON"}],
                "stop_reason": "end_turn"
            }

            result = await run_agent(
                system_prompt="You are helpful",
                user_input="Process",
                tools=registry
            )

        # Should return error dict or raw content
        assert "error" in result or "raw" in result
