"""Tests for tool registry - Slice 2.9."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_can_register_tool_with_handler(self):
        """Can register a tool with its handler function."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        async def handler(query: str) -> list:
            return []

        registry.register(tool, handler)

        assert "search_ideas" in registry._tools

    def test_can_get_all_definitions_as_json(self):
        """Can get all tool definitions in JSON schema format."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()

        tool1 = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )
        tool2 = ToolDefinition(
            name="get_stats",
            description="Get statistics.",
            parameters={}
        )

        registry.register(tool1, AsyncMock())
        registry.register(tool2, AsyncMock())

        definitions = registry.get_tool_definitions()

        assert len(definitions) == 2
        assert definitions[0]["name"] == "search_ideas"
        assert definitions[1]["name"] == "get_stats"
        # Should be valid JSON
        json.dumps(definitions)

    @pytest.mark.asyncio
    async def test_can_invoke_tool_by_name(self):
        """Can invoke a registered tool by name."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        async def handler(query: str) -> list:
            return [{"id": 1, "content": f"Result for: {query}"}]

        registry.register(tool, handler)

        result = await registry.invoke("search_ideas", {"query": "test"})

        assert len(result) == 1
        assert "test" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_raises_error_for_unknown_tool(self):
        """Raises error when invoking unknown tool."""
        from llm.tool_registry import ToolRegistry, ToolNotFoundError

        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError, match="Tool not found: unknown_tool"):
            await registry.invoke("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_validates_arguments_before_invoke(self):
        """Validates arguments before invoking tool."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter, ValidationError

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        registry.register(tool, AsyncMock())

        with pytest.raises(ValidationError, match="Missing required parameter"):
            await registry.invoke("search_ideas", {})

    @pytest.mark.asyncio
    async def test_passes_arguments_to_handler(self):
        """Passes correct arguments to handler function."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Limit", required=False, default=10),
            }
        )

        mock_handler = AsyncMock(return_value=[])
        registry.register(tool, mock_handler)

        await registry.invoke("search_ideas", {"query": "test", "limit": 5})

        mock_handler.assert_called_once_with(query="test", limit=5)

    @pytest.mark.asyncio
    async def test_uses_default_for_missing_optional_params(self):
        """Uses default values for missing optional parameters."""
        from llm.tool_registry import ToolRegistry
        from llm.tool_schema import ToolDefinition, ToolParameter

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Limit", required=False, default=10),
            }
        )

        mock_handler = AsyncMock(return_value=[])
        registry.register(tool, mock_handler)

        await registry.invoke("search_ideas", {"query": "test"})

        # Should be called with default limit
        mock_handler.assert_called_once_with(query="test", limit=10)


class TestIndexingToolsRegistry:
    """Tests for the pre-configured INDEXING_TOOLS registry."""

    def test_indexing_tools_registry_exists(self):
        """INDEXING_TOOLS registry should exist with registered tools."""
        from llm.tool_registry import INDEXING_TOOLS

        definitions = INDEXING_TOOLS.get_tool_definitions()

        # Should have at least the tools we implemented
        tool_names = [d["name"] for d in definitions]
        assert "search_ideas" in tool_names
        assert "get_open_questions" in tool_names
        assert "get_open_todos" in tool_names
        assert "get_current_span" in tool_names
        assert "list_session_spans" in tool_names
        assert "search_entities" in tool_names
        assert "get_recent_ideas" in tool_names

    def test_all_tools_have_valid_schema(self):
        """All registered tools should have valid JSON schema."""
        from llm.tool_registry import INDEXING_TOOLS

        definitions = INDEXING_TOOLS.get_tool_definitions()

        for defn in definitions:
            assert "name" in defn
            assert "description" in defn
            assert "input_schema" in defn
            assert defn["input_schema"]["type"] == "object"
            # Should serialize to JSON
            json.dumps(defn)
