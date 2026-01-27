"""Tool registry for indexing agent - Slice 2.9.

Provides a registry for managing tool definitions and handlers,
allowing the agent to look up and invoke tools by name.
"""

from typing import Any, Callable

from llm.tool_schema import ToolDefinition, ValidationError


class ToolNotFoundError(Exception):
    """Raised when attempting to invoke an unregistered tool."""
    pass


class ToolRegistry:
    """Registry for managing LLM tools and their handlers.

    Stores tool definitions and their async handler functions,
    enabling lookup and invocation by tool name.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {}

    def register(self, tool: ToolDefinition, handler: Callable) -> None:
        """Register a tool with its handler function.

        Args:
            tool: Tool definition with name, description, and parameters
            handler: Async function to handle tool invocations
        """
        self._tools[tool.name] = (tool, handler)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in JSON schema format.

        Returns:
            List of tool definitions as JSON-serializable dicts
        """
        return [tool.to_json_schema() for tool, _ in self._tools.values()]

    async def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Dict of argument name -> value

        Returns:
            Result from the tool handler

        Raises:
            ToolNotFoundError: If tool is not registered
            ValidationError: If arguments fail validation
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")

        tool, handler = self._tools[tool_name]

        # Validate arguments
        tool.validate_args(arguments)

        # Fill in defaults for missing optional parameters
        call_args = {}
        for param_name, param in tool.parameters.items():
            if param_name in arguments:
                call_args[param_name] = arguments[param_name]
            elif not param.required and param.default is not None:
                call_args[param_name] = param.default

        # Invoke handler
        return await handler(**call_args)


# Pre-configured registry with all indexing tools
def _create_indexing_tools_registry() -> ToolRegistry:
    """Create and configure the indexing tools registry."""
    from llm.tool_schema import ToolParameter
    from llm import indexing_tools

    registry = ToolRegistry()

    # search_ideas tool
    registry.register(
        ToolDefinition(
            name="search_ideas",
            description="Search for ideas semantically similar to a query.",
            parameters={
                "query": ToolParameter(type="string", description="Search query text"),
                "limit": ToolParameter(type="integer", description="Maximum number of results", required=False, default=10),
                "session": ToolParameter(type="string", description="Filter by session ID", required=False),
                "intent": ToolParameter(type="string", description="Filter by intent type", required=False),
            }
        ),
        indexing_tools.tool_search_ideas
    )

    # get_open_questions tool
    registry.register(
        ToolDefinition(
            name="get_open_questions",
            description="Get unanswered questions for a session.",
            parameters={
                "session": ToolParameter(type="string", description="Session ID to filter by"),
                "limit": ToolParameter(type="integer", description="Maximum number of results", required=False, default=10),
            }
        ),
        indexing_tools.tool_get_open_questions
    )

    # get_open_todos tool
    registry.register(
        ToolDefinition(
            name="get_open_todos",
            description="Get incomplete todos for a session.",
            parameters={
                "session": ToolParameter(type="string", description="Session ID to filter by"),
                "limit": ToolParameter(type="integer", description="Maximum number of results", required=False, default=10),
            }
        ),
        indexing_tools.tool_get_open_todos
    )

    # get_current_span tool
    registry.register(
        ToolDefinition(
            name="get_current_span",
            description="Get the most recent span for a session.",
            parameters={
                "session": ToolParameter(type="string", description="Session ID"),
            }
        ),
        indexing_tools.tool_get_current_span
    )

    # list_session_spans tool
    registry.register(
        ToolDefinition(
            name="list_session_spans",
            description="List all spans for a session, ordered by start line.",
            parameters={
                "session": ToolParameter(type="string", description="Session ID"),
            }
        ),
        indexing_tools.tool_list_session_spans
    )

    # search_entities tool
    registry.register(
        ToolDefinition(
            name="search_entities",
            description="Search for golden entities by name with fuzzy matching.",
            parameters={
                "name": ToolParameter(type="string", description="Entity name to search for"),
                "type": ToolParameter(type="string", description="Filter by entity type", required=False),
            }
        ),
        indexing_tools.tool_search_entities
    )

    # get_recent_ideas tool
    registry.register(
        ToolDefinition(
            name="get_recent_ideas",
            description="Get recent ideas for a session, ordered by recency.",
            parameters={
                "session": ToolParameter(type="string", description="Session ID to filter by"),
                "limit": ToolParameter(type="integer", description="Maximum number of results", required=False, default=20),
                "intent": ToolParameter(type="string", description="Filter by intent type", required=False),
            }
        ),
        indexing_tools.tool_get_recent_ideas
    )

    return registry


# Singleton instance of the indexing tools registry
INDEXING_TOOLS = _create_indexing_tools_registry()
