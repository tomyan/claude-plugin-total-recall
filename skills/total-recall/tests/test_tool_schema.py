"""Tests for tool definition schema - Slice 2.1."""

import json
import pytest
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_can_create_tool_definition(self):
        """Can create a tool definition with name and description."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for existing ideas semantically similar to a query.",
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="Search query"
                )
            }
        )

        assert tool.name == "search_ideas"
        assert tool.description == "Search for existing ideas semantically similar to a query."
        assert "query" in tool.parameters

    def test_tool_with_multiple_parameters(self):
        """Can create tool with multiple parameters."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Max results", required=False, default=10),
                "session": ToolParameter(type="string", description="Session filter", required=False),
            }
        )

        assert len(tool.parameters) == 3
        assert tool.parameters["limit"].required is False
        assert tool.parameters["limit"].default == 10

    def test_tool_with_no_parameters(self):
        """Can create tool with no parameters."""
        from llm.tool_schema import ToolDefinition

        tool = ToolDefinition(
            name="get_current_state",
            description="Get current indexing state.",
            parameters={}
        )

        assert tool.name == "get_current_state"
        assert len(tool.parameters) == 0


class TestToolParameterTypes:
    """Tests for ToolParameter type handling."""

    def test_string_parameter(self):
        """String parameter type."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(type="string", description="A string value")
        assert param.type == "string"

    def test_integer_parameter(self):
        """Integer parameter type."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(type="integer", description="An integer value")
        assert param.type == "integer"

    def test_boolean_parameter(self):
        """Boolean parameter type."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(type="boolean", description="A boolean value")
        assert param.type == "boolean"

    def test_array_parameter(self):
        """Array parameter type."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(type="array", description="An array value", items_type="string")
        assert param.type == "array"
        assert param.items_type == "string"

    def test_optional_parameter_with_default(self):
        """Optional parameter with default value."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(
            type="integer",
            description="Limit",
            required=False,
            default=10
        )
        assert param.required is False
        assert param.default == 10


class TestJSONSchemaFormat:
    """Tests for JSON schema serialization."""

    def test_serialize_simple_tool(self):
        """Serialize simple tool to JSON schema format."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query")
            }
        )

        schema = tool.to_json_schema()

        assert schema["name"] == "search_ideas"
        assert schema["description"] == "Search for ideas."
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "query" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["properties"]["query"]["type"] == "string"
        assert "query" in schema["input_schema"]["required"]

    def test_serialize_tool_with_optional_params(self):
        """Serialize tool with optional parameters."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Max results", required=False, default=10),
            }
        )

        schema = tool.to_json_schema()

        # Only required params in required list
        assert "query" in schema["input_schema"]["required"]
        assert "limit" not in schema["input_schema"]["required"]
        # Optional param has default in description or schema
        assert schema["input_schema"]["properties"]["limit"]["type"] == "integer"

    def test_serialize_tool_with_array_param(self):
        """Serialize tool with array parameter."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="get_ideas_by_ids",
            description="Get ideas by their IDs.",
            parameters={
                "ids": ToolParameter(type="array", description="List of idea IDs", items_type="integer")
            }
        )

        schema = tool.to_json_schema()

        assert schema["input_schema"]["properties"]["ids"]["type"] == "array"
        assert schema["input_schema"]["properties"]["ids"]["items"]["type"] == "integer"

    def test_schema_is_valid_json(self):
        """Schema should be valid JSON."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool.",
            parameters={
                "param1": ToolParameter(type="string", description="Param 1"),
            }
        )

        schema = tool.to_json_schema()
        # Should be serializable to JSON
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_tool"


class TestArgumentValidation:
    """Tests for tool argument validation."""

    def test_validate_required_params_present(self):
        """Validate that required parameters are present."""
        from llm.tool_schema import ToolDefinition, ToolParameter, ValidationError

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        # Should pass validation
        tool.validate_args({"query": "test"})

    def test_validate_missing_required_param(self):
        """Validation fails for missing required parameter."""
        from llm.tool_schema import ToolDefinition, ToolParameter, ValidationError

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        with pytest.raises(ValidationError, match="Missing required parameter: query"):
            tool.validate_args({})

    def test_validate_wrong_type(self):
        """Validation fails for wrong parameter type."""
        from llm.tool_schema import ToolDefinition, ToolParameter, ValidationError

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "limit": ToolParameter(type="integer", description="Max results"),
            }
        )

        with pytest.raises(ValidationError, match="Expected integer"):
            tool.validate_args({"limit": "not an int"})

    def test_validate_optional_param_absent(self):
        """Validation passes when optional parameter is absent."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="integer", description="Max results", required=False, default=10),
            }
        )

        # Should pass - only required param provided
        tool.validate_args({"query": "test"})

    def test_validate_extra_params_ignored(self):
        """Extra parameters are ignored (flexible schema)."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search_ideas",
            description="Search for ideas.",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
            }
        )

        # Should pass - extra params ignored
        tool.validate_args({"query": "test", "extra": "ignored"})


class TestEdgeCases:
    """Edge case tests for tool schema."""

    def test_empty_string_description(self):
        """Tool can have empty description."""
        from llm.tool_schema import ToolDefinition

        tool = ToolDefinition(
            name="internal_tool",
            description="",
            parameters={}
        )
        assert tool.description == ""

    def test_special_characters_in_name(self):
        """Tool name with underscores works."""
        from llm.tool_schema import ToolDefinition

        tool = ToolDefinition(
            name="get_open_questions_for_session",
            description="Get questions.",
            parameters={}
        )
        assert tool.name == "get_open_questions_for_session"

    def test_null_default_value(self):
        """Parameter can have None as default."""
        from llm.tool_schema import ToolParameter

        param = ToolParameter(
            type="string",
            description="Optional filter",
            required=False,
            default=None
        )
        assert param.default is None

    def test_validate_null_value_for_optional(self):
        """Null value is valid for optional parameter."""
        from llm.tool_schema import ToolDefinition, ToolParameter

        tool = ToolDefinition(
            name="search",
            description="Search.",
            parameters={
                "filter": ToolParameter(type="string", description="Filter", required=False),
            }
        )

        # Should pass - null for optional is ok
        tool.validate_args({"filter": None})
