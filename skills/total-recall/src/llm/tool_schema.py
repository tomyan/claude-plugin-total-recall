"""Tool definition schema for LLM tool use - Slice 2.1.

Provides dataclasses for defining tools that can be converted to JSON schema
format for use with Claude's tool use API.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


class ValidationError(Exception):
    """Raised when tool argument validation fails."""
    pass


@dataclass
class ToolParameter:
    """Definition of a single tool parameter.

    Attributes:
        type: The parameter type (string, integer, boolean, array)
        description: Human-readable description
        required: Whether the parameter is required (default True)
        default: Default value if not provided
        items_type: For array types, the type of array items
    """
    type: str
    description: str
    required: bool = True
    default: Any = None
    items_type: Optional[str] = None


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by an LLM.

    Attributes:
        name: The tool name (used in tool calls)
        description: Human-readable description of what the tool does
        parameters: Dictionary mapping parameter names to ToolParameter objects
    """
    name: str
    description: str
    parameters: dict[str, ToolParameter] = field(default_factory=dict)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format for Claude tool use API.

        Returns a dict with:
            - name: Tool name
            - description: Tool description
            - input_schema: JSON schema for parameters
        """
        properties = {}
        required = []

        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description,
            }

            # Handle array items
            if param.type == "array" and param.items_type:
                prop["items"] = {"type": param.items_type}

            properties[param_name] = prop

            if param.required:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }

    def validate_args(self, args: dict[str, Any]) -> None:
        """Validate arguments against the tool schema.

        Args:
            args: Dictionary of argument name -> value

        Raises:
            ValidationError: If validation fails
        """
        # Check required parameters
        for param_name, param in self.parameters.items():
            if param.required and param_name not in args:
                raise ValidationError(f"Missing required parameter: {param_name}")

        # Type check provided arguments
        for arg_name, arg_value in args.items():
            if arg_name not in self.parameters:
                # Extra parameters are ignored (flexible schema)
                continue

            param = self.parameters[arg_name]

            # None is valid for optional parameters
            if arg_value is None and not param.required:
                continue

            # Skip None check for required (already caught above)
            if arg_value is None:
                continue

            # Type validation
            expected_type = param.type
            if expected_type == "string" and not isinstance(arg_value, str):
                raise ValidationError(f"Expected string for {arg_name}, got {type(arg_value).__name__}")
            elif expected_type == "integer" and not isinstance(arg_value, int):
                raise ValidationError(f"Expected integer for {arg_name}, got {type(arg_value).__name__}")
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                raise ValidationError(f"Expected boolean for {arg_name}, got {type(arg_value).__name__}")
            elif expected_type == "array" and not isinstance(arg_value, list):
                raise ValidationError(f"Expected array for {arg_name}, got {type(arg_value).__name__}")
