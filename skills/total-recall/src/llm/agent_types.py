"""Agent message types for LLM conversation - Slice 3.1.

Defines the message types used in agent conversations with Claude,
including support for tool use.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class ToolCall:
    """Represents a tool call from an assistant message.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Dict of argument name -> value
    """
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_api_response(cls, block: dict) -> "ToolCall":
        """Create ToolCall from Claude API response format.

        Args:
            block: A tool_use content block from API response

        Returns:
            ToolCall instance
        """
        return cls(
            id=block["id"],
            name=block["name"],
            arguments=block.get("input", {})
        )


@dataclass
class AgentMessage:
    """Represents a message in an agent conversation.

    Supports system, user, assistant, and tool_result roles.

    Attributes:
        role: Message role (system, user, assistant, tool_result)
        content: Text content of the message
        tool_calls: List of tool calls (for assistant messages)
        tool_call_id: ID of the tool call this result is for (for tool_result)
    """
    role: Literal["system", "user", "assistant", "tool_result"]
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_api_format(self) -> dict[str, Any]:
        """Convert to Claude API message format.

        Returns:
            Dict suitable for Claude Messages API
        """
        if self.role == "system":
            return {
                "role": "system",
                "content": self.content
            }

        if self.role == "user":
            return {
                "role": "user",
                "content": self.content
            }

        if self.role == "assistant":
            # Assistant messages may have tool calls
            if self.tool_calls:
                content_blocks = []

                # Add text block if there's content
                if self.content:
                    content_blocks.append({
                        "type": "text",
                        "text": self.content
                    })

                # Add tool use blocks
                for call in self.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments
                    })

                return {
                    "role": "assistant",
                    "content": content_blocks
                }
            else:
                return {
                    "role": "assistant",
                    "content": self.content
                }

        if self.role == "tool_result":
            # Tool results are sent as user messages with tool_result blocks
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.content
                    }
                ]
            }

        raise ValueError(f"Unknown role: {self.role}")
