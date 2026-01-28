"""Agent harness for running LLM with tools - Slices 3.2-3.4.

Provides the agent loop for executing multi-turn conversations
with tool use support.
"""

import json
from typing import Any

from llm.agent_types import AgentMessage, ToolCall
from llm.tool_registry import ToolRegistry


async def call_claude_with_tools(
    messages: list[AgentMessage],
    tools: list[dict],
    system: str = None,
) -> dict:
    """Call Claude API with tool definitions.

    Args:
        messages: List of agent messages
        tools: List of tool definitions in JSON schema format
        system: Optional system prompt

    Returns:
        Claude API response dict with content and stop_reason
    """
    from llm.claude import claude_complete

    # Format messages for API
    api_messages = []
    for msg in messages:
        if msg.role == "system":
            # System is passed separately
            if system is None:
                system = msg.content
            continue
        api_messages.append(msg.to_api_format())

    # Build prompt with tool context
    # For now, we'll serialize tools and ask Claude to respond with tool use format
    tool_context = ""
    if tools:
        tool_context = f"\n\nAvailable tools:\n{json.dumps(tools, indent=2)}\n\n"
        tool_context += "To use a tool, respond with JSON in this format:\n"
        tool_context += '{"tool_use": {"name": "tool_name", "arguments": {...}}}\n'

    # Combine messages into a prompt
    prompt_parts = []
    for msg in api_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Handle content blocks
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    text_parts.append(f"Tool result: {block.get('content', '')}")
            content = "\n".join(text_parts)

        prompt_parts.append(f"{role}: {content}")

    full_prompt = "\n\n".join(prompt_parts)

    full_system = (system or "") + tool_context

    # Call Claude
    response_text = await claude_complete(full_prompt, system=full_system)

    # Parse response for tool use
    content_blocks = []
    stop_reason = "end_turn"

    # Check for tool use pattern in response
    try:
        if '{"tool_use"' in response_text or '"type": "tool_use"' in response_text:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^{}]*"tool_use"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                tool_json = json.loads(json_match.group())
                tool_use = tool_json.get("tool_use", {})

                # Add text before tool call if any
                pre_text = response_text[:json_match.start()].strip()
                if pre_text:
                    content_blocks.append({"type": "text", "text": pre_text})

                content_blocks.append({
                    "type": "tool_use",
                    "id": f"call_{hash(json_match.group()) % 100000}",
                    "name": tool_use.get("name", ""),
                    "input": tool_use.get("arguments", {})
                })
                stop_reason = "tool_use"
            else:
                content_blocks.append({"type": "text", "text": response_text})
        else:
            content_blocks.append({"type": "text", "text": response_text})
    except (json.JSONDecodeError, AttributeError):
        content_blocks.append({"type": "text", "text": response_text})

    return {
        "content": content_blocks,
        "stop_reason": stop_reason
    }


async def agent_turn(
    messages: list[AgentMessage],
    tools: ToolRegistry,
) -> AgentMessage:
    """Execute one agent turn with tool handling.

    Args:
        messages: Conversation history
        tools: Tool registry with registered handlers

    Returns:
        Assistant message (may include tool_calls)
    """
    tool_definitions = tools.get_tool_definitions()

    # Find system message if present
    system = None
    for msg in messages:
        if msg.role == "system":
            system = msg.content
            break

    # Pass a copy of messages to preserve state at call time
    response = await call_claude_with_tools(list(messages), tool_definitions, system)

    # Parse response into AgentMessage
    content_blocks = response.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall.from_api_response(block))

    return AgentMessage(
        role="assistant",
        content="\n".join(text_parts),
        tool_calls=tool_calls if tool_calls else None
    )


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    registry: ToolRegistry,
) -> list[AgentMessage]:
    """Execute tool calls and create result messages.

    Args:
        tool_calls: List of tool calls to execute
        registry: Tool registry for invoking tools

    Returns:
        List of tool_result messages
    """
    results = []

    for call in tool_calls:
        try:
            result = await registry.invoke(call.name, call.arguments)
            result_content = json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            result_content = json.dumps({"error": str(e)})

        results.append(AgentMessage(
            role="tool_result",
            content=result_content,
            tool_call_id=call.id
        ))

    return results


async def run_agent(
    system_prompt: str,
    user_input: str,
    tools: ToolRegistry,
    max_turns: int = 10,
) -> dict:
    """Run agent loop until completion or limit.

    Args:
        system_prompt: System prompt for the agent
        user_input: Initial user input
        tools: Tool registry
        max_turns: Maximum number of LLM calls

    Returns:
        Final response parsed as JSON dict
    """
    messages = [
        AgentMessage(role="system", content=system_prompt),
        AgentMessage(role="user", content=user_input),
    ]

    for turn in range(max_turns):
        # Get assistant response
        assistant_msg = await agent_turn(messages, tools)
        messages.append(assistant_msg)

        # Check if done (no tool calls)
        if not assistant_msg.tool_calls:
            break

        # Execute tool calls
        tool_results = await execute_tool_calls(assistant_msg.tool_calls, tools)
        messages.extend(tool_results)

    # Parse final response as JSON
    final_content = messages[-1].content if messages else ""

    try:
        return json.loads(final_content)
    except json.JSONDecodeError:
        # Return raw content on parse failure
        return {"error": "Failed to parse JSON response", "raw": final_content}
