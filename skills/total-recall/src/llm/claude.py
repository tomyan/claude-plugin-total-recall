"""Claude CLI integration for LLM tasks."""

import asyncio
import json
import os
import subprocess
from typing import Any, Optional

from config import logger
from errors import TotalRecallError
from llm.tools import INDEXING_TOOLS, execute_tool


async def claude_complete(
    prompt: str,
    system: Optional[str] = None,
    model: str = "haiku",
    max_retries: int = 2,
    timeout: int = 30
) -> str:
    """Call Claude CLI non-interactively for LLM tasks (async).

    Runs the sync implementation in a thread pool for async compatibility.
    Includes retry logic with exponential backoff for timeouts.

    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model to use (default: haiku for speed)
        max_retries: Number of retries on timeout (default: 2)
        timeout: Timeout in seconds (default: 30)

    Returns:
        The response text
    """
    prompt_preview = prompt[:80].replace('\n', ' ') + '...' if len(prompt) > 80 else prompt.replace('\n', ' ')

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"LLM call attempt {attempt + 1}/{max_retries + 1}: {prompt_preview}")
            result = await asyncio.to_thread(_claude_complete_sync, prompt, system, model, timeout)
            logger.debug(f"LLM call succeeded (attempt {attempt + 1})")
            return result
        except TotalRecallError as e:
            if e.code == "claude_cli_timeout" and attempt < max_retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s...
                logger.warning(f"LLM timeout (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise


def _claude_complete_sync(
    prompt: str,
    system: Optional[str] = None,
    model: str = "haiku",
    timeout: int = 30
) -> str:
    """Call Claude CLI non-interactively for LLM tasks.

    Uses `claude -p` with `--no-session-persistence` to avoid creating
    session transcripts that would pollute the index.

    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model to use (default: haiku for speed)
        timeout: Timeout in seconds (default: 30)

    Returns:
        The response text

    Raises:
        TotalRecallError: If Claude CLI fails
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--no-session-persistence", "--model", model]
    if system:
        cmd.extend(["--system-prompt", system])

    # Set TOTAL_RECALL_INTERNAL to prevent hooks from re-indexing during this call
    env = os.environ.copy()
    env["TOTAL_RECALL_INTERNAL"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            logger.warning(f"Claude CLI failed: {error_msg}")
            raise TotalRecallError(
                f"Claude CLI failed: {error_msg}",
                "claude_cli_error",
                {"stderr": result.stderr, "returncode": result.returncode}
            )

        # Parse JSON output
        # Claude CLI --output-format json returns an array of event objects:
        # [{"type":"system",...}, {"type":"assistant",...}, {"type":"result","result":"..."}]
        try:
            response = json.loads(result.stdout)
            # Find the result event in the array
            if isinstance(response, list):
                for event in response:
                    if isinstance(event, dict) and event.get("type") == "result":
                        return event.get("result", "")
                # No result event found
                logger.warning("No result event in Claude CLI output")
                return ""
            # Fallback for unexpected format
            return response.get("result", "") if isinstance(response, dict) else ""
        except json.JSONDecodeError:
            # If not JSON, return raw stdout (fallback)
            return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise TotalRecallError(
            f"Claude CLI timed out after {timeout} seconds",
            "claude_cli_timeout"
        )
    except FileNotFoundError:
        raise TotalRecallError(
            "Claude CLI not found. Ensure 'claude' is in PATH.",
            "claude_cli_not_found"
        )


def claude_with_tools(
    prompt: str,
    system: Optional[str] = None,
    model: str = "haiku",
    session_context: Optional[str] = None,
    max_turns: int = 5
) -> str:
    """Call Claude with tool use support for richer analysis.

    Allows Claude to search existing ideas, get details, and explore
    the knowledge base before producing final output.

    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model to use (default: haiku)
        session_context: Session identifier for tool context
        max_turns: Maximum tool-use turns before forcing completion

    Returns:
        The final response text

    Raises:
        TotalRecallError: If Claude CLI fails
    """
    # Build tool definitions as JSON for --allowedTools
    tool_names = [t["name"] for t in INDEXING_TOOLS]

    env = os.environ.copy()
    env["TOTAL_RECALL_INTERNAL"] = "1"

    messages = []
    current_prompt = prompt

    for turn in range(max_turns):
        # Build command
        cmd = [
            "claude", "-p", current_prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", model
        ]
        if system:
            cmd.extend(["--system-prompt", system])

        # Add tool allowlist
        for tool_name in tool_names:
            cmd.extend(["--allowedTools", tool_name])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # Longer timeout for tool use
                env=env
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                logger.warning(f"Claude CLI failed: {error_msg}")
                raise TotalRecallError(
                    f"Claude CLI failed: {error_msg}",
                    "claude_cli_error",
                    {"stderr": result.stderr, "returncode": result.returncode}
                )

            # Parse response
            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                return result.stdout.strip()

            # Look for tool use or final result
            tool_calls = []
            final_result = None

            if isinstance(response, list):
                for event in response:
                    if not isinstance(event, dict):
                        continue

                    # Check for tool use in assistant message
                    if event.get("type") == "assistant":
                        message = event.get("message", {})
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool_calls.append({
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "input": block.get("input", {})
                                    })

                    # Check for final result
                    if event.get("type") == "result":
                        final_result = event.get("result", "")

            # If we have tool calls, execute them and continue
            if tool_calls:
                tool_results = []
                for tc in tool_calls:
                    logger.info(f"Executing tool: {tc['name']}({tc['input']})")
                    result_str = execute_tool(tc["name"], tc["input"], session=session_context)
                    tool_results.append({
                        "tool_use_id": tc["id"],
                        "result": result_str
                    })
                    logger.info(f"Tool result: {result_str[:200]}...")

                # Build continuation prompt with tool results
                # Note: Claude CLI doesn't support direct tool_result injection,
                # so we format results as a follow-up message
                results_text = "\n\n".join([
                    f"Tool result for {tc['name']}:\n{tr['result']}"
                    for tc, tr in zip(tool_calls, tool_results)
                ])
                current_prompt = f"Here are the tool results:\n\n{results_text}\n\nNow provide your final analysis based on these results."
                continue

            # No tool calls - return final result
            if final_result is not None:
                return final_result

            # Fallback - look for text in response
            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TotalRecallError(
                "Claude CLI timed out",
                "claude_cli_timeout"
            )

    # Max turns reached
    logger.warning(f"Max tool turns ({max_turns}) reached, returning last response")
    return current_prompt  # Return what we have
