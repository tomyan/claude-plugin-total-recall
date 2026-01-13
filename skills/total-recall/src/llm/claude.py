"""Claude CLI integration for LLM tasks."""

import json
import os
import subprocess
from typing import Optional

from config import logger
from errors import TotalRecallError


def claude_complete(prompt: str, system: Optional[str] = None) -> str:
    """Call Claude CLI non-interactively for LLM tasks.

    Uses `claude -p` with `--no-session-persistence` to avoid creating
    session transcripts that would pollute the index.

    Args:
        prompt: The user prompt
        system: Optional system prompt

    Returns:
        The response text

    Raises:
        TotalRecallError: If Claude CLI fails
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--no-session-persistence"]
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
            timeout=60,  # 60 second timeout
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
            "Claude CLI timed out after 60 seconds",
            "claude_cli_timeout"
        )
    except FileNotFoundError:
        raise TotalRecallError(
            "Claude CLI not found. Ensure 'claude' is in PATH.",
            "claude_cli_not_found"
        )
