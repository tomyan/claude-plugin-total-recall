"""Agent input formatter - Slice 5.2.

Formats batch updates into agent input for the indexing agent.
"""

import json
from typing import Literal

from indexer.batch_collector import BatchUpdate, Message


def format_agent_input(
    updates: list[BatchUpdate],
    mode: Literal["continuous", "backfill"] = "continuous",
) -> str:
    """Format batch updates as JSON input for the indexing agent.

    Args:
        updates: List of batch updates to process
        mode: Processing mode (continuous or backfill)

    Returns:
        JSON string for agent input
    """
    sessions = {}

    for update in updates:
        if update.session not in sessions:
            sessions[update.session] = {
                "session_id": update.session,
                "file_path": update.file_path,
                "messages": [],
            }

        for msg in update.messages:
            sessions[update.session]["messages"].append({
                "role": msg.role,
                "content": msg.content,
                "line_num": msg.line_num,
                "timestamp": msg.timestamp,
            })

    input_data = {
        "mode": mode,
        "sessions": list(sessions.values()),
    }

    return json.dumps(input_data, indent=2)
