#!/bin/bash
# Stop hook for Claude memory indexing
# Runs after each Claude response to index new content
#
# Environment variables used:
#   CLAUDE_SESSION_ID - The current session UUID
#   CLAUDE_PROJECT_DIR - The project directory path
#   OPENAI_TOKEN_MEMORY_EMBEDDINGS - OpenAI API key for embeddings

set -e

# Determine transcript path
# Claude stores transcripts at ~/.claude/projects/<project>/<session>.jsonl
if [ -n "$CLAUDE_SESSION_ID" ] && [ -n "$CLAUDE_PROJECT_DIR" ]; then
    TRANSCRIPT="$HOME/.claude/projects/$CLAUDE_PROJECT_DIR/$CLAUDE_SESSION_ID.jsonl"
elif [ -n "$TRANSCRIPT_PATH" ]; then
    # Allow direct override
    TRANSCRIPT="$TRANSCRIPT_PATH"
else
    # Cannot determine transcript path
    exit 0
fi

# Check transcript exists
if [ ! -f "$TRANSCRIPT" ]; then
    exit 0
fi

# Skip if no API key configured
if [ -z "$OPENAI_TOKEN_MEMORY_EMBEDDINGS" ]; then
    exit 0
fi

# Paths
SKILL_DIR="$(dirname "$(dirname "$(realpath "$0")")")/skills/memgraph"
RUNTIME_DIR="$HOME/.claude-plugin-memgraph"

# Bootstrap runtime if needed
if [ ! -d "$RUNTIME_DIR/.venv" ]; then
    mkdir -p "$RUNTIME_DIR"
    cd "$RUNTIME_DIR"

    # Initialize uv project
    if [ ! -f "pyproject.toml" ]; then
        uv init --name memgraph 2>/dev/null || true
        uv add sqlite-vec openai 2>/dev/null || true
    fi

    # Create venv
    uv sync 2>/dev/null || true
fi

# Run incremental indexing with full topic tracking
uv run python "$SKILL_DIR/src/cli.py" index "$TRANSCRIPT" 2>/dev/null || true
