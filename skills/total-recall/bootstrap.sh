#!/bin/bash
# Bootstrap script - ensures runtime dependencies are ready
# Source code stays in skill folder, only deps and data in runtime

set -e

RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create runtime directory
mkdir -p "$RUNTIME_DIR"

# Initialize uv project if needed
if [ ! -f "$RUNTIME_DIR/pyproject.toml" ]; then
    cd "$RUNTIME_DIR"
    uv init --name total-recall --no-readme
    uv add sqlite-vec openai
fi

# Ensure venv is synced
cd "$RUNTIME_DIR"
uv sync --quiet 2>/dev/null || uv sync

# Initialize database if needed
if [ ! -f "$RUNTIME_DIR/memory.db" ]; then
    uv run python "$SCRIPT_DIR/src/memory_db.py" init
fi

# Return runtime directory path
echo "$RUNTIME_DIR"
