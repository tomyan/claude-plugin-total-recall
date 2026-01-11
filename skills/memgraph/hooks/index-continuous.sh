#!/bin/bash
# Continuous indexing hook - runs on UserPromptSubmit and Stop
# Designed to be fast and run frequently

SKILL_DIR="$HOME/.claude/skills/memgraph"
RUNTIME_DIR="$HOME/.claude-plugin-memgraph"
LOCK_FILE="$RUNTIME_DIR/index.lock"

# Skip if already running (prevents overlapping runs)
if [ -f "$LOCK_FILE" ]; then
    exit 0
fi

# Find the current transcript (most recently modified, not subagent)
TRANSCRIPT=$(find "$HOME/.claude/projects" -name "*.jsonl" -type f ! -path "*/subagents/*" -mmin -10 2>/dev/null | xargs ls -t 2>/dev/null | head -1)

if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    cd "$RUNTIME_DIR" 2>/dev/null || exit 0

    # Run in background with lock
    (
        touch "$LOCK_FILE"
        uv run python "$SKILL_DIR/src/cli.py" index "$TRANSCRIPT" >> "$RUNTIME_DIR/hook.log" 2>&1
        rm -f "$LOCK_FILE"
    ) &
fi
