#!/bin/bash
# Index conversation transcript on Stop hook
# Finds recently modified transcript and indexes new messages

SKILL_DIR="$HOME/.claude/skills/memgraph"
RUNTIME_DIR="$HOME/.claude-plugin-memgraph"

# Find most recently modified transcript (not subagent, modified in last 5 mins)
TRANSCRIPT=$(find "$HOME/.claude/projects" -name "*.jsonl" -type f ! -path "*/subagents/*" -mmin -5 2>/dev/null | head -1)

if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    cd "$RUNTIME_DIR" 2>/dev/null || exit 0
    # Run indexing in background to not block
    nohup uv run python "$SKILL_DIR/src/cli.py" index "$TRANSCRIPT" >> "$RUNTIME_DIR/hook.log" 2>&1 &
fi
