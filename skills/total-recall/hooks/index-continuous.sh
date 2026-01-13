#!/bin/bash
# Continuous indexing hook - runs on UserPromptSubmit and Stop
# Designed to be fast and run frequently

SKILL_DIR="$HOME/.claude/skills/total-recall"
RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
LOCK_FILE="$RUNTIME_DIR/index.lock"

# Skip if called from total-recall itself (prevents unnecessary re-indexing)
if [ -n "$TOTAL_RECALL_INTERNAL" ]; then
    exit 0
fi

# Skip if already running (prevents overlapping runs)
if [ -f "$LOCK_FILE" ]; then
    exit 0
fi

# Find the current transcript (most recently modified, not subagent)
# Use stat to get modification time, sort to find newest (macOS compatible)
TRANSCRIPT=$(find "$HOME/.claude/projects" -name "*.jsonl" -type f ! -path "*/subagents/*" 2>/dev/null | while read f; do
    echo "$(stat -f '%m' "$f" 2>/dev/null) $f"
done | sort -rn | head -1 | cut -d' ' -f2-)

if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    cd "$RUNTIME_DIR" 2>/dev/null || exit 0

    # Run in background with lock
    (
        touch "$LOCK_FILE"
        uv run python "$SKILL_DIR/src/cli.py" index "$TRANSCRIPT" >> "$RUNTIME_DIR/hook.log" 2>&1
        rm -f "$LOCK_FILE"
    ) &
fi
