#!/bin/bash
# Continuous indexing hook - runs on UserPromptSubmit and Stop
# Just enqueues work for the daemon - very fast

SKILL_DIR="$HOME/.claude/skills/total-recall"
RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
DB_PATH="$RUNTIME_DIR/memory.db"
PIDFILE="$RUNTIME_DIR/daemon.pid"

# Skip if setup not complete (database doesn't exist)
# User should run /total-recall to trigger first-run setup
if [ ! -f "$DB_PATH" ]; then
    exit 0
fi

# Skip if called from total-recall itself
if [ -n "$TOTAL_RECALL_INTERNAL" ]; then
    exit 0
fi

# Find the current transcript (most recently modified, not subagent)
TRANSCRIPT=$(find "$HOME/.claude/projects" -name "*.jsonl" -type f ! -path "*/subagents/*" 2>/dev/null | while read f; do
    echo "$(stat -f '%m' "$f" 2>/dev/null) $f"
done | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
    exit 0
fi

# Ensure runtime directory exists
mkdir -p "$RUNTIME_DIR"

# Get file size
FILE_SIZE=$(stat -f '%z' "$TRANSCRIPT" 2>/dev/null || echo 0)

# Enqueue the transcript for processing (fast SQLite insert)
sqlite3 "$DB_PATH" "INSERT INTO work_queue (file_path, file_size) VALUES ('$TRANSCRIPT', $FILE_SIZE);" 2>/dev/null

# Ensure daemon is running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        # Daemon is running, nothing more to do
        exit 0
    fi
    # Stale pidfile, remove it
    rm -f "$PIDFILE"
fi

# Start daemon in background (pass through embeddings API key)
cd "$RUNTIME_DIR" 2>/dev/null || exit 0
PYTHONPATH="$SKILL_DIR/src" OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS="$OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS" nohup uv run python -u "$SKILL_DIR/src/daemon.py" >> "$RUNTIME_DIR/daemon.log" 2>&1 &
