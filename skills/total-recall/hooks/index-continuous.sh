#!/bin/bash
# Continuous indexing hook - runs on UserPromptSubmit and Stop
# Enqueues work for daemon - target: <20ms

RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
DB_PATH="$RUNTIME_DIR/memory.db"
PIDFILE="$RUNTIME_DIR/daemon.pid"

# Skip if setup not complete
[ ! -f "$DB_PATH" ] && exit 0

# Skip if called from total-recall itself
[ -n "$TOTAL_RECALL_INTERNAL" ] && exit 0

# Read stdin (hook input JSON)
read -r INPUT 2>/dev/null || exit 0

# Extract transcript_path using pure bash (no external tools)
# Format: ..."transcript_path": "/path/to/file.jsonl"...
case "$INPUT" in
    *'"transcript_path"'*) ;;
    *) exit 0 ;;
esac
TRANSCRIPT="${INPUT#*\"transcript_path\"*:*\"}"
TRANSCRIPT="${TRANSCRIPT%%\"*}"

# Validate
[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0
[[ "$TRANSCRIPT" == */subagents/* ]] && exit 0

# Enqueue (background, with timeout)
{
    sqlite3 "$DB_PATH" ".timeout 100" \
        "INSERT OR IGNORE INTO work_queue (file_path, file_size) VALUES ('$TRANSCRIPT', 0);" 2>/dev/null
} &

# Check daemon
if [ -f "$PIDFILE" ]; then
    read -r PID < "$PIDFILE" 2>/dev/null
    [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && exit 0
    rm -f "$PIDFILE" 2>/dev/null
fi

# Start daemon (detached)
{
    cd "$RUNTIME_DIR"
    PYTHONPATH="$HOME/.claude/skills/total-recall/src" \
        nohup uv run python -u "$HOME/.claude/skills/total-recall/src/daemon.py" \
        >> "$RUNTIME_DIR/daemon.log" 2>&1 &
} &

exit 0
