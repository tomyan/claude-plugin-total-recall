#!/bin/bash
# Total Recall - First Time Setup
# Single command to initialize everything

set -e

RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
KEY_FILE="$HOME/.config/total-recall/openai-api-key"
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check for API key
if [ ! -f "$KEY_FILE" ] || [ ! -s "$KEY_FILE" ]; then
    echo "ERROR: OpenAI API key required"
    echo ""
    echo "Create the key file first:"
    echo "  mkdir -p ~/.config/total-recall"
    echo "  echo 'your-openai-api-key' > ~/.config/total-recall/openai-api-key"
    exit 1
fi

echo "Setting up Total Recall..."

# Create runtime directory
mkdir -p "$RUNTIME_DIR"

# Initialize Python environment
cd "$RUNTIME_DIR"
uv init --name total-recall --no-readme 2>/dev/null || true
uv add sqlite-vec openai aiosqlite 2>/dev/null || uv sync

# Initialize database
PYTHONPATH="$SKILL_DIR/src" uv run python "$SKILL_DIR/src/memory_db.py" init

echo ""
echo "Setup complete! Total Recall is ready."
echo ""
echo "Next: Run '/total-recall backfill --all' to index your conversation history"
