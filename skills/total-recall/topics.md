---
context: fork
name: memory-topics
description: List topic spans from conversation history
---

# Memory Topics Skill

Lists topic spans across sessions, showing what has been discussed and when.

## Invocation

- `/memory-topics` - List all topics
- `/memory-topics <session>` - List topics for specific session

## Instructions

When the user invokes `/memory-topics`:

### Step 1: List Topics

Run the total-recall CLI:
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
uv run python "$SKILL_DIR/src/cli.py" topics
```

For a specific session:
```bash
uv run python "$SKILL_DIR/src/cli.py" topics -s <session>
```

This returns spans with:
- `id`: Span ID
- `session`: Session name
- `name`: Topic name
- `summary`: Topic summary (if closed)
- `start_line`, `end_line`: Line range
- `depth`: Hierarchy depth

### Step 2: Present Results

Format the topics for the user:

```markdown
## Memory Topics

### Session: <session_name>

1. **<topic_name>** (lines <start>-<end>)
   <summary if available>

2. **<topic_name>** (lines <start>-<end>)
   <summary if available>

### Session: <other_session>
...
```

### Step 3: Offer Actions

After listing topics, offer:
- `/remember <topic>` to search within a topic
- View ideas from a specific topic span

## Notes

- Topics are created when conversation shifts are detected
- Summaries are generated when topic spans close
- Use this to understand what's been indexed
