---
context: fork
name: memory-backfill
description: Index existing conversation history into memory
---

# Memory Backfill Skill

Indexes existing conversation history that occurred before the memory skill was installed.

## Invocation

- `/memory-backfill` - Index current session's history
- `/memory-backfill <file>` - Index specific transcript file
- `/memory-backfill --all` - Index all transcripts in ~/.claude/projects/

## Instructions

When the user invokes `/memory-backfill`:

### Step 1: Determine Scope

```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
```

**Current session (no args):**
The current transcript path is available in the conversation context.

**Specific file:**
Use the provided file path.

**All transcripts (--all):**
```bash
find ~/.claude/projects -name "*.jsonl" -type f
```

### Step 2: Run Backfill

For each transcript file:
```bash
uv run python "$SKILL_DIR/src/cli.py" index "<file_path>"
```

This will:
- Parse the transcript for indexable messages
- Detect topic shifts and create spans
- Classify intent (decision, question, problem, solution, etc.)
- Extract entities and assess confidence
- Track progress for incremental indexing

### Step 3: Check Progress

```bash
uv run python "$SKILL_DIR/src/cli.py" progress "<file_path>"
```

### Step 4: Report Results

```markdown
## Backfill Complete

**Processed:** <file_path>
**Messages indexed:** <count>
**Spans created:** <count>
```

### Step 5: Verify

```bash
uv run python "$SKILL_DIR/src/cli.py" stats
```

## What Gets Indexed

### High Value (Indexed)
- Architecture decisions and rationale
- Technology choices and trade-offs
- Substantive questions and answers
- Solutions to problems encountered
- Key insights and conclusions

### Low Value (Filtered)
- Greetings ("hello", "hi there")
- Acknowledgments ("ok", "thanks", "got it")
- Very short messages (< 20 characters)
- Tool use preambles ("Let me check that file")

## Notes

- Backfill is incremental - running again only indexes new content
- Large transcripts may take time due to embedding API calls
- The stop hook automatically indexes new content after each turn
