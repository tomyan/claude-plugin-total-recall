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

### Step 1: Bootstrap

Ensure runtime is ready:
```bash
RUNTIME=$("$SKILL_DIR/bootstrap.sh")
```

### Step 2: Determine Scope

Based on arguments:

**Current session (no args):**
The current transcript path is available in the conversation context. Look for the session's transcript file.

**Specific file:**
Use the provided file path.

**All transcripts (--all):**
```bash
find ~/.claude/projects -name "*.jsonl" -type f
```

### Step 3: Run Backfill

For each transcript file, use the backfill CLI:

```bash
SKILL_DIR="$HOME/.claude/skills/memgraph"
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/backfill.py" backfill "<file_path>"
```

This will:
- Parse the transcript for indexable messages
- Filter out greetings, acknowledgments, and low-value content (< 20 chars)
- Store each substantive message as an idea with embeddings
- Track progress for incremental indexing (only new content on re-run)

### Step 4: Check Progress

To see indexing progress for a file:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/backfill.py" progress "<file_path>"
```

Returns:
- `last_indexed_line`: Last line that was processed
- `total_lines`: Total lines in the file

### Step 5: Report Results

After processing, report:
```markdown
## Backfill Complete

**Processed:** <file_path>
**Messages indexed:** <count>
**Lines processed:** <start_line> to <end_line>
```

### Step 6: Verify

Check stats after backfill:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" stats
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
