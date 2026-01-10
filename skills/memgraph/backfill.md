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

### Step 3: For Each Transcript File

Read the transcript and process each line:

```bash
# Get session name from path
SESSION=$(cd ~/.claude-plugin-memgraph && uv run python src/memory_db.py session-from-path "<file_path>")
```

For each JSON line in the transcript:
1. Parse the JSON
2. Skip system messages, tool calls, and low-value content
3. For substantive content (decisions, insights, conclusions):

**Identify the type (intent):**
- `decision` - "We decided...", "Going with...", "Chose..."
- `conclusion` - "The key insight is...", "Learned that..."
- `question` - Questions asked (note if answered later)
- `problem` - "The issue is...", "Problem with..."
- `solution` - "Fixed by...", "The solution is..."
- `todo` - "Need to...", "Should implement..."
- `context` - Background information, requirements

**Assess confidence (0.0-1.0):**
- 0.8-1.0: Firm decisions, validated conclusions
- 0.5-0.7: Discussion points, tentative conclusions
- 0.3-0.4: Exploratory ideas, questions

**Extract entities:**
- Projects: control-v1.1, lora-test
- Technologies: ESP32, SX1262, SQLite, LoRa
- Concepts: mesh networking, fail-safe, cartridge design

**Detect topic spans:**
- Look for natural topic boundaries
- Explicit transitions: "let's move on to...", "back to..."
- Domain changes
- Create spans with summaries

### Step 4: Store Ideas

For each extracted idea:
```bash
cd ~/.claude-plugin-memgraph && uv run python src/memory_db.py store \
  "<idea_content>" "<source_file>" <line_number> <span_id> "<intent>" <confidence>
```

### Step 5: Report Progress

After processing, report:
```markdown
## Backfill Complete

**Processed:** <file_path>
**Session:** <session_name>
**Ideas extracted:** <count>
**Topic spans:** <count>
**Entities found:** <list>

### Topics Indexed
1. <topic name>: <brief summary>
2. <topic name>: <brief summary>

### Sample Ideas
- <high-value idea 1>
- <high-value idea 2>
```

### Step 6: Verify

Check stats after backfill:
```bash
cd ~/.claude-plugin-memgraph && uv run python src/memory_db.py stats
```

## Processing Guidelines

### What to Extract (High Value)
- Architecture decisions and rationale
- Technology choices and trade-offs
- Constraints and requirements discovered
- Solutions to problems encountered
- Key insights and conclusions

### What to Skip (Low Value)
- Greetings and acknowledgments
- Debugging output and error traces
- Repeated information
- Pure code without explanation
- Tool invocation details

### Topic Detection

A new topic span starts when:
- User explicitly changes subject
- Significant domain shift (e.g., hardware → software)
- Return to previous topic after digression

Close each span with a 1-2 sentence summary capturing the key points discussed.

## Notes

- Backfill can be run multiple times safely (tracks last indexed line)
- Large transcripts may take a while due to embedding API calls
- Focus on quality over quantity - extract meaningful insights
