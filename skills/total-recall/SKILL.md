---
context: fork
name: total-recall
description: Search past conversations for relevant context
hooks:
  UserPromptSubmit:
    - hooks:
        - type: command
          command: bash ~/.claude/skills/total-recall/hooks/index-continuous.sh
  Stop:
    - hooks:
        - type: command
          command: bash ~/.claude/skills/total-recall/hooks/index-continuous.sh
---

# Memory Retrieval Skill

Searches past conversations for relevant ideas, decisions, and context using semantic vector search.

## Prerequisites

- **uv** - Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS** - Environment variable with OpenAI API key (used for embeddings)

## Invocation

- `/total-recall <query>` - Search past conversations
- `/total-recall backfill` - Index current session's history
- `/total-recall backfill --all` - Index all conversation history
- `/total-recall stats` - Show database statistics
- `/total-recall topics` - List indexed topics

Examples:
- `/total-recall LoRa range testing`
- `/total-recall decisions about relay ratings`
- `/total-recall what did we decide about the cartridge design`

## Instructions

### First-Run Setup

When `/total-recall` is invoked, **always check if setup is needed first**.

**Step 0: Check if runtime environment exists**

```bash
RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
[ -f "$RUNTIME_DIR/memory.db" ] && echo "Setup complete" || echo "Setup needed"
```

**If setup is needed**, run these commands to initialize the environment. Explain each step to the user:

```markdown
## Total Recall - First Time Setup

Setting up your long-term memory system...
```

**1. Create runtime directory** (stores database and logs, separate from skill code):
```bash
mkdir -p "$HOME/.claude-plugin-total-recall"
```

**2. Initialize Python environment** (installs dependencies via uv):
```bash
cd "$HOME/.claude-plugin-total-recall"
uv init --name total-recall --no-readme 2>/dev/null || true
uv add sqlite-vec openai 2>/dev/null || uv sync
```

**3. Initialize database** (creates the memory database with schema):
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
cd "$HOME/.claude-plugin-total-recall" && PYTHONPATH="$SKILL_DIR/src" uv run python "$SKILL_DIR/src/memory_db.py" init
```

After setup completes, respond with:

```markdown
## Setup Complete! ✓

Total Recall is ready. Your long-term memory system will now:
- **Automatically index** new conversations as you work
- **Remember** decisions, problems, solutions, and key context
- **Search** across all your past conversations

**Next steps:**
- Run `/total-recall backfill --all` to index your existing conversation history
- Then search anytime with `/total-recall <query>`

*Tip: Indexing happens in the background - you can keep working while it runs.*
```

Then stop - don't proceed with a search until setup is confirmed complete.

---

### Search: `/total-recall <query>`

When the user invokes `/total-recall` with a query:

### Step 1: Check Setup and Database Status

First, check if the database has any indexed content:
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
uv run python "$SKILL_DIR/src/cli.py" stats
```

**If the database is empty (0 ideas indexed)**, check if the API key is configured:
```bash
[ -n "$OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS" ] && echo "API key is set" || echo "API key missing"
```

**If API key is missing**, respond with:

```markdown
## Memory: <query>

Total Recall needs an OpenAI API key to generate embeddings for semantic search.

**Setup required:**

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Add to your shell profile (~/.zshrc or ~/.bashrc):
   ```bash
   export OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS="sk-..."
   ```
3. Restart your terminal or run `source ~/.zshrc`
4. Run `/total-recall backfill` to index your conversation history

The embedding costs are minimal (~$0.001 per conversation indexed).
```

Then stop.

**If API key is set but database is empty**, respond with:

```markdown
## Memory: <query>

I don't have any memories yet - this is a fresh start!

**Total Recall** is your long-term memory across conversations. As we work together, I'll automatically remember:
- Decisions we make
- Problems we solve
- Questions that come up
- Key context about your projects

**To get started**, run `/total-recall backfill` to index your existing Claude Code conversation history. This takes a few minutes depending on how much history you have.

Once indexed, you can search your memory anytime with `/remember <query>`.
```

Then stop - don't run a search on an empty database.

### Step 2: Search (if database has content)

Run the total-recall CLI. Pass `--cwd` to scope search to the current project:
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
uv run python "$SKILL_DIR/src/cli.py" search "<query>" -n 10 --cwd "$(pwd)"
```

This returns ideas with:
- `content`: The extracted idea
- `intent`: Type (decision, conclusion, question, problem, solution, todo, context)
- `topic`: The topic span this idea belongs to
- `session`: Which conversation session (project)
- `source_file`: Original transcript path
- `source_line`: Line number
- `distance`: Semantic similarity (lower = more similar)

### Step 3: Choose Search Strategy

Based on the query, choose the best search strategy. **Always pass `--cwd "$(pwd)"`** to scope to current project:

**Hybrid Search** - For queries with specific terms:
```bash
uv run python "$SKILL_DIR/src/cli.py" hybrid "<query>" -n 10 --cwd "$(pwd)"
```

**HyDE Search** - For vague/conceptual queries:
```bash
uv run python "$SKILL_DIR/src/cli.py" hyde "<query>" -n 10 --cwd "$(pwd)"
```

**Filtered Search** - For queries with intent:
```bash
uv run python "$SKILL_DIR/src/cli.py" search "<query>" -i decision -n 10 --cwd "$(pwd)"
```

**Global Search** - To search across ALL projects (not just current):
```bash
uv run python "$SKILL_DIR/src/cli.py" search "<query>" -n 10 --global
```

### Step 4: Present Results

**If search returned no results**, respond with:

```markdown
## Memory: <query>

No memories found for this query.

Try:
- Different keywords or phrasing
- A broader search: `uv run python "$SKILL_DIR/src/cli.py" search "<query>" -n 10 --global`
- Check what's indexed: `uv run python "$SKILL_DIR/src/cli.py" sessions`
```

**If search returned results**, format them for the user:

```markdown
## Memory: <query>

### Key Ideas

**Decisions:**
- <decision content> (from: <session>, <date>)

**Conclusions:**
- <conclusion content>

**Related Topics:**
- <topic name>: <topic summary>

### Open Questions
- <any unanswered questions found>

### Context
Found <N> relevant ideas across <M> sessions.
Most relevant from: <session names>
```

### Step 5: Offer Deep Dive

If results seem incomplete or user wants more detail:
- Offer to read the original transcript sections using the `context` command
- Suggest trying different search terms or strategies
- Try a global search with `--global` if project-scoped search found nothing

## Database Stats

To check what's indexed:
```bash
uv run python "$SKILL_DIR/src/cli.py" stats
```

## List Indexed Sessions

See all sessions with idea and topic counts:
```bash
uv run python "$SKILL_DIR/src/cli.py" sessions
```

## Unanswered Questions

To see open questions from past conversations:
```bash
uv run python "$SKILL_DIR/src/cli.py" questions
```

## Get Idea Details

To get a specific idea with its relations:
```bash
uv run python "$SKILL_DIR/src/cli.py" get <idea_id>
```

## Find Similar Ideas

Find ideas semantically similar to a given idea:
```bash
uv run python "$SKILL_DIR/src/cli.py" similar <idea_id> -n 5
uv run python "$SKILL_DIR/src/cli.py" similar <idea_id> --same-session  # Same session only
uv run python "$SKILL_DIR/src/cli.py" similar <idea_id> --other-sessions  # Cross-session only
```

## View Source Context

See the original transcript around an idea:
```bash
uv run python "$SKILL_DIR/src/cli.py" context <idea_id>
uv run python "$SKILL_DIR/src/cli.py" context <idea_id> -B 10 -A 10  # More context
```

## Search by Date Range

Filter search results by time:
```bash
uv run python "$SKILL_DIR/src/cli.py" search "<query>" --since 2024-01-01
uv run python "$SKILL_DIR/src/cli.py" search "<query>" --until 2024-06-01
uv run python "$SKILL_DIR/src/cli.py" search "<query>" --since 2024-01-01 --until 2024-06-01
```

## Export and Import

Backup the memory database:
```bash
uv run python "$SKILL_DIR/src/cli.py" export -o backup.json
uv run python "$SKILL_DIR/src/cli.py" export -s project-name -o project-backup.json  # Single session
```

Restore from backup:
```bash
uv run python "$SKILL_DIR/src/cli.py" import backup.json  # Merge with existing
uv run python "$SKILL_DIR/src/cli.py" import backup.json --replace  # Replace all data
```

## Prune Old Data

Remove old ideas to keep the database lean:
```bash
uv run python "$SKILL_DIR/src/cli.py" prune -d 90  # Dry run - shows what would be removed
uv run python "$SKILL_DIR/src/cli.py" prune -d 90 --execute  # Actually delete
```

## Notes

- Results are from previously indexed conversations
- If nothing found, the conversation may not be indexed yet
- Use `/total-recall backfill` to index existing conversation history

## Graph Revision

The indexed graph is an interpretation of conversations and can be revised. The raw conversation transcripts remain immutable - revisions only affect the interpreted graph.

**Reclassify an idea's intent:**
```bash
uv run python "$SKILL_DIR/src/cli.py" update-intent <idea_id> decision
# Valid intents: decision, conclusion, question, problem, solution, todo, context
```

**Move an idea to a different topic:**
```bash
uv run python "$SKILL_DIR/src/cli.py" move-idea <idea_id> <span_id>
```

**Merge topics:**
```bash
uv run python "$SKILL_DIR/src/cli.py" merge-spans <source_span_id> <target_span_id>
```

**Mark an idea as superseding another:**
```bash
uv run python "$SKILL_DIR/src/cli.py" supersede <old_idea_id> <new_idea_id>
```

## Auto-Categorization

Automatically improve topic organization using LLM analysis.

**Auto-assign topics to projects:**
```bash
uv run python "$SKILL_DIR/src/cli.py" auto-categorize  # Dry run
uv run python "$SKILL_DIR/src/cli.py" auto-categorize --execute  # Apply changes
```

**Run all categorization improvements:**
```bash
uv run python "$SKILL_DIR/src/cli.py" improve
```
This auto-categorizes unassigned topics, renames poorly-named topics, and reports remaining issues.

## Quality Filtering

Keep the database lean by removing low-value content.

**Review ideas against regex filters:**
```bash
uv run python "$SKILL_DIR/src/cli.py" review-ideas  # Dry run
uv run python "$SKILL_DIR/src/cli.py" review-ideas -t <topic_id>  # Specific topic
uv run python "$SKILL_DIR/src/cli.py" review-ideas --execute  # Delete filtered
```

**Use LLM to identify subtle low-value content:**
```bash
uv run python "$SKILL_DIR/src/cli.py" llm-filter  # Dry run
uv run python "$SKILL_DIR/src/cli.py" llm-filter -t <topic_id> -b 30  # Specific topic, batch size 30
uv run python "$SKILL_DIR/src/cli.py" llm-filter --execute  # Delete flagged
```
LLM filtering catches things regex misses: generic statements, context-dependent content, redundant ideas.

## Project Hierarchy

Organize topics into projects for better structure.

**List projects:**
```bash
uv run python "$SKILL_DIR/src/cli.py" projects
```

**Create a project:**
```bash
uv run python "$SKILL_DIR/src/cli.py" create-project "My Project" -d "Description"
```

**Assign a topic to a project:**
```bash
uv run python "$SKILL_DIR/src/cli.py" assign-topic <topic_id> "My Project"
```

**Set a topic's parent (create topic hierarchy):**
```bash
uv run python "$SKILL_DIR/src/cli.py" reparent-topic <topic_id> <parent_topic_id>
```

**Remove a topic from its parent:**
```bash
uv run python "$SKILL_DIR/src/cli.py" unparent-topic <topic_id>
```

**View hierarchy tree:**
```bash
uv run python "$SKILL_DIR/src/cli.py" tree
```

## Timeline Visualization

See activity across time for topics or projects.

**Topic timeline (see activity for a topic across sessions):**
```bash
uv run python "$SKILL_DIR/src/cli.py" timeline --topic "LoRa prototyping"
```

**Project timeline (see recent activity by date):**
```bash
uv run python "$SKILL_DIR/src/cli.py" timeline --project rad-control-v1-1 --days 14
```

Output shows spans grouped by date with key decisions and conclusions highlighted.

## Temporal Queries

Search with natural language time expressions and temporal aggregation.

**Natural language time in search:**
```bash
uv run python "$SKILL_DIR/src/cli.py" search "heating" --when "last week"
uv run python "$SKILL_DIR/src/cli.py" search "decisions" --when "since tuesday"
uv run python "$SKILL_DIR/src/cli.py" search "LoRa" --when "since jan 5"
```

**Search after a specific session:**
```bash
uv run python "$SKILL_DIR/src/cli.py" search "follow up" --after-session abc123 --global
```

**Activity aggregation by period:**
```bash
uv run python "$SKILL_DIR/src/cli.py" activity --by day --days 7
uv run python "$SKILL_DIR/src/cli.py" activity --by week --days 30 -s rad-control-v1-1
```

**Topic activity over time:**
```bash
uv run python "$SKILL_DIR/src/cli.py" topic-activity 42 --by week --days 90
```

---

## Backfill: `/total-recall backfill`

Indexes existing conversation history that occurred before the skill was installed.

### Invocation

- `/total-recall backfill` - Index current session's history
- `/total-recall backfill <file>` - Index specific transcript file
- `/total-recall backfill --all` - Index all transcripts in ~/.claude/projects/

### Instructions

**Step 1: Determine Scope**

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

**Step 2: Run Backfill**

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

**Step 3: Check Progress**

```bash
uv run python "$SKILL_DIR/src/cli.py" progress "<file_path>"
```

**Step 4: Report Results**

```markdown
## Backfill Complete

**Processed:** <file_path>
**Messages indexed:** <count>
**Spans created:** <count>
```

**Step 5: Verify**

```bash
uv run python "$SKILL_DIR/src/cli.py" stats
```

### What Gets Indexed

**High Value (Indexed):**
- Architecture decisions and rationale
- Technology choices and trade-offs
- Substantive questions and answers
- Solutions to problems encountered
- Key insights and conclusions

**Low Value (Filtered):**
- Greetings ("hello", "hi there")
- Acknowledgments ("ok", "thanks", "got it")
- Very short messages (< 20 characters)
- Tool use preambles ("Let me check that file")

### Notes

- Backfill is incremental - running again only indexes new content
- Large transcripts may take time due to embedding API calls
- The stop hook automatically indexes new content after each turn

---

## Stats: `/total-recall stats`

Shows statistics about the memory database.

### Instructions

Run the CLI:
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
uv run python "$SKILL_DIR/src/cli.py" stats
```

This returns:
- `total_ideas`: Number of indexed ideas
- `total_spans`: Number of topic spans
- `total_entities`: Number of extracted entities
- `total_relations`: Number of idea relationships
- `sessions_indexed`: Number of unique sessions
- `by_intent`: Breakdown by idea type (decision, question, etc.)
- `entities_by_type`: Breakdown by entity type

### Present Results

```markdown
## Memory Database Stats

**Ideas:** <total_ideas>
**Topic Spans:** <total_spans>
**Sessions:** <sessions_indexed>
**Entities:** <total_entities>
**Relations:** <total_relations>

### Ideas by Type
- Decisions: <count>
- Conclusions: <count>
- Questions: <count>
- Problems: <count>
- Solutions: <count>

### Entities by Type
- Technologies: <count>
- Projects: <count>
- Concepts: <count>
```

If stats are empty, suggest running `/total-recall backfill` to index existing conversations.

---

## Topics: `/total-recall topics`

Lists topic spans across sessions.

### Invocation

- `/total-recall topics` - List all topics
- `/total-recall topics <session>` - List topics for specific session

### Instructions

Run the CLI:
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

### Present Results

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
