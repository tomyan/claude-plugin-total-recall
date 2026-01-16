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

Jump directly to the section for the user's command:
- `/total-recall <query>` → [Search](#search-total-recall-query)
- `/total-recall backfill` → [Backfill](#backfill-total-recall-backfill)
- `/total-recall stats` → [Stats](#stats-total-recall-stats)
- `/total-recall topics` → [Topics](#topics-total-recall-topics)

If any command fails with "database not found" or "no such file", run [First-Run Setup](#first-run-setup) first.

---

### First-Run Setup

**Only run this section if a command failed because the database doesn't exist.**

**FIRST: Check for required API key:**
```bash
if [ -z "$OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS" ]; then
  echo "❌ ERROR: OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS is not set!"
  echo ""
  echo "This skill requires an OpenAI API key for generating embeddings."
  echo "Without it, search will not work."
  echo ""
  echo "Please set this environment variable and try again:"
  echo "  export OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS='your-openai-api-key'"
  exit 1
fi
echo "✓ API key found"
```

If the above check fails, **STOP** and tell the user they need to set `OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS` before proceeding.

Run these commands to initialize the environment. Explain each step to the user:

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

Run the search directly:
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

Indexes existing conversation history. **Runs in background** - returns immediately.

### Invocation

- `/total-recall backfill --all` - Enqueue all transcripts for background indexing
- `/total-recall backfill <file>` - Enqueue specific file

### Instructions

**Run backfill:**
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
cd "$HOME/.claude-plugin-total-recall" && PYTHONPATH="$SKILL_DIR/src" uv run python "$SKILL_DIR/src/cli.py" backfill --all
```

This returns immediately after enqueueing. Indexing happens in the background via daemon.

**Check progress:**
```bash
cd "$HOME/.claude-plugin-total-recall" && PYTHONPATH="$SKILL_DIR/src" uv run python "$SKILL_DIR/src/cli.py" stats
```

**Report to user:**
```markdown
## Backfill Started

Enqueued **<N>** transcript files for background indexing.

The daemon is processing in the background. You can continue working - indexing won't block you.

Check progress anytime with `/total-recall stats`.
```

### What Gets Indexed

**High Value:** Decisions, conclusions, questions, problems, solutions, todos, key context
**Filtered:** Greetings, acknowledgments, very short messages, tool use preambles

### Notes

- Backfill is incremental - re-running only indexes new content
- Background daemon processes queue automatically
- Hook on each turn keeps current session indexed in real-time

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
