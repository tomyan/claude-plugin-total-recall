---
context: fork
name: memgraph
description: Search past conversations for relevant context
hooks:
  UserPromptSubmit:
    - hooks:
        - type: command
          command: bash ~/.claude/skills/memgraph/hooks/index-continuous.sh
  Stop:
    - hooks:
        - type: command
          command: bash ~/.claude/skills/memgraph/hooks/index-continuous.sh
---

# Memory Retrieval Skill

Searches past conversations for relevant ideas, decisions, and context using semantic vector search.

## Invocation

`/remember <query>`

Examples:
- `/remember LoRa range testing`
- `/remember decisions about relay ratings`
- `/remember what did we decide about the cartridge design`

## Instructions

When the user invokes `/remember <query>`:

### Step 1: Search

Run the memgraph CLI (handles bootstrap automatically). Pass `--cwd` to scope search to the current project:
```bash
SKILL_DIR="$HOME/.claude/skills/memgraph"
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

### Step 2: Choose Search Strategy

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

### Step 3: Present Results

Format the results for the user:

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

### Step 4: Offer Deep Dive

If results seem incomplete or user wants more detail:
- Offer to read the original transcript sections
- Offer to search with different terms
- Suggest `/memory-backfill` if database seems empty

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
- Use `/memory-backfill` to index existing conversation history

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
