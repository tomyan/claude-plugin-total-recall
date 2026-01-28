---
context: fork
name: total-recall
description: Search past conversations for relevant context
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(~/.claude/skills/total-recall/bin/total-recall :*)
  - Bash(bash ~/.claude/skills/total-recall/hooks/:*)
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
- **OpenAI API key** - Create `~/.config/total-recall/openai-api-key` with your key

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

**Prerequisite:** Create your OpenAI API key file first:
```bash
mkdir -p ~/.config/total-recall && echo 'your-openai-api-key' > ~/.config/total-recall/openai-api-key
```

Then run setup:
```bash
bash ~/.claude/skills/total-recall/hooks/setup.sh
```

After setup completes, respond with:

```markdown
## Setup Complete!

Total Recall is ready. Run `/total-recall backfill --all` to index your conversation history.
```

---

### Search: `/total-recall <query>`

Analyze the query and select appropriate search strategies, then run ONE command.

**Strategy Selection Guide:**
- General queries → `hybrid` (combines semantic + keyword)
- Vague/conceptual queries → `hybrid,hyde` (adds hypothetical doc generation)
- "What did we decide about X" → `hybrid,decisions`
- "What are the open tasks/todos" → `todos`
- "What questions/issues came up" → `hybrid,questions` or `hybrid,problems`
- "How did we solve X" → `hybrid,solutions`
- Time-bounded queries → add `--when "last week"` etc.

**Run the search:**
```bash
~/.claude/skills/total-recall/bin/total-recall multi-search "<query>" --strategies "<selected>" -n 15
```

**Options:**
- `--strategies` - Comma-separated: hybrid,hyde,semantic,decisions,todos,questions,problems,solutions
- `--local` - Scope to current project only (default: searches all projects)
- `--when "last week"` - Time filter (natural language)
- `-n 15` - Max results

**Example commands:**
- `/total-recall LoRa range` → `multi-search "LoRa range" --strategies "hybrid"`
- `/total-recall what decisions about auth` → `multi-search "auth" --strategies "hybrid,decisions"`
- `/total-recall open tasks` → `multi-search "tasks" --strategies "todos"`
- `/total-recall recent heating issues` → `multi-search "heating issues" --strategies "hybrid,problems" --when "last week"`

The command returns JSON with ideas containing:
- `content`: The extracted idea
- `intent`: Type (decision, conclusion, question, problem, solution, todo, context)
- `topic`: The topic span this idea belongs to
- `session`: Which conversation session (project)
- `distance`: Semantic similarity (lower = more similar)
- `_strategy`: Which search strategy found this result

### Present Results

Format results concisely grouped by intent type. If no results, say so briefly and suggest different keywords.

## Database Stats

To check what's indexed:
```bash
~/.claude/skills/total-recall/bin/total-recall stats
```

## List Indexed Sessions

See all sessions with idea and topic counts:
```bash
~/.claude/skills/total-recall/bin/total-recall sessions
```

## Unanswered Questions

To see open questions from past conversations:
```bash
~/.claude/skills/total-recall/bin/total-recall questions
```

## Get Idea Details

To get a specific idea with its relations:
```bash
~/.claude/skills/total-recall/bin/total-recall get <idea_id>
```

## Find Similar Ideas

Find ideas semantically similar to a given idea:
```bash
~/.claude/skills/total-recall/bin/total-recall similar <idea_id> -n 5
~/.claude/skills/total-recall/bin/total-recall similar <idea_id> --same-session  # Same session only
~/.claude/skills/total-recall/bin/total-recall similar <idea_id> --other-sessions  # Cross-session only
```

## View Source Context

See the original transcript around an idea:
```bash
~/.claude/skills/total-recall/bin/total-recall context <idea_id>
~/.claude/skills/total-recall/bin/total-recall context <idea_id> -B 10 -A 10  # More context
```

## Search by Date Range

Filter search results by time:
```bash
~/.claude/skills/total-recall/bin/total-recall search "<query>" --since 2024-01-01
~/.claude/skills/total-recall/bin/total-recall search "<query>" --until 2024-06-01
~/.claude/skills/total-recall/bin/total-recall search "<query>" --since 2024-01-01 --until 2024-06-01
```

## Export and Import

Backup the memory database:
```bash
~/.claude/skills/total-recall/bin/total-recall export -o backup.json
~/.claude/skills/total-recall/bin/total-recall export -s project-name -o project-backup.json  # Single session
```

Restore from backup:
```bash
~/.claude/skills/total-recall/bin/total-recall import backup.json  # Merge with existing
~/.claude/skills/total-recall/bin/total-recall import backup.json --replace  # Replace all data
```

## Prune Old Data

Remove old ideas to keep the database lean:
```bash
~/.claude/skills/total-recall/bin/total-recall prune -d 90  # Dry run - shows what would be removed
~/.claude/skills/total-recall/bin/total-recall prune -d 90 --execute  # Actually delete
```

## Notes

- Results are from previously indexed conversations
- If nothing found, the conversation may not be indexed yet
- Use `/total-recall backfill` to index existing conversation history

## Graph Revision

The indexed graph is an interpretation of conversations and can be revised. The raw conversation transcripts remain immutable - revisions only affect the interpreted graph.

**Reclassify an idea's intent:**
```bash
~/.claude/skills/total-recall/bin/total-recall update-intent <idea_id> decision
# Valid intents: decision, conclusion, question, problem, solution, todo, context
```

**Move an idea to a different topic:**
```bash
~/.claude/skills/total-recall/bin/total-recall move-idea <idea_id> <span_id>
```

**Merge topics:**
```bash
~/.claude/skills/total-recall/bin/total-recall merge-spans <source_span_id> <target_span_id>
```

**Mark an idea as superseding another:**
```bash
~/.claude/skills/total-recall/bin/total-recall supersede <old_idea_id> <new_idea_id>
```

## Auto-Categorization

Automatically improve topic organization using LLM analysis.

**Auto-assign topics to projects:**
```bash
~/.claude/skills/total-recall/bin/total-recall auto-categorize  # Dry run
~/.claude/skills/total-recall/bin/total-recall auto-categorize --execute  # Apply changes
```

**Run all categorization improvements:**
```bash
~/.claude/skills/total-recall/bin/total-recall improve
```
This auto-categorizes unassigned topics, renames poorly-named topics, and reports remaining issues.

## Quality Filtering

Keep the database lean by removing low-value content.

**Review ideas against regex filters:**
```bash
~/.claude/skills/total-recall/bin/total-recall review-ideas  # Dry run
~/.claude/skills/total-recall/bin/total-recall review-ideas -t <topic_id>  # Specific topic
~/.claude/skills/total-recall/bin/total-recall review-ideas --execute  # Delete filtered
```

**Use LLM to identify subtle low-value content:**
```bash
~/.claude/skills/total-recall/bin/total-recall llm-filter  # Dry run
~/.claude/skills/total-recall/bin/total-recall llm-filter -t <topic_id> -b 30  # Specific topic, batch size 30
~/.claude/skills/total-recall/bin/total-recall llm-filter --execute  # Delete flagged
```
LLM filtering catches things regex misses: generic statements, context-dependent content, redundant ideas.

## Project Hierarchy

Organize topics into projects for better structure.

**List projects:**
```bash
~/.claude/skills/total-recall/bin/total-recall projects
```

**Create a project:**
```bash
~/.claude/skills/total-recall/bin/total-recall create-project "My Project" -d "Description"
```

**Assign a topic to a project:**
```bash
~/.claude/skills/total-recall/bin/total-recall assign-topic <topic_id> "My Project"
```

**Set a topic's parent (create topic hierarchy):**
```bash
~/.claude/skills/total-recall/bin/total-recall reparent-topic <topic_id> <parent_topic_id>
```

**Remove a topic from its parent:**
```bash
~/.claude/skills/total-recall/bin/total-recall unparent-topic <topic_id>
```

**View hierarchy tree:**
```bash
~/.claude/skills/total-recall/bin/total-recall tree
```

## Timeline Visualization

See activity across time for topics or projects.

**Topic timeline (see activity for a topic across sessions):**
```bash
~/.claude/skills/total-recall/bin/total-recall timeline --topic "LoRa prototyping"
```

**Project timeline (see recent activity by date):**
```bash
~/.claude/skills/total-recall/bin/total-recall timeline --project rad-control-v1-1 --days 14
```

Output shows spans grouped by date with key decisions and conclusions highlighted.

## Temporal Queries

Search with natural language time expressions and temporal aggregation.

**Natural language time in search:**
```bash
~/.claude/skills/total-recall/bin/total-recall search "heating" --when "last week"
~/.claude/skills/total-recall/bin/total-recall search "decisions" --when "since tuesday"
~/.claude/skills/total-recall/bin/total-recall search "LoRa" --when "since jan 5"
```

**Search after a specific session:**
```bash
~/.claude/skills/total-recall/bin/total-recall search "follow up" --after-session abc123 --global
```

**Activity aggregation by period:**
```bash
~/.claude/skills/total-recall/bin/total-recall activity --by day --days 7
~/.claude/skills/total-recall/bin/total-recall activity --by week --days 30 -s rad-control-v1-1
```

**Topic activity over time:**
```bash
~/.claude/skills/total-recall/bin/total-recall topic-activity 42 --by week --days 90
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
~/.claude/skills/total-recall/bin/total-recall backfill --all
```

This returns immediately after enqueueing. Indexing happens in the background via daemon.

**Check progress:**
```bash
~/.claude/skills/total-recall/bin/total-recall stats
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
~/.claude/skills/total-recall/bin/total-recall stats
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
~/.claude/skills/total-recall/bin/total-recall topics
```

For a specific session:
```bash
~/.claude/skills/total-recall/bin/total-recall topics -s <session>
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
