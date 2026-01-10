---
context: fork
name: memory-stats
description: Show memory database statistics
---

# Memory Stats Skill

Shows statistics about the memory database - how many ideas, sessions, and entities are indexed.

## Invocation

`/memory-stats`

## Instructions

When the user invokes `/memory-stats`:

### Step 1: Bootstrap

Ensure runtime is ready:
```bash
SKILL_DIR="$HOME/.claude/skills/memgraph"
RUNTIME=$("$SKILL_DIR/bootstrap.sh")
```

### Step 2: Get Stats

Run the stats command:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" stats
```

This returns:
- `total_ideas`: Number of indexed ideas
- `total_spans`: Number of topic spans
- `total_entities`: Number of extracted entities
- `total_relations`: Number of idea relationships
- `sessions_indexed`: Number of unique sessions
- `by_intent`: Breakdown by idea type (decision, question, etc.)
- `entities_by_type`: Breakdown by entity type

### Step 3: Present Results

Format the stats for the user:

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

## Notes

- If stats are empty, suggest running `/memory-backfill` to index existing conversations
- The stop hook automatically adds new content after each turn
