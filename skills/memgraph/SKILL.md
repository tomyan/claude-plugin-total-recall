---
context: fork
name: remember
description: Search past conversations for relevant context
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

Run the memgraph CLI (handles bootstrap automatically):
```bash
SKILL_DIR="$HOME/.claude/skills/memgraph"
uv run python "$SKILL_DIR/src/cli.py" search "<query>" -n 10
```

This returns ideas with:
- `content`: The extracted idea
- `intent`: Type (decision, conclusion, question, problem, solution, todo, context)
- `topic`: The topic span this idea belongs to
- `session`: Which conversation session
- `source_file`: Original transcript path
- `source_line`: Line number
- `distance`: Semantic similarity (lower = more similar)

### Step 2: Choose Search Strategy

Based on the query, choose the best search strategy:

**Hybrid Search** - For queries with specific terms:
```bash
uv run python "$SKILL_DIR/src/cli.py" hybrid "<query>" -n 10
```

**HyDE Search** - For vague/conceptual queries:
```bash
uv run python "$SKILL_DIR/src/cli.py" hyde "<query>" -n 10
```

**Filtered Search** - For queries with intent:
```bash
uv run python "$SKILL_DIR/src/cli.py" search "<query>" -i decision -n 10
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

## Notes

- Results are from previously indexed conversations
- If nothing found, the conversation may not be indexed yet
- Use `/memory-backfill` to index existing conversation history
