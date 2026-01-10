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

### Step 1: Bootstrap

Ensure the runtime is ready:
```bash
SKILL_DIR="$HOME/.claude/skills/memgraph"
RUNTIME=$("$SKILL_DIR/bootstrap.sh")
```

If this fails, the dependencies need to be installed. Run:
```bash
cd ~/.claude-plugin-memgraph && uv sync
```

### Step 2: Search

Run vector search for relevant ideas:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" search "<query>" 10
```

This returns ideas with:
- `content`: The extracted idea
- `intent`: Type (decision, conclusion, question, problem, solution, todo, context)
- `topic`: The topic span this idea belongs to
- `session`: Which conversation session
- `source_file`: Original transcript path
- `source_line`: Line number
- `distance`: Semantic similarity (lower = more similar)

### Step 3: Search Topic Spans (if needed)

For broader context, also search topic spans:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" search-spans "<query>" 5
```

### Step 4: Hybrid Search (for specific terms)

If the query contains specific terminology the user might remember exactly:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" hybrid "<query>" 10
```

This combines vector similarity with BM25 keyword matching.

### Step 5: Present Results

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

### Step 6: Offer Deep Dive

If results seem incomplete or user wants more detail:
- Offer to read the original transcript sections
- Offer to search with different terms
- Suggest `/memory-backfill` if database seems empty

## Database Stats

To check what's indexed:
```bash
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" stats
```

## Notes

- Results are from previously indexed conversations
- If nothing found, the conversation may not be indexed yet
- Use `/memory-backfill` to index existing conversation history
