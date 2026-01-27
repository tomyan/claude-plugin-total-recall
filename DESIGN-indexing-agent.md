# Indexing Agent Design

## Overview

Replace the current "prompt → response" indexing with an **agent-based approach** where the indexer has tools to query existing knowledge, enabling intelligent semantic analysis.

## Goals (from existing feature set)

The indexing agent must support all existing features:

### From DESIGN.md
- **Relation detection**: supersedes, builds_on, contradicts, answers, relates_to
- **Topic tracking**: detect shifts, build hierarchy, generate summaries
- **Entity extraction**: projects, technologies, concepts, people, files
- **Confidence scoring**: tentative vs firm decisions
- **Content filtering**: skip greetings, acknowledgments, debugging output

### From IMPROVEMENT_PLAN.md
- **Importance at encoding**: weight decisions/conclusions higher for retention
- **Answer detection**: mark questions as answered when solutions appear
- **Cross-session linking**: relate ideas across different projects

### From EXPERT_REVIEW.md
- **LLM-driven curation**: agent decides what's worth remembering (not regex)
- **Spreading activation**: note which existing ideas are relevant

## Current State (Problems)

1. **One LLM call per file** - expensive, no cross-session batching
2. **Static context** - we guess what context to include (last 20 ideas)
3. **No semantic reasoning** - can't detect "this answers question #42"
4. **Regex-based topic detection** - not intelligent
5. **No importance scoring** - all ideas weighted equally at encoding time
6. **No entity deduplication** - same concept extracted multiple times

## Proposed Architecture

### Hook (unchanged, <15ms)
```bash
# Just notifies daemon that a file has new content
sqlite3 "$DB_PATH" "INSERT OR IGNORE INTO work_queue (file_path) VALUES ('$TRANSCRIPT')"
```

### Daemon Batching Strategy

**Continuous mode** (hook-triggered):
- Collect file notifications for 2-3 seconds
- Process all updated files in single agent call
- Multiple sessions in one call (clearly demarcated)

**Backfill mode**:
- Process one session at a time
- Fill context window with session history
- Larger batches, sequential processing

```python
class IndexingDaemon:
    async def continuous_cycle(self):
        # Wait for batch collection window
        await asyncio.sleep(BATCH_WINDOW)  # 2-3 seconds

        # Get all pending files
        files = await get_queue_items(limit=20)
        if not files:
            return

        # Read new content from each file
        updates = []
        for f in files:
            start_byte = await get_byte_position(f.path)
            new_messages = read_from_byte(f.path, start_byte)
            if new_messages:
                updates.append({
                    "session": session_from_path(f.path),
                    "file_path": f.path,
                    "messages": new_messages,
                    "start_byte": start_byte
                })

        if updates:
            # Single agent call for all updates
            await run_indexing_agent(updates, mode="continuous")

    async def backfill_session(self, session: str, file_path: str):
        # Process one session, fill context window
        start_byte = await get_byte_position(file_path)

        while True:
            messages = read_from_byte(file_path, start_byte, max_tokens=50000)
            if not messages:
                break

            await run_indexing_agent([{
                "session": session,
                "file_path": file_path,
                "messages": messages,
                "start_byte": start_byte
            }], mode="backfill")

            start_byte = messages[-1].end_byte
```

### Indexing Agent Tools

The agent can query existing knowledge to make intelligent decisions.

#### Core Search Tools
```python
{
    "name": "search_ideas",
    "description": "Search for existing ideas semantically similar to a query. Use to find duplicates, related ideas, or potential supersession targets.",
    "parameters": {
        "query": "string - search query",
        "limit": "int - max results (default 10)",
        "session": "string - optional, limit to session",
        "intent": "string - optional, filter by type (decision, question, etc)"
    }
}

{
    "name": "get_open_questions",
    "description": "Get unanswered questions from a session. Use to check if new content answers an existing question.",
    "parameters": {
        "session": "string - session identifier",
        "limit": "int - max results (default 10)"
    }
}

{
    "name": "get_open_todos",
    "description": "Get uncompleted todo items from a session. Use to check if new content completes a task.",
    "parameters": {
        "session": "string - session identifier",
        "limit": "int - max results (default 10)"
    }
}
```

#### Topic/Span Tools
```python
{
    "name": "get_current_span",
    "description": "Get the current active topic span for a session, including name, summary, and line range.",
    "parameters": {
        "session": "string - session identifier"
    }
}

{
    "name": "list_session_spans",
    "description": "List all topic spans in a session with their names and summaries.",
    "parameters": {
        "session": "string - session identifier"
    }
}

{
    "name": "find_similar_topics",
    "description": "Find similar topics across all sessions. Use to link related discussions.",
    "parameters": {
        "topic_name": "string - topic name to match",
        "limit": "int - max results (default 5)"
    }
}

{
    "name": "get_topic_hierarchy",
    "description": "Get project/topic hierarchy tree for understanding context structure.",
    "parameters": {
        "session": "string - optional, limit to session"
    }
}
```

#### Entity Tools
```python
{
    "name": "search_entities",
    "description": "Search for existing entities by name. Use to avoid creating duplicate entities.",
    "parameters": {
        "name": "string - entity name (partial match)",
        "type": "string - optional, filter by type (project, technology, concept, file)"
    }
}

{
    "name": "get_entity_ideas",
    "description": "Get ideas linked to an entity. Use to understand entity context.",
    "parameters": {
        "entity_id": "int - entity ID",
        "limit": "int - max results (default 10)"
    }
}
```

#### Context Tools
```python
{
    "name": "get_recent_ideas",
    "description": "Get recent ideas from a session for context.",
    "parameters": {
        "session": "string - session identifier",
        "limit": "int - max ideas (default 20)",
        "intent": "string - optional, filter by type"
    }
}

{
    "name": "get_idea_relations",
    "description": "Get relations for an idea (what it supersedes, builds on, etc).",
    "parameters": {
        "idea_id": "int - idea ID"
    }
}
```

### Agent System Prompt

```
You are an indexing agent that analyzes conversation transcripts and maintains a knowledge graph.

You have tools to query existing knowledge. Use them to:
1. Check if similar ideas already exist (avoid duplicates)
2. Find open questions that new content might answer
3. Understand current topic context before detecting topic shifts
4. Link related ideas across sessions

For each batch of new messages, you should:
1. First, use tools to understand existing state (recent ideas, open questions, current topics)
2. Analyze the new messages
3. Extract new ideas with proper categorization
4. Detect topic changes (explain reasoning)
5. Identify answered questions (link to the answer)
6. Create relations between new and existing ideas

Output your analysis as structured JSON.
```

### Agent Input Format

```json
{
  "mode": "continuous",
  "updates": [
    {
      "session": "clade-plugin-total-recall",
      "file_path": "/path/to/transcript.jsonl",
      "messages": [
        {"line": 900, "role": "human", "content": "why was i doing the async refactor?", "timestamp": "..."},
        {"line": 910, "role": "assistant", "content": "Let me check the design doc...", "timestamp": "..."}
      ]
    },
    {
      "session": "cdp-cli",
      "file_path": "/path/to/other.jsonl",
      "messages": [
        {"line": 200, "role": "human", "content": "add a help command", "timestamp": "..."}
      ]
    }
  ]
}
```

### Agent Output Schema

```json
{
  "ideas": [
    {
      "session": "clade-plugin-total-recall",
      "source_file": "/path/to/transcript.jsonl",
      "source_line": 910,
      "content": "The async refactor was done to fix SQLite lock contention during parallel indexing",
      "intent": "conclusion",
      "confidence": 0.9,
      "importance": 0.8,
      "entities": [
        {"name": "SQLite", "type": "technology"},
        {"name": "async", "type": "concept"},
        {"name": "lock contention", "type": "concept"}
      ],
      "related_to": [42, 45]
    }
  ],

  "topic_updates": [
    {
      "session": "clade-plugin-total-recall",
      "span_id": 42,
      "name": "Async Migration (Complete)",
      "summary": "Completed full async migration of database operations to fix SQLite lock contention. Now moving to indexing agent design."
    }
  ],

  "topic_changes": [
    {
      "session": "clade-plugin-total-recall",
      "from_span_id": 42,
      "new_topic_name": "Indexing Agent Architecture",
      "reason": "Conversation shifted from async implementation to designing agent-based indexing",
      "start_line": 950
    }
  ],

  "answered_questions": [
    {
      "question_idea_id": 38,
      "answered_at_line": 910,
      "answer_summary": "Async refactor was for SQLite lock contention during parallel indexing"
    }
  ],

  "completed_todos": [
    {
      "todo_idea_id": 55,
      "completed_at_line": 920,
      "completion_summary": "Committed async migration changes"
    }
  ],

  "relations": [
    {
      "from_session": "clade-plugin-total-recall",
      "from_line": 910,
      "to_idea_id": 42,
      "relation_type": "builds_on"
    },
    {
      "from_session": "clade-plugin-total-recall",
      "from_line": 915,
      "to_idea_id": 30,
      "relation_type": "supersedes"
    }
  ],

  "entity_links": [
    {
      "idea_source_line": 910,
      "entity_id": 15,
      "entity_name": "SQLite"
    }
  ],

  "skip_lines": [905, 906],

  "activated_ideas": [42, 45, 38]
}
```

**Field definitions:**

| Field | Purpose |
|-------|---------|
| `ideas[].importance` | 0-1 score for retention weighting (decisions=1.0, context=0.3) |
| `ideas[].entities` | Named entities with type for cross-session linking |
| `answered_questions` | Links questions to their answers |
| `completed_todos` | Links todos to their completion |
| `relations[].relation_type` | One of: supersedes, builds_on, contradicts, answers, relates_to |
| `entity_links` | Reuse existing entities instead of creating duplicates |
| `skip_lines` | Lines that are low-value (greetings, acknowledgments) |
| `activated_ideas` | Existing ideas relevant to this batch (for working memory) |

## Agent Reasoning Guidelines

The system prompt should guide the agent to:

### Content Filtering
- **Skip**: Greetings ("hello", "thanks"), acknowledgments ("ok", "got it"), debugging output, tool use preambles
- **Extract**: Decisions, conclusions, questions, problems, solutions, todos, important context

### Importance Scoring
```
importance = 1.0 if intent in (decision, conclusion) else
             0.8 if intent in (question, problem, solution) else
             0.5 if intent == todo else
             0.3  # context
```

### Topic Detection
- Use `get_current_span()` to understand current topic
- Detect shifts when:
  - Explicit transition ("let's move on to...", "back to...")
  - Domain change (different project, technology area)
  - Significant conceptual shift
- Create new span with clear reason

### Relation Detection
- Use `search_ideas()` to find similar existing ideas
- **supersedes**: New decision replaces old ("we changed from X to Y")
- **builds_on**: Extends/refines existing idea
- **contradicts**: In tension with existing idea
- **answers**: Solution responds to question/problem
- **relates_to**: General semantic connection

### Entity Handling
- Use `search_entities()` before creating new entities
- Link to existing entity if >80% match
- Prefer canonical names (e.g., "SQLite" not "sqlite")

### Question/Todo Tracking
- Use `get_open_questions()` to check if new content provides an answer
- Use `get_open_todos()` to check if new content completes a task
- Link explicitly in output

## Implementation Plan

### Phase 1: Tool Infrastructure
1. Create `llm/indexing_tools.py` with tool definitions
2. Create `llm/tool_handlers.py` with database query handlers
3. Create `llm/agent_harness.py` for tool call loop

### Phase 2: Agent Integration
1. Update daemon batching strategy (continuous: 2-3s window)
2. Implement `run_indexing_agent()` with multi-session support
3. Create structured output parser and executor

### Phase 3: Backfill Mode
1. Implement single-session batching (fill context window)
2. Add progress tracking for large backfills
3. Handle session ordering (oldest first)

### Phase 4: Testing & Tuning
1. Test with real conversation history
2. Tune prompts for accuracy
3. Measure LLM call reduction
4. Verify relation accuracy

## Token Budget Considerations

**Continuous mode** (2-3 second batches):
- New messages: ~2-5K tokens typically
- Tool results: ~2-3K tokens
- System prompt: ~1K tokens
- Output: ~1-2K tokens
- **Total: ~10K tokens per cycle**

**Backfill mode** (per session batch):
- New messages: ~30-50K tokens
- Tool results: ~5K tokens
- System prompt: ~1K tokens
- Output: ~5K tokens
- **Total: ~50K tokens per batch**

## Open Questions

1. **Tool call limits**: How many tool calls should agent make before we cut off? (Suggest: 10-15 max)

2. **Cross-session relations**: How aggressively should we link across sessions?
   - Option A: Only link via shared entities
   - Option B: Agent decides based on semantic similarity
   - Option C: Always search cross-session for duplicates

3. **Confidence thresholds**: When is agent confidence too low to store an idea? (Suggest: <0.3)

4. **Deduplication strategy**:
   - Agent detects duplicates via `search_ideas()` tool
   - DB has unique constraint on (source_file, source_line)
   - Should we also dedupe by content hash?

5. **Embedding generation**:
   - Option A: Agent outputs embeddings (not practical)
   - Option B: Post-process executor generates embeddings (current)
   - Option C: Batch embed all new ideas after agent completes

6. **Working memory activation**:
   - Should indexing agent activate ideas it references?
   - Or is activation only for search-time?

7. **Message storage**:
   - Current: Store raw messages in FTS table
   - With agent: Still store messages, or just store ideas?
   - Recommend: Keep message storage for RAG fallback

## Integration with Existing Features

| Existing Feature | How Agent Supports It |
|-----------------|----------------------|
| Hybrid search | Agent-extracted ideas have embeddings + FTS entries |
| HyDE search | Works on agent-extracted ideas |
| Working memory | Agent outputs `activated_ideas` for boosting |
| Forgetting | Agent outputs `importance` for retention scoring |
| Consolidation | Agent creates fewer, higher-quality ideas (less need to consolidate) |
| Query decomposition | Works on agent-extracted ideas |
| Multi-hop reasoning | Agent creates explicit relations |
| Reflection | Works on agent-extracted ideas |
| Timeline | Agent preserves source_line for temporal ordering |
| Entity search | Agent creates/links entities consistently |
