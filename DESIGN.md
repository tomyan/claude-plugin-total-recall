# Claude Memory Graph - Design Document

## 1. Purpose

A long-term memory system for Claude Code that captures, indexes, and retrieves knowledge from past conversations. Enables continuity across sessions by surfacing relevant prior discussions, decisions, and insights.

**Goals:**
- Remember decisions, conclusions, and insights across sessions
- Track how thinking evolves over time
- Surface relevant context without manual searching
- Support multiple retrieval strategies (semantic, temporal, entity-based)
- Minimal overhead - indexing happens automatically

**Non-goals:**
- Not a general-purpose knowledge base
- Not a replacement for project documentation
- Not a collaborative/shared memory system (single user)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │   /remember │     │ Stop Hook   │     │  Session Context    │ │
│  │   (retrieve)│     │  (index)    │     │  (auto-inject)      │ │
│  └──────┬──────┘     └──────┬──────┘     └──────────┬──────────┘ │
└─────────┼───────────────────┼──────────────────────┼────────────┘
          │                   │                      │
          ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Memory Graph System                          │
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │  Retriever  │     │   Indexer   │     │   Topic Tracker     │ │
│  │             │     │             │     │                     │ │
│  │ - Vector    │     │ - Extract   │     │ - Detect shifts     │ │
│  │ - Hybrid    │     │ - Classify  │     │ - Summarize spans   │ │
│  │ - HyDE      │     │ - Embed     │     │ - Build hierarchy   │ │
│  │ - Expand    │     │ - Link      │     │                     │ │
│  └──────┬──────┘     └──────┬──────┘     └──────────┬──────────┘ │
│         │                   │                      │             │
│         └───────────────────┴──────────────────────┘             │
│                             │                                     │
│                             ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    SQLite + sqlite-vec                     │   │
│  │                                                            │   │
│  │  ┌─────────┐  ┌────────┐  ┌──────────┐  ┌───────────────┐ │   │
│  │  │  spans  │  │ ideas  │  │ entities │  │  embeddings   │ │   │
│  │  │  (tree) │  │        │  │          │  │  (vec0)       │ │   │
│  │  └─────────┘  └────────┘  └──────────┘  └───────────────┘ │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Indexer** - Processes new transcript content after each turn (via Stop hook)
2. **Topic Tracker** - Detects topic shifts, builds hierarchical span summaries
3. **Retriever** - Multi-strategy search invoked by `/remember` skill
4. **Database** - SQLite with sqlite-vec for vector search, FTS5 for keyword search

---

## 3. Data Model

### 3.1 Hierarchical Spans

Conversations are chunked into a tree of spans, each with a summary and embedding:

```
Session: control-v1-1
└── Topic: "Plant room system design" (summary + embedding)
    ├── Sub-span: "Relay specifications" (summary + embedding)
    ├── Sub-span: "Safety requirements" (summary + embedding)
    └── Sub-span: "Enclosure design" (summary + embedding)
└── Topic: "LoRa prototyping" (summary + embedding)
    ├── Sub-span: "Ping-pong test setup"
    ├── Sub-span: "TCXO debugging"
    └── Sub-span: "Range testing"
└── Topic: "Memory skill design" (summary + embedding)
    └── ...
```

**Schema:**
```sql
CREATE TABLE spans (
    id INTEGER PRIMARY KEY,
    session TEXT NOT NULL,
    parent_id INTEGER REFERENCES spans(id),
    name TEXT NOT NULL,
    summary TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER,
    depth INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_spans_session ON spans(session);
CREATE INDEX idx_spans_parent ON spans(parent_id);
```

### 3.2 Ideas (Atomic Insights)

Individual decisions, conclusions, insights extracted from conversation:

```sql
CREATE TABLE ideas (
    id INTEGER PRIMARY KEY,
    span_id INTEGER REFERENCES spans(id),
    content TEXT NOT NULL,
    intent TEXT CHECK(intent IN (
        'decision',      -- "We decided to use X"
        'conclusion',    -- "The key insight is..."
        'question',      -- "How should we handle X?"
        'problem',       -- "The issue is..."
        'solution',      -- "Fixed by doing X"
        'todo',          -- "Need to implement X"
        'context'        -- Background information
    )),
    confidence REAL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    answered BOOLEAN,  -- For questions: has it been answered?
    source_file TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ideas_span ON ideas(span_id);
CREATE INDEX idx_ideas_intent ON ideas(intent);
```

### 3.3 Entities

Named entities extracted from ideas for cross-session linking:

```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT CHECK(type IN (
        'project',       -- control-v1.1, lora-test
        'technology',    -- ESP32, SX1262, SQLite
        'concept',       -- mesh networking, fail-safe
        'person',        -- names mentioned
        'file'           -- specific files referenced
    )),
    UNIQUE(name, type)
);

CREATE TABLE idea_entities (
    idea_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (idea_id, entity_id)
);

CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(type);
```

### 3.4 Relations

Edges between ideas capturing evolution and connections:

```sql
CREATE TABLE relations (
    id INTEGER PRIMARY KEY,
    from_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
    to_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
    relation_type TEXT CHECK(relation_type IN (
        'supersedes',    -- New idea replaces old
        'builds_on',     -- New idea extends old
        'contradicts',   -- Ideas in tension
        'answers',       -- Solution answers question/problem
        'relates_to'     -- General association
    )),
    UNIQUE(from_id, to_id, relation_type)
);

CREATE INDEX idx_relations_from ON relations(from_id);
CREATE INDEX idx_relations_to ON relations(to_id);
```

### 3.5 Embeddings (Vector Search)

Separate virtual tables for spans and ideas:

```sql
CREATE VIRTUAL TABLE span_embeddings USING vec0(
    span_id INTEGER PRIMARY KEY,
    embedding FLOAT[1536]
);

CREATE VIRTUAL TABLE idea_embeddings USING vec0(
    idea_id INTEGER PRIMARY KEY,
    embedding FLOAT[1536]
);
```

### 3.6 Full-Text Search

For hybrid retrieval (vector + keyword):

```sql
CREATE VIRTUAL TABLE ideas_fts USING fts5(
    content,
    content='ideas',
    content_rowid='id'
);

CREATE VIRTUAL TABLE spans_fts USING fts5(
    name,
    summary,
    content='spans',
    content_rowid='id'
);
```

### 3.7 Index State

Track indexing progress per transcript file:

```sql
CREATE TABLE index_state (
    file_path TEXT PRIMARY KEY,
    last_line INTEGER DEFAULT 0,
    last_indexed TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Indexing Pipeline

### 4.1 Trigger

**Stop Hook** runs after each Claude response:
- Receives transcript path from hook environment
- Calls indexer with the transcript path
- Indexes only new content since last run

### 4.2 Processing Steps

```
For each new message in transcript:
    1. Parse JSON line
    2. Skip low-value content (greetings, acknowledgments, debugging output)

    3. Topic Detection:
       - Compare to current topic context
       - If topic shift detected:
         a. Close current span (set end_line)
         b. Generate summary of closed span
         c. Embed summary
         d. Start new span

    4. Idea Extraction:
       - Identify substantive content
       - Classify intent (decision, question, problem, solution, etc.)
       - Assess confidence (tentative vs firm)
       - Extract entities mentioned

    5. Storage:
       - Store idea with span linkage
       - Embed idea content
       - Store entity links
       - Update FTS index

    6. Relation Detection:
       - Search for similar existing ideas
       - Detect supersession ("changed from X to Y")
       - Detect build-on ("additionally...")
       - Detect answers (solution follows question)
       - Store relations

    7. Update index state
```

### 4.3 Topic Detection Heuristics

A topic shift is indicated by:
- Explicit transition: "okay, let's move on to...", "back to..."
- Domain change: different project, technology area
- Time gap: significant pause in conversation
- Depth change: zooming in/out significantly

### 4.4 Span Summarization

When closing a span, generate summary that captures:
- Main topic/focus
- Key decisions made
- Conclusions reached
- Open questions remaining

Summary is embedded for span-level retrieval.

---

## 5. Retrieval System

### 5.1 Retrieval Strategies

#### Vector Search (Primary)
```sql
SELECT i.*, e.distance
FROM idea_embeddings e
JOIN ideas i ON i.id = e.idea_id
WHERE e.embedding MATCH ? AND k = ?
ORDER BY e.distance;
```

#### Hybrid Search (Vector + BM25)
- Vector search for semantic similarity
- FTS5 BM25 for exact keyword matches
- Reciprocal Rank Fusion to combine results

```sql
-- BM25 keyword search
SELECT id, bm25(ideas_fts) as score
FROM ideas_fts
WHERE ideas_fts MATCH ?
ORDER BY score;
```

#### HyDE (Hypothetical Document Embeddings)
For vague queries:
1. LLM generates hypothetical answer to query
2. Embed the hypothetical answer
3. Search with that embedding
4. Often retrieves better matches than raw query

#### Graph Expansion
After initial retrieval:
1. Find vector matches
2. Expand via relations (ideas that build_on or supersede matches)
3. Include parent/child spans for context

#### Temporal Filtering
- Filter by session, date range
- Recency weighting for time-sensitive queries
- "What did we discuss last week about X?"

### 5.2 Retrieval Flow

```
/remember <query>

1. Query Analysis:
   - Detect temporal qualifiers ("last week", "recently")
   - Detect entity mentions ("ESP32", "control board")
   - Detect intent filters ("decisions about", "problems with")

2. Multi-Strategy Search:
   a. Vector search on idea_embeddings
   b. Vector search on span_embeddings (coarse)
   c. BM25 on ideas_fts (if keywords detected)
   d. Entity lookup (if entities mentioned)

3. Fusion & Ranking:
   - Combine results via Reciprocal Rank Fusion
   - Apply temporal decay if relevant
   - Apply confidence weighting

4. Graph Expansion:
   - Follow relations from top matches
   - Include parent spans for context

5. Result Assembly:
   - Group by topic/span
   - Include evolution chains if supersession detected
   - Format for presentation
```

### 5.3 Context Injection (Future)

Optionally auto-inject relevant context at session start:
- Find spans related to current project
- Surface recent open questions/TODOs
- Include superseded decisions (for historical context)

---

## 6. Skill Interface

### 6.1 /remember (Retrieval)

```markdown
---
context: fork
---

# Memory Retrieval Skill

Invoke: /remember <query>

Searches past conversations for relevant context.
```

**Usage Examples:**
- `/remember LoRa range testing` - semantic search
- `/remember decisions about relay ratings` - intent-filtered
- `/remember last week ESP32` - temporal + entity
- `/remember open questions` - intent filter

### 6.2 /memory-stats

```markdown
# Memory Stats Skill

Invoke: /memory-stats

Shows database statistics: total ideas, spans, entities, sessions indexed.
```

### 6.3 /memory-topics

```markdown
# Memory Topics Skill

Invoke: /memory-topics [session]

Lists topic spans for a session or across all sessions.
```

### 6.4 /memory-backfill

```markdown
# Memory Backfill Skill

Invoke: /memory-backfill [path]

Indexes existing conversation history.

Modes:
- /memory-backfill              - Index current session's history
- /memory-backfill <file>       - Index specific transcript file
- /memory-backfill <directory>  - Index all transcripts in directory
- /memory-backfill --all        - Index all transcripts in ~/.claude/projects/
```

**Use cases:**
- First-time setup: backfill valuable past conversations
- Current session: index everything discussed before skill was loaded
- Selective import: index specific project histories

---

## 7. Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Database schema (all tables, indexes, virtual tables)
- [ ] Basic store/retrieve functions
- [ ] OpenAI embedding integration
- [ ] CLI for testing (init, store, search, stats)

### Phase 2: Basic Indexer
- [ ] Stop hook integration
- [ ] Transcript parsing (JSON lines)
- [ ] Simple idea extraction (content + source reference)
- [ ] Basic embedding and storage
- [ ] Index state tracking

### Phase 3: Retrieval & Backfill Skills
- [ ] `/remember` skill with vector search
- [ ] `/memory-backfill` skill for importing history
- [ ] Basic result formatting
- [ ] Session filtering

### Phase 4: Topic Tracking
- [ ] Topic shift detection
- [ ] Span creation and hierarchy
- [ ] Span summarization
- [ ] Span embeddings

### Phase 5: Rich Extraction
- [ ] Intent classification (decision, question, problem, etc.)
- [ ] Entity extraction
- [ ] Confidence assessment
- [ ] Relation detection

### Phase 6: Advanced Retrieval
- [ ] Hybrid search (vector + BM25)
- [ ] HyDE implementation
- [ ] Graph expansion
- [ ] Temporal filtering

### Phase 7: Polish
- [ ] `/memory-stats` skill
- [ ] `/memory-topics` skill
- [ ] Error handling and edge cases
- [ ] Performance optimization

---

## 8. Technical Decisions

### SQLite + sqlite-vec over Graph DB
- Simpler, embedded, single file
- Vector search is primary retrieval method
- Graph structure is simple (tree + sparse edges)
- No complex multi-hop traversals needed

### OpenAI Embeddings (text-embedding-3-small)
- 1536 dimensions
- Good quality/cost balance
- Well-supported
- Environment variable: `OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS`

### Python with uv
- sqlite-vec has good Python bindings
- uv for fast, reliable dependency management
- Runtime at `~/.claude-plugin-total-recall/`
- **Self-bootstrapping**: Skills detect missing deps and install on first run

### Stop Hook for Indexing
- Runs after each turn automatically
- No background process needed
- Incremental indexing (only new content)

### Forked Context for Retrieval
- `/remember` runs in isolated context
- Doesn't pollute main conversation
- Returns concise summary

---

## 9. Runtime Bootstrap

### Directory Structure
```
~/.claude-plugin-total-recall/     # Runtime (deps + data only)
├── .venv/                     # Virtual environment (created by uv)
├── memory.db                  # SQLite database
└── pyproject.toml             # Dependencies

~/.claude/skills/total-recall/     # Skill (code lives here)
├── SKILL.md
├── bootstrap.sh
└── src/
    └── memory_db.py
```

### Bootstrap Script

Skills call a bootstrap script before running Python code:

```bash
#!/bin/bash
# bootstrap.sh - ensures runtime is ready

RUNTIME_DIR="$HOME/.claude-plugin-total-recall"
PLUGIN_DIR="$(dirname "$(dirname "$0")")"  # Resolve from skills/total-recall/

# Create runtime directory
mkdir -p "$RUNTIME_DIR"

# Copy/update source files
cp -r "$PLUGIN_DIR/src/"* "$RUNTIME_DIR/src/" 2>/dev/null || mkdir -p "$RUNTIME_DIR/src"
cp "$PLUGIN_DIR/src/"*.py "$RUNTIME_DIR/src/"

# Initialize uv project if needed
if [ ! -f "$RUNTIME_DIR/pyproject.toml" ]; then
    cd "$RUNTIME_DIR"
    uv init --name memgraph
    uv add sqlite-vec openai
fi

# Ensure venv exists
if [ ! -d "$RUNTIME_DIR/.venv" ]; then
    cd "$RUNTIME_DIR"
    uv sync
fi

echo "$RUNTIME_DIR"
```

### Skill Invocation Pattern

Each skill starts with:
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
RUNTIME=$("$SKILL_DIR/bootstrap.sh")
cd "$RUNTIME" && uv run python "$SKILL_DIR/src/memory_db.py" <command> [args]
```

This ensures:
1. First run installs dependencies automatically
2. Subsequent runs are fast (venv already exists)
3. Source always runs from skill folder (no copying needed)

---

## 10. File Structure

```
claude-plugin-memgraph/
├── DESIGN.md              # This document
├── README.md              # Usage documentation
├── plugin.json            # Plugin manifest
│
├── skills/
│   └── memgraph/
│       ├── SKILL.md       # /remember skill definition
│       ├── backfill.md    # /memory-backfill skill
│       ├── stats.md       # /memory-stats skill
│       ├── topics.md      # /memory-topics skill
│       ├── bootstrap.sh   # Runtime bootstrap script
│       │
│       └── src/           # Source code (runs from here, not copied)
│           └── memory_db.py
│
├── hooks/
│   └── stop_hook.sh       # Stop hook script
│
└── tests/
    └── test_db.py
```

---

## 11. Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Summarization** | LLM (indexer agent) | Indexer is already Claude - summarization is natural part of processing |
| **Entity extraction** | LLM | Use the model - indexer identifies entities as it reads |
| **Topic detection** | Hybrid | Embedding distance flags potential shifts, LLM confirms. Cheap signal + smart judgment |
| **Privacy** | Index everything | User controls what's in transcripts - no extra filtering needed |
| **Pruning** | None (for now) | Storage is cheap. Can add pruning later if growth becomes an issue |

**General principle:** Use the model. The indexer agent is Claude - leverage its understanding rather than building parallel heuristics.

---

## 12. Daemon Architecture (v2)

The original design called the LLM per-message during hook execution. This had problems:
- Hook blocks user typing while processing
- No batching = inefficient LLM usage
- Network/sleep issues cause hangs or silent failures
- Context lost between messages

### 12.1 Decoupled Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│                                                                   │
│  ┌─────────────────┐                  ┌─────────────────────────┐│
│  │ UserPromptSubmit│                  │        Stop             ││
│  │      Hook       │                  │        Hook             ││
│  └────────┬────────┘                  └───────────┬─────────────┘│
└───────────┼───────────────────────────────────────┼──────────────┘
            │                                       │
            │  Enqueue file path + size (fast)      │
            ▼                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         work_queue                               │
│                    (SQLite table)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Poll every 2s
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Indexing Daemon                               │
│                  (background process)                            │
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐│
│  │  Message    │     │   Batch     │     │     LLM Call        ││
│  │  Collector  │────▶│  Assembler  │────▶│   (single call)     ││
│  │ (5s window) │     │  (context)  │     │                     ││
│  └─────────────┘     └─────────────┘     └──────────┬──────────┘│
│                                                      │           │
│                                          ┌───────────▼─────────┐│
│                                          │   Action Executor   ││
│                                          │ - Update topic      ││
│                                          │ - Store ideas       ││
│                                          │ - Create spans      ││
│                                          │ - Extract entities  ││
│                                          └───────────┬─────────┘│
│                                                      │           │
│                                          ┌───────────▼─────────┐│
│                                          │  Batch Embeddings   ││
│                                          └─────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Key Principles

1. **Hooks are fast** - Only enqueue file path, don't process
2. **Daemon is resilient** - Timeouts, exponential backoff (max 10s), infinite retry
3. **Batch for efficiency** - Collect messages within time window, single LLM call
4. **Context-aware** - LLM sees hierarchy (project → topic → span) + surrounding messages
5. **Structured protocol** - Well-defined JSON input/output for LLM

### 12.3 LLM Protocol

**Input Context:**
```json
{
  "hierarchy": {
    "project": {"name": "control-v1-1", "description": "..."},
    "topic": {"name": "LoRa prototyping", "summary": "..."},
    "span": {"name": "Range testing", "summary": "..."}
  },
  "recent_messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "new_messages": [
    {"role": "user", "content": "...", "timestamp": "...", "line": 42},
    {"role": "assistant", "content": "...", "timestamp": "...", "line": 43}
  ]
}
```

**Output Protocol:**
```json
{
  "topic_update": {
    "name": "LoRa range testing results",
    "summary": "Testing max range with different antenna configs"
  },
  "new_span": {
    "name": "Antenna comparison",
    "reason": "Shifted from general testing to comparing antenna types"
  },
  "items": [
    {
      "type": "decision",
      "content": "Using the 5dBi antenna for final deployment",
      "confidence": 0.9,
      "source_line": 42,
      "entities": ["5dBi antenna", "SX1262"]
    },
    {
      "type": "question",
      "content": "What's the power consumption difference?",
      "answered": false,
      "source_line": 43
    }
  ],
  "relations": [
    {"from_line": 42, "to_idea_id": 156, "type": "supersedes"}
  ],
  "skip_lines": [44, 45]
}
```

### 12.4 Benefits for Existing Features

| Feature | Benefit |
|---------|---------|
| **Topic detection** | LLM sees full context, detects shifts accurately |
| **Span naming** | Names refined as conversation develops |
| **Intent classification** | Context-aware, not just keywords |
| **Entity extraction** | Batch extract from multiple messages |
| **Relation detection** | LLM identifies connections within batch |
| **Confidence scoring** | Based on conversation tone, not heuristics |
| **Cross-session linking** | Can reference similar topics in prompt |
| **Working memory** | Activate related ideas based on batch content |

### 12.5 Resilience

All operations wrapped with:
- **Timeouts**: DB (30s), File (60s), LLM (120s)
- **Exponential backoff**: 1s → 2s → 4s → 8s → 10s (capped)
- **Infinite retry**: Transient failures don't kill daemon
- **Clear logging**: Every operation logged, errors visible
- **Heartbeat**: Log every 60s to show daemon is alive

---

## 13. Implementation Plan (Elephant Carpaccio)

Thin vertical slices, each delivering working functionality. TDD with adversarial review at each commit.

### Slice 1: Message Batcher
**Goal**: Collect messages within time window, emit batches

```
Input:  Stream of (file_path, byte_position) from queue
Output: Batches of messages with metadata

Tests:
- Messages within 5s window batched together
- Messages > 5s apart in separate batches
- Empty file returns empty batch
- Handles file read errors gracefully
```

**Files**: `src/batcher.py`
**Commit**: "feat: message batcher with time windowing"

---

### Slice 2: Context Builder
**Goal**: Assemble hierarchy context for LLM

```
Input:  Current span_id, session
Output: Hierarchy dict (project → topic → span) with summaries

Tests:
- Returns empty hierarchy for new session
- Includes project if topic assigned
- Includes parent spans if nested
- Handles missing/orphan spans
```

**Files**: `src/context.py`
**Commit**: "feat: context builder for LLM prompt"

---

### Slice 3: LLM Protocol - Input
**Goal**: Format batch + context as LLM prompt

```
Input:  Batch of messages, hierarchy context, recent messages
Output: Structured prompt string

Tests:
- Includes all hierarchy levels
- Recent messages truncated to N
- New messages include line numbers
- Total tokens under limit
```

**Files**: `src/protocol.py`
**Commit**: "feat: LLM input protocol formatting"

---

### Slice 4: LLM Protocol - Output
**Goal**: Parse and validate LLM response

```
Input:  Raw LLM JSON response
Output: Validated actions (topic_update, items, relations, etc.)

Tests:
- Parses valid response
- Handles missing optional fields
- Validates intent types
- Rejects malformed JSON gracefully
- Handles partial responses
```

**Files**: `src/protocol.py` (extend)
**Commit**: "feat: LLM output protocol parsing"

---

### Slice 5: Action Executor - Ideas
**Goal**: Store ideas from LLM response

```
Input:  Parsed items array
Output: Ideas stored in database

Tests:
- Stores idea with correct intent
- Links to current span
- Handles duplicates (source_line unique)
- Records confidence and entities
```

**Files**: `src/executor.py`
**Commit**: "feat: action executor for ideas"

---

### Slice 6: Action Executor - Topic Updates
**Goal**: Update topic/span from LLM response

```
Input:  topic_update, new_span from response
Output: Updated span name/summary, new span if shift

Tests:
- Updates span name if changed
- Updates summary if provided
- Creates child span on topic shift
- Links to existing topic or creates new
```

**Files**: `src/executor.py` (extend)
**Commit**: "feat: action executor for topic updates"

---

### Slice 7: Action Executor - Relations
**Goal**: Create relations between ideas

```
Input:  relations array from response
Output: Relations stored in database

Tests:
- Creates relation between ideas
- Maps source_line to idea_id
- Validates relation_type
- Handles missing target ideas
```

**Files**: `src/executor.py` (extend)
**Commit**: "feat: action executor for relations"

---

### Slice 8: Batch Embeddings
**Goal**: Embed all new ideas in single API call

```
Input:  List of idea IDs to embed
Output: Embeddings stored in idea_embeddings

Tests:
- Batches up to 100 texts per call
- Handles API failures with retry
- Skips already-embedded ideas
- Updates span embedding if summary changed
```

**Files**: `src/embeddings.py`
**Commit**: "feat: batch embedding for ideas"

---

### Slice 9: Daemon Integration
**Goal**: Wire batcher → LLM → executor → embeddings

```
Input:  Work queue items
Output: Fully processed transcripts

Tests:
- End-to-end: queue item → ideas in DB
- Handles LLM timeout with retry
- Updates byte position on success
- Removes queue item on completion
```

**Files**: `src/daemon.py` (rewrite)
**Commit**: "feat: integrated daemon with batched LLM"

---

### Slice 10: Cross-Session Context
**Goal**: Include similar topics from other sessions in prompt

```
Input:  Current topic name/summary
Output: Related topics to include in context

Tests:
- Finds semantically similar topics
- Limits to top 3 related
- Excludes current session
- Handles no similar topics
```

**Files**: `src/context.py` (extend)
**Commit**: "feat: cross-session topic context"

---

### Slice 11: Working Memory Integration
**Goal**: Activate related ideas based on batch content

```
Input:  Processed batch
Output: Working memory updated with activations

Tests:
- Activates ideas mentioned in batch
- Decays old activations
- Prunes very low activations
- Respects session scope
```

**Files**: `src/working_memory.py`
**Commit**: "feat: working memory activation from batches"

---

### Slice 12: Backfill Integration
**Goal**: Backfill uses batched processing

```
Input:  Transcript files to backfill
Output: All content processed via daemon

Tests:
- Enqueues all files
- Progress tracking works
- Handles large transcripts
- Deduplicates existing ideas
```

**Files**: `src/backfill.py` (update)
**Commit**: "feat: backfill via daemon queue"

---

## 14. TDD Workflow

For each slice:

1. **Write failing tests** - Define expected behavior
2. **Implement minimum code** - Make tests pass
3. **Adversarial review** - Challenge assumptions:
   - What if input is malformed?
   - What if network fails mid-operation?
   - What if database is locked?
   - What if LLM returns unexpected format?
4. **Add edge case tests** - Based on review
5. **Refactor** - Clean up while tests pass
6. **Commit** - Atomic, working slice

---

## 15. Fresh Start

No backwards compatibility needed. On first run of new daemon:
1. Delete `~/.claude-plugin-total-recall/memory.db`
2. Reinitialize schema
3. Backfill all transcripts with new batched processing

This ensures clean data with proper context-aware classification from the start.
