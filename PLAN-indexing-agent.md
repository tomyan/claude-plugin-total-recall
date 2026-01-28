# Indexing Agent Implementation Plan

Elephant carpaccio slices - each delivers working, tested functionality.

## Phase 1: Entity Schema (MDM Pattern)

### Slice 1.1: Entity Mentions Table
**Goal:** Create immutable entity mention records

```sql
entity_mentions (
  id TEXT PRIMARY KEY,  -- ULID
  name TEXT NOT NULL,
  metadata JSON,
  source_file TEXT,
  source_line INTEGER,
  golden_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

**Tests:**
- Can create entity mention with ULID
- Metadata stored as JSON
- Source reference preserved
- Created_at auto-populated

**Files:** `db/schema.py`, `tests/test_entity_mentions.py`

---

### Slice 1.2: Golden Entities Table
**Goal:** Create canonical entity records

```sql
golden_entities (
  id TEXT PRIMARY KEY,  -- ULID
  canonical_name TEXT NOT NULL UNIQUE,
  metadata JSON,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT
)
```

**Tests:**
- Can create golden entity
- Canonical name is unique
- Updated_at changes on update
- Can link mention to golden

**Files:** `db/schema.py`, `tests/test_golden_entities.py`

---

### Slice 1.3: Entity Resolution Functions
**Goal:** Functions to resolve mentions to golden records

```python
async def create_entity_mention(name, metadata, source_file, source_line) -> str
async def find_golden_entity(name) -> Optional[dict]  # fuzzy match
async def create_golden_entity(canonical_name, metadata) -> str
async def link_mention_to_golden(mention_id, golden_id) -> None
async def get_entity_mentions(golden_id) -> list[dict]
```

**Tests:**
- Create mention without golden (unresolved)
- Find golden by exact name
- Find golden by fuzzy match (80% similarity)
- Link mention to golden
- Get all mentions for a golden

**Files:** `src/entities.py`, `tests/test_entities.py`

---

## Phase 2: Indexing Agent Tools

### Slice 2.1: Tool Definition Schema
**Goal:** Define tool schema format for agent

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, ToolParameter]

@dataclass
class ToolParameter:
    type: str
    description: str
    required: bool = True
    default: Any = None
```

**Tests:**
- Can define tool with parameters
- Can serialize to JSON schema format
- Can validate tool call arguments

**Files:** `llm/tools.py`, `tests/test_tool_schema.py`

---

### Slice 2.2: Search Ideas Tool
**Goal:** Implement search_ideas tool handler

```python
async def tool_search_ideas(query: str, limit: int = 10,
                            session: str = None, intent: str = None) -> list[dict]
```

**Tests:**
- Returns semantically similar ideas
- Respects limit
- Filters by session when provided
- Filters by intent when provided
- Returns empty list when no matches

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.3: Open Questions Tool
**Goal:** Implement get_open_questions tool handler

```python
async def tool_get_open_questions(session: str, limit: int = 10) -> list[dict]
```

**Tests:**
- Returns questions with answered=FALSE
- Filters by session
- Orders by recency
- Returns id, content, source_line

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.4: Open Todos Tool
**Goal:** Implement get_open_todos tool handler

```python
async def tool_get_open_todos(session: str, limit: int = 10) -> list[dict]
```

**Tests:**
- Returns todos that aren't completed
- Filters by session
- Orders by recency

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.5: Current Span Tool
**Goal:** Implement get_current_span tool handler

```python
async def tool_get_current_span(session: str) -> Optional[dict]
```

**Tests:**
- Returns most recent span for session
- Includes name, summary, start_line
- Returns None for new session

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.6: Session Spans Tool
**Goal:** Implement list_session_spans tool handler

```python
async def tool_list_session_spans(session: str) -> list[dict]
```

**Tests:**
- Returns all spans for session
- Ordered by start_line
- Includes hierarchy (parent_id, depth)

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.7: Search Entities Tool
**Goal:** Implement search_entities tool handler

```python
async def tool_search_entities(name: str, type: str = None) -> list[dict]
```

**Tests:**
- Fuzzy matches entity names
- Filters by type in metadata
- Returns golden entities
- Returns mention count per golden

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.8: Recent Ideas Tool
**Goal:** Implement get_recent_ideas tool handler

```python
async def tool_get_recent_ideas(session: str, limit: int = 20,
                                intent: str = None) -> list[dict]
```

**Tests:**
- Returns ideas ordered by recency
- Filters by session
- Filters by intent
- Includes content, intent, source_line

**Files:** `llm/indexing_tools.py`, `tests/test_indexing_tools.py`

---

### Slice 2.9: Tool Registry
**Goal:** Registry to look up and invoke tools

```python
class ToolRegistry:
    def register(self, tool: ToolDefinition, handler: Callable)
    def get_tool_definitions(self) -> list[dict]  # JSON schema format
    async def invoke(self, tool_name: str, arguments: dict) -> Any

INDEXING_TOOLS = ToolRegistry()
# Register all tools
```

**Tests:**
- Can register tool with handler
- Can get all definitions as JSON
- Can invoke tool by name
- Raises error for unknown tool
- Validates arguments before invoke

**Files:** `llm/tool_registry.py`, `tests/test_tool_registry.py`

---

## Phase 3: Agent Harness

### Slice 3.1: Agent Message Types
**Goal:** Define message types for agent conversation

```python
@dataclass
class AgentMessage:
    role: Literal["system", "user", "assistant", "tool_result"]
    content: str
    tool_calls: list[ToolCall] = None
    tool_call_id: str = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
```

**Tests:**
- Can create system/user/assistant messages
- Can create message with tool calls
- Can create tool result message
- Can serialize to API format

**Files:** `llm/agent_types.py`, `tests/test_agent_types.py`

---

### Slice 3.2: Agent Loop (Single Turn)
**Goal:** Execute one agent turn with tool handling

```python
async def agent_turn(messages: list[AgentMessage],
                     tools: ToolRegistry) -> AgentMessage:
    # Call LLM with messages and tool definitions
    # Return assistant message (may include tool_calls)
```

**Tests:**
- Sends messages to LLM
- Includes tool definitions
- Returns assistant message
- Parses tool calls from response

**Files:** `llm/agent_harness.py`, `tests/test_agent_harness.py`

---

### Slice 3.3: Tool Execution
**Goal:** Execute tool calls and create result messages

```python
async def execute_tool_calls(tool_calls: list[ToolCall],
                             registry: ToolRegistry) -> list[AgentMessage]:
    # Execute each tool call
    # Return tool_result messages
```

**Tests:**
- Executes each tool call
- Creates result message per call
- Handles tool errors gracefully
- Includes tool_call_id in result

**Files:** `llm/agent_harness.py`, `tests/test_agent_harness.py`

---

### Slice 3.4: Agent Loop (Multi-Turn)
**Goal:** Full agent loop until completion or limit

```python
async def run_agent(system_prompt: str,
                    user_input: str,
                    tools: ToolRegistry,
                    max_turns: int = 10) -> dict:
    # Run agent loop until:
    # - Assistant responds without tool calls (done)
    # - Max turns reached
    # Return final response parsed as JSON
```

**Tests:**
- Runs until assistant has no tool calls
- Respects max_turns limit
- Accumulates conversation history
- Parses final response as JSON
- Handles malformed JSON gracefully

**Files:** `llm/agent_harness.py`, `tests/test_agent_harness.py`

---

## Phase 4: Output Executor

### Slice 4.1: Parse Agent Output
**Goal:** Parse and validate agent JSON output

```python
@dataclass
class AgentOutput:
    ideas: list[IdeaOutput]
    topic_updates: list[TopicUpdate]
    topic_changes: list[TopicChange]
    answered_questions: list[AnsweredQuestion]
    completed_todos: list[CompletedTodo]
    relations: list[RelationOutput]
    entity_links: list[EntityLink]
    skip_lines: list[int]
    activated_ideas: list[int]

def parse_agent_output(raw: dict) -> AgentOutput
```

**Tests:**
- Parses all fields from valid output
- Handles missing optional fields
- Validates intent types
- Validates relation types
- Returns empty lists for missing arrays

**Files:** `indexer/output_parser.py`, `tests/test_output_parser.py`

---

### Slice 4.2: Execute Ideas
**Goal:** Store ideas from agent output

```python
async def execute_ideas(ideas: list[IdeaOutput],
                        session: str,
                        source_file: str) -> list[int]:
    # Store each idea
    # Create entity mentions
    # Return idea IDs
```

**Tests:**
- Creates idea with all fields
- Sets importance from output
- Creates entity mentions for entities
- Returns created idea IDs
- Handles duplicates (source_file, source_line)

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.3: Execute Topic Updates
**Goal:** Update spans from agent output

```python
async def execute_topic_updates(updates: list[TopicUpdate]) -> None:
    # Update span name/summary
```

**Tests:**
- Updates span name
- Updates span summary
- Handles non-existent span gracefully

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.4: Execute Topic Changes
**Goal:** Create new spans for topic shifts

```python
async def execute_topic_changes(changes: list[TopicChange],
                                session: str) -> list[int]:
    # Create new span for each change
    # Link to parent span
    # Return new span IDs
```

**Tests:**
- Creates new span
- Sets parent to from_span_id
- Records start_line
- Stores reason in summary

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.5: Execute Answered Questions
**Goal:** Mark questions as answered

```python
async def execute_answered_questions(answers: list[AnsweredQuestion]) -> None:
    # Set answered=TRUE on question ideas
    # Create 'answers' relation
```

**Tests:**
- Sets answered=TRUE on idea
- Creates relation from answer to question
- Handles non-existent question gracefully

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.6: Execute Relations
**Goal:** Create relations between ideas

```python
async def execute_relations(relations: list[RelationOutput],
                           source_file: str,
                           idea_line_map: dict[int, int]) -> int:
    # Map source_line to idea_id
    # Create relation
    # Return count created
```

**Tests:**
- Creates relation with correct type
- Maps from_line to idea_id
- Handles missing target idea
- Deduplicates relations

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.7: Execute Entity Links
**Goal:** Link mentions to existing golden entities

```python
async def execute_entity_links(links: list[EntityLink],
                               source_file: str,
                               idea_line_map: dict[int, int]) -> None:
    # Link entity mention to golden
```

**Tests:**
- Links mention to specified golden
- Handles non-existent golden gracefully

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.8: Execute Activated Ideas
**Goal:** Update working memory activations

```python
async def execute_activated_ideas(idea_ids: list[int], session: str) -> None:
    # Boost activation for each idea
```

**Tests:**
- Increases activation for ideas
- Creates working memory entry if needed
- Updates last_access timestamp

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

### Slice 4.9: Full Executor
**Goal:** Execute complete agent output

```python
async def execute_agent_output(output: AgentOutput,
                               session: str,
                               source_file: str) -> dict:
    # Execute all components
    # Generate embeddings for new ideas
    # Return stats
```

**Tests:**
- Executes all output types
- Returns count of each type executed
- Handles partial failures gracefully
- Generates embeddings for new ideas

**Files:** `indexer/executor.py`, `tests/test_executor.py`

---

## Phase 5: Daemon Integration

### Slice 5.1: Batch Collector
**Goal:** Collect messages across files for batching

```python
@dataclass
class BatchUpdate:
    session: str
    file_path: str
    messages: list[Message]
    start_byte: int
    end_byte: int

async def collect_batch_updates(files: list[str]) -> list[BatchUpdate]:
    # Read new content from each file
    # Parse messages
    # Return updates with content
```

**Tests:**
- Reads from last byte position
- Parses JSONL messages
- Groups by session
- Tracks byte positions

**Files:** `indexer/batch_collector.py`, `tests/test_batch_collector.py`

---

### Slice 5.2: Format Agent Input
**Goal:** Format batch updates for agent

```python
def format_agent_input(updates: list[BatchUpdate],
                       mode: str) -> str:
    # Format as JSON for agent
    # Include mode (continuous/backfill)
```

**Tests:**
- Formats multiple sessions
- Includes all message fields
- Sets mode correctly
- Truncates very long messages

**Files:** `indexer/agent_input.py`, `tests/test_agent_input.py`

---

### Slice 5.3: Indexing Agent System Prompt
**Goal:** Create system prompt for indexing agent

```python
INDEXING_SYSTEM_PROMPT = """
You are an indexing agent...
[Full prompt with guidelines]
"""
```

**Tests:**
- Prompt includes tool usage instructions
- Prompt includes output schema
- Prompt includes filtering guidelines
- Prompt includes importance scoring

**Files:** `indexer/prompts.py`, `tests/test_prompts.py`

---

### Slice 5.4: Run Indexing Agent
**Goal:** Top-level function to run indexing

```python
async def run_indexing_agent(updates: list[BatchUpdate],
                             mode: str = "continuous") -> dict:
    # Format input
    # Run agent with tools
    # Parse output
    # Execute output
    # Update byte positions
    # Return stats
```

**Tests:**
- Runs full pipeline
- Updates byte positions on success
- Returns execution stats
- Handles agent errors

**Files:** `indexer/run.py`, `tests/test_indexer_run.py`

---

### Slice 5.5: Continuous Mode Daemon
**Goal:** Update daemon for continuous batching

```python
async def continuous_cycle(self):
    # Wait for batch window (2-3s)
    # Get pending files from queue
    # Collect batch updates
    # Run indexing agent
    # Remove processed from queue
```

**Tests:**
- Waits for batch window
- Processes multiple files together
- Single agent call per cycle
- Handles empty queue

**Files:** `daemon.py`, `tests/test_daemon_continuous.py`

---

### Slice 5.6: Backfill Mode
**Goal:** Implement backfill with session batching

```python
async def backfill_session(self, file_path: str):
    # Process one session
    # Fill context window per batch
    # Continue until file complete
```

**Tests:**
- Processes single session
- Respects token limit per batch
- Continues from last position
- Handles large files

**Files:** `daemon.py`, `tests/test_daemon_backfill.py`

---

## Phase 6: Testing & Polish

### Slice 6.1: Integration Test - Continuous
**Goal:** End-to-end test of continuous indexing

**Tests:**
- Hook enqueues file
- Daemon picks up after batch window
- Agent extracts ideas
- Ideas searchable

**Files:** `tests/test_integration_continuous.py`

---

### Slice 6.2: Integration Test - Backfill
**Goal:** End-to-end test of backfill

**Tests:**
- Backfill command enqueues files
- Daemon processes session by session
- All content indexed
- Progress trackable

**Files:** `tests/test_integration_backfill.py`

---

### Slice 6.3: Performance Baseline
**Goal:** Measure and document performance

**Metrics:**
- LLM calls per file (before vs after)
- Token usage per file
- Indexing latency
- Relation accuracy

**Files:** `tests/test_performance.py`, `BENCHMARK.md`

---

## Progress Tracking

| Slice | Status | Commit |
|-------|--------|--------|
| 1.1 Entity Mentions | ✅ | (earlier) |
| 1.2 Golden Entities | ✅ | (earlier) |
| 1.3 Entity Resolution | ✅ | (earlier) |
| 2.1 Tool Schema | ✅ | 0d6bdeb |
| 2.2 Search Ideas Tool | ✅ | f54272c |
| 2.3 Open Questions Tool | ✅ | f54272c |
| 2.4 Open Todos Tool | ✅ | f54272c |
| 2.5 Current Span Tool | ✅ | f54272c |
| 2.6 Session Spans Tool | ✅ | f54272c |
| 2.7 Search Entities Tool | ✅ | f54272c |
| 2.8 Recent Ideas Tool | ✅ | f54272c |
| 2.9 Tool Registry | ✅ | 2ee298d |
| 3.1 Agent Message Types | ✅ | 7df0bbf |
| 3.2 Agent Loop Single | ✅ | 05e0047 |
| 3.3 Tool Execution | ✅ | 05e0047 |
| 3.4 Agent Loop Multi | ✅ | 05e0047 |
| 4.1 Parse Output | ✅ | dfdd0a8 |
| 4.2 Execute Ideas | ✅ | 6bd130a |
| 4.3 Execute Topic Updates | ✅ | 6bd130a |
| 4.4 Execute Topic Changes | ✅ | 6bd130a |
| 4.5 Execute Answered | ✅ | 6bd130a |
| 4.6 Execute Relations | ✅ | 6bd130a |
| 4.7 Execute Entity Links | ✅ | 6bd130a |
| 4.8 Execute Activated | ✅ | 6bd130a |
| 4.9 Full Executor | ✅ | 6bd130a |
| 5.1 Batch Collector | ✅ | e7926a4 |
| 5.2 Format Agent Input | ✅ | e7926a4 |
| 5.3 System Prompt | ✅ | e7926a4 |
| 5.4 Run Indexing Agent | ✅ | e7926a4 |
| 5.5 Continuous Daemon | ⏳ | |
| 5.6 Backfill Mode | ⏳ | |
| 6.1 Integration Continuous | ⏳ | |
| 6.2 Integration Backfill | ⏳ | |
| 6.3 Performance | ⏳ | |
