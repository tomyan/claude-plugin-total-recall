# Memgraph Improvement Plan

Based on the expert review, this document outlines the implementation plan for all identified improvements using Test-Driven Development with adversarial review cycles.

## Development Methodology

Each improvement follows this cycle:

```
┌─────────────────────────────────────────────────────────────┐
│  1. RED: Write failing tests that specify desired behavior  │
│     ↓                                                       │
│  2. ADVERSARIAL REVIEW: Challenge test coverage & design    │
│     ↓                                                       │
│  3. GREEN: Implement minimum code to pass tests             │
│     ↓                                                       │
│  4. ADVERSARIAL REVIEW: Challenge implementation quality    │
│     ↓                                                       │
│  5. REFACTOR: Improve code without changing behavior        │
│     ↓                                                       │
│  6. ADVERSARIAL REVIEW: Any more refactors needed?          │
│     ↓ (loop until clean)                                    │
│  7. COMMIT & PUSH                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Code Organization (Foundation)

### 1.1 Split memory_db.py into Modules

**Goal:** Break 5,400-line monolith into focused modules for maintainability.

**Target Structure:**
```
skills/memgraph/src/
├── __init__.py
├── cli.py              # CLI entry point (existing)
├── indexer.py          # Indexing pipeline (existing)
├── transcript.py       # Transcript parsing (existing)
├── db/
│   ├── __init__.py
│   ├── connection.py   # Database connection, initialization
│   ├── schema.py       # Table definitions, migrations
│   ├── ideas.py        # Idea CRUD operations
│   ├── spans.py        # Span CRUD operations
│   ├── topics.py       # Topic CRUD operations
│   ├── relations.py    # Relation operations
│   └── entities.py     # Entity operations
├── search/
│   ├── __init__.py
│   ├── vector.py       # Vector search (sqlite-vec)
│   ├── keyword.py      # FTS5 keyword search
│   ├── hybrid.py       # Hybrid search (RRF)
│   ├── hyde.py         # HyDE search
│   └── temporal.py     # Temporal search
├── embeddings/
│   ├── __init__.py
│   ├── provider.py     # Abstract embedding provider
│   ├── openai.py       # OpenAI implementation
│   ├── cache.py        # Embedding cache
│   └── local.py        # Future: local model support
├── llm/
│   ├── __init__.py
│   ├── claude.py       # Claude CLI integration
│   └── tasks.py        # Topic naming, summarization, etc.
├── analysis/
│   ├── __init__.py
│   ├── intent.py       # Intent classification
│   ├── entities.py     # Entity extraction
│   └── topics.py       # Topic detection, shift detection
└── config.py           # Centralized configuration
```

**TDD Approach:**
```
RED:
  - Write tests importing from new module paths
  - Tests should fail with ImportError initially

GREEN:
  - Create module structure
  - Move functions to appropriate modules
  - Add re-exports to maintain backward compatibility

REFACTOR:
  - Remove circular dependencies
  - Ensure consistent error handling across modules
  - Add module-level docstrings
```

**Adversarial Review Questions:**
- Are module boundaries clean? No circular imports?
- Is backward compatibility maintained for existing callers?
- Are public APIs clearly defined in `__init__.py`?
- Is there a single source of truth for each concept?

---

### 1.2 Centralize Configuration

**Goal:** Replace hardcoded thresholds with a configuration system.

**Config Schema:**
```python
@dataclass
class MemgraphConfig:
    # Topic Detection
    topic_shift_threshold: float = 0.55
    topic_shift_strong_delta: float = 0.15
    topic_shift_history_size: int = 3
    topic_shift_min_divergent: int = 2

    # Relation Detection
    relation_similarity_threshold: float = 0.75
    relation_lookback_count: int = 10

    # Cross-Session Linking
    cross_session_link_threshold: float = 0.80

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_cache_size: int = 10000

    # Search
    hybrid_bm25_weight: float = 0.3
    hybrid_vector_weight: float = 0.7
    default_search_limit: int = 10

    # Consolidation (Phase 2)
    consolidation_age_threshold_days: int = 30
    consolidation_min_ideas: int = 3

    # Forgetting (Phase 2)
    decay_rate: float = 0.1
    min_retention_days: int = 7

    # Working Memory (Phase 2)
    working_memory_capacity: int = 50
    working_memory_decay_rate: float = 0.1
```

**TDD Approach:**
```
RED:
  - Test loading config from file
  - Test environment variable overrides
  - Test default values
  - Test validation (e.g., thresholds in 0-1 range)

GREEN:
  - Implement config dataclass
  - Add file loading (YAML or TOML)
  - Add env var override support

REFACTOR:
  - Replace all hardcoded values with config references
  - Add config to function signatures where needed
```

---

## Phase 2: Cognitive Improvements (Core Value)

### 2.1 Working Memory

**Goal:** Track recently accessed ideas to provide context-aware retrieval.

**Design:**
```python
class WorkingMemory:
    """
    Tracks activation levels of recently accessed ideas.

    Cognitive model:
    - Ideas are activated when retrieved or mentioned
    - Activation decays over time
    - High-activation ideas are boosted in search
    - Provides "what we've been discussing" context
    """

    def __init__(self, capacity: int, decay_rate: float):
        self.activations: dict[int, float] = {}
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.access_times: dict[int, datetime] = {}

    def activate(self, idea_id: int, strength: float = 1.0) -> None:
        """Boost activation when idea is accessed."""

    def decay_all(self) -> None:
        """Apply time-based decay to all activations."""

    def get_active_context(self, limit: int = 10) -> list[int]:
        """Get most active idea IDs for context."""

    def boost_search_results(self, results: list[dict]) -> list[dict]:
        """Re-rank search results by activation."""

    def save_state(self) -> None:
        """Persist to database for session continuity."""

    def load_state(self, session: str) -> None:
        """Restore from database."""
```

**Database Addition:**
```sql
CREATE TABLE working_memory (
    session TEXT NOT NULL,
    idea_id INTEGER REFERENCES ideas(id),
    activation REAL NOT NULL,
    last_access TEXT NOT NULL,
    PRIMARY KEY (session, idea_id)
);
```

**TDD Approach:**
```
RED:
  - Test activation increases on access
  - Test decay reduces activation over time
  - Test capacity limit evicts lowest activation
  - Test search boost increases scores of active ideas
  - Test persistence across sessions

GREEN:
  - Implement WorkingMemory class
  - Add database table and persistence
  - Integrate with search pipeline

REFACTOR:
  - Optimize decay calculation (batch updates)
  - Add activation strength based on access type (search vs mention)
```

**Adversarial Review Questions:**
- Does decay model match cognitive research?
- Is persistence efficient (not too many DB writes)?
- Does boost factor distort search quality?
- How does this interact with temporal search?

---

### 2.2 Memory Consolidation

**Goal:** Periodically merge old, similar ideas into higher-level summaries.

**Design:**
```python
class MemoryConsolidator:
    """
    Consolidates old memories into higher-level representations.

    Cognitive model:
    - Fresh memories: detailed, specific
    - Old memories: abstracted, gist-based
    - Important memories (decisions): preserved verbatim
    - Context memories: consolidated into summaries
    """

    def consolidate_session(
        self,
        session: str,
        age_threshold_days: int = 30,
        min_ideas_to_consolidate: int = 3
    ) -> ConsolidationResult:
        """
        Consolidate old ideas in a session.

        Process:
        1. Find ideas older than threshold
        2. Group by topic
        3. For each topic group:
           a. Separate high-value (decisions, conclusions) from context
           b. Preserve high-value verbatim
           c. Summarize context ideas into consolidated idea
           d. Mark originals as consolidated (not deleted)
        4. Update embeddings for consolidated ideas
        """

    def should_preserve(self, idea: dict) -> bool:
        """
        Determine if idea should be preserved verbatim.

        Preserve if:
        - Intent is 'decision' or 'conclusion'
        - Confidence > 0.8
        - Has outgoing 'supersedes' relation (important evolution)
        - Manually starred by user
        """

    def generate_consolidation_summary(
        self,
        ideas: list[dict],
        topic_name: str
    ) -> str:
        """Use LLM to summarize multiple ideas into one."""
```

**Database Addition:**
```sql
ALTER TABLE ideas ADD COLUMN consolidated_into INTEGER REFERENCES ideas(id);
ALTER TABLE ideas ADD COLUMN is_consolidated BOOLEAN DEFAULT FALSE;
ALTER TABLE ideas ADD COLUMN preserve_verbatim BOOLEAN DEFAULT FALSE;

CREATE INDEX idx_ideas_consolidated ON ideas(consolidated_into);
```

**TDD Approach:**
```
RED:
  - Test old context ideas get consolidated
  - Test decisions are preserved verbatim
  - Test consolidated summary contains key points
  - Test original ideas marked but not deleted
  - Test search still finds consolidated content
  - Test consolidation is idempotent

GREEN:
  - Implement MemoryConsolidator
  - Add CLI command: `consolidate --session X --age 30`
  - Add automatic trigger option in hooks

REFACTOR:
  - Optimize batch processing for large sessions
  - Add dry-run mode to preview consolidation
```

**Adversarial Review Questions:**
- Is the preservation heuristic correct? Missing any important cases?
- Does consolidation lose important nuance?
- Can users undo consolidation?
- How does consolidation affect relation graph?

---

### 2.3 Automatic Forgetting

**Goal:** Implement decay-based forgetting to keep index lean and improve retrieval.

**Design:**
```python
class ForgettingMechanism:
    """
    Implements strategic forgetting based on memory research.

    Factors affecting retention:
    - Recency: newer memories decay slower
    - Frequency: often-accessed memories are retained
    - Importance: high-confidence decisions never forgotten
    - Relevance: memories related to active work retained
    """

    def calculate_retention_score(self, idea: dict) -> float:
        """
        Calculate how strongly an idea should be retained.

        Score = (
            0.3 * recency_score +      # Based on created_at
            0.3 * access_score +        # Based on retrieval count
            0.2 * importance_score +    # Based on intent + confidence
            0.2 * relevance_score       # Based on working memory activation
        )
        """

    def identify_forgettable(
        self,
        session: str,
        threshold: float = 0.2,
        min_age_days: int = 7
    ) -> list[int]:
        """Find ideas below retention threshold."""

    def forget(
        self,
        idea_ids: list[int],
        mode: Literal["soft", "hard"] = "soft"
    ) -> int:
        """
        Forget ideas.

        Soft: Mark as forgotten, exclude from search, keep in DB
        Hard: Delete from database entirely
        """
```

**Database Addition:**
```sql
ALTER TABLE ideas ADD COLUMN access_count INTEGER DEFAULT 0;
ALTER TABLE ideas ADD COLUMN last_accessed TEXT;
ALTER TABLE ideas ADD COLUMN forgotten BOOLEAN DEFAULT FALSE;

CREATE INDEX idx_ideas_forgotten ON ideas(forgotten);
```

**TDD Approach:**
```
RED:
  - Test recency affects retention score
  - Test frequently accessed ideas retained
  - Test decisions never auto-forgotten
  - Test soft forget excludes from search
  - Test hard forget removes from database
  - Test working memory activation boosts retention

GREEN:
  - Implement ForgettingMechanism
  - Add CLI command: `forget --session X --threshold 0.2`
  - Track access_count on search retrieval

REFACTOR:
  - Batch retention calculation for efficiency
  - Add visualization of retention scores
```

---

## Phase 3: Search Improvements

### 3.1 Query Decomposition

**Goal:** Parse complex queries into atomic sub-queries for better retrieval.

**Design:**
```python
class QueryDecomposer:
    """
    Decomposes complex queries into atomic sub-queries.

    Handles:
    - Multiple topics: "auth and database" → ["auth", "database"]
    - Relationships: "how X relates to Y" → search both, find connections
    - Temporal comparisons: "before vs after" → two temporal searches
    - Aggregations: "all decisions about X" → filtered search
    """

    def decompose(self, query: str) -> DecomposedQuery:
        """
        Parse query into components.

        Returns:
            DecomposedQuery(
                sub_queries=["auth implementation", "database schema"],
                relationship_type="relates_to",
                temporal_constraint=None,
                aggregation="all",
                intent_filter="decision"
            )
        """

    def execute_decomposed(
        self,
        decomposed: DecomposedQuery,
        limit: int = 10
    ) -> list[dict]:
        """
        Execute sub-queries and combine results.

        For relationship queries:
        1. Search for each sub-query
        2. Find ideas that connect the result sets
        3. Rank by connection strength
        """
```

**TDD Approach:**
```
RED:
  - Test "X and Y" decomposes to two queries
  - Test "how X relates to Y" finds connecting ideas
  - Test "decisions about X" applies intent filter
  - Test combined results are deduplicated
  - Test ranking reflects all sub-query relevance

GREEN:
  - Implement QueryDecomposer with pattern matching
  - Add LLM fallback for complex queries
  - Integrate with search CLI

REFACTOR:
  - Cache decomposition for repeated queries
  - Optimize multi-query execution (parallel)
```

---

### 3.2 Relevance Verification

**Goal:** Post-retrieval filtering to improve precision.

**Design:**
```python
class RelevanceVerifier:
    """
    Uses LLM to verify search result relevance.

    Applied after retrieval to filter out false positives
    that have high embedding similarity but low semantic relevance.
    """

    def verify_batch(
        self,
        query: str,
        results: list[dict],
        threshold: float = 3.0  # 1-5 scale
    ) -> list[dict]:
        """
        Score and filter results by relevance.

        Uses batch LLM call for efficiency:
        "Rate 1-5 how relevant each result is to the query."
        """

    def explain_relevance(
        self,
        query: str,
        result: dict
    ) -> str:
        """Generate explanation of why result is relevant."""
```

**TDD Approach:**
```
RED:
  - Test irrelevant results filtered out
  - Test relevant results retained
  - Test scores are sensible (decisions about X score high for "decisions about X")
  - Test batch processing is efficient
  - Test graceful fallback when LLM unavailable

GREEN:
  - Implement RelevanceVerifier
  - Add --verify flag to search commands
  - Add relevance scores to result output

REFACTOR:
  - Optimize batch size for LLM calls
  - Cache verification results for repeated searches
```

---

### 3.3 Multi-Hop Reasoning

**Goal:** Follow relation chains to answer complex queries.

**Design:**
```python
class ReasoningTracer:
    """
    Traces reasoning chains through the relation graph.

    Enables queries like:
    - "What led to decision X?" (trace back through builds_on)
    - "What changed after Y?" (trace forward through supersedes)
    - "How did we get from A to B?" (find path)
    """

    def trace_backward(
        self,
        idea_id: int,
        relation_types: list[str] = ["builds_on", "answers"],
        max_hops: int = 3
    ) -> list[ReasoningStep]:
        """Trace what led to this idea."""

    def trace_forward(
        self,
        idea_id: int,
        relation_types: list[str] = ["supersedes", "builds_on"],
        max_hops: int = 3
    ) -> list[ReasoningStep]:
        """Trace what followed from this idea."""

    def find_path(
        self,
        from_id: int,
        to_id: int,
        max_hops: int = 5
    ) -> list[ReasoningStep]:
        """Find connection path between two ideas."""
```

**TDD Approach:**
```
RED:
  - Test backward trace finds antecedents
  - Test forward trace finds consequences
  - Test path finding connects related ideas
  - Test max_hops limits traversal depth
  - Test cycles don't cause infinite loops

GREEN:
  - Implement graph traversal algorithms
  - Add CLI commands: `trace --backward/--forward <idea_id>`
  - Add path visualization

REFACTOR:
  - Optimize with BFS/DFS as appropriate
  - Add pruning for low-relevance branches
```

---

## Phase 4: Embedding Abstraction

### 4.1 Provider Abstraction

**Goal:** Support multiple embedding providers.

**Design:**
```python
class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider."""

class LocalEmbeddings(EmbeddingProvider):
    """Local sentence-transformers provider."""

class CohereEmbeddings(EmbeddingProvider):
    """Cohere embed-v3 provider."""
```

**TDD Approach:**
```
RED:
  - Test provider interface contract
  - Test OpenAI provider works as before
  - Test local provider returns correct dimensions
  - Test provider can be swapped via config

GREEN:
  - Implement abstract base class
  - Refactor OpenAI code to implement interface
  - Add local provider stub

REFACTOR:
  - Ensure cache works across providers
  - Handle dimension mismatches gracefully
```

---

## Phase 5: Reflection Mechanism

### 5.1 Periodic Reflection

**Goal:** Generate meta-insights from accumulated memories.

**Design:**
```python
class ReflectionEngine:
    """
    Periodically synthesizes higher-level insights.

    Inspired by Generative Agents (Park et al., 2023):
    "What did I learn? What patterns emerged?"
    """

    def reflect_on_session(
        self,
        session: str,
        time_window_days: int = 7
    ) -> list[Insight]:
        """
        Generate insights from recent session activity.

        Questions to answer:
        - What were the main themes this week?
        - What decisions were made and why?
        - What problems remain unresolved?
        - What patterns are emerging?
        """

    def reflect_on_topic(
        self,
        topic_id: int
    ) -> TopicInsight:
        """
        Generate insight about a specific topic's evolution.

        - How has understanding changed?
        - What were the key turning points?
        - What's the current state?
        """
```

**TDD Approach:**
```
RED:
  - Test reflection identifies main themes
  - Test reflection summarizes decisions
  - Test reflection finds unresolved questions
  - Test insights are stored for retrieval

GREEN:
  - Implement ReflectionEngine
  - Add CLI command: `reflect --session X --days 7`
  - Store insights as special idea type

REFACTOR:
  - Add scheduled reflection (weekly)
  - Add reflection quality scoring
```

---

## Implementation Order

```
Phase 1: Foundation (Week 1-2)
├── 1.1 Split memory_db.py
└── 1.2 Centralize configuration

Phase 2: Cognitive Core (Week 3-5)
├── 2.1 Working Memory
├── 2.2 Memory Consolidation
└── 2.3 Automatic Forgetting

Phase 3: Search Improvements (Week 6-7)
├── 3.1 Query Decomposition
├── 3.2 Relevance Verification
└── 3.3 Multi-Hop Reasoning

Phase 4: Flexibility (Week 8)
└── 4.1 Embedding Abstraction

Phase 5: Intelligence (Week 9-10)
└── 5.1 Reflection Mechanism
```

---

## Success Criteria

### Phase 1
- [ ] No file > 1000 lines
- [ ] All imports work from new paths
- [ ] All existing tests pass
- [ ] Config file supported

### Phase 2
- [ ] Working memory boosts recent context
- [ ] Old memories consolidated without data loss
- [ ] Forgetting reduces index size by 20%+
- [ ] Search quality maintained or improved

### Phase 3
- [ ] Complex queries return better results
- [ ] Precision improved by relevance verification
- [ ] Reasoning chains visualized

### Phase 4
- [ ] Local embedding model works
- [ ] Provider swappable via config

### Phase 5
- [ ] Weekly insights generated
- [ ] Insights improve retrieval context

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Comprehensive test coverage before refactoring |
| Performance regression | Benchmark before/after each phase |
| Data loss in consolidation | Soft delete, backup before major changes |
| LLM costs | Rate limiting, caching, local fallbacks |
| Complexity creep | Strict module boundaries, documentation |

---

## Review Checkpoints

After each phase:
1. Run full test suite (must pass 100%)
2. Run performance benchmarks
3. Manual testing of key workflows
4. Code review for maintainability
5. Documentation update
6. Version bump and release notes
