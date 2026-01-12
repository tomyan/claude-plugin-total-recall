# Memgraph Improvement Plan

Based on the expert review, this document outlines the implementation plan using **elephant carpaccio** - the thinnest possible vertical slices that each deliver value independently.

## Principles

1. **Each slice is deployable** - Can be committed and used immediately
2. **Each slice delivers value** - Even if small, it improves something
3. **Each slice takes hours, not days** - If it takes longer, slice thinner
4. **Tests pass after every slice** - Never break existing functionality

---

## Phase 1: Code Organization

### 1.1 Module Extraction (one at a time)

Each extraction is a separate commit:

- [x] **1.1a** Extract `errors.py` - MemgraphError class (~20 lines)
- [x] **1.1b** Extract `llm/claude.py` - claude_complete function (~80 lines)
- [x] **1.1c** Extract `config.py` - Just constants: DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIM
- [x] **1.1d** Extract `embeddings/cache.py` - Cache functions only (save/load/clear/stats)
- [x] **1.1e** Extract `embeddings/openai.py` - get_embedding, get_embeddings_batch
- [x] **1.1f** Extract `embeddings/serialize.py` - serialize/deserialize_embedding
- [x] **1.1g** Extract `db/connection.py` - get_db only (not init_db yet)
- [x] **1.1h** Extract `db/schema.py` - init_db and schema creation
- [x] **1.1i** Extract `db/migrations.py` - migrate_schema, migrate_spans_to_topics
- [x] **1.1j** Extract `search/vector.py` - search_ideas, find_similar_ideas
- [x] **1.1k** Extract `search/hybrid.py` - hybrid_search
- [x] **1.1l** Extract `search/hyde.py` - generate_hypothetical_doc, hyde_search

**Done when:** Each module works independently, memory_db.py imports and re-exports for backward compatibility, all 164 tests pass.

### 1.2 Configuration (incremental)

- [x] **1.2a** Create `config.py` with dataclass, hardcoded defaults only
- [x] **1.2b** Replace ONE hardcoded threshold (topic_shift_threshold) with config
- [x] **1.2c** Add config file loading (TOML)
- [x] **1.2d** Add env var override for one value (prove pattern)
- [x] **1.2e** Migrate remaining thresholds to config (batch)

**Done when:** `memgraph.toml` can override any threshold, env vars work.

---

## Phase 2: Cognitive Improvements

### 2.1 Access Tracking (foundation for working memory & forgetting)

- [x] **2.1a** Add `access_count` column to ideas table (migration)
- [x] **2.1b** Increment access_count when idea returned by search
- [x] **2.1c** Add `last_accessed` column to ideas table
- [x] **2.1d** Update last_accessed on search retrieval
- [x] **2.1e** Add `stats` output showing most/least accessed ideas

**Done when:** Every search updates access tracking, visible in stats.

### 2.2 Working Memory (thin slices)

- [x] **2.2a** Create `working_memory` table (session, idea_id, activation, last_access)
- [x] **2.2b** Record activation when idea is retrieved (activation = 1.0)
- [x] **2.2c** Add working memory functions (activate_idea, get_active_ideas, decay_working_memory, boost_results_by_activation)
- [x] **2.2d** Integrate activation recording into search functions
- [ ] **2.2e** Add `--boost-active` flag to search (re-rank by activation)
- [ ] **2.2f** Make boost the default, add `--no-boost` to disable

**Done when:** Search results are influenced by recent activity.

### 2.3 Soft Forgetting (thin slices)

- [x] **2.3a** Add `forgotten` BOOLEAN column to ideas (default FALSE)
- [x] **2.3b** Exclude forgotten=TRUE from search results
- [x] **2.3c** Add `forget <idea_id>` command (sets forgotten=TRUE)
- [x] **2.3d** Add `unforget <idea_id>` command (sets forgotten=FALSE)
- [x] **2.3e** Add `--include-forgotten` flag to search
- [x] **2.3f** Add `forgotten` command listing forgotten ideas

**Done when:** Users can manually forget/unforget ideas. ✓

### 2.4 Auto-Forget Candidates (thin slices)

- [x] **2.4a** Add `retention_score()` function (recency + access_count + importance)
- [x] **2.4b** Add `forgettable` command showing low-retention ideas (dry-run)
- [x] **2.4c** Add `--execute` flag to `forgettable` to actually forget
- [x] **2.4d** Add `--threshold` flag to control retention cutoff
- [x] **2.4e** Never auto-forget decisions/conclusions (importance override)

**Done when:** `forgettable --execute` safely removes low-value ideas. ✓

### 2.5 Consolidation (thin slices)

- [x] **2.5a** Add `consolidated_into` column to ideas table
- [x] **2.5b** Add `is_consolidated` BOOLEAN column (marks summary ideas)
- [x] **2.5c** Add `consolidatable` command showing candidate groups (dry-run)
- [x] **2.5d** Add `should_preserve()` - protect decisions, conclusions, high-confidence
- [x] **2.5e** Add `consolidate <topic_id>` command (creates summary, links originals)
- [x] **2.5f** Exclude consolidated originals from search (show summary instead)
- [x] **2.5g** Add `--show-originals` flag to see pre-consolidation ideas

**Done when:** Old context ideas can be consolidated into summaries. ✓

---

## Phase 3: Search Improvements

### 3.1 Intent Filtering (already partially exists)

- [x] **3.1a** Verify `--intent` flag works on all search commands
- [x] **3.1b** Add `decisions` shortcut command (`search --intent decision`)
- [x] **3.1c** Add `questions` shortcut showing unanswered questions
- [x] **3.1d** Add `todos` shortcut showing todo items

**Done when:** Quick access to filtered search by intent. ✓

### 3.2 Query Decomposition (thin slices)

- [x] **3.2a** Detect "X and Y" pattern, run two searches, merge results
- [x] **3.2b** Detect "decisions about X" pattern, apply intent filter
- [x] **3.2c** Detect "how X relates to Y" pattern, find connecting ideas
- [x] **3.2d** Add `--decompose` flag to show query interpretation

**Done when:** Complex queries automatically decomposed. ✓

### 3.3 Relevance Verification (thin slices)

- [x] **3.3a** Add `--verify` flag that uses LLM to score results 1-5
- [x] **3.3b** Filter out results scoring < 3
- [x] **3.3c** Show relevance score in output when `--verify` used
- [x] **3.3d** Add `--explain` flag for per-result relevance explanation

**Done when:** `search --verify` improves precision. ✓

### 3.4 Multi-Hop Reasoning (thin slices)

- [x] **3.4a** Add `trace <idea_id>` command showing related ideas (1-hop)
- [x] **3.4b** Add `--backward` flag (what led to this idea)
- [x] **3.4c** Add `--forward` flag (what followed from this idea)
- [x] **3.4d** Add `--hops N` flag for multi-hop traversal
- [x] **3.4e** Add `path <from_id> <to_id>` command finding connection

**Done when:** Can trace reasoning chains through relations. ✓

---

## Phase 4: Embedding Abstraction

### 4.1 Provider Interface (thin slices)

- [x] **4.1a** Create `EmbeddingProvider` abstract base class
- [x] **4.1b** Refactor OpenAI code to `OpenAIEmbeddings(EmbeddingProvider)`
- [x] **4.1c** Add provider selection to config (default: openai)
- [x] **4.1d** Add `LocalEmbeddings` stub (raises NotImplementedError)
- [ ] **4.1e** Implement `LocalEmbeddings` with sentence-transformers

**Done when:** Can switch embedding providers via config. (partial - local not yet implemented)

---

## Phase 5: Reflection

### 5.1 Session Reflection (thin slices)

- [x] **5.1a** Add `reflect` command generating session summary (LLM)
- [x] **5.1b** Store reflection as special idea (intent='reflection')
- [x] **5.1c** Add `--days N` flag to reflect on recent activity
- [x] **5.1d** Add `insights` command showing stored reflections
- [x] **5.1e** Include reflections in search results (searchable as ideas)

**Done when:** Can generate and retrieve session insights. ✓

### 5.2 Topic Reflection (thin slices)

- [x] **5.2a** Add `reflect-topic <topic_id>` command
- [x] **5.2b** Generate "how understanding evolved" summary
- [x] **5.2c** Identify key turning points in topic
- [ ] **5.2d** Store as topic-linked reflection (optional)

**Done when:** Can reflect on individual topic evolution. ✓

---

## Progress Tracking

### Completed
- [x] 1.1a - errors.py extracted
- [x] 1.1b - llm/claude.py extracted
- [x] 1.1c - config.py with constants
- [x] 1.1d - embeddings/cache.py extracted
- [x] 1.1e - embeddings/openai.py extracted
- [x] 1.1f - embeddings/serialize.py extracted
- [x] 1.1g - db/connection.py extracted
- [x] 1.1h - db/schema.py extracted
- [x] 1.1i - db/migrations.py extracted
- [x] 1.1j - search/vector.py extracted
- [x] 1.1k - search/hybrid.py extracted
- [x] 1.1l - search/hyde.py extracted
- [x] Remove LLM fallbacks (HyDE, suggest_topic_name, etc.)
- [x] 2.1a-e - Access tracking complete
- [x] 2.2a-d - Working memory foundation complete
- [x] 2.3a-f - Soft forgetting complete
- [x] 2.4a-e - Auto-forget candidates complete
- [x] 2.5a-g - Consolidation complete
- [x] 3.1a-d - Intent filtering shortcuts complete
- [x] 3.2a-d - Query decomposition complete
- [x] 3.3a-d - Relevance verification complete
- [x] 3.4a-e - Multi-hop reasoning complete
- [x] 4.1a-d - Embedding provider abstraction complete
- [x] 5.1a-e - Session reflection complete
- [x] 5.2a-c - Topic reflection complete

### Complete!

---

## Success Criteria

Each slice must:
1. Pass all 164 existing tests
2. Be independently useful (even if small)
3. Not require other slices to function
4. Have clear "done" definition

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes | Import and re-export for backward compatibility |
| Circular imports | Extract in dependency order (errors → config → db → search) |
| Lost functionality | Run full test suite after every slice |
| Scope creep | If slice takes > 2 hours, slice thinner |
