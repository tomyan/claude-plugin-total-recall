# Total Recall Improvement Plan

Generated: 2026-01-28

This plan addresses all issues identified in the project assessment, organized by priority and effort.

---

## Phase 1: Critical (Blocking Core Functionality)

### 1.1 Re-enable Continuous Indexing Hooks
**Priority:** P0 | **Effort:** 30 min | **Status:** Not Started

The hooks are currently disabled in SKILL.md, meaning users get no automatic indexing.

**Tasks:**
- [ ] Uncomment hooks in SKILL.md (lines 11-20)
- [ ] Verify hook scripts work with current daemon
- [ ] Test end-to-end: new conversation → hook triggers → daemon processes → searchable
- [ ] Add hook latency logging (target: <20ms)

**Files:**
- `skills/total-recall/SKILL.md`
- `skills/total-recall/hooks/index-continuous.sh`

---

---

## Phase 2: Short-term (This Week)

### 2.1 Async Naming Cleanup
**Priority:** P1 | **Effort:** 4-6 hours | **Status:** Not Started

34 functions still have `_async` suffix which is redundant in an async-first codebase.

**Tasks:**
- [ ] Identify all functions with `_async` suffix
- [ ] Rename to remove suffix (keep async behavior)
- [ ] Update all call sites
- [ ] Update tests
- [ ] Run full test suite

**Reference:** `PLAN-async-cleanup.md`

---

### 2.2 Complete Integration Tests
**Priority:** P1 | **Effort:** 6-8 hours | **Status:** Not Started

Several integration tests have TODO placeholders and aren't runnable.

**Tasks:**
- [ ] Complete `test_indexer_integration.py` TODOs
- [ ] Complete `test_integration_backfill.py` TODOs
- [ ] Add edge case tests for async retry logic
- [ ] Add timeout/streaming edge case tests
- [ ] Ensure all tests pass: `pytest skills/total-recall/tests/`

**Files:**
- `skills/total-recall/tests/test_indexer_integration.py`
- `skills/total-recall/tests/test_integration_backfill.py`

---

### 2.3 Expose Working Memory Boost
**Priority:** P1 | **Effort:** 2 hours | **Status:** Not Started

Infrastructure exists (`working_memory` table, activation tracking) but `--boost-active` flag not implemented.

**Tasks:**
- [ ] Add `--boost-active` flag to search commands
- [ ] Implement boost logic in search functions
- [ ] Add tests for activation boost
- [ ] Document in SKILL.md

**Files:**
- `skills/total-recall/src/cli.py`
- `skills/total-recall/src/search/vector.py`
- `skills/total-recall/src/search/hybrid.py`

---

### 2.4 Improve Daemon UX
**Priority:** P1 | **Effort:** 2 hours | **Status:** Partial

`daemon-status` exists but more commands would help.

**Tasks:**
- [ ] Add `daemon start` command (explicit start)
- [ ] Add `daemon stop` command (graceful shutdown)
- [ ] Add `daemon logs` command (tail recent logs)
- [ ] Add `daemon restart` command
- [ ] Document in SKILL.md

**Files:**
- `skills/total-recall/src/cli.py`
- `skills/total-recall/SKILL.md`

---

### 2.5 Distinguish Transient vs Permanent Errors
**Priority:** P1 | **Effort:** 3 hours | **Status:** Not Started

Current infinite retry doesn't distinguish error types, could mask permanent failures.

**Tasks:**
- [ ] Categorize error types (transient: timeout, rate limit; permanent: auth, invalid key)
- [ ] Retry only transient errors indefinitely
- [ ] Fail fast on permanent errors with clear message
- [ ] Add max_attempts parameter for specific contexts
- [ ] Log error category with each retry

**Files:**
- `skills/total-recall/src/llm/claude.py`
- `skills/total-recall/src/embeddings/openai.py`
- `skills/total-recall/src/errors.py`

---

## Phase 3: Medium-term (Next 2 Weeks)

### 3.1 Performance Tuning
**Priority:** P2 | **Effort:** 8 hours | **Status:** Not Started

Need benchmarks with real workloads to optimize settings.

**Tasks:**
- [ ] Create benchmark suite with varied transcript sizes
- [ ] Benchmark parallel worker counts (1, 2, 4, 8)
- [ ] Benchmark batch sizes (2K, 4K, 6K, 8K tokens)
- [ ] Benchmark timeout values
- [ ] Document optimal settings for different hardware
- [ ] Add configurable settings via config.toml

**Current settings:**
- `PARALLEL_WORKERS = 2`
- `FILE_TIMEOUT = 180`
- `target_tokens = 6000`

**Files:**
- `skills/total-recall/src/daemon.py`
- `skills/total-recall/src/batch_processor.py`
- `skills/total-recall/tests/test_performance.py`

---

### 3.2 Schema Migration Guide
**Priority:** P2 | **Effort:** 4 hours | **Status:** Not Started

No documentation for handling schema changes.

**Tasks:**
- [ ] Document current migration system (`src/db/migrations.py`)
- [ ] Create step-by-step guide for users
- [ ] Add version checking on startup
- [ ] Add `migrate` CLI command
- [ ] Test migration path from v1 to current

**Files:**
- `skills/total-recall/docs/MIGRATION.md` (new)
- `skills/total-recall/src/cli.py`
- `skills/total-recall/src/db/migrations.py`

---

### 3.3 Embedding Cache Pruning
**Priority:** P2 | **Effort:** 3 hours | **Status:** Not Started

50K entry cache could grow unbounded over time.

**Tasks:**
- [ ] Add LRU eviction when cache exceeds max size
- [ ] Add `cache prune` CLI command
- [ ] Add configurable max cache size
- [ ] Add age-based pruning option
- [ ] Log cache evictions

**Files:**
- `skills/total-recall/src/embeddings/cache.py`
- `skills/total-recall/src/cli.py`

---

### 3.4 Database Index Optimization
**Priority:** P2 | **Effort:** 2 hours | **Status:** Not Started

Some commonly-filtered columns may need indexes.

**Tasks:**
- [ ] Analyze slow queries with EXPLAIN
- [ ] Add indexes for: `ideas.intent`, `ideas.created_at`, `spans.session`
- [ ] Benchmark before/after
- [ ] Add to schema.py

**Files:**
- `skills/total-recall/src/db/schema.py`

---

### 3.5 Deployment Guide
**Priority:** P2 | **Effort:** 4 hours | **Status:** Not Started

No documentation for production setup.

**Tasks:**
- [ ] Document prerequisites (Python 3.11+, uv, sqlite-vec)
- [ ] Document file locations and permissions
- [ ] Document API key setup
- [ ] Document daemon management (systemd user service option)
- [ ] Add troubleshooting section
- [ ] Add upgrade instructions

**Files:**
- `skills/total-recall/docs/DEPLOYMENT.md` (new)

---

### 3.6 Improve Idea Deduplication Logging
**Priority:** P2 | **Effort:** 1 hour | **Status:** Not Started

Unique constraint on (source_file, source_line) silently skips duplicates.

**Tasks:**
- [ ] Log when ideas are deduplicated
- [ ] Add dedup count to processing stats
- [ ] Consider alternative dedup strategy (content hash?)

**Files:**
- `skills/total-recall/src/indexer/executor.py`

---

## Phase 4: Long-term (Nice to Have)

### 4.1 Alternative Embedding Providers
**Priority:** P3 | **Effort:** 8 hours | **Status:** Not Started

Currently only supports OpenAI text-embedding-3-small.

**Tasks:**
- [ ] Abstract embedding provider interface
- [ ] Add local embedding option (sentence-transformers)
- [ ] Add Claude Embeddings API option (when available)
- [ ] Add provider selection in config.toml
- [ ] Handle dimension differences between providers

**Files:**
- `skills/total-recall/src/embeddings/provider.py` (new)
- `skills/total-recall/src/embeddings/openai.py`
- `skills/total-recall/src/embeddings/local.py` (new)

---

### 4.2 Incremental Backfill Resume
**Priority:** P3 | **Effort:** 4 hours | **Status:** Not Started

Interrupted backfills must restart from beginning.

**Tasks:**
- [ ] Track backfill progress in database
- [ ] Add `--resume` flag to backfill command
- [ ] Skip already-processed files
- [ ] Show progress (X of Y files)

**Files:**
- `skills/total-recall/src/backfill.py`
- `skills/total-recall/src/cli.py`

---

### 4.3 Graph Visualization
**Priority:** P3 | **Effort:** 16 hours | **Status:** Not Started

Visual topic/idea browser would aid exploration.

**Tasks:**
- [ ] Add `visualize` CLI command that generates HTML
- [ ] Use D3.js or similar for graph rendering
- [ ] Show topic hierarchy
- [ ] Show idea relations
- [ ] Allow filtering by intent, time, session

**Files:**
- `skills/total-recall/src/visualize.py` (new)
- `skills/total-recall/src/cli.py`

---

### 4.4 Retention Scoring Validation
**Priority:** P3 | **Effort:** 8 hours | **Status:** Not Started

Current formula (recency × 0.3 + access × 0.3 + importance × 0.4) is untested.

**Tasks:**
- [ ] Collect usage data (access patterns, forgetting decisions)
- [ ] Analyze correlation between score and user actions
- [ ] A/B test alternative formulas
- [ ] Document findings and rationale

---

### 4.5 PID Management Improvements
**Priority:** P3 | **Effort:** 4 hours | **Status:** Not Started

Current PID file approach has race condition risks.

**Tasks:**
- [ ] Consider flock-based locking
- [ ] Add stale PID detection (process exists but wrong)
- [ ] Option for systemd user service
- [ ] Document recommended approach

**Files:**
- `skills/total-recall/src/daemon.py`
- `skills/total-recall/docs/DEPLOYMENT.md`

---

## Summary

| Phase | Items | Total Effort |
|-------|-------|--------------|
| Phase 1: Critical | 1 | ~30 min |
| Phase 2: Short-term | 5 | ~17 hours |
| Phase 3: Medium-term | 6 | ~22 hours |
| Phase 4: Long-term | 5 | ~40 hours |
| **Total** | **17** | **~80 hours** |

---

## Next Actions

1. **Immediately:** Complete Phase 1 items (re-enable hooks, commit staged files)
2. **This week:** Complete Phase 2 items (async cleanup, integration tests, daemon UX)
3. **Ongoing:** Work through Phase 3 as time permits
4. **Backlog:** Phase 4 items for future sprints

---

## Progress Tracking

Update this section as work progresses:

```
Phase 1: [ ] 1.1
Phase 2: [ ] 2.1  [ ] 2.2  [ ] 2.3  [ ] 2.4  [ ] 2.5
Phase 3: [ ] 3.1  [ ] 3.2  [ ] 3.3  [ ] 3.4  [ ] 3.5  [ ] 3.6
Phase 4: [ ] 4.1  [ ] 4.2  [ ] 4.3  [ ] 4.4  [ ] 4.5
```
