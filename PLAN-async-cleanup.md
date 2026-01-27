# Async Cleanup Plan

## Current State

- **261 tests passing**, 16 failing
- 34 functions with `_async` suffix that should be renamed
- 11 source files affected
- Failing tests are in sync test files that need async updates

## Goals

1. Remove `_async` suffix from all function names (everything is async now)
2. Update all callers
3. Fix all failing tests
4. Clean codebase with consistent naming

## Failing Tests (16 total)

| File | Failures | Root Cause |
|------|----------|------------|
| test_backfill.py | 6 | Patches `call_llm` (now `call_llm_async`) |
| test_batch_processor.py | 8 | Patches `call_llm` (now `call_llm_async`) |
| test_classification.py | 2 | Test expectation issues |

## Phase 1: Rename Functions (Remove _async suffix)

### Slice 1.1: embeddings/openai.py
- `get_embedding_async` → `get_embedding`
- `get_embeddings_batch_async` → `get_embeddings_batch`
- Update `embeddings/__init__.py` exports
- Update all callers

### Slice 1.2: search/vector.py
- `search_ideas_async` → `search_ideas`
- `find_similar_ideas_async` → `find_similar_ideas`
- `enrich_with_relations_async` → `enrich_with_relations`
- `search_spans_async` → `search_spans`
- `_update_access_tracking_async` → `_update_access_tracking`
- Update all callers

### Slice 1.3: search/hybrid.py
- `hybrid_search_async` → `hybrid_search`
- Update all callers

### Slice 1.4: search/hyde.py
- `generate_hypothetical_doc_async` → `generate_hypothetical_doc`
- `hyde_search_async` → `hyde_search`
- Update all callers

### Slice 1.5: llm/claude.py
- `claude_complete_async` → `claude_complete`
- Update all callers

### Slice 1.6: batch_processor.py
- `call_llm_async` → `call_llm`
- `execute_ideas_async` → `execute_ideas`
- `execute_topic_update_async` → `execute_topic_update`
- `execute_new_span_async` → `execute_new_span`
- `execute_relations_async` → `execute_relations`
- `embed_ideas_async` → `embed_ideas`
- `embed_messages_async` → `embed_messages`
- `process_transcript_async` → `process_transcript`
- Update all callers

### Slice 1.7: async_executor.py
- `store_idea_async` → `store_idea`
- `store_ideas_batch_async` → `store_ideas_batch`
- `create_span_async` → `create_span`
- `close_span_async` → `close_span`
- `update_span_embedding_async` → `update_span_embedding`
- `add_relation_async` → `add_relation`
- `get_open_span_async` → `get_open_span`
- `detect_semantic_topic_shift_async` → `detect_semantic_topic_shift`
- `flush_all_async` → `flush_all`
- **Decision**: Keep as separate module or merge into executor.py?

### Slice 1.8: daemon.py & cli_async.py
- `shutdown_async_modules` → `shutdown_modules`
- Any other _async functions
- Update callers

### Slice 1.9: db/async_connection.py
- Review and rename if needed
- Update callers

## Phase 2: Fix Failing Tests

### Slice 2.1: test_batch_processor.py (8 tests)
- Update to use `@pytest.mark.asyncio`
- Fix patches to use correct function names
- Use `AsyncMock` instead of `Mock` where needed

### Slice 2.2: test_backfill.py (6 tests)
- Update to use `@pytest.mark.asyncio`
- Fix patches to use correct function names
- Use `AsyncMock` instead of `Mock` where needed

### Slice 2.3: test_classification.py (2 tests)
- Review test expectations
- Fix filtering logic or update test cases

## Phase 3: Cleanup

### Slice 3.1: Remove dead code
- Check for any remaining sync wrappers
- Remove unused imports
- Clean up any backwards-compat code

### Slice 3.2: Final verification
- Run full test suite
- Verify all 277 tests pass
- Check for any import warnings

## Execution Order

1. Phase 1 slices in order (1.1 → 1.9) - each slice is independent
2. Phase 2 slices (2.1 → 2.3) - can be done in parallel
3. Phase 3 slices (3.1 → 3.2)

## Notes

- Each slice should be committed separately
- Run tests after each slice to catch regressions early
- Use `replace_all` in Edit tool for bulk renames
