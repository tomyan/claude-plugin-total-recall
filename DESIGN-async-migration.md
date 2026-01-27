# Design: Async Migration Plan

## Goal
Migrate total-recall to fully async architecture for:
- Better concurrent database access
- Queued writes with backoff
- Non-blocking operations throughout

## Current State
- Synchronous codebase using `threading.Thread` for parallelism
- SQLite with WAL mode, but still hitting lock contention
- Cache module has thread-based write queue (just implemented)

## Target Architecture
- Async/await throughout
- `asyncio.Queue` for write operations
- Connection pooling with async SQLite (aiosqlite)
- Graceful degradation with retries

## Migration Slices (TDD approach)

### Slice 1: Async Database Connection Layer
**Goal:** Create async database connection utilities alongside sync ones

**Files:**
- `db/async_connection.py` (new)

**Tests first:**
```python
# test_async_connection.py
async def test_get_async_db_returns_connection():
    db = await get_async_db()
    assert db is not None
    await db.close()

async def test_async_db_can_execute_query():
    db = await get_async_db()
    cursor = await db.execute("SELECT 1 as value")
    row = await cursor.fetchone()
    assert row['value'] == 1
    await db.close()

async def test_async_db_has_wal_mode():
    db = await get_async_db()
    cursor = await db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0].upper() == 'WAL'
    await db.close()
```

**Implementation:**
```python
# db/async_connection.py
import aiosqlite
import config

async def get_async_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(config.DB_PATH), timeout=30.0)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=30000")
    db.row_factory = aiosqlite.Row
    return db
```

**Dependencies:** Add `aiosqlite` to pyproject.toml

---

### Slice 2: Async Retry Utility
**Goal:** Create async retry with exponential backoff

**Files:**
- `utils/async_retry.py` (new)

**Tests first:**
```python
# test_async_retry.py
async def test_retry_succeeds_first_try():
    call_count = 0
    async def succeed():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await retry_with_backoff(succeed)
    assert result == "success"
    assert call_count == 1

async def test_retry_succeeds_after_failures():
    call_count = 0
    async def fail_twice_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    result = await retry_with_backoff(fail_twice_then_succeed)
    assert result == "success"
    assert call_count == 3

async def test_retry_respects_max_delay():
    # Verify backoff caps at max_delay
    ...
```

**Implementation:**
```python
# utils/async_retry.py
import asyncio
import random
import sqlite3

async def retry_with_backoff(
    func,
    *args,
    max_delay: float = 5.0,
    **kwargs
):
    delay = 0.1
    while True:
        try:
            return await func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                jittered = delay * (0.5 + random.random())
                await asyncio.sleep(jittered)
                delay = min(delay * 2, max_delay)
            else:
                raise
```

---

### Slice 3: Async Write Queue
**Goal:** Create async queue processor for database writes

**Files:**
- `utils/write_queue.py` (new)

**Tests first:**
```python
# test_write_queue.py
async def test_queue_processes_items():
    processed = []
    async def processor(item):
        processed.append(item)

    queue = WriteQueue(processor)
    await queue.start()
    await queue.put("item1")
    await queue.put("item2")
    await queue.flush()
    await queue.stop()

    assert processed == ["item1", "item2"]

async def test_queue_retries_on_failure():
    attempts = []
    async def failing_processor(item):
        attempts.append(item)
        if len(attempts) < 3:
            raise sqlite3.OperationalError("locked")

    queue = WriteQueue(failing_processor)
    await queue.start()
    await queue.put("item")
    await queue.flush()
    await queue.stop()

    assert len(attempts) == 3

async def test_queue_drains_on_shutdown():
    ...
```

**Implementation:**
```python
# utils/write_queue.py
import asyncio
from typing import Callable, Any

class WriteQueue:
    def __init__(self, processor: Callable):
        self._queue = asyncio.Queue()
        self._processor = processor
        self._task = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._worker())

    async def stop(self):
        self._running = False
        await self._queue.put(None)  # Sentinel
        if self._task:
            await self._task

    async def put(self, item):
        await self._queue.put(item)

    async def flush(self):
        await self._queue.join()

    async def _worker(self):
        while self._running:
            item = await self._queue.get()
            if item is None:
                break
            try:
                await retry_with_backoff(self._processor, item)
            finally:
                self._queue.task_done()
```

---

### Slice 4: Async Cache Module
**Goal:** Rewrite cache.py with async/await

**Files:**
- `embeddings/async_cache.py` (new, then replace cache.py)

**Tests first:**
```python
# test_async_cache.py
async def test_cache_miss_returns_none():
    result = await get_cached_embedding("nonexistent")
    assert result is None

async def test_cache_hit_returns_embedding():
    embedding = [0.1] * 1536
    await cache_embedding("test text", embedding)
    await flush_writes()

    result = await get_cached_embedding("test text")
    assert result == embedding

async def test_cache_tracks_hit_count():
    await cache_embedding("test", [0.1] * 1536)
    await flush_writes()

    await get_cached_embedding("test")
    await get_cached_embedding("test")

    stats = await get_cache_stats()
    assert stats["total_hits"] == 2

async def test_cache_evicts_when_full():
    # Set small max size for test
    ...

async def test_cache_source_context():
    with cache_source("search"):
        await cache_embedding("query", [0.1] * 1536)

    stats = await get_cache_stats()
    assert stats["entries_by_source"]["search"]["count"] == 1
```

---

### Slice 5: Async Search Functions
**Goal:** Make search functions async

**Files:**
- `search/async_vector.py` (new)
- `search/async_hybrid.py` (new)
- `search/async_hyde.py` (new)

**Tests first:**
```python
# test_async_search.py
async def test_hybrid_search_returns_results():
    # Setup: insert test ideas with embeddings
    results = await hybrid_search("test query", limit=5)
    assert isinstance(results, list)

async def test_hybrid_search_uses_cache():
    # First search populates cache
    await hybrid_search("test query")
    stats_before = await get_cache_stats()

    # Second search should hit cache
    await hybrid_search("test query")
    stats_after = await get_cache_stats()

    assert stats_after["total_hits"] > stats_before["total_hits"]
```

---

### Slice 6: Async CLI Adapter
**Goal:** Bridge async functions to sync CLI

**Files:**
- `cli_async.py` (new adapter)

**Pattern:**
```python
def run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)

# In CLI:
elif args.command == "hybrid":
    results = run_async(async_hybrid_search(args.query, ...))
```

---

### Slice 7: Async Executor/Indexing
**Goal:** Make indexing pipeline async

**Files:**
- `async_executor.py` (new)

---

### Slice 8: Integration & Cutover
**Goal:** Replace sync modules with async versions

**Steps:**
1. Run both implementations in parallel, compare results
2. Feature flag to switch between sync/async
3. Validate in production with flag
4. Remove sync implementations

---

## Dependency Order

```
Slice 1 (async_connection)
    ↓
Slice 2 (async_retry)
    ↓
Slice 3 (write_queue)
    ↓
Slice 4 (async_cache) ←── depends on 1,2,3
    ↓
Slice 5 (async_search) ←── depends on 4
    ↓
Slice 6 (cli_adapter)
    ↓
Slice 7 (async_executor)
    ↓
Slice 8 (cutover)
```

## Testing Strategy

1. **Unit tests for each slice** - Run with `pytest-asyncio`
2. **Integration tests** - Test async modules together
3. **Comparison tests** - Same inputs, verify sync and async produce same outputs
4. **Load tests** - Verify concurrency improvements

## Dependencies to Add

```toml
[project.dependencies]
aiosqlite = ">=0.19.0"

[project.optional-dependencies]
dev = [
    "pytest-asyncio>=0.21.0",
]
```

## Rollback Plan

Each slice is additive (new files) until Slice 8. Can abort at any point before cutover with no impact to existing functionality.
