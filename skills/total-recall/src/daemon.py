#!/usr/bin/env python3
"""
Total Recall Indexing Daemon (Async)

A background service that processes conversation transcripts with:
- Fully async database operations
- Parallel file processing via asyncio.gather
- Automatic retry with exponential backoff
- Clean shutdown with embedding cache flush
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from db.async_connection import get_async_db
from utils.async_retry import retry_with_backoff

# Runtime directory (derived from DB path)
RUNTIME_DIR = config.DB_PATH.parent
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
root_logger.addHandler(handler)

logger = logging.getLogger("daemon")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Optional OpenAI for LLM analysis
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

# =============================================================================
# Configuration
# =============================================================================

# Daemon settings
POLL_INTERVAL = 2  # seconds between queue checks
IDLE_TIMEOUT = 300  # seconds of idle before exit (0 = never)
PARALLEL_WORKERS = 8  # number of transcripts to process in parallel
HEARTBEAT_INTERVAL = 60  # log heartbeat every N seconds


# =============================================================================
# Async Database Operations
# =============================================================================

async def db_get_queue_items(limit: int = 4) -> list[dict]:
    """Get multiple items from work queue and remove them atomically."""
    async def do_query():
        db = await get_async_db()
        try:
            cursor = await db.execute("""
                SELECT id, file_path, file_size FROM work_queue
                ORDER BY queued_at ASC LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            items = [{"id": row["id"], "file_path": row["file_path"], "file_size": row["file_size"]} for row in rows]

            # Remove them immediately (claim)
            if items:
                ids = [item["id"] for item in items]
                placeholders = ",".join("?" * len(ids))
                await db.execute(f"DELETE FROM work_queue WHERE id IN ({placeholders})", ids)
                await db.commit()

            return items
        finally:
            await db.close()

    return await retry_with_backoff(do_query)


# =============================================================================
# Processing Context
# =============================================================================

@dataclass
class ProcessingContext:
    """Tracks processing state for the daemon."""
    messages_processed: int = 0
    ideas_stored: int = 0
    files_processed: int = 0
    errors: list = field(default_factory=list)


# =============================================================================
# Async File Processing
# =============================================================================

async def process_single_file(file_path: str) -> dict:
    """Process a single transcript file asynchronously."""
    from batch_processor import process_transcript_async, ProcessingError
    from backfill import session_from_path

    session = session_from_path(file_path)

    try:
        logger.info(f"Processing: {file_path}")
        result = await process_transcript_async(file_path, session=session)
        logger.info(f"  Result: batches={result.get('batches_processed', 0)}, ideas={result.get('ideas_stored', 0)}")
        return {
            "file_path": file_path,
            "batches_processed": result.get('batches_processed', 0),
            "ideas_stored": result.get('ideas_stored', 0),
            "error": None
        }
    except ProcessingError as e:
        logger.error(f"  Failed to process {file_path}: {e}")
        return {"file_path": file_path, "batches_processed": 0, "ideas_stored": 0, "error": str(e)}
    except Exception as e:
        logger.error(f"  Unexpected error processing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return {"file_path": file_path, "batches_processed": 0, "ideas_stored": 0, "error": str(e)}


async def process_queue_batch(ctx: ProcessingContext, max_workers: int = PARALLEL_WORKERS) -> bool:
    """Process multiple queue items in parallel using asyncio.

    Returns:
        True if any items were processed, False if queue empty
    """
    items = await db_get_queue_items(limit=max_workers)
    if not items:
        return False

    file_paths = [item["file_path"] for item in items]
    logger.info(f"Processing {len(file_paths)} files in parallel")

    # Use semaphore to limit concurrency
    sem = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(fp: str) -> dict:
        async with sem:
            return await process_single_file(fp)

    # Process all files concurrently
    results = await asyncio.gather(*[process_with_semaphore(fp) for fp in file_paths])

    for result in results:
        ctx.messages_processed += result["batches_processed"]
        ctx.ideas_stored += result["ideas_stored"]
        ctx.files_processed += 1
        if result["error"]:
            ctx.errors.append(result["error"])

    return True


# =============================================================================
# Pidfile Watcher Thread
# =============================================================================

class PidfileWatcher(threading.Thread):
    """Preemptive thread that monitors pidfile and exits if missing.

    This runs independently of the async event loop, ensuring the daemon
    exits even if the main loop is blocked in a long-running operation.
    """

    def __init__(self, pidfile_path: Path, my_pid: int, check_interval: float = 1.0):
        super().__init__(daemon=True, name="pidfile-watcher")
        self.pidfile_path = pidfile_path
        self.my_pid = my_pid
        self.check_interval = check_interval
        self.stop_event = threading.Event()

    def run(self):
        """Check pidfile periodically, exit process if invalid."""
        logger.info(f"Pidfile watcher started (checking every {self.check_interval}s)")

        while not self.stop_event.wait(self.check_interval):
            if not self._check_pidfile():
                logger.info("Pidfile watcher: forcing exit")
                os._exit(0)  # Force immediate exit

    def _check_pidfile(self) -> bool:
        """Check if pidfile exists and contains our PID."""
        if not self.pidfile_path.exists():
            logger.info(f"Pidfile {self.pidfile_path} no longer exists")
            return False

        try:
            stored_pid = int(self.pidfile_path.read_text().strip())
            if stored_pid != self.my_pid:
                logger.info(f"Pidfile contains PID {stored_pid}, but we are {self.my_pid}")
                return False
        except (ValueError, OSError) as e:
            logger.info(f"Cannot read pidfile ({e})")
            return False

        return True

    def stop(self):
        """Signal the watcher to stop."""
        self.stop_event.set()


# =============================================================================
# Async Daemon Main Loop
# =============================================================================

class AsyncIndexingDaemon:
    """Main async daemon class."""

    def __init__(self):
        self.running = True
        self.ctx = ProcessingContext()
        self.last_activity = time.time()
        self.last_heartbeat = time.time()
        self.last_pidfile_check = 0
        self.my_pid = os.getpid()
        self.pidfile_path = (RUNTIME_DIR / "daemon.pid").resolve()
        self.pidfile_watcher = None

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _check_pidfile(self) -> bool:
        """Check if pidfile exists and contains our PID."""
        now = time.time()
        if now - self.last_pidfile_check < 1.0:
            return True

        self.last_pidfile_check = now

        if not self.pidfile_path.exists():
            logger.info(f"Pidfile {self.pidfile_path} no longer exists, exiting")
            return False

        try:
            stored_pid = int(self.pidfile_path.read_text().strip())
            if stored_pid != self.my_pid:
                logger.info(f"Pidfile contains PID {stored_pid}, but we are {self.my_pid}, exiting")
                return False
        except (ValueError, OSError) as e:
            logger.info(f"Cannot read pidfile ({e}), exiting")
            return False

        return True

    def _log_heartbeat(self):
        """Log periodic heartbeat."""
        now = time.time()
        if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
            logger.info(
                f"[HEARTBEAT] Alive - processed {self.ctx.files_processed} files, "
                f"{self.ctx.messages_processed} batches, {self.ctx.ideas_stored} ideas"
            )
            self.last_heartbeat = now

    async def run(self):
        """Main async daemon loop."""
        logger.info("=" * 60)
        logger.info("Total Recall Async Daemon starting")
        logger.info(f"  DB: {config.DB_PATH}")
        logger.info(f"  Poll interval: {POLL_INTERVAL}s")
        logger.info(f"  Idle timeout: {IDLE_TIMEOUT}s")
        logger.info(f"  Parallel workers: {PARALLEL_WORKERS}")
        logger.info(f"  OpenAI available: {HAS_OPENAI}")
        logger.info("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Write pidfile
        self.pidfile_path.write_text(str(self.my_pid))
        logger.info(f"PID {self.my_pid} written to {self.pidfile_path}")

        # Start preemptive pidfile watcher thread
        self.pidfile_watcher = PidfileWatcher(self.pidfile_path, self.my_pid)
        self.pidfile_watcher.start()

        try:
            while self.running:
                try:
                    # Check pidfile
                    if not self._check_pidfile():
                        break

                    # Process queue items in parallel
                    had_work = await process_queue_batch(self.ctx)

                    if had_work:
                        self.last_activity = time.time()
                    else:
                        # Check idle timeout
                        if IDLE_TIMEOUT > 0:
                            idle_time = time.time() - self.last_activity
                            if idle_time > IDLE_TIMEOUT:
                                logger.info(f"Idle for {idle_time:.0f}s, exiting")
                                break

                        # Sleep before next poll (async)
                        await asyncio.sleep(POLL_INTERVAL)

                    # Heartbeat
                    self._log_heartbeat()

                except Exception as e:
                    logger.error(f"[MAIN LOOP] Unexpected error: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(POLL_INTERVAL)

        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Clean shutdown with cache flush."""
        logger.info("Daemon shutting down")

        # Stop pidfile watcher
        if self.pidfile_watcher:
            self.pidfile_watcher.stop()

        # Flush embedding cache
        try:
            from embeddings.cache import flush_write_queue, get_embedding_cache_stats, shutdown
            stats = await get_embedding_cache_stats()
            if stats["size"] > 0:
                await flush_write_queue()
                logger.info(f"Flushed embedding cache ({stats['size']} entries)")
            await shutdown()
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

        # Remove pidfile
        self.pidfile_path.unlink(missing_ok=True)
        logger.info(f"Final stats: {self.ctx.files_processed} files, "
                   f"{self.ctx.messages_processed} batches, {self.ctx.ideas_stored} ideas")


def main():
    """Entry point."""
    # Check for API key FIRST
    from config import get_openai_api_key, OPENAI_KEY_FILE

    if not get_openai_api_key():
        logger.error("=" * 70)
        logger.error("ERROR: OpenAI API key required for daemon operation")
        logger.error("")
        logger.error("The daemon needs an OpenAI API key to:")
        logger.error("  1. Generate embeddings for semantic search")
        logger.error("  2. Analyze conversations to extract ideas")
        logger.error("")
        logger.error("Please set your API key:")
        logger.error(f"  echo 'your-key' > {OPENAI_KEY_FILE}")
        logger.error("")
        logger.error("Or set OPENAI_API_KEY environment variable")
        logger.error("=" * 70)
        sys.exit(1)

    daemon = AsyncIndexingDaemon()
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
