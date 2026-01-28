"""Agent-based daemon for continuous indexing - Slice 5.5.

Uses the indexing agent for batch processing of transcript files.
"""

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from db.async_connection import get_async_db
from indexer.batch_collector import collect_batch_updates
from indexer.run import run_indexing_agent
from utils.async_retry import retry_with_backoff

# Runtime directory
RUNTIME_DIR = config.DB_PATH.parent
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logger = logging.getLogger("daemon_agent")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BATCH_WINDOW = 2.0  # seconds to wait for more files
DEFAULT_POLL_INTERVAL = 2  # seconds between queue checks when idle
DEFAULT_IDLE_TIMEOUT = 300  # seconds of idle before exit (0 = never)
DEFAULT_MAX_FILES_PER_BATCH = 10  # max files to process in single agent call
DEFAULT_BACKFILL_TARGET_TOKENS = 30000  # target tokens per backfill batch
HEARTBEAT_INTERVAL = 60  # log heartbeat every N seconds


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
# Async Database Operations
# =============================================================================

async def db_get_queue_items(limit: int = 10) -> list[dict]:
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


async def db_get_queue_count() -> int:
    """Get number of items in queue."""
    async def do_query():
        db = await get_async_db()
        try:
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM work_queue")
            row = await cursor.fetchone()
            return row["cnt"]
        finally:
            await db.close()

    return await retry_with_backoff(do_query)


# =============================================================================
# Agent Daemon
# =============================================================================

class AgentDaemon:
    """Daemon using indexing agent for continuous processing."""

    def __init__(
        self,
        batch_window_seconds: float = DEFAULT_BATCH_WINDOW,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
        max_files_per_batch: int = DEFAULT_MAX_FILES_PER_BATCH,
        backfill_target_tokens: int = DEFAULT_BACKFILL_TARGET_TOKENS,
    ):
        self.batch_window_seconds = batch_window_seconds
        self.poll_interval = poll_interval
        self.idle_timeout = idle_timeout
        self.max_files_per_batch = max_files_per_batch
        self.backfill_target_tokens = backfill_target_tokens

        self.running = True
        self.ctx = ProcessingContext()
        self.last_activity = time.time()
        self.last_heartbeat = time.time()
        self.last_pidfile_check = 0
        self.my_pid = os.getpid()
        self.pidfile_path = (RUNTIME_DIR / "daemon.pid").resolve()

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
                f"{self.ctx.ideas_stored} ideas"
            )
            self.last_heartbeat = now

    async def _update_byte_position(self, file_path: str, byte_position: int) -> None:
        """Update the byte position for a file.

        Args:
            file_path: Path to transcript file
            byte_position: New byte position
        """
        async def do_update():
            db = await get_async_db()
            try:
                await db.execute("""
                    INSERT INTO index_state (file_path, byte_position)
                    VALUES (?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        byte_position = excluded.byte_position,
                        last_indexed = datetime('now')
                """, (file_path, byte_position))
                await db.commit()
            finally:
                await db.close()

        await retry_with_backoff(do_update)

    async def continuous_cycle(self) -> bool:
        """Run one continuous processing cycle.

        Returns:
            True if work was done, False if queue was empty
        """
        # Wait for batch window to allow files to accumulate
        if self.batch_window_seconds > 0:
            await asyncio.sleep(self.batch_window_seconds)

        # Get files from queue
        items = await db_get_queue_items(limit=self.max_files_per_batch)
        if not items:
            return False

        file_paths = [item["file_path"] for item in items]
        logger.info(f"Processing {len(file_paths)} files in batch")

        try:
            # Collect batch updates from all files
            updates = await collect_batch_updates(file_paths)

            if not updates:
                logger.info("No new content to process")
                return True

            # Run indexing agent on all updates at once
            stats = await run_indexing_agent(updates, mode="continuous")

            # Update context
            self.ctx.ideas_stored += stats.get("ideas_created", 0)
            self.ctx.files_processed += stats.get("sessions_processed", 0)

            logger.info(
                f"Batch complete: {stats.get('ideas_created', 0)} ideas, "
                f"{stats.get('sessions_processed', 0)} sessions"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            logger.error(traceback.format_exc())
            self.ctx.errors.append(str(e))
            return True  # Still return True to indicate we tried

    async def backfill_session(self, file_path: str) -> dict:
        """Backfill a single session from a transcript file.

        Processes the file in batches, respecting the token limit per batch.
        Uses mode='backfill' for the indexing agent.

        Args:
            file_path: Path to the transcript file

        Returns:
            Dict with processing stats including:
            - ideas_created: Number of ideas extracted
            - sessions_processed: Number of sessions processed (1 for single file)
            - bytes_processed: Total bytes processed
            - batches: Number of agent batches used
            - error: Error message if processing failed
        """
        from indexer.batch_collector import get_byte_position, BatchUpdate, Message
        import json

        if not Path(file_path).exists():
            return {"error": "file_not_found", "sessions_processed": 0}

        total_stats = {
            "ideas_created": 0,
            "sessions_processed": 0,
            "bytes_processed": 0,
            "batches": 0,
        }

        try:
            file_size = Path(file_path).stat().st_size
            session = Path(file_path).stem

            # Process file in chunks until complete
            while True:
                start_byte = get_byte_position(file_path)
                if start_byte >= file_size:
                    break

                # Collect messages up to token limit
                messages = []
                current_tokens = 0
                end_byte = start_byte

                with open(file_path, "rb") as f:
                    f.seek(start_byte)

                    # Count lines for proper line numbering
                    line_num = 0
                    if start_byte > 0:
                        f.seek(0)
                        for _ in range(start_byte):
                            if f.read(1) == b"\n":
                                line_num += 1
                        f.seek(start_byte)

                    for line in f:
                        line_num += 1
                        line_bytes = len(line)

                        try:
                            data = json.loads(line.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            end_byte += line_bytes
                            continue

                        msg_type = data.get("type")
                        if msg_type not in ("user", "assistant"):
                            end_byte += line_bytes
                            continue

                        message_data = data.get("message", {})
                        content = message_data.get("content", "")

                        # Handle content blocks
                        if isinstance(content, list):
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            content = "\n".join(text_parts)

                        if not content.strip():
                            end_byte += line_bytes
                            continue

                        # Estimate tokens (~4 chars per token)
                        msg_tokens = len(content) // 4

                        # Check if adding this message exceeds limit
                        if current_tokens + msg_tokens > self.backfill_target_tokens and messages:
                            # Don't add this message, process current batch
                            break

                        messages.append(Message(
                            role=msg_type,
                            content=content,
                            line_num=line_num,
                            timestamp=data.get("timestamp", ""),
                        ))
                        current_tokens += msg_tokens
                        end_byte += line_bytes

                if not messages:
                    # No more messages to process
                    break

                # Create batch update
                updates = [BatchUpdate(
                    session=session,
                    file_path=file_path,
                    messages=messages,
                    start_byte=start_byte,
                    end_byte=end_byte,
                )]

                # Run indexing agent with backfill mode
                stats = await run_indexing_agent(updates, mode="backfill")

                # Update byte position (ensure progress even if agent doesn't)
                await self._update_byte_position(file_path, end_byte)

                # Accumulate stats
                total_stats["ideas_created"] += stats.get("ideas_created", 0)
                total_stats["bytes_processed"] += end_byte - start_byte
                total_stats["batches"] += 1

            total_stats["sessions_processed"] = 1
            return total_stats

        except Exception as e:
            logger.error(f"Backfill error for {file_path}: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "ideas_created": total_stats["ideas_created"],
                "sessions_processed": 0,
                "bytes_processed": total_stats["bytes_processed"],
            }

    async def run(self):
        """Main async daemon loop."""
        logger.info("=" * 60)
        logger.info("Total Recall Agent Daemon starting")
        logger.info(f"  DB: {config.DB_PATH}")
        logger.info(f"  Batch window: {self.batch_window_seconds}s")
        logger.info(f"  Poll interval: {self.poll_interval}s")
        logger.info(f"  Idle timeout: {self.idle_timeout}s")
        logger.info("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Write pidfile
        self.pidfile_path.write_text(str(self.my_pid))
        logger.info(f"PID {self.my_pid} written to {self.pidfile_path}")

        try:
            while self.running:
                try:
                    # Check pidfile
                    if not self._check_pidfile():
                        break

                    # Process queue
                    had_work = await self.continuous_cycle()

                    if had_work:
                        self.last_activity = time.time()
                    else:
                        # Check idle timeout
                        if self.idle_timeout > 0:
                            idle_time = time.time() - self.last_activity
                            if idle_time > self.idle_timeout:
                                logger.info(f"Idle for {idle_time:.0f}s, exiting")
                                break

                        # Sleep before next poll
                        await asyncio.sleep(self.poll_interval)

                    # Heartbeat
                    self._log_heartbeat()

                except Exception as e:
                    logger.error(f"[MAIN LOOP] Unexpected error: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(self.poll_interval)

        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Clean shutdown with cache flush."""
        logger.info("Agent daemon shutting down")

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
        logger.info(
            f"Final stats: {self.ctx.files_processed} files, "
            f"{self.ctx.ideas_stored} ideas"
        )


def main():
    """Entry point."""
    # Check for API key FIRST
    from config import get_openai_api_key, OPENAI_KEY_FILE

    if not get_openai_api_key():
        logger.error("=" * 70)
        logger.error("ERROR: OpenAI API key required for daemon operation")
        logger.error("")
        logger.error("Please set your API key:")
        logger.error(f"  echo 'your-key' > {OPENAI_KEY_FILE}")
        logger.error("")
        logger.error("Or set OPENAI_API_KEY environment variable")
        logger.error("=" * 70)
        sys.exit(1)

    daemon = AgentDaemon()
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
