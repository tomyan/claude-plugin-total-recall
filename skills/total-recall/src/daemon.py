#!/usr/bin/env python3
"""
Total Recall Indexing Daemon

A background service that processes conversation transcripts with robust error handling:
- Timeouts on all operations
- Exponential backoff with retry on failures
- Clear logging of all errors
- Handles sleep/wake and network issues gracefully
"""

import functools
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from db.connection import get_db

# Runtime directory (derived from DB path)
RUNTIME_DIR = config.DB_PATH.parent
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging - force reconfigure since config.py may have set it up already
# Remove any existing handlers and set up fresh
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

# Retry settings
MAX_BACKOFF_SECONDS = 10  # Cap exponential backoff at 10s
INITIAL_BACKOFF_SECONDS = 1
MAX_RETRIES = None  # None = infinite retries

# Timeouts (seconds)
TIMEOUT_DB_QUERY = 30
TIMEOUT_FILE_READ = 60
TIMEOUT_LLM_CALL = 120

# Daemon settings
POLL_INTERVAL = 2  # seconds between queue checks
IDLE_TIMEOUT = 300  # seconds of idle before exit (0 = never)
PARALLEL_WORKERS = 8  # number of transcripts to process in parallel
HEARTBEAT_INTERVAL = 60  # log heartbeat every N seconds

# Processing settings
BATCH_TIME_WINDOW = 30  # batch messages within this window
CONTEXT_WINDOW_SIZE = 5  # recent messages for context

# Valid intents
VALID_INTENTS = {"decision", "conclusion", "question", "problem", "solution", "todo", "context"}

T = TypeVar('T')


# =============================================================================
# Retry and Timeout Infrastructure
# =============================================================================

class OperationTimeout(Exception):
    """Raised when an operation times out."""
    pass


class OperationFailed(Exception):
    """Raised when an operation fails after retries."""
    pass


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to a function using SIGALRM."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            def handler(signum, frame):
                raise OperationTimeout(f"{func.__name__} timed out after {timeout_seconds}s")

            # Set alarm (only works on Unix)
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(timeout_seconds))
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def retry_with_backoff(
    operation_name: str,
    max_retries: Optional[int] = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    max_backoff: float = MAX_BACKOFF_SECONDS,
):
    """Decorator for retry with exponential backoff.

    Args:
        operation_name: Human-readable name for logging
        max_retries: Max attempts (None = infinite)
        initial_backoff: Starting backoff in seconds
        max_backoff: Maximum backoff cap in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            backoff = initial_backoff

            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log the failure
                    logger.error(
                        f"[{operation_name}] Attempt {attempt} failed: {type(e).__name__}: {e}"
                    )

                    # Check if we've exhausted retries
                    if max_retries is not None and attempt >= max_retries:
                        logger.error(f"[{operation_name}] Exhausted {max_retries} retries, giving up")
                        raise OperationFailed(f"{operation_name} failed after {max_retries} attempts") from e

                    # Log backoff
                    logger.info(f"[{operation_name}] Backing off {backoff:.1f}s before retry...")
                    time.sleep(backoff)

                    # Exponential backoff with cap
                    backoff = min(backoff * 2, max_backoff)
        return wrapper
    return decorator


# =============================================================================
# Database Operations (with timeout and retry)
# =============================================================================

@retry_with_backoff("db_get_queue_item")
def db_get_queue_item() -> Optional[dict]:
    """Get next item from work queue."""
    items = db_get_queue_items(limit=1)
    return items[0] if items else None


def db_get_queue_items(limit: int = 4) -> list[dict]:
    """Get multiple items from work queue and remove them atomically.

    Items are removed immediately to allow parallel processing.
    """
    @with_timeout(TIMEOUT_DB_QUERY)
    def _query():
        db = get_db()
        try:
            # Get items
            cursor = db.execute("""
                SELECT id, file_path, file_size FROM work_queue
                ORDER BY queued_at ASC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            items = [{"id": row["id"], "file_path": row["file_path"], "file_size": row["file_size"]} for row in rows]

            # Remove them immediately (claim)
            if items:
                ids = [item["id"] for item in items]
                placeholders = ",".join("?" * len(ids))
                db.execute(f"DELETE FROM work_queue WHERE id IN ({placeholders})", ids)
                db.commit()

            return items
        finally:
            db.close()
    return _query()


@retry_with_backoff("db_remove_queue_item")
def db_remove_queue_item(item_id: int):
    """Remove item from work queue."""
    @with_timeout(TIMEOUT_DB_QUERY)
    def _query():
        db = get_db()
        try:
            db.execute("DELETE FROM work_queue WHERE id = ?", (item_id,))
            db.commit()
        finally:
            db.close()
    _query()


@retry_with_backoff("db_get_byte_position")
def db_get_byte_position(file_path: str) -> int:
    """Get last indexed byte position for a file."""
    @with_timeout(TIMEOUT_DB_QUERY)
    def _query():
        db = get_db()
        try:
            cursor = db.execute(
                "SELECT byte_position FROM index_state WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return row["byte_position"] if row else 0
        finally:
            db.close()
    return _query()


@retry_with_backoff("db_update_byte_position")
def db_update_byte_position(file_path: str, byte_position: int):
    """Update byte position for a file."""
    @with_timeout(TIMEOUT_DB_QUERY)
    def _query():
        db = get_db()
        try:
            db.execute("""
                INSERT INTO index_state (file_path, byte_position, last_indexed)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    byte_position = excluded.byte_position,
                    last_indexed = excluded.last_indexed
            """, (file_path, byte_position))
            db.commit()
        finally:
            db.close()
    _query()


@retry_with_backoff("db_store_idea")
def db_store_idea(content: str, source_file: str, source_line: int,
                  intent: Optional[str], confidence: float, message_time: Optional[str]):
    """Store an idea in the database."""
    @with_timeout(TIMEOUT_DB_QUERY)
    def _query():
        db = get_db()
        try:
            db.execute("""
                INSERT OR IGNORE INTO ideas
                (content, source_file, source_line, intent, confidence, message_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (content, source_file, source_line, intent, confidence, message_time))
            db.commit()
        finally:
            db.close()
    _query()


# =============================================================================
# File Operations (with timeout and retry)
# =============================================================================

@retry_with_backoff("file_read_from_position")
def file_read_from_position(file_path: str, start_byte: int) -> tuple[list[dict], int]:
    """Read new lines from file starting at byte position.

    Returns:
        Tuple of (list of parsed messages, new byte position)
    """
    @with_timeout(TIMEOUT_FILE_READ)
    def _read():
        messages = []

        with open(file_path, 'r') as f:
            f.seek(start_byte)
            line_num = 0

            # Count lines up to start position for accurate line numbers
            if start_byte > 0:
                f.seek(0)
                for _ in range(start_byte):
                    try:
                        char = f.read(1)
                        if char == '\n':
                            line_num += 1
                        if f.tell() >= start_byte:
                            break
                    except:
                        break
                f.seek(start_byte)

            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    msg_type = data.get("type")
                    if msg_type not in ("user", "assistant"):
                        continue

                    message = data.get("message", {})
                    content = _extract_content(message)

                    if content and len(content) >= 20:
                        messages.append({
                            "line_num": line_num,
                            "type": msg_type,
                            "content": content,
                            "timestamp": data.get("timestamp", ""),
                        })
                except json.JSONDecodeError:
                    continue

            new_position = f.tell()

        return messages, new_position

    return _read()


def _extract_content(message: dict) -> str:
    """Extract text content from message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)
    return ""


# =============================================================================
# LLM Operations (with timeout and retry)
# =============================================================================

_logged_heuristic_fallback = False

def llm_classify_message(content: str, context: list[str]) -> dict:
    """Classify a message using LLM with fallback to heuristics.

    Returns:
        Dict with 'indexable', 'intent', 'confidence'
    """
    global _logged_heuristic_fallback

    if not HAS_OPENAI:
        if not _logged_heuristic_fallback:
            logger.info("Using heuristic classification (OpenAI not installed)")
            _logged_heuristic_fallback = True
        return _heuristic_classify(content)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        if not _logged_heuristic_fallback:
            logger.info("Using heuristic classification (OPENAI_API_KEY not set)")
            _logged_heuristic_fallback = True
        return _heuristic_classify(content)

    return _llm_classify_with_retry(content, context)


@retry_with_backoff("llm_classify_message", max_retries=5)
def _llm_classify_with_retry(content: str, context: list[str]) -> dict:
    """Internal LLM classification with retry."""

    @with_timeout(TIMEOUT_LLM_CALL)
    def _call():
        client = OpenAI()

        context_str = "\n".join(f"- {c[:200]}" for c in context[-3:]) if context else "(none)"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """Analyze this message from a coding conversation.
Return JSON: {"indexable": bool, "intent": string, "confidence": float}

indexable: false if greeting, acknowledgment, filler, or tool preamble
intent: one of "decision", "conclusion", "question", "problem", "solution", "todo", "context"
confidence: 0.0-1.0"""
            }, {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nMessage:\n{content[:1000]}"
            }],
            response_format={"type": "json_object"},
            timeout=TIMEOUT_LLM_CALL
        )

        result = json.loads(response.choices[0].message.content)

        # Validate
        if result.get("intent") not in VALID_INTENTS:
            result["intent"] = "context"
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        result["indexable"] = bool(result.get("indexable", True))

        return result

    return _call()


def _heuristic_classify(content: str) -> dict:
    """Simple heuristic classification when LLM unavailable."""
    content_lower = content.lower().strip()

    # Filter obvious non-indexable
    if len(content) < 30:
        return {"indexable": False, "intent": "context", "confidence": 0.3}

    if content_lower.startswith(("let me", "i'll ", "now let")):
        return {"indexable": False, "intent": "context", "confidence": 0.4}

    # Simple intent detection
    if "?" in content:
        return {"indexable": True, "intent": "question", "confidence": 0.6}
    if any(w in content_lower for w in ["decided", "decision", "chose", "will use"]):
        return {"indexable": True, "intent": "decision", "confidence": 0.6}
    if any(w in content_lower for w in ["todo", "need to", "should "]):
        return {"indexable": True, "intent": "todo", "confidence": 0.5}
    if any(w in content_lower for w in ["problem", "issue", "error", "bug"]):
        return {"indexable": True, "intent": "problem", "confidence": 0.5}
    if any(w in content_lower for w in ["solution", "fix", "solved"]):
        return {"indexable": True, "intent": "solution", "confidence": 0.5}

    return {"indexable": True, "intent": "context", "confidence": 0.5}


# =============================================================================
# Main Processing Logic
# =============================================================================

@dataclass
class ProcessingContext:
    """Maintains state during processing."""
    recent_messages: list = field(default_factory=list)
    messages_processed: int = 0
    ideas_stored: int = 0


def process_file(file_path: str, ctx: ProcessingContext) -> dict:
    """Process a single transcript file.

    Returns:
        Dict with processing stats
    """
    logger.info(f"Processing: {file_path}")

    # Get current position
    start_byte = db_get_byte_position(file_path)
    logger.info(f"  Starting from byte {start_byte}")

    # Check file exists and get size
    if not os.path.exists(file_path):
        logger.warning(f"  File not found, skipping: {file_path}")
        return {"status": "skipped", "reason": "file_not_found"}

    file_size = os.path.getsize(file_path)
    if start_byte >= file_size:
        logger.info(f"  Already fully indexed ({file_size} bytes)")
        return {"status": "already_indexed"}

    # Read new messages
    messages, new_position = file_read_from_position(file_path, start_byte)
    logger.info(f"  Read {len(messages)} new messages, now at byte {new_position}")

    if not messages:
        db_update_byte_position(file_path, new_position)
        return {"status": "no_new_messages"}

    # Process each message
    ideas_stored = 0
    for msg in messages:
        ctx.messages_processed += 1

        # Classify
        classification = llm_classify_message(msg["content"], ctx.recent_messages)

        # Update context window
        ctx.recent_messages.append(msg["content"][:500])
        if len(ctx.recent_messages) > CONTEXT_WINDOW_SIZE:
            ctx.recent_messages.pop(0)

        # Skip non-indexable
        if not classification["indexable"]:
            continue

        # Store idea
        db_store_idea(
            content=msg["content"],
            source_file=file_path,
            source_line=msg["line_num"],
            intent=classification["intent"],
            confidence=classification["confidence"],
            message_time=msg.get("timestamp")
        )
        ideas_stored += 1
        ctx.ideas_stored += 1

    # Update position
    db_update_byte_position(file_path, new_position)

    logger.info(f"  Stored {ideas_stored} ideas")
    return {"status": "processed", "messages": len(messages), "ideas": ideas_stored}


def process_single_file(file_path: str) -> dict:
    """Process a single transcript file. Thread-safe.

    Returns:
        Dict with batches_processed, ideas_stored, error (if any)
    """
    from batch_processor import process_transcript, ProcessingError
    from backfill import session_from_path

    session = session_from_path(file_path)

    try:
        logger.info(f"Processing: {file_path}")
        result = process_transcript(file_path, session=session)
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


def process_queue_item(ctx: ProcessingContext) -> bool:
    """Process one item from the queue (legacy single-threaded).

    Returns:
        True if an item was processed, False if queue empty
    """
    item = db_get_queue_item()
    if not item:
        return False

    result = process_single_file(item["file_path"])
    ctx.messages_processed += result["batches_processed"]
    ctx.ideas_stored += result["ideas_stored"]
    return True


def process_queue_batch(ctx: ProcessingContext, max_workers: int = PARALLEL_WORKERS) -> bool:
    """Process multiple queue items in parallel.

    Returns:
        True if any items were processed, False if queue empty
    """
    items = db_get_queue_items(limit=max_workers)
    if not items:
        return False

    file_paths = [item["file_path"] for item in items]
    logger.info(f"Processing {len(file_paths)} files in parallel")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in file_paths}

        for future in as_completed(futures):
            result = future.result()
            ctx.messages_processed += result["batches_processed"]
            ctx.ideas_stored += result["ideas_stored"]

    return True


# =============================================================================
# Daemon Main Loop
# =============================================================================

class IndexingDaemon:
    """Main daemon class."""

    def __init__(self):
        self.running = True
        self.ctx = ProcessingContext()
        self.last_activity = time.time()
        self.last_heartbeat = time.time()
        self.last_pidfile_check = 0
        self.my_pid = os.getpid()
        # Use absolute path for pidfile
        self.pidfile_path = (RUNTIME_DIR / "daemon.pid").resolve()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _check_pidfile(self) -> bool:
        """Check if pidfile exists and contains our PID. Called every second.

        Returns:
            True if we should continue running, False if we should exit
        """
        now = time.time()
        if now - self.last_pidfile_check < 1.0:
            return True  # Not time to check yet

        self.last_pidfile_check = now

        # Check if pidfile exists
        if not self.pidfile_path.exists():
            logger.info(f"Pidfile {self.pidfile_path} no longer exists, exiting")
            return False

        # Check if pidfile contains our PID
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
                f"[HEARTBEAT] Alive - processed {self.ctx.messages_processed} messages, "
                f"stored {self.ctx.ideas_stored} ideas"
            )
            self.last_heartbeat = now

    def run(self):
        """Main daemon loop."""
        logger.info("=" * 60)
        logger.info("Total Recall Daemon starting")
        logger.info(f"  DB: {config.DB_PATH}")
        logger.info(f"  Poll interval: {POLL_INTERVAL}s")
        logger.info(f"  Idle timeout: {IDLE_TIMEOUT}s")
        logger.info(f"  OpenAI available: {HAS_OPENAI}")
        logger.info("=" * 60)

        # Write pidfile (using absolute path)
        self.pidfile_path.write_text(str(self.my_pid))
        logger.info(f"PID {self.my_pid} written to {self.pidfile_path}")

        try:
            while self.running:
                try:
                    # Check pidfile every second (method self-throttles)
                    if not self._check_pidfile():
                        break

                    # Process queue items in parallel
                    had_work = process_queue_batch(self.ctx)

                    if had_work:
                        self.last_activity = time.time()
                    else:
                        # Check idle timeout
                        if IDLE_TIMEOUT > 0:
                            idle_time = time.time() - self.last_activity
                            if idle_time > IDLE_TIMEOUT:
                                logger.info(f"Idle for {idle_time:.0f}s, exiting")
                                break

                        # Sleep before next poll
                        time.sleep(POLL_INTERVAL)

                    # Heartbeat
                    self._log_heartbeat()

                except Exception as e:
                    # Catch-all for unexpected errors in main loop
                    logger.error(f"[MAIN LOOP] Unexpected error: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(POLL_INTERVAL)

        finally:
            # Cleanup
            logger.info("Daemon shutting down")

            # Save embedding cache to disk
            try:
                from embeddings.cache import save_embedding_cache, get_embedding_cache_stats
                stats = get_embedding_cache_stats()
                if stats["size"] > 0:
                    save_embedding_cache()
                    logger.info(f"Saved embedding cache ({stats['size']} entries)")
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {e}")

            self.pidfile_path.unlink(missing_ok=True)
            logger.info(f"Final stats: {self.ctx.messages_processed} messages, {self.ctx.ideas_stored} ideas")


def main():
    """Entry point."""
    # Check for API key FIRST - refuse to run without it
    from config import get_openai_api_key, OPENAI_KEY_FILE
    if not get_openai_api_key():
        logger.error("=" * 70)
        logger.error("❌ OpenAI API key not found!")
        logger.error("")
        logger.error("The daemon requires this API key to generate embeddings.")
        logger.error("Without embeddings, search will not work.")
        logger.error("")
        logger.error(f"To fix: echo 'your-key' > {OPENAI_KEY_FILE}")
        logger.error("   Or:  export OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS='your-key'")
        logger.error("=" * 70)
        sys.exit(1)

    # Check if already running (use absolute path)
    pidfile = (RUNTIME_DIR / "daemon.pid").resolve()
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            logger.error(f"Daemon already running with PID {pid}")
            sys.exit(1)
        except (ValueError, OSError):
            # Stale pidfile
            pidfile.unlink(missing_ok=True)

    daemon = IndexingDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
