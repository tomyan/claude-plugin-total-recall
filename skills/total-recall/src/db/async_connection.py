"""Async database connection management for total-recall."""

import aiosqlite
import sqlite_vec

import config
from errors import TotalRecallError


async def get_async_db() -> aiosqlite.Connection:
    """Get async database connection with sqlite-vec loaded.

    Returns:
        aiosqlite.Connection configured with WAL mode, busy timeout, and sqlite-vec
    """
    try:
        db = await aiosqlite.connect(str(config.DB_PATH), timeout=30.0)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=30000")

        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        ext_path = sqlite_vec.loadable_path()
        await db.load_extension(ext_path)
        await db.enable_load_extension(False)

        db.row_factory = aiosqlite.Row
        return db
    except Exception as e:
        config.logger.error(f"Async database error: {e}")
        raise TotalRecallError(
            f"Failed to open database at {config.DB_PATH}: {e}",
            "database_error",
            {"path": str(config.DB_PATH), "original_error": str(e)}
        ) from e
