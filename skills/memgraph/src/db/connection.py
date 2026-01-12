"""Database connection management for memgraph."""

import sqlite3

import sqlite_vec

import config
from errors import MemgraphError


def get_db() -> sqlite3.Connection:
    """Get database connection with sqlite-vec loaded."""
    try:
        db = sqlite3.connect(str(config.DB_PATH))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.row_factory = sqlite3.Row
        return db
    except sqlite3.DatabaseError as e:
        config.logger.error(f"Database error: {e}")
        raise MemgraphError(
            f"Failed to open database at {config.DB_PATH}: {e}",
            "database_error",
            {"path": str(config.DB_PATH), "original_error": str(e)}
        ) from e
    except Exception as e:
        config.logger.error(f"Unexpected error opening database: {e}")
        raise MemgraphError(
            f"Failed to open database: {e}",
            "database_error",
            {"original_error": str(e)}
        ) from e
