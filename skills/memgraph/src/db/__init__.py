"""Database operations for memgraph."""

from db.connection import get_db
from db.schema import init_db

__all__ = ["get_db", "init_db"]
