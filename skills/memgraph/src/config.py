"""Configuration and constants for memgraph."""

import logging
from pathlib import Path

# Database location
DB_PATH = Path.home() / ".claude-plugin-memgraph" / "memory.db"

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Logging setup
LOG_PATH = Path.home() / ".claude-plugin-memgraph" / "memgraph.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
    ]
)
logger = logging.getLogger("memgraph")
