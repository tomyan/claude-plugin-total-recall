"""Configuration and constants for total-recall."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python


@dataclass
class TotalRecallConfig:
    """Configuration for total-recall.

    All thresholds and tuneable parameters in one place.
    """
    # Database
    db_path: Path = field(default_factory=lambda: Path.home() / ".claude-plugin-total-recall" / "memory.db")

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Topic linking thresholds
    topic_similarity_threshold: float = 0.8  # For auto-linking topics
    duplicate_topic_threshold: float = 0.85  # For finding duplicate topics

    # Topic shift detection
    topic_shift_threshold: float = 0.55  # Below = topic divergence

    # Clustering
    coherence_threshold: float = 0.7  # For cluster quality
    cluster_merge_threshold: float = 0.85  # For merging similar clusters

    # Default confidence
    default_confidence: float = 0.5

    # Logging
    log_path: Path = field(default_factory=lambda: Path.home() / ".claude-plugin-total-recall" / "total-recall.log")


def _find_config_file() -> Optional[Path]:
    """Find total-recall.toml config file.

    Searches in order:
    1. TOTAL_RECALL_CONFIG env var path
    2. Current working directory
    3. ~/.config/total-recall/
    4. ~/.claude-plugin-total-recall/
    """
    # Check env var first
    env_path = os.environ.get("TOTAL_RECALL_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Search paths
    search_paths = [
        Path.cwd() / "total-recall.toml",
        Path.home() / ".config" / "total-recall" / "total-recall.toml",
        Path.home() / ".claude-plugin-total-recall" / "total-recall.toml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def _load_config_from_toml(path: Path) -> TotalRecallConfig:
    """Load config from TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Start with defaults
    config = TotalRecallConfig()

    # Override with TOML values
    if "database" in data:
        if "path" in data["database"]:
            config.db_path = Path(data["database"]["path"])

    if "embedding" in data:
        if "model" in data["embedding"]:
            config.embedding_model = data["embedding"]["model"]
        if "dim" in data["embedding"]:
            config.embedding_dim = data["embedding"]["dim"]

    if "thresholds" in data:
        t = data["thresholds"]
        if "topic_similarity" in t:
            config.topic_similarity_threshold = t["topic_similarity"]
        if "duplicate_topic" in t:
            config.duplicate_topic_threshold = t["duplicate_topic"]
        if "topic_shift" in t:
            config.topic_shift_threshold = t["topic_shift"]
        if "coherence" in t:
            config.coherence_threshold = t["coherence"]
        if "cluster_merge" in t:
            config.cluster_merge_threshold = t["cluster_merge"]
        if "default_confidence" in t:
            config.default_confidence = t["default_confidence"]

    if "logging" in data:
        if "path" in data["logging"]:
            config.log_path = Path(data["logging"]["path"])

    return config


def _apply_env_overrides(config: TotalRecallConfig) -> TotalRecallConfig:
    """Apply environment variable overrides to config.

    Env vars:
        TOTAL_RECALL_DB_PATH: Override database path
        TOTAL_RECALL_TOPIC_SHIFT_THRESHOLD: Override topic shift threshold
    """
    # Database path
    if db_path := os.environ.get("TOTAL_RECALL_DB_PATH"):
        config.db_path = Path(db_path)

    # Topic shift threshold
    if threshold := os.environ.get("TOTAL_RECALL_TOPIC_SHIFT_THRESHOLD"):
        try:
            config.topic_shift_threshold = float(threshold)
        except ValueError:
            pass  # Ignore invalid values

    return config


def _init_config() -> TotalRecallConfig:
    """Initialize config, loading from TOML if available, then applying env overrides."""
    config_file = _find_config_file()
    if config_file:
        config = _load_config_from_toml(config_file)
    else:
        config = TotalRecallConfig()

    # Env vars override TOML and defaults
    config = _apply_env_overrides(config)

    return config


# Global config instance
_config = _init_config()


def get_config() -> TotalRecallConfig:
    """Get the current configuration."""
    return _config


# Legacy module-level constants for backward compatibility
DB_PATH = _config.db_path
EMBEDDING_MODEL = _config.embedding_model
EMBEDDING_DIM = _config.embedding_dim
LOG_PATH = _config.log_path

# OpenAI API key file location (config, not state)
OPENAI_KEY_FILE = Path.home() / ".config" / "total-recall" / "openai-api-key"


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from file or environment variable.

    Checks in order:
    1. File at ~/.claude-plugin-total-recall/openai-api-key
    2. OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS env var

    Returns:
        API key string or None if not found
    """
    # Check file first (more reliable across processes)
    if OPENAI_KEY_FILE.exists():
        try:
            key = OPENAI_KEY_FILE.read_text().strip()
            if key:
                return key
        except Exception:
            pass

    # Fall back to env var
    return os.environ.get("OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS")

# Logging setup
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
    ]
)
logger = logging.getLogger("total-recall")
