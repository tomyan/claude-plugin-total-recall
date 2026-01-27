"""Entity MDM (Master Data Management) functions.

Implements interim records (entity_mentions) and golden records (golden_entities)
for robust entity management with correction capability.
"""

import json
import time
from typing import Optional
from db.connection import get_db


def generate_ulid() -> str:
    """Generate a ULID (Universally Unique Lexicographically Sortable Identifier).

    Simple implementation: timestamp (10 chars) + random (16 chars) = 26 chars.
    Uses Crockford's Base32 alphabet.
    """
    import random

    # Crockford's Base32 alphabet (excludes I, L, O, U to avoid confusion)
    ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

    # Timestamp component (48 bits = 10 chars)
    # Milliseconds since Unix epoch
    timestamp_ms = int(time.time() * 1000)
    timestamp_chars = []
    for _ in range(10):
        timestamp_chars.append(ALPHABET[timestamp_ms & 0x1F])
        timestamp_ms >>= 5
    timestamp_part = "".join(reversed(timestamp_chars))

    # Random component (80 bits = 16 chars)
    random_part = "".join(random.choice(ALPHABET) for _ in range(16))

    return timestamp_part + random_part


def create_entity_mention(
    name: str,
    metadata: dict,
    source_file: str,
    source_line: int,
    golden_id: str = None
) -> str:
    """Create an entity mention (interim record).

    Args:
        name: Entity name as extracted
        metadata: JSON metadata (type, etc.)
        source_file: Source transcript file
        source_line: Line number in source
        golden_id: Optional link to golden record

    Returns:
        ULID of created mention
    """
    mention_id = generate_ulid()

    db = get_db()
    db.execute("""
        INSERT INTO entity_mentions (id, name, metadata, source_file, source_line, golden_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        mention_id,
        name,
        json.dumps(metadata),
        source_file,
        source_line,
        golden_id
    ))
    db.commit()
    db.close()

    return mention_id


def create_golden_entity(canonical_name: str, metadata: dict) -> str:
    """Create a golden entity (canonical/master record).

    Args:
        canonical_name: Canonical name for the entity
        metadata: JSON metadata

    Returns:
        ULID of created golden entity

    Raises:
        sqlite3.IntegrityError: If canonical_name already exists
    """
    golden_id = generate_ulid()

    db = get_db()
    db.execute("""
        INSERT INTO golden_entities (id, canonical_name, metadata, created_at, updated_at)
        VALUES (?, ?, ?, datetime('now'), datetime('now'))
    """, (
        golden_id,
        canonical_name,
        json.dumps(metadata)
    ))
    db.commit()
    db.close()

    return golden_id


def update_golden_entity(golden_id: str, metadata: dict = None, canonical_name: str = None) -> None:
    """Update a golden entity.

    Args:
        golden_id: ID of golden entity to update
        metadata: New metadata (optional)
        canonical_name: New canonical name (optional)
    """
    db = get_db()

    if metadata is not None and canonical_name is not None:
        db.execute("""
            UPDATE golden_entities
            SET metadata = ?, canonical_name = ?, updated_at = datetime('now')
            WHERE id = ?
        """, (json.dumps(metadata), canonical_name, golden_id))
    elif metadata is not None:
        db.execute("""
            UPDATE golden_entities
            SET metadata = ?, updated_at = datetime('now')
            WHERE id = ?
        """, (json.dumps(metadata), golden_id))
    elif canonical_name is not None:
        db.execute("""
            UPDATE golden_entities
            SET canonical_name = ?, updated_at = datetime('now')
            WHERE id = ?
        """, (canonical_name, golden_id))

    db.commit()
    db.close()


def link_mention_to_golden(mention_id: str, golden_id: str) -> None:
    """Link an entity mention to a golden record.

    Args:
        mention_id: ID of entity mention
        golden_id: ID of golden entity to link to
    """
    db = get_db()
    db.execute("""
        UPDATE entity_mentions
        SET golden_id = ?
        WHERE id = ?
    """, (golden_id, mention_id))
    db.commit()
    db.close()


def find_golden_entity(name: str, threshold: float = 0.8) -> Optional[dict]:
    """Find a golden entity by name with fuzzy matching.

    Args:
        name: Name to search for
        threshold: Similarity threshold (0-1) for fuzzy match

    Returns:
        Golden entity dict or None if not found
    """
    db = get_db()

    # First try exact case-insensitive match
    cursor = db.execute("""
        SELECT id, canonical_name, metadata, created_at, updated_at
        FROM golden_entities
        WHERE LOWER(canonical_name) = LOWER(?)
    """, (name,))

    row = cursor.fetchone()
    if row:
        db.close()
        return {
            "id": row["id"],
            "canonical_name": row["canonical_name"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    # Try fuzzy match
    cursor = db.execute("SELECT id, canonical_name, metadata, created_at, updated_at FROM golden_entities")

    best_match = None
    best_similarity = 0

    for row in cursor:
        similarity = _string_similarity(name.lower(), row["canonical_name"].lower())
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = row

    db.close()

    if best_match:
        return {
            "id": best_match["id"],
            "canonical_name": best_match["canonical_name"],
            "metadata": json.loads(best_match["metadata"]) if best_match["metadata"] else {},
            "created_at": best_match["created_at"],
            "updated_at": best_match["updated_at"]
        }

    return None


def _string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0-1).

    Uses longest common subsequence ratio.
    """
    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    # Simple LCS-based similarity
    len1, len2 = len(s1), len(s2)

    # Create matrix for LCS
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[len1][len2]
    return (2.0 * lcs_length) / (len1 + len2)


def get_entity_mentions(golden_id: str) -> list[dict]:
    """Get all entity mentions linked to a golden entity.

    Args:
        golden_id: ID of golden entity

    Returns:
        List of mention dicts
    """
    db = get_db()
    cursor = db.execute("""
        SELECT id, name, metadata, source_file, source_line, created_at
        FROM entity_mentions
        WHERE golden_id = ?
        ORDER BY created_at DESC
    """, (golden_id,))

    mentions = []
    for row in cursor:
        mentions.append({
            "id": row["id"],
            "name": row["name"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "source_file": row["source_file"],
            "source_line": row["source_line"],
            "created_at": row["created_at"]
        })

    db.close()
    return mentions
