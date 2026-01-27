"""Tests for entity MDM (Master Data Management) pattern."""

import json
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def setup_database():
    """Initialize test database before each test."""
    import config
    import tempfile
    import shutil

    # Use temp database for tests
    temp_dir = tempfile.mkdtemp()
    config.DB_PATH = Path(temp_dir) / "test_memory.db"

    from db.schema import init_db
    init_db()

    yield

    # Cleanup (use rmtree to handle SQLite journal files)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEntityMentionsTable:
    """Tests for entity_mentions table - Slice 1.1"""

    def test_entity_mentions_table_exists(self):
        """Entity mentions table should exist after init."""
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='entity_mentions'
        """)
        result = cursor.fetchone()
        db.close()

        assert result is not None, "entity_mentions table should exist"

    def test_can_create_entity_mention_with_ulid(self):
        """Can create entity mention with ULID id."""
        from db.connection import get_db
        from entities import create_entity_mention

        mention_id = create_entity_mention(
            name="SQLite",
            metadata={"type": "technology", "category": "database"},
            source_file="/path/to/transcript.jsonl",
            source_line=42
        )

        # ULID is 26 characters
        assert len(mention_id) == 26

        # Verify stored
        db = get_db()
        cursor = db.execute(
            "SELECT * FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["name"] == "SQLite"

    def test_metadata_stored_as_json(self):
        """Metadata should be stored as JSON and retrievable."""
        from db.connection import get_db
        from entities import create_entity_mention

        metadata = {"type": "technology", "category": "database", "version": "3.40"}

        mention_id = create_entity_mention(
            name="SQLite",
            metadata=metadata,
            source_file="/path/to/file.jsonl",
            source_line=10
        )

        db = get_db()
        cursor = db.execute(
            "SELECT metadata FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        stored_metadata = json.loads(row["metadata"])
        assert stored_metadata == metadata

    def test_source_reference_preserved(self):
        """Source file and line should be preserved."""
        from db.connection import get_db
        from entities import create_entity_mention

        mention_id = create_entity_mention(
            name="React",
            metadata={"type": "technology"},
            source_file="/home/user/.claude/projects/abc.jsonl",
            source_line=999
        )

        db = get_db()
        cursor = db.execute(
            "SELECT source_file, source_line FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row["source_file"] == "/home/user/.claude/projects/abc.jsonl"
        assert row["source_line"] == 999

    def test_created_at_auto_populated(self):
        """created_at should be automatically set."""
        from db.connection import get_db
        from entities import create_entity_mention

        mention_id = create_entity_mention(
            name="Python",
            metadata={"type": "technology"},
            source_file="/path/to/file.jsonl",
            source_line=1
        )

        db = get_db()
        cursor = db.execute(
            "SELECT created_at FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row["created_at"] is not None
        # Should be ISO format datetime
        assert "T" in row["created_at"] or "-" in row["created_at"]

    def test_golden_id_initially_null(self):
        """New mentions should have null golden_id (unresolved)."""
        from db.connection import get_db
        from entities import create_entity_mention

        mention_id = create_entity_mention(
            name="PostgreSQL",
            metadata={"type": "technology"},
            source_file="/path/to/file.jsonl",
            source_line=5
        )

        db = get_db()
        cursor = db.execute(
            "SELECT golden_id FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row["golden_id"] is None


class TestGoldenEntitiesTable:
    """Tests for golden_entities table - Slice 1.2"""

    def test_golden_entities_table_exists(self):
        """Golden entities table should exist after init."""
        from db.connection import get_db

        db = get_db()
        cursor = db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='golden_entities'
        """)
        result = cursor.fetchone()
        db.close()

        assert result is not None, "golden_entities table should exist"

    def test_can_create_golden_entity(self):
        """Can create golden entity record."""
        from db.connection import get_db
        from entities import create_golden_entity

        golden_id = create_golden_entity(
            canonical_name="SQLite",
            metadata={"type": "technology", "category": "database"}
        )

        assert len(golden_id) == 26  # ULID

        db = get_db()
        cursor = db.execute(
            "SELECT * FROM golden_entities WHERE id = ?",
            (golden_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row is not None
        assert row["canonical_name"] == "SQLite"

    def test_canonical_name_is_unique(self):
        """Cannot create two golden entities with same canonical name."""
        from entities import create_golden_entity
        import sqlite3

        create_golden_entity(canonical_name="React", metadata={})

        with pytest.raises(sqlite3.IntegrityError):
            create_golden_entity(canonical_name="React", metadata={})

    def test_updated_at_changes_on_update(self):
        """updated_at should change when entity is updated."""
        from db.connection import get_db
        from entities import create_golden_entity, update_golden_entity
        import time

        golden_id = create_golden_entity(
            canonical_name="FastAPI",
            metadata={"type": "technology"}
        )

        db = get_db()
        cursor = db.execute(
            "SELECT updated_at FROM golden_entities WHERE id = ?",
            (golden_id,)
        )
        original_updated = cursor.fetchone()["updated_at"]
        db.close()

        # SQLite datetime has second precision, wait >1 second
        time.sleep(1.1)

        update_golden_entity(
            golden_id,
            metadata={"type": "technology", "category": "framework"}
        )

        db = get_db()
        cursor = db.execute(
            "SELECT updated_at FROM golden_entities WHERE id = ?",
            (golden_id,)
        )
        new_updated = cursor.fetchone()["updated_at"]
        db.close()

        assert new_updated != original_updated

    def test_can_link_mention_to_golden(self):
        """Can link entity mention to golden record."""
        from db.connection import get_db
        from entities import (
            create_entity_mention,
            create_golden_entity,
            link_mention_to_golden
        )

        # Create golden
        golden_id = create_golden_entity(
            canonical_name="SQLite",
            metadata={"type": "technology"}
        )

        # Create mention
        mention_id = create_entity_mention(
            name="sqlite",  # lowercase variant
            metadata={"type": "technology"},
            source_file="/path/file.jsonl",
            source_line=10
        )

        # Link them
        link_mention_to_golden(mention_id, golden_id)

        # Verify
        db = get_db()
        cursor = db.execute(
            "SELECT golden_id FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row["golden_id"] == golden_id


class TestEdgeCases:
    """Adversarial tests for edge cases."""

    def test_empty_name_allowed(self):
        """Empty name should be allowed (validation is caller's job)."""
        from entities import create_entity_mention

        # Should not raise
        mention_id = create_entity_mention(
            name="",
            metadata={},
            source_file="/path/file.jsonl",
            source_line=1
        )
        assert len(mention_id) == 26

    def test_none_metadata_handled(self):
        """None metadata should be stored as null."""
        from db.connection import get_db
        from entities import create_entity_mention

        mention_id = create_entity_mention(
            name="Test",
            metadata=None,
            source_file="/path/file.jsonl",
            source_line=1
        )

        db = get_db()
        cursor = db.execute(
            "SELECT metadata FROM entity_mentions WHERE id = ?",
            (mention_id,)
        )
        row = cursor.fetchone()
        db.close()

        assert row["metadata"] == "null"  # JSON null

    def test_special_characters_in_name(self):
        """Names with special characters should work."""
        from entities import create_entity_mention, create_golden_entity

        # SQL-like characters
        create_entity_mention(
            name="'; DROP TABLE entities; --",
            metadata={},
            source_file="/path/file.jsonl",
            source_line=1
        )

        # Unicode
        create_golden_entity(
            canonical_name="日本語テスト",
            metadata={"type": "test"}
        )

        # Quotes
        create_golden_entity(
            canonical_name='Test "with" quotes',
            metadata={}
        )

    def test_ulid_uniqueness(self):
        """ULIDs should be unique across rapid creation."""
        from entities import generate_ulid

        ulids = set()
        for _ in range(1000):
            ulid = generate_ulid()
            assert ulid not in ulids, f"Duplicate ULID: {ulid}"
            ulids.add(ulid)

    def test_link_to_nonexistent_golden_succeeds(self):
        """Linking to non-existent golden should succeed (no FK enforcement in SQLite by default)."""
        from entities import create_entity_mention, link_mention_to_golden

        mention_id = create_entity_mention(
            name="Test",
            metadata={},
            source_file="/path/file.jsonl",
            source_line=1
        )

        # Link to non-existent golden - should not raise
        link_mention_to_golden(mention_id, "NONEXISTENT_GOLDEN_ID")

    def test_fuzzy_match_short_strings(self):
        """Very short strings should not match unless identical."""
        from entities import create_golden_entity, find_golden_entity

        create_golden_entity(canonical_name="Go", metadata={})

        # Exact match should work
        assert find_golden_entity("Go") is not None
        assert find_golden_entity("go") is not None

        # Different short string should not match
        assert find_golden_entity("Py") is None
        assert find_golden_entity("C") is None


class TestEntityResolution:
    """Tests for entity resolution functions - Slice 1.3"""

    def test_find_golden_entity_exact_match(self):
        """Find golden entity by exact name match."""
        from entities import create_golden_entity, find_golden_entity

        golden_id = create_golden_entity(
            canonical_name="PostgreSQL",
            metadata={"type": "technology"}
        )

        result = find_golden_entity("PostgreSQL")

        assert result is not None
        assert result["id"] == golden_id
        assert result["canonical_name"] == "PostgreSQL"

    def test_find_golden_entity_case_insensitive(self):
        """Find golden entity with case-insensitive match."""
        from entities import create_golden_entity, find_golden_entity

        create_golden_entity(
            canonical_name="PostgreSQL",
            metadata={"type": "technology"}
        )

        # Various case variations
        assert find_golden_entity("postgresql") is not None
        assert find_golden_entity("POSTGRESQL") is not None
        assert find_golden_entity("PostgreSQL") is not None

    def test_find_golden_entity_fuzzy_match(self):
        """Find golden entity with fuzzy match (>80% similarity)."""
        from entities import create_golden_entity, find_golden_entity

        create_golden_entity(
            canonical_name="PostgreSQL",
            metadata={"type": "technology"}
        )

        # "postgres" is similar enough to "PostgreSQL"
        result = find_golden_entity("postgres")
        assert result is not None

        # "pg" is too short/different
        result = find_golden_entity("pg")
        assert result is None

    def test_find_golden_entity_returns_none_for_no_match(self):
        """Returns None when no matching golden entity."""
        from entities import find_golden_entity

        result = find_golden_entity("NonExistentThing")
        assert result is None

    def test_get_entity_mentions_for_golden(self):
        """Get all mentions linked to a golden entity."""
        from entities import (
            create_golden_entity,
            create_entity_mention,
            link_mention_to_golden,
            get_entity_mentions
        )

        golden_id = create_golden_entity(
            canonical_name="React",
            metadata={"type": "technology"}
        )

        # Create several mentions
        m1 = create_entity_mention(
            name="react",
            metadata={},
            source_file="/a.jsonl",
            source_line=1
        )
        m2 = create_entity_mention(
            name="React",
            metadata={},
            source_file="/b.jsonl",
            source_line=2
        )
        m3 = create_entity_mention(
            name="ReactJS",
            metadata={},
            source_file="/c.jsonl",
            source_line=3
        )

        # Link them
        link_mention_to_golden(m1, golden_id)
        link_mention_to_golden(m2, golden_id)
        link_mention_to_golden(m3, golden_id)

        # Get mentions
        mentions = get_entity_mentions(golden_id)

        assert len(mentions) == 3
        names = {m["name"] for m in mentions}
        assert names == {"react", "React", "ReactJS"}

    def test_get_entity_mentions_empty_for_no_links(self):
        """Returns empty list when no mentions linked."""
        from entities import create_golden_entity, get_entity_mentions

        golden_id = create_golden_entity(
            canonical_name="Vue",
            metadata={}
        )

        mentions = get_entity_mentions(golden_id)
        assert mentions == []
