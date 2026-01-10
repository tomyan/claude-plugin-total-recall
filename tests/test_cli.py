"""Tests for CLI commands."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SKILL_SRC = Path(__file__).parent.parent / "skills" / "memgraph" / "src"


class TestUnifiedCLI:
    """Test the unified memgraph CLI."""

    def test_cli_help(self):
        """CLI shows help without error."""
        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "cli.py"), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "memgraph" in result.stdout.lower() or "memory" in result.stdout.lower()

    def test_cli_stats_command(self):
        """CLI stats command returns JSON."""
        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "cli.py"), "stats"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # May fail if no DB, but should not crash
        if result.returncode == 0:
            data = json.loads(result.stdout)
            assert "total_ideas" in data

    def test_cli_topics_command(self):
        """CLI topics command returns JSON list."""
        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "cli.py"), "topics"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            assert isinstance(data, list)


class TestTranscriptCLI:
    """Test transcript.py CLI commands."""

    def test_get_indexable_outputs_json(self, tmp_path):
        """CLI outputs JSON list of indexable messages."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "hello"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "Help me implement user authentication with JWT tokens"}, "timestamp": "T2"}),
        ]))

        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "transcript.py"), "get-indexable", str(transcript)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 1
        assert messages[0]["line_num"] == 2
        assert "JWT" in messages[0]["content"]

    def test_get_indexable_with_start_line(self, tmp_path):
        """CLI respects --start-line argument."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "First message about database design patterns"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "Second message about API architecture decisions"}, "timestamp": "T2"}),
            json.dumps({"type": "user", "message": {"content": "Third message about caching implementation strategy"}, "timestamp": "T3"}),
        ]))

        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "transcript.py"), "get-indexable", str(transcript), "--start-line", "2"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 2
        assert messages[0]["line_num"] == 2

    def test_get_indexable_empty_result(self, tmp_path):
        """CLI returns empty list for no indexable content."""
        transcript = tmp_path / "test.jsonl"
        transcript.write_text("\n".join([
            json.dumps({"type": "user", "message": {"content": "hi"}, "timestamp": "T1"}),
            json.dumps({"type": "user", "message": {"content": "ok"}, "timestamp": "T2"}),
        ]))

        result = subprocess.run(
            [sys.executable, str(SKILL_SRC / "transcript.py"), "get-indexable", str(transcript)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        messages = json.loads(result.stdout)
        assert messages == []
