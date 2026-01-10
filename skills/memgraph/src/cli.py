#!/usr/bin/env python3
"""Unified CLI for memgraph memory system.

Single entry point that handles bootstrap internally.
Usage: memgraph <command> [args]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure we can import our modules
SKILL_DIR = Path(__file__).parent.parent
SRC_DIR = SKILL_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

RUNTIME_DIR = Path.home() / ".claude-plugin-memgraph"


def safe_result(data: Any) -> dict:
    """Wrap successful result."""
    return {"success": True, "data": data}


def error_result(error: str, code: str = "unknown", details: dict = None) -> dict:
    """Wrap error result."""
    result = {"success": False, "error": error, "error_code": code}
    if details:
        result["details"] = details
    return result


def run_search_command(query: str, limit: int = 10, session: str = None, intent: str = None) -> dict:
    """Run search command with error handling."""
    try:
        import memory_db
        results = memory_db.search_ideas(query, limit=limit, session=session, intent=intent)
        return safe_result(results)
    except memory_db.MemgraphError as e:
        return error_result(str(e), e.error_code, e.details)
    except Exception as e:
        return error_result(f"Search failed: {e}", "search_error")


def run_stats_command() -> dict:
    """Run stats command with error handling."""
    try:
        import memory_db
        stats = memory_db.get_stats()
        return safe_result(stats)
    except memory_db.MemgraphError as e:
        return error_result(str(e), e.error_code, e.details)
    except Exception as e:
        return error_result(f"Failed to get stats: {e}", "stats_error")


def ensure_runtime():
    """Ensure runtime directory and dependencies exist."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    pyproject = RUNTIME_DIR / "pyproject.toml"
    if not pyproject.exists():
        # Initialize uv project
        subprocess.run(
            ["uv", "init", "--name", "memgraph", "--no-readme"],
            cwd=RUNTIME_DIR,
            capture_output=True
        )
        subprocess.run(
            ["uv", "add", "sqlite-vec", "openai"],
            cwd=RUNTIME_DIR,
            capture_output=True
        )

    venv = RUNTIME_DIR / ".venv"
    if not venv.exists():
        subprocess.run(["uv", "sync"], cwd=RUNTIME_DIR, capture_output=True)


def run_command(args):
    """Run a command with proper imports and error handling."""
    import memory_db

    try:
        if args.command == "init":
            memory_db.init_db()
            print(json.dumps({"success": True, "message": "Database initialized"}))

        elif args.command == "search":
            result = run_search_command(
                args.query,
                limit=args.limit,
                session=args.session,
                intent=args.intent
            )
            if result["success"]:
                print(json.dumps(result["data"], indent=2, default=str))
            else:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

        elif args.command == "hybrid":
            results = memory_db.hybrid_search(args.query, limit=args.limit)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "hyde":
            results = memory_db.hyde_search(args.query, limit=args.limit)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "stats":
            result = run_stats_command()
            if result["success"]:
                print(json.dumps(result["data"], indent=2))
            else:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

        elif args.command == "backfill":
            from backfill import backfill_transcript
            result = backfill_transcript(args.file, args.start_line)
            print(json.dumps(result))

        elif args.command == "index":
            from indexer import index_transcript
            result = index_transcript(args.file, args.start_line)
            print(json.dumps(result))

        elif args.command == "topics":
            db = memory_db.get_db()
            if args.session:
                cursor = db.execute("""
                    SELECT id, session, name, summary, start_line, end_line, depth
                    FROM spans WHERE session = ? ORDER BY start_line
                """, (args.session,))
            else:
                cursor = db.execute("""
                    SELECT id, session, name, summary, start_line, end_line, depth
                    FROM spans ORDER BY session, start_line
                """)
            results = [dict(row) for row in cursor]
            db.close()
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "progress":
            from backfill import get_progress
            result = get_progress(args.file)
            print(json.dumps(result))

        elif args.command == "questions":
            questions = memory_db.get_unanswered_questions(args.session)
            print(json.dumps(questions, indent=2, default=str))

    except memory_db.MemgraphError as e:
        print(json.dumps(e.to_dict()), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e), "error_code": "unexpected"}), file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="memgraph",
        description="Memory graph for Claude conversations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Initialize the database")

    # search
    search_p = subparsers.add_parser("search", help="Vector search for ideas")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    search_p.add_argument("-s", "--session", help="Filter by session")
    search_p.add_argument("-i", "--intent", help="Filter by intent")

    # hybrid
    hybrid_p = subparsers.add_parser("hybrid", help="Hybrid vector+keyword search")
    hybrid_p.add_argument("query", help="Search query")
    hybrid_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # hyde
    hyde_p = subparsers.add_parser("hyde", help="HyDE search (hypothetical doc)")
    hyde_p.add_argument("query", help="Search query")
    hyde_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")

    # stats
    subparsers.add_parser("stats", help="Show database statistics")

    # backfill
    backfill_p = subparsers.add_parser("backfill", help="Backfill a transcript")
    backfill_p.add_argument("file", help="Transcript file path")
    backfill_p.add_argument("--start-line", type=int, help="Start from line")

    # index (full indexing with topic tracking)
    index_p = subparsers.add_parser("index", help="Index with topic tracking")
    index_p.add_argument("file", help="Transcript file path")
    index_p.add_argument("--start-line", type=int, help="Start from line")

    # topics
    topics_p = subparsers.add_parser("topics", help="List topic spans")
    topics_p.add_argument("-s", "--session", help="Filter by session")

    # progress
    progress_p = subparsers.add_parser("progress", help="Show indexing progress")
    progress_p.add_argument("file", help="Transcript file path")

    # questions
    questions_p = subparsers.add_parser("questions", help="List unanswered questions")
    questions_p.add_argument("-s", "--session", help="Filter by session")

    args = parser.parse_args()

    # Ensure runtime is ready
    ensure_runtime()

    # Initialize DB if needed
    import memory_db
    if not memory_db.DB_PATH.exists():
        memory_db.init_db()

    run_command(args)


if __name__ == "__main__":
    main()
