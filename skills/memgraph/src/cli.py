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


def resolve_session(args) -> str | None:
    """Resolve the session to search based on args.

    Priority:
    1. If --global is set, return None (search all)
    2. If --session is explicitly set, use it
    3. If --cwd is set, try to derive session from it
    4. Otherwise return None (will search all for backwards compat)

    Args:
        args: Parsed arguments with session, cwd, and global_search attributes

    Returns:
        Session name to filter by, or None for global search
    """
    import memory_db

    # Global search - no session filter
    if getattr(args, 'global_search', False):
        return None

    # Explicit session takes priority
    if getattr(args, 'session', None):
        return args.session

    # Try to derive from cwd
    cwd = getattr(args, 'cwd', None)
    if cwd:
        session = memory_db.get_session_for_cwd(cwd)
        if session:
            return session

    # Default: return None (global search for backwards compatibility)
    # In future, could change this to require --global for cross-project search
    return None


def run_search_command(
    query: str,
    limit: int = 10,
    session: str = None,
    intent: str = None,
    since: str = None,
    until: str = None
) -> dict:
    """Run search command with error handling."""
    try:
        import memory_db
        if since or until:
            # Use temporal search when date filters provided
            results = memory_db.search_ideas_temporal(query, limit=limit, since=since, until=until)
            # Apply additional filters
            if session:
                results = [r for r in results if r.get('session') == session]
            if intent:
                results = [r for r in results if r.get('intent') == intent]
        else:
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
            # Resolve session from args (--global, --session, --cwd)
            session = resolve_session(args)
            result = run_search_command(
                args.query,
                limit=args.limit,
                session=session,
                intent=args.intent,
                since=args.since,
                until=args.until
            )
            if result["success"]:
                # Include session info in output for transparency
                output = result["data"]
                if session:
                    print(f"# Searching project: {session}", file=sys.stderr)
                print(json.dumps(output, indent=2, default=str))
            else:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

        elif args.command == "hybrid":
            # Resolve session from args
            session = resolve_session(args)
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            results = memory_db.hybrid_search(args.query, limit=args.limit, session=session)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "hyde":
            # Resolve session from args
            session = resolve_session(args)
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            results = memory_db.hyde_search(args.query, limit=args.limit, session=session)
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
            # List deduplicated topics
            topics = memory_db.list_topics()
            print(json.dumps(topics, indent=2, default=str))

        elif args.command == "spans":
            # List physical transcript spans
            db = memory_db.get_db()
            if args.session:
                cursor = db.execute("""
                    SELECT s.id, s.topic_id, s.session, s.name, s.summary,
                           s.start_line, s.end_line, s.depth, t.name as topic_name
                    FROM spans s
                    LEFT JOIN topics t ON t.id = s.topic_id
                    WHERE s.session = ? ORDER BY s.start_line
                """, (args.session,))
            else:
                cursor = db.execute("""
                    SELECT s.id, s.topic_id, s.session, s.name, s.summary,
                           s.start_line, s.end_line, s.depth, t.name as topic_name
                    FROM spans s
                    LEFT JOIN topics t ON t.id = s.topic_id
                    ORDER BY s.session, s.start_line
                """)
            results = [dict(row) for row in cursor]
            db.close()
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "merge-topics":
            result = memory_db.merge_topics(args.source, args.target)
            print(json.dumps({"success": True, **result}))

        elif args.command == "review":
            issues = memory_db.review_topics()
            print(json.dumps(issues, indent=2, default=str))

        elif args.command == "rename-topic":
            if args.name:
                success = memory_db.rename_topic(args.id, args.name)
            else:
                # Use LLM to suggest name
                suggested = memory_db.suggest_topic_name(args.id)
                if suggested:
                    if args.auto:
                        success = memory_db.rename_topic(args.id, suggested)
                        print(json.dumps({"success": success, "new_name": suggested}))
                        return
                    else:
                        print(json.dumps({"suggested_name": suggested, "apply_with": f"rename-topic {args.id} --name '{suggested}'"}))
                        return
                else:
                    print(json.dumps({"error": "Could not suggest name", "error_code": "suggestion_failed"}))
                    return
            print(json.dumps({"success": success}))

        elif args.command == "migrate-timestamps":
            result = memory_db.migrate_timestamps_from_transcripts()
            print(json.dumps(result))

        elif args.command == "progress":
            from backfill import get_progress
            result = get_progress(args.file)
            print(json.dumps(result))

        elif args.command == "questions":
            questions = memory_db.get_unanswered_questions(args.session)
            print(json.dumps(questions, indent=2, default=str))

        elif args.command == "get":
            result = memory_db.get_idea_with_relations(args.id)
            if result:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(json.dumps({"error": f"Idea {args.id} not found"}), file=sys.stderr)
                sys.exit(1)

        elif args.command == "similar":
            results = memory_db.find_similar_ideas(
                args.id,
                limit=args.limit,
                same_session=args.same_session,
                session=args.session
            )
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "prune":
            result = memory_db.prune_old_ideas(
                older_than_days=args.days,
                session=args.session,
                dry_run=not args.execute
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "update-intent":
            result = memory_db.update_idea_intent(args.id, args.intent)
            print(json.dumps({"updated": result, "id": args.id, "intent": args.intent}))

        elif args.command == "move-idea":
            result = memory_db.move_idea_to_span(args.id, args.span_id)
            print(json.dumps({"moved": result, "idea_id": args.id, "span_id": args.span_id}))

        elif args.command == "merge-spans":
            result = memory_db.merge_spans(args.source, args.target)
            print(json.dumps(result, indent=2))

        elif args.command == "supersede":
            memory_db.supersede_idea(args.old_id, args.new_id)
            print(json.dumps({"success": True, "old_id": args.old_id, "new_id": args.new_id}))

        elif args.command == "export":
            data = memory_db.export_data(session=args.session)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(json.dumps({"success": True, "output": args.output, "stats": data["stats"]}))
            else:
                print(json.dumps(data, indent=2, default=str))

        elif args.command == "import":
            with open(args.file, 'r') as f:
                data = json.load(f)
            result = memory_db.import_data(data, replace=args.replace)
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "context":
            result = memory_db.get_context(
                args.id,
                lines_before=args.before,
                lines_after=args.after
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "sessions":
            sessions = memory_db.list_sessions()
            print(json.dumps(sessions, indent=2, default=str))

        elif args.command == "projects":
            projects = memory_db.list_projects()
            print(json.dumps(projects, indent=2, default=str))

        elif args.command == "create-project":
            project_id = memory_db.create_project(args.name, args.description)
            print(json.dumps({"success": True, "id": project_id, "name": args.name}))

        elif args.command == "assign-topic":
            # Try to parse project as ID first, then lookup by name
            try:
                project_id = int(args.project)
            except ValueError:
                project = memory_db.get_project_by_name(args.project)
                if not project:
                    print(json.dumps({"error": f"Project '{args.project}' not found"}), file=sys.stderr)
                    sys.exit(1)
                project_id = project["id"]
            memory_db.assign_topic_to_project(args.topic_id, project_id)
            print(json.dumps({"success": True, "topic_id": args.topic_id, "project_id": project_id}))

        elif args.command == "unparent-topic":
            memory_db.unparent_topic(args.topic_id)
            print(json.dumps({"success": True, "topic_id": args.topic_id, "parent_id": None}))

        elif args.command == "delete-topic":
            result = memory_db.delete_topic(args.topic_id, delete_ideas=args.delete_ideas)
            print(f"Deleted topic {args.topic_id}")
            print(f"  Ideas deleted: {result['ideas_deleted']}")
            print(f"  Spans deleted: {result['spans_deleted']}")

        elif args.command == "tree":
            tree = memory_db.get_project_tree()

            def count_ideas(topic):
                """Recursively count ideas including children."""
                total = topic.get("idea_count", 0)
                for child in topic.get("children", []):
                    total += count_ideas(child)
                return total

            def render_topic(topic, prefix="  ", is_last=True):
                """Recursively render a topic and its children."""
                connector = "└─" if is_last else "├─"
                idea_count = topic['idea_count']
                child_count = len(topic.get('children', []))
                suffix = f" (+{child_count} sub)" if child_count else ""
                print(f"{prefix}{connector} {topic['name']} ({idea_count} ideas){suffix}")

                children = topic.get('children', [])
                for i, child in enumerate(children):
                    child_prefix = prefix + ("   " if is_last else "│  ")
                    render_topic(child, child_prefix, i == len(children) - 1)

            # Render as visual tree
            for project in tree:
                idea_total = sum(count_ideas(t) for t in project.get("topics", []))
                topic_count = len(project['topics'])
                print(f"{'=' * 60}")
                if project["id"]:
                    print(f"[{project['name']}] ({topic_count} topics, {idea_total} ideas)")
                else:
                    print(f"{project['name']} ({topic_count} topics, {idea_total} ideas)")
                if project.get("description"):
                    print(f"  {project['description']}")
                topics = project.get("topics", [])
                for i, topic in enumerate(topics):
                    render_topic(topic, "  ", i == len(topics) - 1)

        elif args.command == "review-ideas":
            result = memory_db.review_ideas_against_filters(
                topic_id=args.topic,
                dry_run=not args.execute
            )
            # Pretty print summary
            print(f"Reviewed: {result['total_reviewed']} ideas")
            print(f"Would filter: {result['would_filter']}")
            print(f"Would keep: {result['would_keep']}")
            if result['filter_reasons']:
                print("\nFilter reasons:")
                for reason, count in sorted(result['filter_reasons'].items(), key=lambda x: -x[1]):
                    print(f"  {reason}: {count}")
            if result['samples']:
                print(f"\nSample filtered ideas (first {len(result['samples'])}):")
                for s in result['samples']:
                    print(f"  [{s['reason']}] {s['content'][:70]}...")
            if not result['dry_run']:
                print(f"\nRemoved: {result.get('removed', 0)} ideas")

        elif args.command == "auto-categorize":
            result = memory_db.auto_categorize_topics(dry_run=not args.execute)
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)
            print(f"Reviewed: {result.get('reviewed', 0)} unassigned topics")
            if result.get("changes"):
                print(f"\n{'Would assign' if result.get('dry_run') else 'Assigned'}:")
                for c in result["changes"]:
                    print(f"  {c['topic_name']} -> {c['suggested_project']}")
            else:
                print("No changes needed")

        elif args.command == "improve":
            result = memory_db.improve_categorization()
            print("Categorization improvements:")
            if result.get("categorization", {}).get("changes"):
                print(f"  Assigned {len(result['categorization']['changes'])} topics to projects")
            if result.get("renamed_topics"):
                print(f"  Renamed {len(result['renamed_topics'])} topics:")
                for r in result["renamed_topics"]:
                    print(f"    '{r['old_name']}' -> '{r['new_name']}'")
            if result.get("remaining_issues"):
                ri = result["remaining_issues"]
                if ri.get("catch_all_topics"):
                    print(f"  {ri['catch_all_topics']} catch-all topics need manual review")
                if ri.get("empty_topics"):
                    print(f"  {ri['empty_topics']} empty topics could be removed")

        elif args.command == "llm-filter":
            result = memory_db.llm_filter_ideas(
                topic_id=args.topic,
                batch_size=args.batch,
                dry_run=not args.execute
            )
            print(f"Reviewed: {result['total_reviewed']} ideas")
            print(f"Flagged for removal: {result['flagged']}")
            if result.get('samples'):
                print(f"\nSamples flagged (showing first {len(result['samples'])}):")
                for s in result['samples']:
                    print(f"  [{s['reason']}] {s['content'][:70]}...")
            if not result.get('dry_run'):
                print(f"\nRemoved: {result.get('removed', 0)} ideas")

        elif args.command == "cluster":
            result = memory_db.cluster_topics(min_cluster_size=args.min_size)
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            print(f"Analyzed {result['total_ideas']} ideas across {result['total_topics']} topics\n")

            # Topic coherence (worst first)
            print("Topic Coherence (lower = more scattered):")
            for t in result['topic_coherence'][:10]:
                bar = "█" * int(t['coherence_score'] * 20)
                print(f"  {t['coherence_score']:.2f} {bar} {t['name']} ({t['idea_count']})")

            # Split candidates
            if result['split_candidates']:
                print(f"\nTopics to consider splitting ({len(result['split_candidates'])}):")
                for s in result['split_candidates'][:5]:
                    print(f"  • {s['name']} (coherence: {s['coherence_score']}, {s['idea_count']} ideas)")
                    print(f"    Run: recluster {s['topic_id']}")

            # Merge candidates
            if result['merge_candidates']:
                print(f"\nTopics to consider merging ({len(result['merge_candidates'])}):")
                for m in result['merge_candidates'][:5]:
                    print(f"  • {m['topic1_name']} + {m['topic2_name']} (similarity: {m['similarity']})")
                    print(f"    Run: {m['suggestion']}")

            # Misplaced ideas
            if result['misplaced_ideas']:
                print(f"\nPotentially misplaced ideas ({result['summary']['misplaced_count']} total, showing top 10):")
                for m in result['misplaced_ideas'][:10]:
                    print(f"  [{m['idea_id']}] {m['content'][:50]}...")
                    print(f"    {m['current_topic']} → {m['suggested_topic']} ({m['distance_improvement']}% closer)")

        elif args.command == "recluster":
            result = memory_db.recluster_topic(args.topic_id, num_clusters=args.clusters)
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            print(f"Topic: {result['topic_name']} ({result['total_ideas']} ideas)")
            print(f"Found {result['num_clusters']} natural clusters:\n")

            for c in result['clusters']:
                if c['idea_count'] == 0:
                    continue
                print(f"━━━ Cluster {c['cluster_id']+1}: {c['suggested_name'] or '(unnamed)'} ({c['idea_count']} ideas) ━━━")
                for idea in c['sample_ideas']:
                    print(f"  • {idea['content'][:80]}...")
                print()

        elif args.command == "split-topic":
            result = memory_db.split_topic(
                args.topic_id,
                num_clusters=args.clusters,
                min_cluster_size=args.min_size,
                delete_junk=not args.keep_junk
            )
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            print(f"Split '{result['original_name']}' into {len(result['created_topics'])} sub-topics:\n")
            for t in result['created_topics']:
                print(f"  • {t['name']} ({t['idea_count']} ideas)")
            print(f"\nMoved: {result['moved_ideas']} ideas")
            print(f"Deleted (junk): {result['deleted_ideas']} ideas")
            print(f"Kept in original: {result['kept_in_original']} ideas")
            print(f"\nRun 'tree' to see the new hierarchy.")

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
    search_p.add_argument("-s", "--session", help="Filter by session (auto-detected from --cwd if not specified)")
    search_p.add_argument("-i", "--intent", help="Filter by intent")
    search_p.add_argument("--since", help="Only ideas after this date (ISO format, e.g. 2024-01-01)")
    search_p.add_argument("--until", help="Only ideas before this date (ISO format)")
    search_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    search_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")

    # hybrid
    hybrid_p = subparsers.add_parser("hybrid", help="Hybrid vector+keyword search")
    hybrid_p.add_argument("query", help="Search query")
    hybrid_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    hybrid_p.add_argument("-s", "--session", help="Filter by session (auto-detected from --cwd if not specified)")
    hybrid_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    hybrid_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")

    # hyde
    hyde_p = subparsers.add_parser("hyde", help="HyDE search (hypothetical doc)")
    hyde_p.add_argument("query", help="Search query")
    hyde_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    hyde_p.add_argument("-s", "--session", help="Filter by session (auto-detected from --cwd if not specified)")
    hyde_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    hyde_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")

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

    # topics (deduplicated conceptual subjects)
    subparsers.add_parser("topics", help="List deduplicated topics")

    # spans (physical transcript sections)
    spans_p = subparsers.add_parser("spans", help="List transcript spans")
    spans_p.add_argument("-s", "--session", help="Filter by session")

    # merge-topics
    merge_topics_p = subparsers.add_parser("merge-topics", help="Merge two topics")
    merge_topics_p.add_argument("source", type=int, help="Source topic ID (will be deleted)")
    merge_topics_p.add_argument("target", type=int, help="Target topic ID")

    # review
    subparsers.add_parser("review", help="Review topics for quality issues")

    # rename-topic
    rename_p = subparsers.add_parser("rename-topic", help="Rename a topic (or suggest name with LLM)")
    rename_p.add_argument("id", type=int, help="Topic ID")
    rename_p.add_argument("--name", help="New name (if not provided, LLM suggests one)")
    rename_p.add_argument("--auto", action="store_true", help="Automatically apply LLM suggestion")

    # migrate-timestamps
    subparsers.add_parser("migrate-timestamps", help="Populate timestamps from transcripts for existing data")

    # progress
    progress_p = subparsers.add_parser("progress", help="Show indexing progress")
    progress_p.add_argument("file", help="Transcript file path")

    # questions
    questions_p = subparsers.add_parser("questions", help="List unanswered questions")
    questions_p.add_argument("-s", "--session", help="Filter by session")

    # get (retrieve idea with relations)
    get_p = subparsers.add_parser("get", help="Get idea by ID with relations")
    get_p.add_argument("id", type=int, help="Idea ID")

    # similar (find similar ideas)
    similar_p = subparsers.add_parser("similar", help="Find ideas similar to a given idea")
    similar_p.add_argument("id", type=int, help="Idea ID")
    similar_p.add_argument("-n", "--limit", type=int, default=5, help="Max results")
    similar_p.add_argument("-s", "--session", help="Filter by session")
    similar_p.add_argument("--same-session", dest="same_session", action="store_true", default=None,
                          help="Only same session")
    similar_p.add_argument("--other-sessions", dest="same_session", action="store_false",
                          help="Only other sessions")

    # prune (remove old ideas)
    prune_p = subparsers.add_parser("prune", help="Remove old ideas from database")
    prune_p.add_argument("-d", "--days", type=int, default=90, help="Remove ideas older than N days")
    prune_p.add_argument("-s", "--session", help="Only prune from specific session")
    prune_p.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")

    # Graph revision commands
    update_intent_p = subparsers.add_parser("update-intent", help="Update an idea's intent")
    update_intent_p.add_argument("id", type=int, help="Idea ID")
    update_intent_p.add_argument("intent", choices=["decision", "conclusion", "question", "problem", "solution", "todo", "context"])

    move_idea_p = subparsers.add_parser("move-idea", help="Move an idea to a different span")
    move_idea_p.add_argument("id", type=int, help="Idea ID")
    move_idea_p.add_argument("span_id", type=int, help="Target span ID")

    merge_spans_p = subparsers.add_parser("merge-spans", help="Merge one span into another")
    merge_spans_p.add_argument("source", type=int, help="Source span ID (will be deleted)")
    merge_spans_p.add_argument("target", type=int, help="Target span ID")

    supersede_p = subparsers.add_parser("supersede", help="Mark one idea as superseding another")
    supersede_p.add_argument("old_id", type=int, help="Old idea ID (being superseded)")
    supersede_p.add_argument("new_id", type=int, help="New idea ID (superseding)")

    # export
    export_p = subparsers.add_parser("export", help="Export data as JSON")
    export_p.add_argument("-s", "--session", help="Only export specific session")
    export_p.add_argument("-o", "--output", help="Output file path")

    # import
    import_p = subparsers.add_parser("import", help="Import data from JSON backup")
    import_p.add_argument("file", help="JSON file to import")
    import_p.add_argument("--replace", action="store_true", help="Replace existing data (default: merge)")

    # context
    context_p = subparsers.add_parser("context", help="Show source transcript context for an idea")
    context_p.add_argument("id", type=int, help="Idea ID")
    context_p.add_argument("-B", "--before", type=int, default=5, help="Lines before (default: 5)")
    context_p.add_argument("-A", "--after", type=int, default=5, help="Lines after (default: 5)")

    # sessions
    subparsers.add_parser("sessions", help="List all indexed sessions with stats")

    # Project commands
    subparsers.add_parser("projects", help="List all projects")

    create_project_p = subparsers.add_parser("create-project", help="Create a new project")
    create_project_p.add_argument("name", help="Project name")
    create_project_p.add_argument("-d", "--description", help="Project description")

    assign_topic_p = subparsers.add_parser("assign-topic", help="Assign a topic to a project")
    assign_topic_p.add_argument("topic_id", type=int, help="Topic ID")
    assign_topic_p.add_argument("project", help="Project name or ID")

    unparent_p = subparsers.add_parser("unparent-topic", help="Remove a topic from its parent (make it top-level)")
    unparent_p.add_argument("topic_id", type=int, help="Topic ID")

    delete_topic_p = subparsers.add_parser("delete-topic", help="Delete a topic and optionally its ideas")
    delete_topic_p.add_argument("topic_id", type=int, help="Topic ID to delete")
    delete_topic_p.add_argument("--delete-ideas", action="store_true", help="Also delete all ideas in this topic")

    subparsers.add_parser("tree", help="Show project -> topic hierarchy")

    # Review ideas against current filters
    review_ideas_p = subparsers.add_parser("review-ideas", help="Review ideas against current filters")
    review_ideas_p.add_argument("-t", "--topic", type=int, help="Only review specific topic ID")
    review_ideas_p.add_argument("--execute", action="store_true", help="Actually delete filtered ideas (default is dry-run)")

    # Auto-categorize topics
    auto_cat_p = subparsers.add_parser("auto-categorize", help="Auto-assign unassigned topics to projects")
    auto_cat_p.add_argument("--execute", action="store_true", help="Actually assign (default is dry-run)")

    # Improve all categorization (auto-categorize + rename bad topics)
    subparsers.add_parser("improve", help="Run all categorization improvements")

    # LLM-based filtering for subtle low-value content
    llm_filter_p = subparsers.add_parser("llm-filter", help="Use LLM to identify low-value ideas regex can't catch")
    llm_filter_p.add_argument("-t", "--topic", type=int, help="Only filter specific topic ID")
    llm_filter_p.add_argument("-b", "--batch", type=int, default=20, help="Batch size for LLM calls")
    llm_filter_p.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")

    # Clustering analysis
    cluster_p = subparsers.add_parser("cluster", help="Analyze topic coherence and suggest reorganization")
    cluster_p.add_argument("-m", "--min-size", type=int, default=5, help="Min ideas for split candidate")

    # Recluster a specific topic
    recluster_p = subparsers.add_parser("recluster", help="Analyze a topic and suggest how to split it")
    recluster_p.add_argument("topic_id", type=int, help="Topic ID to analyze")
    recluster_p.add_argument("-n", "--clusters", type=int, help="Number of clusters (auto if not specified)")

    # Split a topic based on clustering
    split_p = subparsers.add_parser("split-topic", help="Split a topic into sub-topics based on clustering")
    split_p.add_argument("topic_id", type=int, help="Topic ID to split")
    split_p.add_argument("-n", "--clusters", type=int, help="Number of clusters (auto if not specified)")
    split_p.add_argument("-m", "--min-size", type=int, default=3, help="Min ideas for a sub-topic")
    split_p.add_argument("--keep-junk", action="store_true", help="Keep ideas in tiny clusters (default: delete)")

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
