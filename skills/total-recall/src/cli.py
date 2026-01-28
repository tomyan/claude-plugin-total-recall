#!/usr/bin/env python3
"""Unified CLI for total-recall memory system.

Single entry point that handles bootstrap internally.
Usage: total-recall <command> [args]
"""

import argparse
import asyncio
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

RUNTIME_DIR = Path.home() / ".claude-plugin-total-recall"


def safe_result(data: Any) -> dict:
    """Wrap successful result."""
    return {"success": True, "data": data}


def error_result(error: str, code: str = "unknown", details: dict = None) -> dict:
    """Wrap error result."""
    result = {"success": False, "error": error, "error_code": code}
    if details:
        result["details"] = details

    # Print critical errors prominently (both stdout and stderr)
    if code == "missing_api_key":
        from config import OPENAI_KEY_FILE
        msg = f"""
================================================================================
❌ CONFIGURATION ERROR: {error}

To fix this:
  mkdir -p ~/.config/total-recall
  echo 'your-openai-api-key' > {OPENAI_KEY_FILE}

Then retry your search.
================================================================================
"""
        print(msg)
        print(msg, file=sys.stderr)

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
        session = asyncio.run(memory_db.get_session_for_cwd(cwd))
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
    until: str = None,
    recent: str = None,
    auto_analyze: bool = True,
    include_forgotten: bool = False,
    show_originals: bool = False,
    show_decomposition: bool = False,
    verify: bool = False,
    explain: bool = False
) -> dict:
    """Run search command with error handling.

    Args:
        auto_analyze: If True, uses analyze_query() to detect temporal/intent
                     filters from natural language (unless explicitly provided)
        include_forgotten: If True, include forgotten ideas in results
        show_originals: If True, show original ideas that were consolidated
        show_decomposition: If True, show how query was decomposed
        verify: If True, use LLM to verify result relevance
        explain: If True, include LLM explanation for relevance
    """
    try:
        import memory_db

        # Auto-analyze query for temporal/intent filters if not explicitly provided
        detected = {}
        if auto_analyze:
            detected = asyncio.run(memory_db.analyze_query(query))
            # Only use detected filters if not explicitly provided
            if not recent and not since and not until:
                recent = detected.get("temporal")
            if not intent:
                intent = detected.get("intent_filter")

        # Check if query should be decomposed
        decomp = asyncio.run(memory_db.decompose_query(query))
        use_decomposed = decomp["type"] != "simple" or show_decomposition

        if since or until or recent:
            # Use temporal search when date filters provided
            results = asyncio.run(memory_db.search_ideas_temporal(
                query, limit=limit, since=since, until=until,
                relative=recent, session=session,
                include_forgotten=include_forgotten
            ))
            # Apply intent filter if specified
            if intent:
                results = [r for r in results if r.get('intent') == intent]
        elif use_decomposed:
            # Use decomposed search for complex queries
            decomp_result = asyncio.run(memory_db.decomposed_search(
                query, limit=limit, session=session,
                show_decomposition=show_decomposition
            ))
            results = decomp_result["results"]
            if decomp_result.get("decomposition"):
                detected["decomposition"] = decomp_result["decomposition"]
        else:
            results = asyncio.run(memory_db.search_ideas(
                query, limit=limit, session=session, intent=intent,
                include_forgotten=include_forgotten,
                show_originals=show_originals
            ))

        # Apply LLM relevance verification if requested
        if verify or explain:
            results = asyncio.run(memory_db.verify_relevance(
                query, results, threshold=3, explain=explain
            ))
            detected["verified"] = True

        # Add metadata about detected filters
        return safe_result({
            "results": results,
            "detected_filters": detected if detected else None
        })
    except memory_db.TotalRecallError as e:
        return error_result(str(e), e.error_code, e.details)
    except Exception as e:
        return error_result(f"Search failed: {e}", "search_error")


def run_stats_command() -> dict:
    """Run stats command with error handling."""
    try:
        import memory_db
        stats = asyncio.run(memory_db.get_stats())
        return safe_result(stats)
    except memory_db.TotalRecallError as e:
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
            ["uv", "init", "--name", "total-recall", "--no-readme"],
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
            asyncio.run(memory_db.init_db())
            print(json.dumps({"success": True, "message": "Database initialized"}))

        elif args.command == "search":
            # Resolve session from args (--global, --session, --cwd)
            session = resolve_session(args)
            # --when takes precedence over --recent
            recent = args.when if args.when else args.recent
            # --after-session sets since to the end of that session
            since = args.since
            if args.after_session:
                session_range = asyncio.run(memory_db.get_session_time_range(args.after_session))
                if session_range:
                    since = session_range['end_time']
                    print(f"# Searching after session '{args.after_session}' (since {since[:10]})", file=sys.stderr)
                else:
                    print(f"# Warning: Session '{args.after_session}' not found", file=sys.stderr)
            result = run_search_command(
                args.query,
                limit=args.limit,
                session=session,
                intent=args.intent,
                since=since,
                until=args.until,
                recent=recent,
                include_forgotten=getattr(args, 'include_forgotten', False),
                show_originals=getattr(args, 'show_originals', False),
                show_decomposition=getattr(args, 'decompose', False),
                verify=getattr(args, 'verify', False),
                explain=getattr(args, 'explain', False)
            )
            if result["success"]:
                data = result["data"]
                results = data.get("results", data) if isinstance(data, dict) else data
                detected = data.get("detected_filters") if isinstance(data, dict) else None

                # Show session and detected filters for transparency
                if session:
                    print(f"# Searching project: {session}", file=sys.stderr)
                if detected:
                    filters = []
                    if detected.get("temporal"):
                        filters.append(f"temporal={detected['temporal']}")
                    if detected.get("intent_filter"):
                        filters.append(f"intent={detected['intent_filter']}")
                    if filters:
                        print(f"# Detected filters: {', '.join(filters)}", file=sys.stderr)

                print(json.dumps(results, indent=2, default=str))
            else:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

        elif args.command == "hybrid":
            # Resolve session from args
            session = resolve_session(args)
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            # Resolve temporal filters (--when takes precedence over --recent)
            since, until = args.since, args.until
            time_filter = args.when if args.when else args.recent
            if time_filter:
                since, until = asyncio.run(memory_db.resolve_temporal_qualifier(time_filter))
                print(f"# Time filter: {time_filter} ({since[:10]} to {until[:10]})", file=sys.stderr)
            results = asyncio.run(memory_db.hybrid_search(
                args.query, limit=args.limit, session=session,
                since=since, until=until
            ))
            # Apply intent filter if specified
            intent = getattr(args, 'intent', None)
            if intent:
                results = [r for r in results if r.get('intent') == intent]
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "hyde":
            # Resolve session from args
            session = resolve_session(args)
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            # Resolve temporal filters (--when takes precedence over --recent)
            since, until = args.since, args.until
            time_filter = args.when if args.when else args.recent
            if time_filter:
                since, until = asyncio.run(memory_db.resolve_temporal_qualifier(time_filter))
                print(f"# Time filter: {time_filter} ({since[:10]} to {until[:10]})", file=sys.stderr)
            results = asyncio.run(memory_db.hyde_search(
                args.query, limit=args.limit, session=session,
                since=since, until=until
            ))
            # Apply intent filter if specified
            intent = getattr(args, 'intent', None)
            if intent:
                results = [r for r in results if r.get('intent') == intent]
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "multi-search":
            # Parallel multi-strategy search using asyncio

            # Parse strategies
            strategies = [s.strip() for s in args.strategies.split(',')]
            valid_strategies = {'hybrid', 'hyde', 'semantic', 'decisions', 'todos', 'questions', 'problems', 'solutions'}
            invalid = set(strategies) - valid_strategies
            if invalid:
                print(json.dumps(error_result(f"Invalid strategies: {invalid}", "invalid_strategy")))
                sys.exit(1)

            # Resolve session (--local means use cwd, otherwise global)
            session = None
            if args.local:
                if args.cwd:
                    session = asyncio.run(memory_db.get_session_for_cwd(args.cwd))
                    if session:
                        print(f"# Scoped to project: {session}", file=sys.stderr)
                    else:
                        print("# Warning: --local specified but couldn't detect project from cwd", file=sys.stderr)

            # Resolve temporal filters
            since, until = args.since, args.until
            time_filter = args.when if args.when else args.recent
            if time_filter:
                since, until = asyncio.run(memory_db.resolve_temporal_qualifier(time_filter))
                print(f"# Time filter: {time_filter} ({since[:10]} to {until[:10]})", file=sys.stderr)

            print(f"# Running strategies: {', '.join(strategies)}", file=sys.stderr)

            # Map intent-based strategies to their filter values
            intent_map = {
                'decisions': 'decision',
                'todos': 'todo',
                'questions': 'question',
                'problems': 'problem',
                'solutions': 'solution',
            }

            async def run_strategy(strategy: str) -> tuple[str, list[dict]]:
                """Run a single search strategy and return (strategy_name, results)."""
                try:
                    if strategy == 'hybrid':
                        results = await memory_db.hybrid_search(
                            args.query, limit=args.limit, session=session,
                            since=since, until=until
                        )
                    elif strategy == 'hyde':
                        results = await memory_db.hyde_search(
                            args.query, limit=args.limit, session=session,
                            since=since, until=until
                        )
                    elif strategy == 'semantic':
                        results = await memory_db.search_ideas(
                            args.query, limit=args.limit, session=session
                        )
                        # Apply temporal filters manually for semantic search
                        if since:
                            results = [r for r in results if r.get('message_time', r.get('created_at', '')) >= since]
                        if until:
                            results = [r for r in results if r.get('message_time', r.get('created_at', '')) <= until]
                    elif strategy in intent_map:
                        # Intent-filtered search using hybrid as base
                        results = await memory_db.hybrid_search(
                            args.query, limit=args.limit, session=session,
                            since=since, until=until
                        )
                        results = [r for r in results if r.get('intent') == intent_map[strategy]]
                    else:
                        results = []
                    return (strategy, results)
                except Exception as e:
                    print(f"# Strategy {strategy} failed: {e}", file=sys.stderr)
                    return (strategy, [])

            async def run_all_strategies():
                tasks = [run_strategy(s) for s in strategies]
                return await asyncio.gather(*tasks)

            # Run all strategies in parallel
            all_results = {}
            seen_ids = set()
            combined = []

            strategy_results = asyncio.run(run_all_strategies())
            for strategy, results in strategy_results:
                all_results[strategy] = len(results)
                for r in results:
                    if r['id'] not in seen_ids:
                        seen_ids.add(r['id'])
                        r['_strategy'] = strategy  # Track which strategy found it
                        combined.append(r)

            # Sort by distance (if available) and limit
            combined.sort(key=lambda x: x.get('distance', 999))
            combined = combined[:args.limit]

            # Report strategy contributions
            for strategy, count in all_results.items():
                print(f"#   {strategy}: {count} results", file=sys.stderr)
            print(f"# Combined: {len(combined)} unique results", file=sys.stderr)

            print(json.dumps(combined, indent=2, default=str))

        elif args.command == "expand":
            # Search with topic expansion
            session = resolve_session(args)
            if session:
                print(f"# Primary session: {session}", file=sys.stderr)
            results = asyncio.run(memory_db.search_with_topic_expansion(
                args.query,
                limit=args.limit,
                session=session,
                expand_limit=args.expand_limit
            ))
            # Format output
            print(f"# Primary results: {len(results['primary_results'])}", file=sys.stderr)
            if results['linked_results']:
                linked_count = sum(len(v) for v in results['linked_results'].values())
                print(f"# Linked results: {linked_count} from {len(results['linked_results'])} sessions", file=sys.stderr)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "topic-links":
            links = asyncio.run(memory_db.get_topic_links(args.topic_id))
            print(json.dumps(links, indent=2, default=str))

        elif args.command == "link-topics":
            link_id = asyncio.run(memory_db.link_topics(
                args.topic_id,
                args.related_id,
                similarity=args.similarity,
                link_type='manual'
            ))
            print(json.dumps({"link_id": link_id, "success": True}))

        elif args.command == "auto-link":
            # Get all topics and auto-link them
            async def do_auto_link():
                topics = await memory_db.list_topics()
                all_links = []
                for topic in topics:
                    # Get the session for this topic
                    from db.async_connection import get_async_db
                    db = await get_async_db()
                    try:
                        cursor = await db.execute(
                            "SELECT DISTINCT session FROM spans WHERE topic_id = ?",
                            (topic['id'],)
                        )
                        rows = await cursor.fetchall()
                        sessions = [r['session'] for r in rows]
                    finally:
                        await db.close()

                    if not sessions:
                        continue

                    # Find related topics (excluding this topic's sessions)
                    related = await memory_db.find_related_topics(
                        topic['id'],
                        exclude_sessions=sessions,
                        min_similarity=args.min_similarity
                    )

                    for rel in related:
                        link_info = {
                            "topic_id": topic['id'],
                            "topic_name": topic['name'],
                            "related_id": rel['id'],
                            "related_name": rel['name'],
                            "similarity": rel['similarity'],
                            "related_session": rel.get('session')
                        }

                        if args.dry_run:
                            all_links.append(link_info)
                        else:
                            link_id = await memory_db.link_topics(
                                topic['id'], rel['id'],
                                similarity=rel['similarity'],
                                link_type='semantic'
                            )
                            link_info['link_id'] = link_id
                            all_links.append(link_info)

                return all_links

            all_links = asyncio.run(do_auto_link())
            print(json.dumps({
                "dry_run": args.dry_run,
                "links_found": len(all_links),
                "links": all_links
            }, indent=2, default=str))

        elif args.command == "refresh-embeddings":
            # Find spans without embeddings
            async def do_refresh():
                from db.async_connection import get_async_db
                db = await get_async_db()
                try:
                    sql = """
                        SELECT s.id, s.name, s.session
                        FROM spans s
                        LEFT JOIN span_embeddings se ON se.span_id = s.id
                        WHERE se.span_id IS NULL
                    """
                    params = []
                    if args.session:
                        sql += " AND s.session = ?"
                        params.append(args.session)

                    cursor = await db.execute(sql, params)
                    spans_to_update = [dict(row) for row in await cursor.fetchall()]
                finally:
                    await db.close()

                print(f"Found {len(spans_to_update)} spans without embeddings", file=sys.stderr)

                updated = 0
                for span in spans_to_update:
                    try:
                        await memory_db.update_span_embedding(span['id'])
                        updated += 1
                        print(f"  Updated: {span['name'][:50]}... ({span['session']})", file=sys.stderr)
                    except Exception as e:
                        print(f"  Failed: {span['name'][:50]}... - {e}", file=sys.stderr)

                return {"spans_found": len(spans_to_update), "spans_updated": updated}

            result = asyncio.run(do_refresh())
            print(json.dumps(result, indent=2))

        elif args.command == "embed-ideas":
            # Find ideas without embeddings
            async def do_embed_ideas():
                from db.async_connection import get_async_db
                db = await get_async_db()
                try:
                    cursor = await db.execute("""
                        SELECT i.id, i.content
                        FROM ideas i
                        LEFT JOIN idea_embeddings ie ON ie.idea_id = i.id
                        WHERE ie.idea_id IS NULL
                    """)
                    ideas_to_embed = [dict(row) for row in await cursor.fetchall()]
                finally:
                    await db.close()

                print(f"Found {len(ideas_to_embed)} ideas without embeddings", file=sys.stderr)

                if not ideas_to_embed:
                    return {"ideas_found": 0, "ideas_embedded": 0}
                else:
                    from executor import embed_ideas
                    idea_ids = [idea['id'] for idea in ideas_to_embed]
                    embedded = embed_ideas(idea_ids)
                    print(f"Embedded {embedded} ideas", file=sys.stderr)
                    return {
                        "ideas_found": len(ideas_to_embed),
                        "ideas_embedded": embedded
                    }

            result = asyncio.run(do_embed_ideas())
            print(json.dumps(result, indent=2))

        elif args.command == "stats":
            result = run_stats_command()
            if result["success"]:
                print(json.dumps(result["data"], indent=2))
            else:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

        elif args.command == "cache-stats":
            from embeddings.cache import get_embedding_cache_stats, clear_embedding_cache
            if getattr(args, 'clear', False):
                asyncio.run(clear_embedding_cache())
                print(json.dumps({"success": True, "message": "Cache cleared"}))
            else:
                stats = asyncio.run(get_embedding_cache_stats())
                print(json.dumps(stats, indent=2, default=str))

        elif args.command == "backfill":
            # Check for API key FIRST - refuse to backfill without it
            from config import get_openai_api_key
            if not get_openai_api_key():
                print(error_result(
                    "OpenAI API key not found. "
                    "Backfill requires this for generating embeddings.",
                    "missing_api_key"
                ))
                sys.exit(1)

            from backfill import enqueue_file, enqueue_all, get_progress
            import subprocess

            # Enqueue files
            if args.all:
                result = enqueue_all()
                print(f"Enqueued {result['files_queued']} files ({result['files_already_queued']} already queued)")
            elif args.file:
                result = enqueue_file(args.file)
                if result.get("error"):
                    print(f"Error: {result['error']}", file=sys.stderr)
                    sys.exit(1)
                print(f"Enqueued: {result['file_path']} ({result.get('status', 'unknown')})")
            else:
                print("Usage: backfill <file> or backfill --all", file=sys.stderr)
                sys.exit(1)

            # Ensure daemon is running
            runtime_dir = Path.home() / ".claude-plugin-total-recall"
            pidfile = runtime_dir / "daemon.pid"
            daemon_running = False

            if pidfile.exists():
                try:
                    pid = int(pidfile.read_text().strip())
                    os.kill(pid, 0)  # Check if process exists
                    daemon_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    pidfile.unlink(missing_ok=True)

            if not daemon_running:
                skill_dir = Path.home() / ".claude" / "skills" / "total-recall"
                env = os.environ.copy()
                env["PYTHONPATH"] = str(skill_dir / "src")
                subprocess.Popen(
                    ["uv", "run", "python", "-u", str(skill_dir / "src" / "daemon.py")],
                    cwd=str(runtime_dir),
                    stdout=open(runtime_dir / "daemon.log", "a"),
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True
                )
                print("Started background daemon")

            # Show progress
            progress = get_progress()
            print(f"Queue: {progress['queue']['pending_files']} files pending")
            print(f"Indexed: {progress['indexed']['files']} files, {progress['indexed']['percent']}% complete")
            print("\nIndexing runs in background - check progress with: cli.py progress")

        elif args.command == "index":
            from indexer import index_transcript
            result = index_transcript(args.file, args.start_line)
            print(json.dumps(result))

        elif args.command == "topics":
            # List deduplicated topics
            topics = asyncio.run(memory_db.list_topics())
            print(json.dumps(topics, indent=2, default=str))

        elif args.command == "spans":
            # List physical transcript spans
            async def do_list_spans():
                from db.async_connection import get_async_db
                db = await get_async_db()
                try:
                    if args.session:
                        cursor = await db.execute("""
                            SELECT s.id, s.topic_id, s.session, s.name, s.summary,
                                   s.start_line, s.end_line, s.depth, t.name as topic_name
                            FROM spans s
                            LEFT JOIN topics t ON t.id = s.topic_id
                            WHERE s.session = ? ORDER BY s.start_line
                        """, (args.session,))
                    else:
                        cursor = await db.execute("""
                            SELECT s.id, s.topic_id, s.session, s.name, s.summary,
                                   s.start_line, s.end_line, s.depth, t.name as topic_name
                            FROM spans s
                            LEFT JOIN topics t ON t.id = s.topic_id
                            ORDER BY s.session, s.start_line
                        """)
                    return [dict(row) for row in await cursor.fetchall()]
                finally:
                    await db.close()

            results = asyncio.run(do_list_spans())
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "merge-topics":
            result = asyncio.run(memory_db.merge_topics(args.source, args.target))
            print(json.dumps({"success": True, **result}))

        elif args.command == "review":
            issues = asyncio.run(memory_db.review_topics())
            print(json.dumps(issues, indent=2, default=str))

        elif args.command == "rename-topic":
            if args.name:
                success = asyncio.run(memory_db.rename_topic(args.id, args.name))
            else:
                # Use LLM to suggest name
                suggested = asyncio.run(memory_db.suggest_topic_name(args.id))
                if suggested:
                    if args.auto:
                        success = asyncio.run(memory_db.rename_topic(args.id, suggested))
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
            result = asyncio.run(memory_db.migrate_timestamps_from_transcripts())
            print(json.dumps(result))

        elif args.command == "progress":
            from backfill import get_progress
            result = get_progress(args.file)
            print(json.dumps(result))

        elif args.command == "questions":
            questions = asyncio.run(memory_db.get_unanswered_questions(args.session))
            print(json.dumps(questions, indent=2, default=str))

        elif args.command == "decisions":
            # Shortcut for search --intent decision
            session = resolve_session(args)
            results = asyncio.run(memory_db.search_ideas(
                args.query or "",
                limit=args.limit,
                session=session,
                intent="decision"
            ))
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "todos":
            # Shortcut for search --intent todo
            session = resolve_session(args)
            results = asyncio.run(memory_db.search_ideas(
                args.query or "",
                limit=args.limit,
                session=session,
                intent="todo"
            ))
            if session:
                print(f"# Searching project: {session}", file=sys.stderr)
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "get":
            result = asyncio.run(memory_db.get_idea_with_relations(args.id))
            if result:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(json.dumps({"error": f"Idea {args.id} not found"}), file=sys.stderr)
                sys.exit(1)

        elif args.command == "similar":
            results = asyncio.run(memory_db.find_similar_ideas(
                args.id,
                limit=args.limit,
                same_session=args.same_session,
                session=args.session
            ))
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "trace":
            direction = "both"
            if args.backward and not args.forward:
                direction = "backward"
            elif args.forward and not args.backward:
                direction = "forward"
            result = asyncio.run(memory_db.trace_idea(
                args.id,
                direction=direction,
                hops=args.hops,
                limit=args.limit
            ))
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            # Pretty print trace
            idea = result["idea"]
            print(f"\nIdea {idea['id']}: {idea['content'][:80]}...")
            print(f"  Intent: {idea['intent']} | Topic: {idea.get('topic', 'N/A')}")

            for hop_num, hop_ideas in result["hops"].items():
                print(f"\n{'─' * 60}")
                print(f"Hop {hop_num}: {len(hop_ideas)} connected ideas")
                for h in hop_ideas:
                    arrow = "→" if h["direction"] == "forward" else "←"
                    print(f"  {arrow} [{h['id']:5}] [{h['relation']:12}] {h['content'][:60]}...")

        elif args.command == "path":
            result = asyncio.run(memory_db.find_path(
                args.from_id,
                args.to_id,
                max_hops=args.max_hops
            ))
            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            # Pretty print path
            print(f"\nPath found ({result['hops']} hops):\n")
            for i, idea in enumerate(result["path"]):
                if i > 0:
                    print("    ↓")
                print(f"  [{idea['id']:5}] {idea.get('content', '?')[:70]}...")
                if idea.get("intent"):
                    print(f"           Intent: {idea['intent']} | Topic: {idea.get('topic', 'N/A')}")

        elif args.command == "prune":
            result = asyncio.run(memory_db.prune_old_ideas(
                older_than_days=args.days,
                session=args.session,
                dry_run=not args.execute
            ))
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "update-intent":
            result = asyncio.run(memory_db.update_idea_intent(args.id, args.intent))
            print(json.dumps({"updated": result, "id": args.id, "intent": args.intent}))

        elif args.command == "move-idea":
            result = asyncio.run(memory_db.move_idea_to_span(args.id, args.span_id))
            print(json.dumps({"moved": result, "idea_id": args.id, "span_id": args.span_id}))

        elif args.command == "merge-spans":
            result = asyncio.run(memory_db.merge_spans(args.source, args.target))
            print(json.dumps(result, indent=2))

        elif args.command == "supersede":
            asyncio.run(memory_db.supersede_idea(args.old_id, args.new_id))
            print(json.dumps({"success": True, "old_id": args.old_id, "new_id": args.new_id}))

        elif args.command == "export":
            data = asyncio.run(memory_db.export_data(session=args.session))
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(json.dumps({"success": True, "output": args.output, "stats": data["stats"]}))
            else:
                print(json.dumps(data, indent=2, default=str))

        elif args.command == "import":
            with open(args.file, 'r') as f:
                data = json.load(f)
            result = asyncio.run(memory_db.import_data(data, replace=args.replace))
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "context":
            result = asyncio.run(memory_db.get_context(
                args.id,
                lines_before=args.before,
                lines_after=args.after
            ))
            print(json.dumps(result, indent=2, default=str))

        elif args.command == "sessions":
            sessions = asyncio.run(memory_db.list_sessions())
            print(json.dumps(sessions, indent=2, default=str))

        elif args.command == "projects":
            projects = asyncio.run(memory_db.list_projects())
            print(json.dumps(projects, indent=2, default=str))

        elif args.command == "create-project":
            project_id = asyncio.run(memory_db.create_project(args.name, args.description))
            print(json.dumps({"success": True, "id": project_id, "name": args.name}))

        elif args.command == "assign-topic":
            # Try to parse project as ID first, then lookup by name
            try:
                project_id = int(args.project)
            except ValueError:
                project = asyncio.run(memory_db.get_project_by_name(args.project))
                if not project:
                    print(json.dumps({"error": f"Project '{args.project}' not found"}), file=sys.stderr)
                    sys.exit(1)
                project_id = project["id"]
            asyncio.run(memory_db.assign_topic_to_project(args.topic_id, project_id))
            print(json.dumps({"success": True, "topic_id": args.topic_id, "project_id": project_id}))

        elif args.command == "unparent-topic":
            asyncio.run(memory_db.unparent_topic(args.topic_id))
            print(json.dumps({"success": True, "topic_id": args.topic_id, "parent_id": None}))

        elif args.command == "reparent-topic":
            asyncio.run(memory_db.reparent_topic(args.topic_id, args.parent_id))
            print(json.dumps({"success": True, "topic_id": args.topic_id, "parent_id": args.parent_id}))

        elif args.command == "delete-topic":
            result = asyncio.run(memory_db.delete_topic(args.topic_id, delete_ideas=args.delete_ideas))
            print(f"Deleted topic {args.topic_id}")
            print(f"  Ideas deleted: {result['ideas_deleted']}")
            print(f"  Spans deleted: {result['spans_deleted']}")

        elif args.command == "tree":
            tree = asyncio.run(memory_db.get_project_tree())

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
            result = asyncio.run(memory_db.review_ideas_against_filters(
                topic_id=args.topic,
                dry_run=not args.execute
            ))
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
            result = asyncio.run(memory_db.auto_categorize_topics(dry_run=not args.execute))
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
            result = asyncio.run(memory_db.improve_categorization())
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
            result = asyncio.run(memory_db.llm_filter_ideas(
                topic_id=args.topic,
                batch_size=args.batch,
                dry_run=not args.execute
            ))
            print(f"Reviewed: {result['total_reviewed']} ideas")
            print(f"Flagged for removal: {result['flagged']}")
            if result.get('samples'):
                print(f"\nSamples flagged (showing first {len(result['samples'])}):")
                for s in result['samples']:
                    print(f"  [{s['reason']}] {s['content'][:70]}...")
            if not result.get('dry_run'):
                print(f"\nRemoved: {result.get('removed', 0)} ideas")

        elif args.command == "cluster":
            result = asyncio.run(memory_db.cluster_topics(min_cluster_size=args.min_size))
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
            result = asyncio.run(memory_db.recluster_topic(args.topic_id, num_clusters=args.clusters))
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
            result = asyncio.run(memory_db.split_topic(
                args.topic_id,
                num_clusters=args.clusters,
                min_cluster_size=args.min_size,
                delete_junk=not args.keep_junk
            ))
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

        elif args.command == "timeline":
            from datetime import datetime

            if args.topic:
                # Topic timeline
                result = asyncio.run(memory_db.get_topic_timeline(topic_name=args.topic))
                if "error" in result:
                    print(json.dumps(result), file=sys.stderr)
                    sys.exit(1)

                topic = result['topic']
                print(f"\nTopic: {topic['name']}")
                if topic.get('summary'):
                    print(f"  {topic['summary']}")
                print(f"\n  First seen: {topic.get('first_seen', 'unknown')}")
                print(f"  Last seen: {topic.get('last_seen', 'unknown')}")
                print(f"  Total: {result['total_spans']} spans, {result['total_ideas']} ideas")
                print()

                # Show timeline by date
                for date_key in sorted(result['timeline'].keys(), reverse=True):
                    spans = result['timeline'][date_key]
                    try:
                        date_display = datetime.strptime(date_key, '%Y-%m-%d').strftime('%b %d, %Y')
                    except:
                        date_display = date_key
                    print(f"  {date_display}")

                    for span in spans:
                        time_range = ""
                        if span.get('start_time') and span.get('end_time'):
                            try:
                                start = datetime.fromisoformat(span['start_time'].replace('Z', '+00:00'))
                                end = datetime.fromisoformat(span['end_time'].replace('Z', '+00:00'))
                                time_range = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                            except:
                                pass
                        session_info = f"Session: {span['session']}"
                        print(f"    {time_range:12} {session_info} ({span['idea_count']} ideas)")

                        # Show key ideas
                        for idea in span.get('key_ideas', []):
                            prefix = "→" if idea['intent'] == 'decision' else "•"
                            content = idea['content'][:70] + "..." if len(idea['content']) > 70 else idea['content']
                            print(f"      {prefix} {content}")
                    print()

            elif args.project:
                # Project timeline
                session = args.project
                if args.cwd:
                    session = asyncio.run(memory_db.get_session_for_cwd(args.cwd)) or args.project

                result = asyncio.run(memory_db.get_project_timeline(session, days=args.days))
                if "error" in result:
                    print(json.dumps(result), file=sys.stderr)
                    sys.exit(1)

                print(f"\nProject: {result['session']} (last {result['days']} days)")
                print(f"  Total: {result['total_spans']} spans, {result['total_ideas']} ideas")
                print()

                # Show activity by date
                for date_key in sorted(result['timeline'].keys(), reverse=True):
                    day_data = result['timeline'][date_key]
                    try:
                        date_display = datetime.strptime(date_key, '%Y-%m-%d').strftime('%b %d')
                    except:
                        date_display = date_key

                    topics = ", ".join(day_data['topics'][:5])
                    if len(day_data['topics']) > 5:
                        topics += f" (+{len(day_data['topics']) - 5} more)"
                    print(f"  {date_display}: {topics}")
                print()

            else:
                print("Error: Specify --topic or --project", file=sys.stderr)
                sys.exit(1)

        elif args.command == "activity":
            # Resolve session from args
            session = resolve_session(args)
            result = asyncio.run(memory_db.get_activity_by_period(
                period=args.by,
                days=args.days,
                session=session
            ))

            print(f"\nActivity by {args.by} (last {args.days} days)")
            if session:
                print(f"  Project: {session}")
            print(f"  Total ideas: {result['total_ideas']}")
            print()

            # Show by period
            for period_key in sorted(result['by_period'].keys(), reverse=True):
                data = result['by_period'][period_key]
                bar = "█" * min(data['total'] // 5, 20)
                extras = []
                if data['decisions']:
                    extras.append(f"{data['decisions']}D")
                if data['conclusions']:
                    extras.append(f"{data['conclusions']}C")
                if data['questions']:
                    extras.append(f"{data['questions']}Q")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                sessions_str = f" [{data['session_count']} sessions]" if data['session_count'] > 1 else ""
                print(f"  {period_key}: {data['total']:4d} {bar}{extra_str}{sessions_str}")

            print()
            print("Intent breakdown:")
            for intent, count in sorted(result['by_intent'].items(), key=lambda x: -x[1]):
                print(f"  {intent}: {count}")

        elif args.command == "forget":
            success = asyncio.run(memory_db.forget_idea(args.id))
            if success:
                print(json.dumps({"success": True, "id": args.id, "status": "forgotten"}))
            else:
                print(json.dumps({"success": False, "error": f"Idea {args.id} not found"}), file=sys.stderr)
                sys.exit(1)

        elif args.command == "unforget":
            success = asyncio.run(memory_db.unforget_idea(args.id))
            if success:
                print(json.dumps({"success": True, "id": args.id, "status": "restored"}))
            else:
                print(json.dumps({"success": False, "error": f"Idea {args.id} not found"}), file=sys.stderr)
                sys.exit(1)

        elif args.command == "forgotten":
            results = asyncio.run(memory_db.get_forgotten_ideas(limit=args.limit))
            print(json.dumps(results, indent=2, default=str))

        elif args.command == "forgettable":
            result = asyncio.run(memory_db.auto_forget_ideas(
                threshold=args.threshold,
                limit=args.limit,
                session=args.session,
                dry_run=not args.execute
            ))
            print(f"Retention threshold: {result['threshold']}")
            print(f"Found {result['candidates']} forgettable ideas")
            if result['samples']:
                print(f"\nSamples (lowest retention first):")
                for s in result['samples']:
                    score_bar = "█" * int(s['retention_score'] * 10)
                    print(f"  [{s['id']:5}] {s['retention_score']:.2f} {score_bar:10} [{s['intent']:10}] {s['content'][:50]}...")
            if not result['dry_run']:
                print(f"\nForgotten: {result['forgotten']} ideas")
            else:
                print(f"\nDry run - use --execute to actually forget")

        elif args.command == "consolidatable":
            candidates = asyncio.run(memory_db.get_consolidation_candidates(
                session=args.session,
                min_ideas=args.min_ideas,
                max_age_days=args.max_age
            ))
            if not candidates:
                print("No topics ready for consolidation")
            else:
                print(f"Topics ready for consolidation ({len(candidates)}):\n")
                for c in candidates:
                    print(f"  Topic {c['topic_id']:4}: {c['topic_name'][:40]:40} ({c['candidate_count']} ideas)")
                print(f"\nRun: consolidate <topic_id> to consolidate a topic")

        elif args.command == "consolidate":
            result = asyncio.run(memory_db.consolidate_topic(
                topic_id=args.topic_id,
                min_ideas=args.min_ideas,
                max_age_days=args.max_age,
                dry_run=not args.execute
            ))
            print(f"Topic: {result.get('topic_name', result['topic_id'])}")
            print(f"Consolidatable ideas: {result['candidates']}")
            if result.get('samples'):
                print(f"\nSample ideas to consolidate:")
                for s in result['samples']:
                    print(f"  [{s['id']:5}] [{s['intent']:10}] {s['content'][:60]}...")
            if result.get('error'):
                print(f"\nError: {result['error']}")
            elif not result['dry_run'] and result.get('summary_id'):
                print(f"\nConsolidated {result['consolidated']} ideas")
                print(f"Summary idea ID: {result['summary_id']}")
                print(f"Summary: {result.get('summary', '')[:150]}...")
            elif result['dry_run']:
                print(f"\nDry run - use --execute to consolidate")

        elif args.command == "reflect":
            result = asyncio.run(memory_db.generate_session_reflection(
                session=resolve_session(args),
                days=args.days
            ))
            if result.get("error"):
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

            print(f"\nSession Reflection ({result['days']} days, {result['ideas_count']} ideas)")
            print("=" * 60)
            if result.get("by_intent"):
                print("Activity by intent:", ", ".join(f"{k}:{v}" for k, v in result["by_intent"].items()))
            print()
            print(result["reflection"])

            if args.store:
                idea_id = asyncio.run(memory_db.store_reflection(result["reflection"], session=resolve_session(args)))
                print(f"\nStored as reflection idea #{idea_id}")
                result["stored"] = True
                result["idea_id"] = idea_id

        elif args.command == "reflect-topic":
            result = asyncio.run(memory_db.reflect_on_topic(args.topic_id))
            if result.get("error"):
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

            print(f"\nTopic Reflection: {result['topic_name']}")
            print("=" * 60)
            print(f"Period: {result.get('first_seen', '?')[:10]} to {result.get('last_seen', '?')[:10]}")
            print(f"Ideas: {result['ideas_count']}")
            print()
            print(result["reflection"])

        elif args.command == "insights":
            results = asyncio.run(memory_db.get_reflections(
                session=resolve_session(args),
                limit=args.limit
            ))
            if not results:
                print("No reflections found")
            else:
                print(f"\nReflections ({len(results)}):\n")
                for r in results:
                    time_str = (r.get("created_at") or "")[:10]
                    session_str = r.get("session") or "global"
                    print(f"[{r['id']:5}] [{time_str}] {session_str}")
                    print(f"  {r['content'][:200]}...")
                    print()

        elif args.command == "topic-activity":
            result = asyncio.run(memory_db.get_topic_activity(
                topic_id=args.topic_id,
                period=args.by,
                days=args.days
            ))

            if "error" in result:
                print(json.dumps(result), file=sys.stderr)
                sys.exit(1)

            topic = result['topic']
            print(f"\nTopic: {topic['name']}")
            if topic.get('summary'):
                print(f"  {topic['summary']}")
            print(f"\n  Activity by {args.by} (last {args.days} days)")
            print(f"  Total ideas: {result['total_ideas']}")
            print()

            # Show by period
            for period_key in sorted(result['by_period'].keys(), reverse=True):
                data = result['by_period'][period_key]
                bar = "█" * min(data['total'] // 2, 20)
                sessions = ", ".join(data['sessions'][:2])
                if len(data['sessions']) > 2:
                    sessions += f" (+{len(data['sessions']) - 2})"
                print(f"  {period_key}: {data['total']:3d} {bar}  [{sessions}]")

                # Show key ideas
                for idea in data.get('key_ideas', []):
                    prefix = "→" if idea['intent'] == 'decision' else "•"
                    print(f"    {prefix} {idea['content']}")
            print()

    except memory_db.TotalRecallError as e:
        print(json.dumps(e.to_dict()), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e), "error_code": "unexpected"}), file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="total-recall",
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
    search_p.add_argument("--recent", help="Relative time filter (e.g. 1d, 1w, 1m, 3m, 1y)")
    search_p.add_argument("--when", help="Natural language time (e.g. 'last week', 'since tuesday', 'since jan 5')")
    search_p.add_argument("--after-session", help="Only ideas after this session ended (session name)")
    search_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    search_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")
    search_p.add_argument("--include-forgotten", action="store_true",
                          help="Include forgotten ideas in results")
    search_p.add_argument("--show-originals", action="store_true",
                          help="Show original ideas that were consolidated")
    search_p.add_argument("--decompose", action="store_true",
                          help="Show how query was decomposed (for complex queries)")
    search_p.add_argument("--verify", action="store_true",
                          help="Use LLM to verify and score result relevance")
    search_p.add_argument("--explain", action="store_true",
                          help="Include LLM explanation for relevance scores (implies --verify)")

    # hybrid
    hybrid_p = subparsers.add_parser("hybrid", help="Hybrid vector+keyword search")
    hybrid_p.add_argument("query", help="Search query")
    hybrid_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    hybrid_p.add_argument("-s", "--session", help="Filter by session (auto-detected from --cwd if not specified)")
    hybrid_p.add_argument("-i", "--intent", help="Filter by intent")
    hybrid_p.add_argument("--since", help="Only ideas after this date (ISO format)")
    hybrid_p.add_argument("--until", help="Only ideas before this date (ISO format)")
    hybrid_p.add_argument("--recent", help="Relative time filter (e.g. 1d, 1w, 1m, 3m, 1y)")
    hybrid_p.add_argument("--when", help="Natural language time (e.g. 'last week', 'since tuesday', 'since jan 5')")
    hybrid_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    hybrid_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")

    # hyde
    hyde_p = subparsers.add_parser("hyde", help="HyDE search (hypothetical doc)")
    hyde_p.add_argument("query", help="Search query")
    hyde_p.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    hyde_p.add_argument("-s", "--session", help="Filter by session (auto-detected from --cwd if not specified)")
    hyde_p.add_argument("-i", "--intent", help="Filter by intent")
    hyde_p.add_argument("--since", help="Only ideas after this date (ISO format)")
    hyde_p.add_argument("--until", help="Only ideas before this date (ISO format)")
    hyde_p.add_argument("--recent", help="Relative time filter (e.g. 1d, 1w, 1m, 3m, 1y)")
    hyde_p.add_argument("--when", help="Natural language time (e.g. 'last week', 'since tuesday', 'since jan 5')")
    hyde_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    hyde_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                          help="Search across all projects (default: current project only)")

    # multi-search (parallel strategy search - LLM selects strategies)
    multi_p = subparsers.add_parser("multi-search", help="Parallel multi-strategy search (LLM selects strategies)")
    multi_p.add_argument("query", help="Search query")
    multi_p.add_argument("--strategies", "-S", default="hybrid",
                         help="Comma-separated strategies: hybrid,hyde,semantic,decisions,todos,questions,problems,solutions")
    multi_p.add_argument("-n", "--limit", type=int, default=15, help="Max total results")
    multi_p.add_argument("--local", "-l", action="store_true", help="Scope to current project (default: global)")
    multi_p.add_argument("--cwd", help="Current working directory (for --local project detection)")
    multi_p.add_argument("--since", help="Only ideas after this date (ISO format)")
    multi_p.add_argument("--until", help="Only ideas before this date (ISO format)")
    multi_p.add_argument("--recent", help="Relative time filter (e.g. 1d, 1w, 1m)")
    multi_p.add_argument("--when", help="Natural language time (e.g. 'last week', 'since tuesday')")

    # expand (search with topic expansion)
    expand_p = subparsers.add_parser("expand", help="Search with cross-session topic expansion")
    expand_p.add_argument("query", help="Search query")
    expand_p.add_argument("-n", "--limit", type=int, default=10, help="Max results per session")
    expand_p.add_argument("-s", "--session", help="Primary session to search")
    expand_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    expand_p.add_argument("--expand-limit", type=int, default=5, help="Max linked topics to expand to")

    # topic-links (view links for a topic)
    links_p = subparsers.add_parser("topic-links", help="View links for a topic")
    links_p.add_argument("topic_id", type=int, help="Topic ID")

    # link-topics (manually link two topics)
    link_p = subparsers.add_parser("link-topics", help="Manually link two topics")
    link_p.add_argument("topic_id", type=int, help="First topic ID")
    link_p.add_argument("related_id", type=int, help="Second topic ID")
    link_p.add_argument("--similarity", type=float, default=1.0, help="Similarity score (0-1)")

    # auto-link (auto-link all topics)
    autolink_p = subparsers.add_parser("auto-link", help="Auto-link topics across sessions")
    autolink_p.add_argument("--min-similarity", type=float, default=0.8, help="Min similarity threshold")
    autolink_p.add_argument("--dry-run", action="store_true", help="Show what would be linked")

    # refresh-embeddings (update span embeddings for spans missing them)
    refresh_p = subparsers.add_parser("refresh-embeddings", help="Update span embeddings for spans missing them")
    refresh_p.add_argument("-s", "--session", help="Only refresh spans in this session")

    # embed-ideas (generate embeddings for ideas missing them)
    embed_p = subparsers.add_parser("embed-ideas", help="Generate embeddings for ideas missing them")
    embed_p.add_argument("--all", action="store_true", help="Embed all ideas without embeddings")

    # stats
    subparsers.add_parser("stats", help="Show database statistics")

    # cache-stats
    cache_stats_p = subparsers.add_parser("cache-stats", help="Show embedding cache statistics")
    cache_stats_p.add_argument("--clear", action="store_true", help="Clear the cache and reset stats")

    # backfill (async - enqueues for daemon)
    backfill_p = subparsers.add_parser("backfill", help="Enqueue transcripts for background indexing")
    backfill_p.add_argument("file", nargs="?", help="Transcript file path (optional)")
    backfill_p.add_argument("--all", action="store_true", help="Enqueue all transcripts")

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

    # decisions (shortcut for search --intent decision)
    decisions_p = subparsers.add_parser("decisions", help="List decisions (shortcut for search --intent decision)")
    decisions_p.add_argument("query", nargs="?", default="", help="Optional search query")
    decisions_p.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    decisions_p.add_argument("-s", "--session", help="Filter by session")
    decisions_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    decisions_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                             help="Search across all projects")

    # todos (shortcut for search --intent todo)
    todos_p = subparsers.add_parser("todos", help="List todo items (shortcut for search --intent todo)")
    todos_p.add_argument("query", nargs="?", default="", help="Optional search query")
    todos_p.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    todos_p.add_argument("-s", "--session", help="Filter by session")
    todos_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    todos_p.add_argument("-g", "--global", dest="global_search", action="store_true",
                         help="Search across all projects")

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

    # trace (follow relation chains)
    trace_p = subparsers.add_parser("trace", help="Trace relation chains from an idea")
    trace_p.add_argument("id", type=int, help="Starting idea ID")
    trace_p.add_argument("-n", "--limit", type=int, default=10, help="Max ideas per hop")
    trace_p.add_argument("--hops", type=int, default=1, help="Number of hops to follow (1-3)")
    trace_p.add_argument("--backward", action="store_true", help="Only follow incoming relations")
    trace_p.add_argument("--forward", action="store_true", help="Only follow outgoing relations")

    # path (find connection between ideas)
    path_p = subparsers.add_parser("path", help="Find path between two ideas")
    path_p.add_argument("from_id", type=int, help="Starting idea ID")
    path_p.add_argument("to_id", type=int, help="Target idea ID")
    path_p.add_argument("--max-hops", type=int, default=4, help="Maximum hops to search (1-6)")

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

    reparent_p = subparsers.add_parser("reparent-topic", help="Set a topic's parent (create hierarchy)")
    reparent_p.add_argument("topic_id", type=int, help="Topic ID to reparent")
    reparent_p.add_argument("parent_id", type=int, help="New parent topic ID")

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

    # Timeline visualization
    timeline_p = subparsers.add_parser("timeline", help="Show activity timeline for a topic or project")
    timeline_p.add_argument("--topic", "-t", help="Topic name to show timeline for")
    timeline_p.add_argument("--project", "-p", help="Project/session to show timeline for")
    timeline_p.add_argument("--days", "-d", type=int, default=7, help="Days to look back (for --project)")
    timeline_p.add_argument("--cwd", help="Current working directory (for auto-detecting project)")

    # Activity aggregation
    activity_p = subparsers.add_parser("activity", help="Show idea activity aggregated by time period")
    activity_p.add_argument("--by", choices=["day", "week", "month"], default="day", help="Aggregation period")
    activity_p.add_argument("--days", "-d", type=int, default=7, help="Days to look back")
    activity_p.add_argument("-s", "--session", help="Filter by session")
    activity_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")

    # Topic activity
    topic_activity_p = subparsers.add_parser("topic-activity", help="Show activity for a specific topic over time")
    topic_activity_p.add_argument("topic_id", type=int, help="Topic ID")
    topic_activity_p.add_argument("--by", choices=["day", "week", "month"], default="week", help="Aggregation period")
    topic_activity_p.add_argument("--days", "-d", type=int, default=90, help="Days to look back")

    # Soft forgetting commands
    forget_p = subparsers.add_parser("forget", help="Mark an idea as forgotten (soft delete)")
    forget_p.add_argument("id", type=int, help="Idea ID to forget")

    unforget_p = subparsers.add_parser("unforget", help="Restore a forgotten idea")
    unforget_p.add_argument("id", type=int, help="Idea ID to restore")

    forgotten_p = subparsers.add_parser("forgotten", help="List all forgotten ideas")
    forgotten_p.add_argument("-n", "--limit", type=int, default=50, help="Max results")

    # Auto-forget candidates
    forgettable_p = subparsers.add_parser("forgettable", help="Show ideas that are candidates for auto-forgetting")
    forgettable_p.add_argument("-t", "--threshold", type=float, default=0.3,
                               help="Max retention score to include (0-1, default 0.3)")
    forgettable_p.add_argument("-n", "--limit", type=int, default=100, help="Max ideas to show")
    forgettable_p.add_argument("-s", "--session", help="Filter by session")
    forgettable_p.add_argument("--execute", action="store_true", help="Actually forget (default is dry-run)")

    # Consolidation commands
    consolidatable_p = subparsers.add_parser("consolidatable", help="Show topics ready for consolidation")
    consolidatable_p.add_argument("-s", "--session", help="Filter by session")
    consolidatable_p.add_argument("--min-ideas", type=int, default=5, help="Min ideas for consolidation")
    consolidatable_p.add_argument("--max-age", type=int, default=30, help="Max age in days for ideas")

    consolidate_p = subparsers.add_parser("consolidate", help="Consolidate old context ideas in a topic")
    consolidate_p.add_argument("topic_id", type=int, help="Topic ID to consolidate")
    consolidate_p.add_argument("--min-ideas", type=int, default=5, help="Min ideas for consolidation")
    consolidate_p.add_argument("--max-age", type=int, default=30, help="Max age in days for ideas")
    consolidate_p.add_argument("--execute", action="store_true", help="Actually consolidate (default is dry-run)")

    # Reflection commands
    reflect_p = subparsers.add_parser("reflect", help="Generate reflection on recent activity")
    reflect_p.add_argument("-s", "--session", help="Session to reflect on")
    reflect_p.add_argument("-d", "--days", type=int, default=7, help="Days to look back")
    reflect_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    reflect_p.add_argument("-g", "--global", dest="global_search", action="store_true", help="Reflect on all sessions")
    reflect_p.add_argument("--store", action="store_true", help="Store reflection as a special idea")

    reflect_topic_p = subparsers.add_parser("reflect-topic", help="Generate reflection on a topic's evolution")
    reflect_topic_p.add_argument("topic_id", type=int, help="Topic ID to reflect on")

    insights_p = subparsers.add_parser("insights", help="Show stored reflections")
    insights_p.add_argument("-s", "--session", help="Filter by session")
    insights_p.add_argument("-n", "--limit", type=int, default=20, help="Max reflections")
    insights_p.add_argument("--cwd", help="Current working directory (for auto-detecting session)")
    insights_p.add_argument("-g", "--global", dest="global_search", action="store_true", help="Show all reflections")

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
