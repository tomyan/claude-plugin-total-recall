# Memgraph - Conversation Memory for Claude Code

A skill that indexes your Claude Code conversations for semantic search, allowing you to recall past decisions, ideas, and context.

## Features

### Search & Retrieval
- **Hybrid search** - Combines vector similarity (semantic) with BM25 keyword matching
- **HyDE search** - Uses LLM to generate hypothetical answers for better query matching
- **Query decomposition** - Automatically handles complex queries like "decisions about X and Y"
- **Intent filtering** - Filter by type: `decisions`, `questions`, `todos`, `conclusions`
- **Relevance verification** - Optional LLM-based scoring to improve precision (`--verify`)
- **Project scoping** - Search within current project or across all conversations

### Memory Organization
- **Automatic topic detection** - Conversations chunked by semantic shifts with hysteresis
- **Hierarchical structure** - Projects → Topics → Spans → Ideas
- **Entity extraction** - Tracks files, technologies, concepts mentioned
- **Relation tracking** - Links between ideas (supersedes, builds_on, answers, contradicts)
- **Cross-session linking** - Connects related topics across different conversations

### Cognitive Features
- **Soft forgetting** - Mark ideas as forgotten without deletion; restore anytime
- **Retention scoring** - Automatic identification of low-value ideas based on recency, access, and importance
- **Memory consolidation** - Summarize old context ideas while preserving decisions
- **Working memory** - Tracks recently accessed ideas with activation decay
- **Reflection** - Generate insights about sessions or topic evolution

### Graph Exploration
- **Multi-hop reasoning** - Trace idea relationships across 1-3 hops
- **Path finding** - Discover how two ideas connect through relations
- **Topic timeline** - View activity on a topic across sessions
- **Clustering analysis** - Identify misplaced ideas and merge candidates

### Developer Experience
- **Continuous indexing** - Hooks index conversations in real-time
- **CLI interface** - Comprehensive commands for search, management, and analysis
- **Configurable thresholds** - TOML config with environment variable overrides
- **164 tests** - Well-tested core functionality

## Prerequisites

- **uv** - Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Claude Code** - Authenticated and in PATH (used for LLM tasks like topic naming, HyDE search)
- **OPENAI_TOKEN_MEMORY_EMBEDDINGS** - Environment variable with OpenAI API key (used only for embeddings)

## Installation

### 1. Copy the skill

```bash
cp -r skills/memgraph ~/.claude/skills/
```

### 2. Add hooks to settings.json

Add the following to `~/.claude/settings.json` to enable continuous indexing:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/skills/memgraph/hooks/index-continuous.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/skills/memgraph/hooks/index-continuous.sh"
          }
        ]
      }
    ]
  }
}
```

If you already have hooks configured, merge the new hooks with your existing ones.

### 3. Restart Claude Code

Restart Claude Code for the hooks to take effect.

### 4. Initial backfill (optional)

To index your existing conversation history:

```
/memory-backfill --all
```

Or to index just the current session:

```
/memory-backfill
```

## Usage

### Search past conversations

```
/memgraph <query>
```

Examples:
- `/memgraph decisions about authentication`
- `/memgraph what did we decide about the API design`
- `/memgraph LoRa range testing results`

### Project scoping

By default, searches are scoped to the **current project**. This means `/memgraph` in your project directory only returns results from conversations in that project.

To search across **all projects**:
```bash
uv run python ~/.claude/skills/memgraph/src/cli.py search "<query>" --global
```

To explicitly scope to a project:
```bash
uv run python ~/.claude/skills/memgraph/src/cli.py search "<query>" --cwd /path/to/project
```

### Check database stats

```bash
uv run python ~/.claude/skills/memgraph/src/cli.py stats
```

### View topics

```bash
uv run python ~/.claude/skills/memgraph/src/cli.py topics
```

## How it works

1. **Continuous indexing**: Hooks trigger on each user prompt and when Claude stops, incrementally indexing new messages
2. **Semantic chunking**: Conversations are split into topic spans based on semantic shifts
3. **Idea extraction**: Each message is analyzed and classified by intent (decision, question, problem, solution, etc.)
4. **Vector embeddings**: Ideas are embedded using OpenAI's text-embedding-3-small for semantic search
5. **LLM tasks**: Topic naming, HyDE search, and filtering use Claude CLI (`claude -p`) for non-interactive queries
6. **Entity extraction**: Key entities (files, technologies, concepts) are tracked and linked

## Data location

- **Skill code**: `~/.claude/skills/memgraph/`
- **Runtime data**: `~/.claude-plugin-memgraph/`
  - `memory.db` - SQLite database with vector embeddings
  - `embedding_cache.json` - Cached embeddings to reduce API calls
  - `hook.log` - Hook execution log

## Troubleshooting

### Hooks not firing

1. Check `~/.claude/settings.json` is valid JSON
2. Restart Claude Code after modifying settings
3. Check `~/.claude-plugin-memgraph/hook.log` for errors

### Empty search results

1. Run `/memory-backfill` to index existing history
2. Check `uv run python ~/.claude/skills/memgraph/src/cli.py stats` to see what's indexed
3. Verify `OPENAI_TOKEN_MEMORY_EMBEDDINGS` is set

### Bootstrap issues

The skill auto-bootstraps on first use. If you encounter issues:

```bash
bash ~/.claude/skills/memgraph/bootstrap.sh
```

This creates the runtime environment at `~/.claude-plugin-memgraph/` with required dependencies.
