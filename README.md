# Total Recall - Conversation Memory for Claude Code

A skill that indexes your Claude Code conversations for semantic search, allowing you to recall past decisions, ideas, and context.

## Features

### Search & Retrieval
- **Dual-layer search** - Semantic search on both extracted ideas AND raw messages
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
- **OpenAI API key** - For embeddings. Create the key file:
  ```bash
  echo 'your-openai-api-key' > ~/.config/total-recall/openai-api-key
  ```
  Or set env var: `OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS`

## Installation

### 1. Copy the skill

```bash
cp -r skills/total-recall ~/.claude/skills/
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
            "command": "bash ~/.claude/skills/total-recall/hooks/index-continuous.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/skills/total-recall/hooks/index-continuous.sh"
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
/total-recall backfill --all
```

Or to index just the current session:

```
/total-recall backfill
```

## Usage

### Search past conversations

```
/total-recall <query>
```

Examples:
- `/total-recall decisions about authentication`
- `/total-recall what did we decide about the API design`
- `/total-recall LoRa range testing results`

### Project scoping

By default, searches are scoped to the **current project**. This means `/total-recall` in your project directory only returns results from conversations in that project.

To search across **all projects**:
```bash
uv run python ~/.claude/skills/total-recall/src/cli.py search "<query>" --global
```

To explicitly scope to a project:
```bash
uv run python ~/.claude/skills/total-recall/src/cli.py search "<query>" --cwd /path/to/project
```

### Check database stats

```bash
uv run python ~/.claude/skills/total-recall/src/cli.py stats
```

### View topics

```bash
uv run python ~/.claude/skills/total-recall/src/cli.py topics
```

## How it works

1. **Background daemon**: A persistent daemon processes conversations in the background, triggered by hooks on each user prompt
2. **Semantic chunking**: Conversations are split into topic spans based on semantic shifts
3. **Idea extraction**: Each message is analyzed and classified by intent (decision, question, problem, solution, etc.)
4. **Dual embeddings**: Both extracted ideas AND raw messages are embedded for comprehensive semantic search
5. **Embedding cache**: 50K entry cache reduces API calls across sessions (persisted to disk)
6. **LLM tasks**: Topic naming, HyDE search, and filtering use Claude CLI (`claude -p`) for non-interactive queries
7. **Entity extraction**: Key entities (files, technologies, concepts) are tracked and linked

## Data location

- **Skill code**: `~/.claude/skills/total-recall/`
- **Runtime data**: `~/.claude-plugin-total-recall/`
  - `memory.db` - SQLite database with vector embeddings (ideas, messages, topics)
  - `embedding_cache.json` - 50K entry cache to reduce API calls
  - `daemon.log` - Background daemon processing log
  - `daemon.pid` - PID file for the running daemon

## Troubleshooting

### Hooks not firing

1. Check `~/.claude/settings.json` is valid JSON
2. Restart Claude Code after modifying settings
3. Check `~/.claude-plugin-total-recall/daemon.log` for errors

### Empty search results

1. Run `/total-recall backfill --all` to index existing history
2. Check `/total-recall stats` to see what's indexed
3. Verify `OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS` is set

### Daemon not running

Check the daemon log:
```bash
tail -50 ~/.claude-plugin-total-recall/daemon.log
```

The daemon auto-starts when hooks fire. If needed, manually restart:
```bash
kill $(cat ~/.claude-plugin-total-recall/daemon.pid)
rm ~/.claude-plugin-total-recall/daemon.pid
# Daemon will restart on next hook trigger
```

### Missing API key

The skill requires an OpenAI API key for generating embeddings. Without it:
- Backfill will refuse to run
- Daemon will refuse to start
- Search will not work

Create the key file (recommended - works across all shells/processes):
```bash
mkdir -p ~/.config/total-recall
echo 'your-openai-api-key' > ~/.config/total-recall/openai-api-key
```

Or set environment variable:
```bash
export OPENAI_TOKEN_TOTAL_RECALL_EMBEDDINGS="your-openai-api-key"
```
