# Design: Parallel Search with LLM-Selected Strategies

## Problem

The current skill UX requires multiple bash commands for search, each needing approval:
- Search command
- Optional HyDE search
- Optional intent-filtered search
- etc.

This creates friction with multiple "Waiting for approval" prompts, undercutting the value of the skill.

## User Requirements

1. **Global by default** - Search all conversations, not just current project
2. **LLM selects strategies** - Trust the model to decide which search strategies are appropriate for the query (not pattern matching in code)
3. **Run strategies in parallel** - Execute selected strategies concurrently and combine results
4. **Single command** - One bash invocation = one permission prompt
5. **Same quality** - Don't lose functionality (HyDE, intent filtering, temporal queries, etc.)

## Solution: `multi-search` CLI Command

### Command Signature

```bash
uv run python "$SKILL_DIR/src/cli.py" multi-search "<query>" \
  --strategies "hybrid,hyde,decisions" \
  --limit 15 \
  [--local] \
  [--when "last week"]
```

### Available Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `hybrid` | Vector + BM25 keyword | Default for most queries |
| `hyde` | Hypothetical document embedding | Vague/conceptual queries |
| `semantic` | Pure vector search | Simple similarity matching |
| `decisions` | Filter to intent=decision | "What did we decide about X" |
| `todos` | Filter to intent=todo | "What are the open tasks" |
| `questions` | Filter to intent=question | "What questions came up" |
| `problems` | Filter to intent=problem | "What issues/problems" |
| `solutions` | Filter to intent=solution | "How did we solve X" |

### Behavior

1. Parses comma-separated `--strategies` list
2. Runs all selected strategies in parallel using ThreadPoolExecutor
3. Deduplicates results by idea ID
4. Sorts by distance (relevance)
5. Returns combined JSON results
6. Default: global search (all projects)
7. `--local` flag: scope to current project

### Implementation Status

**DONE**: The `multi-search` command handler is added to cli.py (lines 317-431)

**TODO**: Add argparse definition for multi-search command. Add after hyde_p definition:

```python
# multi-search (parallel strategy search - LLM selects strategies)
multi_p = subparsers.add_parser("multi-search", help="Parallel multi-strategy search (LLM selects strategies)")
multi_p.add_argument("query", help="Search query")
multi_p.add_argument("--strategies", "-S", help="Comma-separated strategies: hybrid,hyde,semantic,decisions,todos,questions,problems,solutions")
multi_p.add_argument("-n", "--limit", type=int, default=15, help="Max total results")
multi_p.add_argument("--local", "-l", action="store_true", help="Scope to current project (default: global)")
multi_p.add_argument("--cwd", help="Current working directory (for --local project detection)")
multi_p.add_argument("--since", help="Only ideas after this date (ISO format)")
multi_p.add_argument("--until", help="Only ideas before this date (ISO format)")
multi_p.add_argument("--recent", help="Relative time filter (e.g. 1d, 1w, 1m)")
multi_p.add_argument("--when", help="Natural language time (e.g. 'last week', 'since tuesday')")
```

**TODO**: Update SKILL.md search section to use multi-search with LLM strategy selection.

## Updated SKILL.md Search Section

Replace the current search instructions with:

```markdown
### Search: `/total-recall <query>`

Analyze the query and select appropriate search strategies, then run ONE command:

**Strategy Selection Guide:**
- General queries → `hybrid` (combines semantic + keyword)
- Vague/conceptual queries → `hybrid,hyde` (add hypothetical doc)
- "What did we decide about X" → `hybrid,decisions`
- "What are the open tasks/todos" → `todos`
- "What questions/issues came up" → `hybrid,questions` or `hybrid,problems`
- "How did we solve X" → `hybrid,solutions`
- Time-bounded queries → add `--when "last week"` etc.

**Run the search:**
```bash
SKILL_DIR="$HOME/.claude/skills/total-recall"
cd "$HOME/.claude-plugin-total-recall" && PYTHONPATH="$SKILL_DIR/src" uv run python "$SKILL_DIR/src/cli.py" multi-search "<query>" --strategies "<selected>" -n 15
```

**Options:**
- `--strategies` - Comma-separated: hybrid,hyde,semantic,decisions,todos,questions,problems,solutions
- `--local` - Scope to current project only (default: searches all projects)
- `--when "last week"` - Time filter (natural language)
- `-n 15` - Max results

**Example commands:**
- `/total-recall LoRa range` → `multi-search "LoRa range" --strategies "hybrid"`
- `/total-recall what decisions about auth` → `multi-search "auth" --strategies "hybrid,decisions"`
- `/total-recall open tasks` → `multi-search "tasks" --strategies "todos"`
- `/total-recall recent heating issues` → `multi-search "heating issues" --strategies "hybrid,problems" --when "last week"`
```

## Avoiding Permission Prompts

### Option 1: Skill Hooks (Already Using)
The skill already declares hooks in SKILL.md frontmatter that run without prompts:
```yaml
hooks:
  UserPromptSubmit:
    - hooks:
        - type: command
          command: bash ~/.claude/skills/total-recall/hooks/index-continuous.sh
```
These run automatically on events, no approval needed.

### Option 2: Single Command Design (This Design)
By consolidating to ONE bash command per skill invocation, there's only ONE approval prompt. The user approves running the search command once, and all strategies execute in parallel within that single process.

### Option 3: Pre-approved Commands (Not Available for Skills)
Claude Code's `allowedPrompts` in ExitPlanMode only works for plan mode, not skills. Skills cannot pre-declare bash commands that bypass approval.

### Option 4: User's Claude Settings
Users can configure trusted commands in their Claude Code settings, but this is user-side configuration, not skill-declared.

### Recommendation
The **single command design** is the right approach. The skill instructs Claude to run ONE `multi-search` command that internally parallelizes strategies. One command = one prompt.

## Files Modified

1. `/Users/tom/.claude/skills/total-recall/src/cli.py` - Added multi-search command handler (partially done)
2. `/Users/tom/.claude/skills/total-recall/SKILL.md` - Needs update to use multi-search

## Key Context from Session

- Embeddings are generated for both ideas AND raw messages
- Embedding cache increased to 50K entries, persisted on daemon shutdown
- API key stored in file: `~/.config/total-recall/openai-api-key`
- User wants to import Claude.ai and ChatGPT exports later (exports pending)
- Current indexing: ~7600 ideas, 660 topics across sessions
