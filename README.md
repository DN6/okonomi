# Okonomi

A terminal agent for creative tasks — it calls tools directly, runs code through
a sandboxed shell with the session directory as its persistent workspace, and
carries session memory across runs.

Okonomi is a personal learning project for exploring the design choices that go
into building an agent harness for creative media work. It is **not** stable: it
changes often and breaks compatibility without notice. Don't depend on it.

```shell
uv tool install git+https://github.com/dn6/okonomi.git
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/), on macOS or Linux. Shell commands
always run in an OS sandbox (Seatbelt on macOS, bubblewrap on Linux) that limits writes to the
working directory, temp dirs and `$OKONOMI_HOME`; okonomi refuses to start without one. On Linux,
install bubblewrap (`sudo apt install bubblewrap` or `sudo dnf install bubblewrap`).

<details>
<summary><b>What it explores</b></summary>

The things I mostly use it to poke at:

- **TUI design** — a prompt-toolkit REPL with slash commands and full-screen
  pickers, and how command handlers hand control to modal UI.
- **Context and long-term memory** — compressing the live context window over a
  long session, plus durable cross-session recall (BM25 over past conversations,
  titles, and media prompts) and DAG-style session branching.
- **Creative research strategies** — structured exploration via the `research`
  skill, web search/fetch, and fanning research out across subagents.
- **Media creation workflows** — a sandboxed shell as the execution surface and
  the session directory as the persistent workspace for generating and editing
  media, with outputs auto-tracked as session artifacts.
- **Training data** — every session records model calls, candidate sets and the
  picks the user made, exported as preference pairs for a taste model.

Expect these to be rebuilt, replaced, or removed as the experiments continue.

</details>

<details>
<summary><b>Setup</b></summary>

The model provider is auto-detected from whichever API key you set. Create a `.env`
file in the project root, or export the variables in your shell — set **one** of:

```bash
OPENROUTER_API_KEY=...   # provider: openrouter — default anthropic/claude-sonnet-5
ANTHROPIC_API_KEY=...    # provider: anthropic  — default claude-sonnet-5
OPENAI_API_KEY=...       # provider: openai     — default openai/gpt-5.6-terra
HF_TOKEN=...             # provider: huggingface — default moonshotai/Kimi-K3
```

If more than one key is set, pick the provider explicitly (valid strings:
`anthropic`, `openai`, `huggingface`, `openrouter`):

```bash
OKONOMI_PROVIDER=openrouter   # which provider to use when several keys are present
OKONOMI_MODEL=...             # override the default model id for the chosen provider
OKONOMI_SUBAGENT_MODELS=a,b   # model ids the agent may pick for sub-agents (default: its own model only)
```

> Media generation (image/video) routes through OpenRouter-specific model ids, so
> those tools only work on the `openrouter` provider.

Everything else is optional:

```bash
# Storage root for sessions, research, skills and favorites (default: ~/.okonomi)
OKONOMI_HOME=/path/to/okonomi/home

# Prompt before dangerous `bash` commands (manual, the default) or run them without asking (off).
# Hardline patterns and deny rules are blocked either way.
OKONOMI_APPROVALS=manual

# Dotenv file of credentials injected into `bash` commands (re-read per command, values redacted from output)
OKONOMI_ENV_FILE=~/.okonomi/tool.env
```

When a `bash` command matches a dangerous pattern, the prompt offers: Enter (run once),
`s` (approve the pattern for this session), `a` (always, saved to `$OKONOMI_HOME/approvals.json`),
`n` (deny), or typed feedback for the model. Without an interactive UI the command is refused.
`approvals.json` also takes deny rules, regexes that block a command unconditionally:

```json
{"allow": ["git reset --hard (destroys uncommitted changes)"], "deny": ["\\bgit\\s+push\\b"]}
```

Sessions, generated files, saved research and skills all live under `$OKONOMI_HOME`
(`~/.okonomi` by default). Nothing is written outside it unless you ask for it.

</details>

<details>
<summary><b>Running</b></summary>

```shell
# Interactive session
okonomi

# One-off task
okonomi --task "Design a futuristic logo"

# Resume the most recent session
okonomi --resume

# Resume a specific session by id, title, or content words
okonomi --resume <id|title|words>

# Write preference data (candidate sets, chosen/rejected pairs, tiers, metrics) for every session
okonomi --export ./preferences
```

</details>

<details>
<summary><b>Slash commands</b></summary>

| Command | Does |
|---|---|
| `/find [query]` | Search sessions, conversations, artifacts and the mind wiki. With nothing typed it lists this session's artifacts, favorites, recent sessions and mind pages. Enter adds a file to your message, Ctrl-O opens it. |
| `/resume [query]` | Pick a past session to resume |
| `/tree` | Browse the session tree, jump back, edit and resubmit to branch |
| `/viewer` | Open the media viewer for this session in the browser |
| `/mind [note]` | Add to the mind wiki, or file what this conversation established |
| `/bg <task>` | Run a task in the background |
| `/tasks` | List background tasks; Enter shows a task's output |
| `/skills [install <spec>]` | List skills, or install one from GitHub. Run a skill with `/<name>` |
| `/mcp [add\|remove …]` | List, add or remove MCP servers |
| `/title`, `/save`, `/clear`, `/reload`, `/timeout`, `/verbose`, `/help`, `/exit` | Session housekeeping |

`/find` takes GitHub-style qualifiers: `type:artifact|session|event|mind`, `kind:image|video|audio|3d|html|text`,
`is:favorite`, `session:<id prefix or title word>`. Finished tool calls print as one line each; click the line to
open the full call and output, which is written under the session's `scratch/logs/`.

</details>

<details>
<summary><b>Store</b></summary>

Everything lives in one store: `$OKONOMI_HOME/okonomi.db` holds sessions, the event
stream, artifacts, favorites and the search index over the mind wiki; `mind/` holds the
wiki pages, versioned in git by `remember`. Each `sessions/<id>/` has three parts:

```
sessions/<id>/
├── traces.jsonl    one line per model call, written by the harness
├── artifacts/      outputs — $OKONOMI_SESSION_DIR; swept into the DB and shown in the viewer
│   └── references/ images you supplied, kept apart from generated candidates
└── scratch/        scripts, intermediate state, installs — $OKONOMI_SCRATCH_DIR; never tracked
    └── logs/       one markdown file per tool call, linked from the transcript
```

</details>

<details>
<summary><b>Development</b></summary>

```shell
git clone https://github.com/dn6/okonomi.git && cd okonomi && uv pip install -e .
OKONOMI_HOME=$(mktemp -d) uv run pytest
```

One test module per subsystem: `test_agent`, `test_tui`, `test_store`, `test_terminal`, `test_training`,
`test_skills`, `test_viewer`, `test_mcp`.

</details>
