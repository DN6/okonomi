# Okonomi

A terminal agent for creative tasks — it calls tools directly, runs code through
a sandboxed shell with the session directory as its persistent workspace, and
carries session memory across runs.

Okonomi is a personal learning project for exploring the design choices that go
into building an agent harness for creative media work. It is **not** stable: it
changes often and breaks compatibility without notice. Don't depend on it.

## What it explores

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
  media, with outputs auto-tracked as session assets.

Expect these to be rebuilt, replaced, or removed as the experiments continue.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip), on macOS or Linux.
Shell commands always run in an OS sandbox (Seatbelt on macOS, bubblewrap on Linux) that limits
writes to the working directory, temp dirs and `$OKONOMI_HOME`; okonomi refuses to start without one.
On Linux, install bubblewrap (`sudo apt install bubblewrap` or `sudo dnf install bubblewrap`).

```shell
git clone https://github.com/dn6/okonomi.git
cd okonomi
uv pip install -e .
```

## Setup

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

## Running

```shell
# Interactive session
okonomi

# One-off task
okonomi --task "Design a futuristic logo"

# Resume the most recent session
okonomi --resume

# Load a specific session by ID
okonomi --session <id>

# Write preference data (candidate sets, chosen/rejected pairs, tiers, metrics) for every session
okonomi --export ./preferences
```

Everything lives in one store: `$OKONOMI_HOME/okonomi.db` holds sessions, the event
stream, artifacts, favorites and the search index over the mind wiki; `mind/` holds the
wiki pages, versioned in git by `remember`. Each `sessions/<id>/` has three parts:

```
sessions/<id>/
├── traces.jsonl    one line per model call, written by the harness
├── artifacts/      outputs — $OKONOMI_SESSION_DIR; swept into the DB and shown in the viewer
└── scratch/        scripts, intermediate state, installs — $OKONOMI_SCRATCH_DIR; never tracked
```
