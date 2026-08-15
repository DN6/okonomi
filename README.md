# Okonomi

A terminal agent for creative tasks — it calls tools directly, keeps a persistent
Python REPL as a workspace for generating and editing media, and carries session
memory across runs.

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
- **Media creation workflows** — a persistent Python REPL as the workspace for
  generating and editing media, with outputs auto-tracked as session assets.

Expect these to be rebuilt, replaced, or removed as the experiments continue.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

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
```

> Media generation (image/video) routes through OpenRouter-specific model ids, so
> those tools only work on the `openrouter` provider.

Everything else is optional:

```bash
# Storage root for sessions, research, skills and favourites (default: ~/.okonomi)
OKONOMI_HOME=/path/to/okonomi/home

# Skip the shell-command safety checks in the `bash` tool
OKONOMI_YOLO=false
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

# Show executed code and full tool output
okonomi --verbose
```
