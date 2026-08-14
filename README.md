# Okonomi

A terminal agent that works through a persistent Python REPL, with session memory across runs.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

```shell
git clone https://github.com/dn6/okonomi.git
cd okonomi
uv pip install -e .
```

## Setup

Okonomi routes all model calls through [OpenRouter](https://openrouter.ai). Create a `.env`
file in the project root, or export the variables in your shell.

```bash
# Required — get a key from https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key
```

Everything else is optional:

```bash
# Enables the web_search / web_fetch tools — https://app.tavily.com
TAVILY_API_KEY=your_tavily_api_key

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
