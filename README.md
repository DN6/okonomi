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

## Setup

The model provider is auto-detected from whichever API key you set. Create a `.env`
file in the project root, or export the variables in your shell — set **one** of:

```bash
OPENROUTER_API_KEY=...   # provider: openrouter — default anthropic/claude-sonnet-5
ANTHROPIC_API_KEY=...    # provider: anthropic  — default claude-sonnet-5
OPENAI_API_KEY=...       # provider: openai     — default openai/gpt-5.6-terra
HF_TOKEN=...             # provider: huggingface — default moonshotai/Kimi-K3
```

Options live in `$OKONOMI_HOME/settings.json`; every key is optional:

```json
{
  "provider": "openrouter",
  "model": "anthropic/claude-sonnet-5",
  "subagentModels": ["a", "b"],
  "approvals": {
    "mode": "manual",
    "allow": ["git reset --hard (destroys uncommitted changes)"],
    "deny": ["\\bgit\\s+push\\b"]
  },
  "mcpServers": {
    "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}
  }
}
```

- `provider`: which provider to use when several keys are present (`anthropic`, `openai`, `huggingface`, `openrouter`).
- `model`: overrides the default model id for the chosen provider.
- `subagentModels`: model ids the agent may pick for sub-agents (default: its own model only).
- `approvals.mode`: `manual` (the default) prompts before dangerous `bash` commands, `off` runs them without asking. Hardline patterns and deny rules are blocked either way.
- `approvals.allow`: dangerous patterns approved with `a` at the prompt. `approvals.deny`: regexes that block a command unconditionally.
- `mcpServers`: MCP servers in the shape shared by Claude Desktop, Cursor and Codex; `/mcp add` and `/mcp remove` edit it.

An environment variable overrides the matching key: `OKONOMI_PROVIDER`, `OKONOMI_MODEL`,
`OKONOMI_SUBAGENT_MODELS` (comma-separated) and `OKONOMI_APPROVALS`.

> Media generation (image/video) routes through OpenRouter-specific model ids, so
> those tools only work on the `openrouter` provider.

Everything else is optional:

```bash
# Storage root for settings, sessions, research, skills and favorites (default: ~/.okonomi)
OKONOMI_HOME=/path/to/okonomi/home

# Dotenv file of credentials injected into `bash` commands (re-read per command, values redacted from output)
OKONOMI_ENV_FILE=~/.okonomi/tool.env
```

When a `bash` command matches a dangerous pattern, the prompt offers: Enter (run once),
`s` (approve the pattern for this session), `a` (always, saved to `approvals.allow` in `settings.json`),
`n` (deny), or typed feedback for the model. Without an interactive UI the command is refused.

Sessions, generated files, saved research and skills all live under `$OKONOMI_HOME`
(`~/.okonomi` by default). Nothing is written outside it unless you ask for it.

## Usage

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

Type `/help` inside a session for the slash commands.
