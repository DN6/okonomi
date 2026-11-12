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
