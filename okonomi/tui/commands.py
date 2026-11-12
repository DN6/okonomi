"""Slash command handling for the TUI.

Intercepts ``/foo`` inputs in the TuiSession before they reach the agent. Each
command is a handler registered in ``COMMANDS``; it emits output via the
EventEmitter and optionally returns a ``CommandResult`` telling TuiSession to
perform a UI action (open a picker, exit, forward text to the agent, …). The
registry is the single source for dispatch, ``/help``, and completion.
"""

from __future__ import annotations

import shlex
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from .. import db, mcp
from .. import session as session_store
from ..events import EventEmitter, SystemNote
from ..tools.skills import (
    BUNDLED_SKILLS_DIR,
    USER_SKILLS_DIR,
    _discover_skills,
    _slugify_heading,
    install_skill_from_github,
)


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


class Action(str, Enum):
    """A follow-on action for TuiSession to perform after a command runs."""

    EXIT = "exit"
    VERBOSE = "verbose"
    FIND_PICKER = "find_picker"
    TREE_PICKER = "tree_picker"
    BG = "bg"
    FORWARD = "forward"  # send payload to the agent as the next user message


@dataclass
class CommandResult:
    action: Action
    payload: str = ""
    scope: str = ""  # FIND_PICKER only: "" (everything) | "session"


Handler = Callable[[str, "OkonomiAgent", EventEmitter], "CommandResult | None"]


@dataclass
class Command:
    name: str
    description: str
    handler: Handler


COMMANDS: dict[str, Command] = {}


def _command(name: str, description: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        COMMANDS[name] = Command(name, description, fn)
        return fn

    return register


def is_slash_command(text: str) -> bool:
    return text.startswith("/") and len(text) > 1


def _skill_commands() -> list[tuple[str, str, str]]:
    """``(slug, real_name, description)`` per skill. The slug is the token
    typed after ``/``, namespaced by origin like Claude Code plugin skills:
    ``okonomi:<name>`` for bundled skills, ``user:<name>`` for user skills
    (skill names may contain spaces, so the name part is slugged);
    ``real_name`` is what ``load_skill`` expects. Built-in commands win on a
    slug clash, and the first skill wins on slug collisions between skills."""
    # Empty on any discovery error so slash dispatch/completion degrade gracefully.
    try:
        skills = _discover_skills([USER_SKILLS_DIR, BUNDLED_SKILLS_DIR])
    except Exception:
        skills = {}
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for name, info in skills.items():
        prefix = "user" if info.get("source") == "user" else "okonomi"
        slug = f"{prefix}:{_slugify_heading(name)}"
        if f"/{slug}" in COMMANDS or slug in seen:
            continue
        seen.add(slug)
        out.append((slug, name, info.get("description", "")))
    return out


def discover_skill_commands() -> list[tuple[str, str]]:
    """``(/slug, description)`` for each skill, for the slash completer."""
    return [(f"/{slug}", desc) for slug, _name, desc in _skill_commands()]


def handle(command: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    """Dispatch a slash command to its registered handler.

    Returns a ``CommandResult`` when TuiSession must act (exit, picker, forward
    to the agent, …), or None when the command was fully handled in place.
    A ``/<skill-name>`` that matches no built-in forwards a directive to the
    agent to load and follow that skill.
    """
    cmd = command.strip()
    name, _, arg = cmd.partition(" ")
    entry = COMMANDS.get(name.lower())
    if entry is None:
        slug = name[1:]  # strip leading '/'
        # Match the namespaced slug exactly, or the bare name without its
        # okonomi:/user: prefix so `/research` keeps working.
        real = next(
            (n for s, n, _ in _skill_commands() if s == slug or s.split(":", 1)[1] == slug),
            None,
        )
        if real is not None:
            # Inject the full skill body, with any trailing text as extra instruction.
            result = agent.skills._load_full(real)
            if "content" not in result:
                emitter.emit(SystemNote(result.get("error", "Skill not found"), level="error"))
                return None
            body = f"Follow this skill:\n\n{result['content']}"
            return CommandResult(Action.FORWARD, f"{body}\n\n{arg.strip()}" if arg.strip() else body)
        emitter.emit(SystemNote(f"Unknown command: {cmd}. Type /help for a list.", level="warn"))
        return None
    return entry.handler(arg.strip(), agent, emitter)


# --- Handlers (definition order drives /help and completion order) ----------


@_command("/help", "Show available commands")
def _help(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    lines = ["Commands:"]
    for c in COMMANDS.values():
        lines.append(f"  {c.name:<14}  {c.description}")
    lines += [
        "",
        "Keys:",
        "  ↑ / ↓        Input history",
        "  Ctrl+R       Search conversation history / branch from a match",
        "  Ctrl+C       Interrupt running turn / clear draft / exit when idle",
        "  Ctrl+D       Exit",
    ]
    emitter.emit(SystemNote("\n".join(lines)))


@_command("/skills", "List skills, or install one from GitHub (`/skills install <owner/repo/path | URL>`)")
def _skills(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    sub, _, spec = arg.partition(" ")
    if sub == "install":
        spec = spec.strip()
        if not spec:
            emitter.emit(
                SystemNote("Usage: /skills install <owner/repo/path/to/skill | github.com tree URL>", level="warn")
            )
            return None
        emitter.emit(SystemNote(f"Installing skill from {spec} …"))

        # Network I/O off the prompt_toolkit loop; the emitter is thread-safe.
        def _run() -> None:
            result = install_skill_from_github(spec)
            if not result["success"]:
                emitter.emit(SystemNote(result["error"], level="error"))
                return
            counts = agent.reload()
            emitter.emit(
                SystemNote(
                    f"Installed skill '{result['name']}' → {result['path']} (reloaded — {counts['skills']} skills)."
                )
            )

        threading.Thread(target=_run, name="skill-install", daemon=True).start()
        return None
    if arg:
        emitter.emit(SystemNote(f"Unknown subcommand '{sub}'. Usage: /skills [install <spec>]", level="warn"))
        return None
    cmds = _skill_commands()
    if not cmds:
        emitter.emit(SystemNote("No skills found."))
        return None
    lines = ["Skills — run with /<name>:"]
    for slug, _real, desc in sorted(cmds):
        lines.append(f"  /{slug:<30} {desc}")
    emitter.emit(SystemNote("\n".join(lines)))


@_command("/bg", "Run a task in the background (`/bg <task>`)")
def _bg(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    if not arg:
        emitter.emit(SystemNote("Usage: /bg <task>", level="warn"))
        return None
    return CommandResult(Action.BG, arg)


@_command("/clear", "Clear conversation history")
def _clear(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    agent.state.message_history = []
    agent.state.context = []
    rewound = agent.session.rewind()
    emitter.emit(SystemNote(f"Conversation cleared ({rewound} events rewound)."))


@_command("/resume", "Pick a past session to resume (`/resume [query]`)")
def _resume(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.FIND_PICKER, arg, scope="session")


@_command("/find", "Search sessions, conversations, artifacts, and the mind wiki (`/find [query]`)")
def _find(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.FIND_PICKER, arg)


@_command("/tree", "Browse the session tree — jump back, edit & resubmit to branch")
def _tree(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    if db.count_events(db.connection(), agent.session.session_id, active_only=False) == 0:
        emitter.emit(SystemNote("No history yet."))
        return None
    return CommandResult(Action.TREE_PICKER)


@_command("/title", "Set/show session title")
def _title(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    if arg:
        if agent.session.set_title(arg):
            emitter.emit(SystemNote(f"Title set: {arg}"))
        else:
            emitter.emit(SystemNote("Failed to set title (session not saved yet).", level="error"))
    else:
        # Re-fetch: the out-of-band titler writes to the DB, not the live object.
        meta = session_store.Session.load(agent.session.session_id)
        emitter.emit(SystemNote(f"Title: {(meta.title if meta else '') or '(none)'}"))


_MCP_USAGE = "Usage: /mcp | /mcp add <name> <url | command [args...]> [-e KEY=VALUE ...] | /mcp remove <name>"


@_command("/mcp", "List, add or remove MCP servers (`/mcp add <name> <url|command ...>`)")
def _mcp(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    try:
        words = shlex.split(arg)
    except ValueError as e:
        emitter.emit(SystemNote(f"Could not parse arguments: {e}", level="error"))
        return None
    sub = words[0] if words else ""

    if sub == "add":
        env: dict[str, str] = {}
        rest: list[str] = []
        it = iter(words[1:])
        for w in it:
            if w == "-e":
                kv = next(it, "")
                key, eq, value = kv.partition("=")
                if not eq or not key:
                    emitter.emit(SystemNote(f"-e expects KEY=VALUE, got '{kv}'", level="warn"))
                    return None
                env[key] = value
            else:
                rest.append(w)
        if len(rest) < 2:
            emitter.emit(SystemNote(_MCP_USAGE, level="warn"))
            return None
        name, target, *args = rest
        result = mcp.add_server(name, target, args, env)
    elif sub == "remove":
        if len(words) != 2:
            emitter.emit(SystemNote(_MCP_USAGE, level="warn"))
            return None
        name = words[1]
        result = mcp.remove_server(name)
    elif sub:
        emitter.emit(SystemNote(_MCP_USAGE, level="warn"))
        return None
    else:
        servers = mcp.load_servers()
        if not servers:
            emitter.emit(SystemNote(f"No MCP servers configured ({mcp.OKONOMI_MCP_CONFIG}).\n{_MCP_USAGE}"))
            return None
        lines = [f"MCP servers ({mcp.OKONOMI_MCP_CONFIG}):"]
        for name, server in servers.items():
            lines.append(f"  {name:<16} {mcp.describe(server)}")
        emitter.emit(SystemNote("\n".join(lines)))
        return None

    if not result["success"]:
        emitter.emit(SystemNote(result["error"], level="error"))
        return None
    counts = agent.reload()
    verb = "Added" if sub == "add" else "Removed"
    emitter.emit(SystemNote(f"{verb} MCP server '{name}' — {counts['mcp']} servers configured (agent reloaded)."))
    return None


@_command("/reload", "Rebuild the agent — picks up edited skills, tool docstrings, system prompt")
def _reload(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    counts = agent.reload()
    emitter.emit(SystemNote(f"Reloaded — {counts['skills']} skills, {counts['tools']} tools."))


@_command("/save", "Save current session")
def _save(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    agent.save_session(status="completed")
    emitter.emit(SystemNote("Session saved."))


@_command("/viewer", "Open the media viewer for this session in the browser")
def _viewer(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    result = agent.viewer.open_viewer()
    if "url" in result:
        emitter.emit(SystemNote(f"Viewer: {result['url']}"))
        return
    emitter.emit(SystemNote(result.get("error", "Could not open the viewer."), level="error"))


@_command("/tasks", "List active and recent background tasks (term-N, agents)")
def _tasks(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    import time as _time

    tasks = agent.tasks.entries
    if not tasks:
        emitter.emit(SystemNote("No background tasks."))
        return
    now = _time.time()
    running = [(jid, j) for jid, j in tasks.items() if j.get("status") == "running"]
    finished = [(jid, j) for jid, j in tasks.items() if j.get("status") != "running"]
    lines: list[str] = []
    if running:
        lines.append(f"Running ({len(running)}):")
        for jid, j in sorted(running):
            elapsed = now - j["started_at"]
            lines.append(f"  ● {jid:<14} {j['kind']:<8} {j.get('label','')} ({elapsed:.0f}s)")
    if finished:
        lines.append(f"Finished ({len(finished)}):")
        for jid, j in sorted(finished):
            dur = (j.get("ended_at") or now) - j["started_at"]
            glyph = "✓" if j.get("status") == "ok" else "✗"
            lines.append(f"  {glyph} {jid:<14} {j['kind']:<8} {j.get('label','')} ({dur:.0f}s)")
    emitter.emit(SystemNote("\n".join(lines)))


@_command("/timeout", "Show or set per-turn timeout in seconds (`/timeout <seconds>`, 0 = off)")
def _timeout(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    if not arg:
        current = agent.timeout
        label = "disabled" if current <= 0 else f"{current:.0f}s"
        emitter.emit(SystemNote(f"Per-turn timeout: {label}. Use `/timeout <seconds>` to change (0 disables)."))
        return
    try:
        seconds = float(arg)
    except ValueError:
        emitter.emit(SystemNote(f"Invalid value: {arg!r}. Use a number of seconds.", level="warn"))
        return
    agent.timeout = seconds
    label = "disabled" if seconds <= 0 else f"{seconds:.0f}s"
    emitter.emit(SystemNote(f"Per-turn timeout set to {label}."))


@_command("/mind", "Add to the mind wiki: `/mind <note>`, or `/mind` to file what this conversation established")
def _mind(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    what = (
        f"this note: {arg}"
        if arg
        else "what we've established in this conversation (decisions, preferences, findings worth keeping)"
    )
    return CommandResult(
        Action.FORWARD,
        f"Add {what} to the mind wiki. `recall` first; extend an existing page with `remember(..., update=True)` "
        "rather than creating a duplicate, and link related pages.",
    )


@_command("/verbose", "Toggle full tool code/output (REPL only)")
def _verbose(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.VERBOSE)


@_command("/exit", "End session")
def _exit(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.EXIT)


# Names + descriptions for the slash completer, derived from the registry so
# there's nothing to keep in sync.
SLASH_COMMANDS = [(c.name, c.description) for c in COMMANDS.values()]
