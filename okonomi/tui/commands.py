"""Slash command handling for the TUI.

Intercepts ``/foo`` inputs in the TuiSession before they reach the agent. Each
command is a handler registered in ``COMMANDS``; it emits output via the
EventEmitter and optionally returns a ``CommandResult`` telling TuiSession to
perform a UI action (open a picker, exit, forward text to the agent, …). The
registry is the single source for dispatch, ``/help``, and completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from .. import session as session_store
from ..constants import OKONOMI_LOVED_DIR
from ..events import AssistantStream, EventEmitter, SystemNote
from ..utils import copy_to_clipboard
from . import assets


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


class Action(str, Enum):
    """A follow-on action for TuiSession to perform after a command runs."""

    EXIT = "exit"
    VERBOSE = "verbose"
    SESSION_PICKER = "session_picker"
    ASSET_PICKER = "asset_picker"
    BG = "bg"
    BTW = "btw"
    FORWARD = "forward"  # send payload to the agent as the next user message


@dataclass
class CommandResult:
    action: Action
    payload: str = ""


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


# Names + descriptions for the slash completer, derived from the registry so
# there's nothing to keep in sync. Populated after the handlers below register.
SLASH_COMMANDS: list[tuple[str, str]] = []


def is_slash_command(text: str) -> bool:
    return text.startswith("/") and len(text) > 1


def handle(command: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    """Dispatch a slash command to its registered handler.

    Returns a ``CommandResult`` when TuiSession must act (exit, picker, forward
    to the agent, …), or None when the command was fully handled in place.
    """
    cmd = command.strip()
    name, _, arg = cmd.partition(" ")
    entry = COMMANDS.get(name.lower())
    if entry is None:
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
        "  Ctrl+C       Interrupt running turn / clear draft",
        "  Ctrl+D       Exit",
    ]
    emitter.emit(SystemNote("\n".join(lines)))


@_command("/assets", "List generated assets in this session")
def _assets(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    try:
        agent._sync_session_assets(agent._get_dependencies())
    except Exception:
        pass
    artifacts = [a for a in agent.state.artifacts if a.artifact_type in ("image", "video", "audio", "3d", "html")]
    if not artifacts:
        emitter.emit(SystemNote("No assets generated yet."))
        return None
    return CommandResult(Action.ASSET_PICKER)


@_command("/bg", "Run a task in the background (`/bg <task>`)")
def _bg(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    if not arg:
        emitter.emit(SystemNote("Usage: /bg <task>", level="warn"))
        return None
    return CommandResult(Action.BG, arg)


@_command("/btw", "Interrupt the current turn with a side question (`/btw <question>`)")
def _btw(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    if not arg:
        emitter.emit(SystemNote("Usage: /btw <question>", level="warn"))
        return None
    return CommandResult(Action.BTW, arg)


@_command("/clear", "Clear conversation history")
def _clear(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    agent.state.message_history = []
    agent.state.context = []
    rewound = agent.session.rewind()
    emitter.emit(SystemNote(f"Conversation cleared ({rewound} events rewound)."))


@_command("/session", "Browse and resume a past session")
def _session(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.SESSION_PICKER)


@_command("/last", "Resume the most recent session")
def _last(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    sessions = session_store.list_sessions(limit=1)
    if not sessions:
        emitter.emit(SystemNote("No previous sessions.", level="warn"))
        return
    if agent.load_session(sessions[0].session_id):
        emitter.emit(SystemNote(f"Resumed: {sessions[0].display_name}"))
    else:
        emitter.emit(SystemNote("Failed to load last session.", level="error"))


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


@_command("/heart", "♥ the last generated asset, or pass a URL / local path")
def _heart(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    assets.heart(agent, emitter, arg)


@_command("/loved", "List loved assets — optional filter: images / video / audio / text / 3d / internal / external")
def _loved(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    assets.list_loved(agent, emitter, filter_arg=arg.lower())


@_command("/prune", "Delete empty sessions older than 24h")
def _prune(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    count = session_store.prune_empty_sessions()
    emitter.emit(SystemNote(f"Pruned {count} empty session{'s' if count != 1 else ''}."))


@_command("/reload", "Rebuild the agent — picks up edited skills, tool docstrings, system prompt")
def _reload(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    counts = agent.reload()
    emitter.emit(SystemNote(f"Reloaded — {counts['skills']} skills, {counts['tools']} tools."))


@_command("/save", "Save current session")
def _save(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    agent.save_session(status="completed")
    emitter.emit(SystemNote("Session saved."))


@_command("/skill", "Load a skill by name (`/skill <name>`)")
def _skill(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult | None":
    if not arg:
        emitter.emit(SystemNote("Usage: /skill <name>", level="warn"))
        return None
    # Load the *full* skill body — the user's explicit /skill forces the whole
    # skill into context (vs the model's own progressive TOC-only load_skill).
    result = agent.skills._load_full(arg)
    if "content" in result:
        return CommandResult(Action.FORWARD, f"Follow this skill:\n\n{result['content']}")
    emitter.emit(SystemNote(result.get("error", "Skill not found"), level="error"))
    return None


@_command("/jobs", "List active and recent background jobs (bg-exec-N, term-N)")
def _jobs(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    import time as _time

    jobs = getattr(agent, "_bg_logs", {}) or {}
    if not jobs:
        emitter.emit(SystemNote("No background jobs."))
        return
    now = _time.time()
    running = [(jid, j) for jid, j in jobs.items() if j.get("status") == "running"]
    finished = [(jid, j) for jid, j in jobs.items() if j.get("status") != "running"]
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
    lines += ["", "Use `/logs <id>` to view captured output."]
    emitter.emit(SystemNote("\n".join(lines)))


@_command("/logs", "Show captured stdout/stderr for a background job (`/logs <id>`)")
def _logs(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    jobs = getattr(agent, "_bg_logs", {}) or {}
    if not arg:
        ids = ", ".join(sorted(jobs.keys())) or "(none)"
        emitter.emit(SystemNote(f"Usage: /logs <id>. Known jobs: {ids}"))
        return
    entry = jobs.get(arg)
    if entry is None:
        emitter.emit(SystemNote(f"No job named {arg!r}. Run /jobs to list.", level="warn"))
        return
    try:
        output = entry["_reader"]()
    except Exception as e:  # noqa: BLE001
        output = f"(reader error: {e})"
    # Keep the snapshot bounded so the chat doesn't get flooded.
    MAX = 10_000
    if len(output) > MAX:
        output = f"…[truncated {len(output) - MAX} chars]…\n" + output[-MAX:]
    header = f"{arg} [{entry.get('status','?')}] {entry.get('label','')}"
    emitter.emit(SystemNote(f"{header}\n{output}"))


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


@_command("/copy", "Copy an assistant message to the clipboard (`/copy [n]`)")
def _copy(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    texts = [ev.text for ev in emitter.history() if isinstance(ev, AssistantStream) and ev.text.strip()]
    if not texts:
        emitter.emit(SystemNote("Nothing to copy yet.", level="warn"))
        return
    n = 1
    if arg:
        try:
            n = max(1, int(arg))
        except ValueError:
            emitter.emit(SystemNote("Usage: /copy [n] — copies the nth-from-last assistant message.", level="warn"))
            return
    if n > len(texts):
        emitter.emit(SystemNote(f"Only {len(texts)} assistant message(s) so far.", level="warn"))
        return
    text = texts[-n]
    if copy_to_clipboard(text):
        emitter.emit(SystemNote(f"Copied {len(text)} characters to clipboard."))
    else:
        emitter.emit(SystemNote("Clipboard copy failed (no pbcopy/xclip/xsel found).", level="warn"))


@_command("/verbose", "Toggle full tool code/output (REPL only)")
def _verbose(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.VERBOSE)


@_command("/exit", "End session")
def _exit(arg: str, agent: "OkonomiAgent", emitter: EventEmitter) -> "CommandResult":
    return CommandResult(Action.EXIT)


SLASH_COMMANDS = [(c.name, c.description) for c in COMMANDS.values()]
