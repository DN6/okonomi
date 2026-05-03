"""Slash command handling for the TUI.

Intercepts ``/foo`` style inputs in the TuiSession before they reach the
agent. Each handler emits output via the EventEmitter and optionally
returns a string to forward to the agent as the user's next message.

Ported from the prompt_toolkit cli.py's ``_handle_command``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .events import EventEmitter


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/assets", "List generated assets in this session"),
    ("/bg", "Run a task in the background (`/bg <task>`)"),
    ("/btw", "Interrupt the current turn with a side question (`/btw <question>`)"),
    ("/clear", "Clear conversation history"),
    ("/reset", "Clear conversation history"),
    ("/session", "Browse and resume a past session"),
    ("/last", "Resume the most recent session"),
    ("/title", "Set/show session title"),
    ("/fav", "Favourite the last generated media asset"),
    ("/favs", "List all favourited assets"),
    ("/prune", "Delete empty sessions older than 24h"),
    ("/save", "Save current session"),
    ("/skill", "Load a skill by name (`/skill <name>`)"),
    ("/exit", "End session"),
    ("/quit", "End session"),
]


def is_slash_command(text: str) -> bool:
    return text.startswith("/") and len(text) > 1


def handle(
    command: str,
    agent: "OkonomiAgent",
    emitter: EventEmitter,
) -> str | None:
    """Dispatch a slash command.

    Returns:
        A string to forward to the agent as the next user message (e.g. from
        `/skill`), or None if the command was fully handled.
    """
    cmd = command.strip()
    low = cmd.lower()

    if low == "/help":
        lines = ["Commands:"]
        for sc, desc in SLASH_COMMANDS:
            lines.append(f"  {sc:<14}  {desc}")
        lines.append("")
        lines.append("Keys:")
        lines.append("  Ctrl+O       Expand/collapse latest tool call")
        lines.append("  Tab          Focus most recent tool call")
        lines.append("  Esc          Return focus to input")
        lines.append("  ↑ / ↓        Input history")
        lines.append("  Ctrl+C       Interrupt running turn")
        lines.append("  Ctrl+D       Exit")
        emitter.note("\n".join(lines))
        return None

    if low in ("/clear", "/reset"):
        agent.message_history = []
        emitter.note("Conversation cleared.")
        return None

    if low == "/assets":
        outputs = agent.generated_outputs
        if not outputs:
            emitter.note("No assets generated yet.")
            return None
        lines = [f"Assets ({len(outputs)}):"]
        for i, o in enumerate(outputs, 1):
            kind = o.asset_type or "file"
            path = o.local_path or o.asset_url or ""
            prompt = (o.prompt or "").strip()
            if len(prompt) > 60:
                prompt = prompt[:60] + "..."
            lines.append(f"  {i}. [{kind}] {path}")
            if prompt:
                lines.append(f"     {prompt}")
        emitter.note("\n".join(lines))
        return None

    if low == "/save":
        agent.save_session(status="saved")
        emitter.note("Session saved.")
        return None

    if low == "/prune":
        count = agent.session_manager.prune_empty_sessions()
        emitter.note(f"Pruned {count} empty session{'s' if count != 1 else ''}.")
        return None

    if low == "/last":
        sessions = agent.session_manager.list_sessions(limit=1)
        if not sessions:
            emitter.note("No previous sessions.", level="warn")
            return None
        if agent.load_session(sessions[0].session_id):
            emitter.note(f"Resumed: {sessions[0].display_name}")
        else:
            emitter.note("Failed to load last session.", level="error")
        return None

    if low.startswith("/title"):
        title_text = cmd[len("/title"):].strip()
        if title_text:
            if agent.session_manager.set_title(agent.session_id, title_text):
                emitter.note(f"Title set: {title_text}")
            else:
                emitter.note(
                    "Failed to set title (conflict or session not saved yet).",
                    level="error",
                )
        else:
            index = agent.session_manager._read_index()
            current = ""
            for entry in index:
                if entry["session_id"] == agent.session_id:
                    current = entry.get("title", "")
                    break
            emitter.note(f"Title: {current or '(none)'}")
        return None

    if low == "/fav":
        _fav_last(agent, emitter)
        return None

    if low == "/favs":
        _list_favs(agent, emitter)
        return None

    if low.startswith("/skill "):
        name = cmd[len("/skill "):].strip()
        result = agent.skills.load_skill(name)
        if "instructions" in result:
            # Feed the skill instructions to the agent as the next message.
            return f"Follow this skill: {result['instructions']}"
        emitter.note(result.get("error", "Skill not found"), level="error")
        return None

    if low.startswith("/bg "):
        task = cmd[len("/bg "):].strip()
        if not task:
            emitter.note("Usage: /bg <task>", level="warn")
            return None
        # Sentinel picked up by TuiSession to spawn a parallel agent.
        return f"__bg__:{task}"

    if low.startswith("/btw"):
        rest = cmd[len("/btw"):].strip()
        if not rest:
            emitter.note("Usage: /btw <question>", level="warn")
            return None
        # Sentinel picked up by TuiSession to cancel the in-flight turn
        # and redirect the agent to this side question.
        return f"__btw__:{rest}"

    if low == "/session":
        # Sentinel — TuiSession opens a modal picker.
        return "__session_picker__"

    if low in ("/exit", "/quit"):
        # Handled by TuiSession's app exit — return a sentinel-like note;
        # TuiSession checks for this and calls app.exit().
        return "__exit__"

    emitter.note(f"Unknown command: {cmd}. Type /help for a list.", level="warn")
    return None


def _fav_last(agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    import os
    import shutil
    from pathlib import Path

    outputs = agent.generated_outputs
    if not outputs:
        emitter.note("No generated media in this session.", level="warn")
        return

    last = outputs[-1]
    local = last.local_path
    if not local or not os.path.exists(local):
        if last.asset_url and os.path.exists(last.asset_url):
            local = last.asset_url
        else:
            emitter.note("Last output has no local file yet. Try /save first.", level="warn")
            return

    favs_dir = Path.home() / ".okonomi" / "favourites"
    favs_dir.mkdir(parents=True, exist_ok=True)
    dst = favs_dir / Path(local).name
    if dst.exists():
        emitter.note(f"Already favourited: {dst.name}")
        return
    shutil.copy2(local, dst)
    emitter.note(f"Favourited → {dst}")


def _list_favs(agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    from pathlib import Path

    favs_dir = Path.home() / ".okonomi" / "favourites"
    if not favs_dir.exists():
        emitter.note("No favourites yet. Use /fav on a generated asset.")
        return
    files = sorted(favs_dir.iterdir())
    if not files:
        emitter.note("No favourites yet.")
        return
    lines = [f"Favourites ({len(files)}):"]
    for f in files:
        lines.append(f"  {f.name}")
    emitter.note("\n".join(lines))
