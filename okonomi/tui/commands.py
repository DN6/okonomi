"""Slash command handling for the TUI.

Intercepts ``/foo`` style inputs in the TuiSession before they reach the
agent. Each handler emits output via the EventEmitter and optionally
returns a string to forward to the agent as the user's next message.

Ported from the prompt_toolkit cli.py's ``_handle_command``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..events import EventEmitter


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
    ("/heart", "♥ the last generated asset, or pass a URL / local path"),
    ("/loved", "List loved assets — optional filter: images / video / audio / text / 3d / internal / external"),
    ("/prune", "Delete empty sessions older than 24h"),
    ("/reload", "Rebuild the agent — picks up edited skills, tool docstrings, system prompt"),
    ("/save", "Save current session"),
    ("/skill", "Load a skill by name (`/skill <name>`)"),
    ("/jobs", "List active and recent background jobs (bg-exec-N, term-N)"),
    ("/logs", "Show captured stdout/stderr for a background job (`/logs <id>`)"),
    ("/timeout", "Show or set per-turn timeout in seconds (`/timeout <seconds>`, 0 = off)"),
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
        agent.state.message_history = []
        agent.state.context = []
        emitter.note("Conversation cleared.")
        return None

    if low == "/jobs":
        import time as _time

        jobs = getattr(agent, "_bg_logs", {}) or {}
        if not jobs:
            emitter.note("No background jobs.")
            return None
        now = _time.time()
        running = [(jid, j) for jid, j in jobs.items() if j.get("status") == "running"]
        finished = [(jid, j) for jid, j in jobs.items() if j.get("status") != "running"]
        lines = []
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
        lines.append("")
        lines.append("Use `/logs <id>` to view captured output.")
        emitter.note("\n".join(lines))
        return None

    if low == "/logs" or low.startswith("/logs "):
        rest = cmd[len("/logs"):].strip()
        jobs = getattr(agent, "_bg_logs", {}) or {}
        if not rest:
            ids = ", ".join(sorted(jobs.keys())) or "(none)"
            emitter.note(f"Usage: /logs <id>. Known jobs: {ids}")
            return None
        entry = jobs.get(rest)
        if entry is None:
            emitter.note(f"No job named {rest!r}. Run /jobs to list.", level="warn")
            return None
        try:
            output = entry["_reader"]()
        except Exception as e:  # noqa: BLE001
            output = f"(reader error: {e})"
        # Keep the snapshot bounded so the chat doesn't get flooded.
        MAX = 10_000
        if len(output) > MAX:
            output = f"…[truncated {len(output) - MAX} chars]…\n" + output[-MAX:]
        header = f"{rest} [{entry.get('status','?')}] {entry.get('label','')}"
        emitter.note(f"{header}\n{output}")
        return None

    if low == "/timeout" or low.startswith("/timeout "):
        rest = cmd[len("/timeout"):].strip()
        if not rest:
            current = agent.timeout
            label = "disabled" if current <= 0 else f"{current:.0f}s"
            emitter.note(f"Per-turn timeout: {label}. Use `/timeout <seconds>` to change (0 disables).")
            return None
        try:
            seconds = float(rest)
        except ValueError:
            emitter.note(f"Invalid value: {rest!r}. Use a number of seconds.", level="warn")
            return None
        agent.timeout = seconds
        label = "disabled" if seconds <= 0 else f"{seconds:.0f}s"
        emitter.note(f"Per-turn timeout set to {label}.")
        return None

    if low == "/assets":
        try:
            agent._sync_session_assets(agent._get_dependencies())
        except Exception:
            pass
        artifacts = [a for a in agent.state.artifacts if a.artifact_type in ("image", "video", "audio", "3d", "html")]
        if not artifacts:
            emitter.note("No assets generated yet.")
            return None
        return "__asset_picker__"

    if low == "/save":
        agent.save_session(status="completed")
        emitter.note("Session saved.")
        return None

    if low == "/prune":
        count = agent.session_manager.prune_empty_sessions()
        emitter.note(f"Pruned {count} empty session{'s' if count != 1 else ''}.")
        return None

    if low == "/reload":
        counts = agent.reload()
        emitter.note(f"Reloaded — {counts['skills']} skills, {counts['tools']} tools.")
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
        title_text = cmd[len("/title") :].strip()
        if title_text:
            if agent.session_manager.set_title(agent.state.info.session_id, title_text):
                emitter.note(f"Title set: {title_text}")
            else:
                emitter.note(
                    "Failed to set title (conflict or session not saved yet).",
                    level="error",
                )
        else:
            meta = agent.session_manager.get_info(agent.state.info.session_id)
            emitter.note(f"Title: {(meta.title if meta else '') or '(none)'}")
        return None

    if low == "/heart" or low.startswith("/heart "):
        arg = cmd[len("/heart") :].strip()
        _heart_dispatch(agent, emitter, arg)
        return None

    if low == "/loved" or low.startswith("/loved "):
        arg = cmd[len("/loved") :].strip().lower()
        _list_loved(agent, emitter, filter_arg=arg)
        return None

    if low.startswith("/skill "):
        name = cmd[len("/skill ") :].strip()
        # Slash command loads the *full* skill body so the user gets the
        # explicit "force this skill into context" behavior they asked for.
        # The progressive load_skill (TOC only) is what the model uses on
        # its own; the user's explicit /skill should never be coy.
        result = agent.skills._load_full(name)
        if "content" in result:
            return f"Follow this skill:\n\n{result['content']}"
        emitter.note(result.get("error", "Skill not found"), level="error")
        return None

    if low.startswith("/bg "):
        task = cmd[len("/bg ") :].strip()
        if not task:
            emitter.note("Usage: /bg <task>", level="warn")
            return None
        # Sentinel picked up by TuiSession to spawn a parallel agent.
        return f"__bg__:{task}"

    if low.startswith("/btw"):
        rest = cmd[len("/btw") :].strip()
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


_LOVED_DIR = Path.home() / ".okonomi" / "loved"

# Extensions where stdlib `mimetypes` either lacks an entry or classifies
# them as application/* even though they're plain text. Override here so
# they land in the "text" bucket. 3D formats are similarly missing from
# the stdlib registry.
_3D_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".usdz"}
_TEXT_EXTS = {
    ".md", ".markdown", ".rst", ".txt", ".log",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".csv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".sh", ".html", ".css",
}

_MIME_PREFIX_MAP = {"image": "image", "video": "video", "audio": "audio", "text": "text"}


def _guess_artifact_type(name: str) -> str:
    """Map a filename or URL to an ``artifact_type`` tag.

    Uses ``mimetypes`` for the broad set (image/video/audio/text + their
    long tail of extensions like ``.svg``, ``.flac``); explicit overrides
    cover the 3D formats and common text extensions the stdlib misses.
    Falls back to ``"external"`` for anything unrecognized.
    """
    import mimetypes

    clean = name.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(clean).suffix.lower()
    if suffix in _3D_EXTS:
        return "3d"
    if suffix in _TEXT_EXTS:
        return "text"
    mime, _ = mimetypes.guess_type(clean)
    if not mime:
        return "external"
    return _MIME_PREFIX_MAP.get(mime.split("/", 1)[0], "external")


def _symlink_into_loved(src: Path, label: str) -> Path:
    """Place a symlink in ``~/.okonomi/loved/`` pointing at ``src``.

    Returns the symlink path; idempotent — re-hearting reuses an
    existing link.
    """
    _LOVED_DIR.mkdir(parents=True, exist_ok=True)
    link = _LOVED_DIR / f"{label}_{src.name}"
    if not link.exists():
        link.symlink_to(src)
    return link


def _heart_dispatch(agent: "OkonomiAgent", emitter: EventEmitter, arg: str) -> None:
    """Route ``/heart [<arg>]`` to the right handler.

    - empty                                  → heart last generated asset
    - URL (http/https/file/data)             → external URL heart
    - existing local file path               → external file heart
    - anything else                          → error (no silent fallback)
    """
    if not arg:
        _heart_last_artifact(agent, emitter)
        return
    if _looks_like_url(arg):
        _heart_url(agent, emitter, arg)
        return
    p = _resolve_local_path(arg)
    if p is not None:
        _heart_local_path(agent, emitter, p)
        return
    emitter.note(
        f"{arg!r} is neither a URL nor an existing file. "
        "Use `/heart` alone to heart the last generated asset.",
        level="warn",
    )


def _looks_like_url(text: str) -> bool:
    return text.startswith(("http://", "https://", "file://", "data:"))


def _resolve_local_path(text: str) -> Path | None:
    try:
        p = Path(text).expanduser()
    except Exception:
        return None
    return p if p.is_file() else None


def _heart_last_artifact(agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    media = [a for a in agent.state.artifacts if a.artifact_type in ("image", "video", "audio", "3d")]
    if not media:
        emitter.note("No generated media in this session.", level="warn")
        return
    last = media[-1]
    if not last.path or not Path(last.path).exists():
        emitter.note("Last output has no local file yet — wait for the download to finish.", level="warn")
        return

    sm = agent.session_manager
    row = sm._db.execute(
        "SELECT id FROM artifacts WHERE session_id = ? AND (url = ? OR path = ?) ORDER BY id DESC LIMIT 1",
        (agent.state.info.session_id, last.url or "", last.path or ""),
    ).fetchone()
    if row is None:
        emitter.note("Asset not yet indexed — try again in a moment.", level="warn")
        return

    sm.favorite(artifact_id=row["id"])
    src = Path(last.path).resolve()
    _symlink_into_loved(src, agent.state.info.session_id)
    emitter.note(f"♥ {src.name}")


def _heart_url(agent: "OkonomiAgent", emitter: EventEmitter, url: str) -> None:
    artifact_type = _guess_artifact_type(url)
    agent.session_manager.favorite(url=url, artifact_type=artifact_type)
    emitter.note(f"♥ {url}  [{artifact_type}]")


def _heart_local_path(agent: "OkonomiAgent", emitter: EventEmitter, path: Path) -> None:
    resolved = path.resolve()
    artifact_type = _guess_artifact_type(resolved.name)
    agent.session_manager.favorite(url=str(resolved), artifact_type=artifact_type)
    _symlink_into_loved(resolved, "local")
    emitter.note(f"♥ {resolved.name}  [{artifact_type}]  ({resolved})")


# Filter aliases for `/loved <filter>`. Maps user-typed word → (kind, value)
# where kind is "type" (artifact_type column) or "source" (artifact_source).
_LOVED_FILTERS: dict[str, tuple[str, str]] = {
    "image":    ("type", "image"),
    "images":   ("type", "image"),
    "video":    ("type", "video"),
    "videos":   ("type", "video"),
    "audio":    ("type", "audio"),
    "audios":   ("type", "audio"),
    "text":     ("type", "text"),
    "texts":    ("type", "text"),
    "3d":       ("type", "3d"),
    "external": ("source", "external"),
    "internal": ("source", "internal"),
}


def _list_loved(agent: "OkonomiAgent", emitter: EventEmitter, *, filter_arg: str = "") -> None:
    artifact_type: str | None = None
    artifact_source: str | None = None
    label = ""
    if filter_arg:
        match = _LOVED_FILTERS.get(filter_arg)
        if match is None:
            options = ", ".join(sorted(set(_LOVED_FILTERS.keys())))
            emitter.note(f"Unknown filter {filter_arg!r}. Try: {options}", level="warn")
            return
        kind, value = match
        if kind == "type":
            artifact_type = value
        else:
            artifact_source = value
        label = f" {filter_arg}"

    favs = agent.session_manager.list_favorites(
        artifact_type=artifact_type, artifact_source=artifact_source
    )
    if not favs:
        msg = f"Nothing loved matching {filter_arg!r}." if filter_arg else "Nothing loved yet. Use /heart to start."
        emitter.note(msg)
        return
    lines = [f"Loved{label} ({len(favs)}) — browse in {_LOVED_DIR}:"]
    for f in favs:
        target = f.get("url") or ""
        name = Path(target).name or target or "(no target)"
        artifact_type = f.get("artifact_type") or ""
        session = (f.get("session_title") or "")[:30]
        line = f"  ♥ {name}"
        if artifact_type:
            line += f"  [{artifact_type}]"
        if session:
            line += f"  · {session}"
        lines.append(line)
    emitter.note("\n".join(lines))
