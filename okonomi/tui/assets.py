"""Asset favoriting ("heart") and the multi-select picker actions.

One home for everything that touches ``~/.okonomi/loved/`` and the favorites
table, so ``/heart`` (command) and the asset picker's heart action share a
single code path — favorite in the DB *and* symlink local files into the loved
directory, rather than the two divergent implementations they used to have.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..events import EventEmitter, SystemNote
from ..utils import copy_to_clipboard


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


LOVED_DIR = Path.home() / ".okonomi" / "loved"

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


def guess_artifact_type(name: str) -> str:
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

    Returns the symlink path; idempotent — re-hearting reuses an existing link.
    """
    LOVED_DIR.mkdir(parents=True, exist_ok=True)
    link = LOVED_DIR / f"{label}_{src.name}"
    if not link.exists():
        link.symlink_to(src)
    return link


def _favorite_ref(
    agent: "OkonomiAgent",
    ref: str,
    *,
    artifact_type: str = "",
    symlink_label: str,
) -> str:
    """Favorite a single asset ref (URL or local path) and, when it's a local
    file, symlink it into the loved directory. Returns the artifact_type used
    (guessed from the ref when not supplied)."""
    kind = artifact_type or guess_artifact_type(ref)
    agent.session_manager.favorite(url=ref, artifact_type=kind)
    p = Path(ref)
    try:
        is_file = p.is_file()
    except OSError:
        is_file = False
    if is_file:
        _symlink_into_loved(p.resolve(), symlink_label)
    return kind


# --- /heart -----------------------------------------------------------------


def heart(agent: "OkonomiAgent", emitter: EventEmitter, arg: str) -> None:
    """Route ``/heart [<arg>]`` to the right handler.

    - empty                        → heart last generated asset (by DB id)
    - URL (http/https/file/data)   → favorite the URL
    - existing local file path     → favorite + symlink the file
    - anything else                → error (no silent fallback)
    """
    if not arg:
        _heart_last_artifact(agent, emitter)
        return
    if arg.startswith(("http://", "https://", "file://", "data:")):
        kind = _favorite_ref(agent, arg, symlink_label="external")
        emitter.emit(SystemNote(f"♥ {arg}  [{kind}]"))
        return
    p = _resolve_local_path(arg)
    if p is not None:
        resolved = p.resolve()
        kind = _favorite_ref(agent, str(resolved), symlink_label="local")
        emitter.emit(SystemNote(f"♥ {resolved.name}  [{kind}]  ({resolved})"))
        return
    emitter.emit(SystemNote(
        f"{arg!r} is neither a URL nor an existing file. "
        "Use `/heart` alone to heart the last generated asset.",
        level="warn",
    ))


def _resolve_local_path(text: str) -> Path | None:
    try:
        p = Path(text).expanduser()
    except Exception:
        return None
    return p if p.is_file() else None


def _heart_last_artifact(agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    media = [a for a in agent.state.artifacts if a.artifact_type in ("image", "video", "audio", "3d")]
    if not media:
        emitter.emit(SystemNote("No generated media in this session.", level="warn"))
        return
    last = media[-1]
    if not last.path or not Path(last.path).exists():
        emitter.emit(SystemNote("Last output has no local file yet — wait for the download to finish.", level="warn"))
        return

    sm = agent.session_manager
    row = sm._db.execute(
        "SELECT id FROM artifacts WHERE session_id = ? AND (url = ? OR path = ?) ORDER BY id DESC LIMIT 1",
        (agent.state.info.session_id, last.url or "", last.path or ""),
    ).fetchone()
    if row is None:
        emitter.emit(SystemNote("Asset not yet indexed — try again in a moment.", level="warn"))
        return

    sm.favorite(artifact_id=row["id"])
    src = Path(last.path).resolve()
    _symlink_into_loved(src, agent.state.info.session_id)
    emitter.emit(SystemNote(f"♥ {src.name}"))


# --- /loved -----------------------------------------------------------------

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


def list_loved(agent: "OkonomiAgent", emitter: EventEmitter, *, filter_arg: str = "") -> None:
    artifact_type: str | None = None
    artifact_source: str | None = None
    label = ""
    if filter_arg:
        match = _LOVED_FILTERS.get(filter_arg)
        if match is None:
            options = ", ".join(sorted(set(_LOVED_FILTERS.keys())))
            emitter.emit(SystemNote(f"Unknown filter {filter_arg!r}. Try: {options}", level="warn"))
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
        emitter.emit(SystemNote(msg))
        return
    lines = [f"Loved{label} ({len(favs)}) — browse in {LOVED_DIR}:"]
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
    emitter.emit(SystemNote("\n".join(lines)))


# --- Asset-picker actions (open / copy / heart) -----------------------------


def handle_asset_action(
    action: str,
    paths: list[str],
    agent: "OkonomiAgent",
    emitter: EventEmitter,
) -> None:
    """Dispatch a multi-asset action from the picker.

    ``action`` is one of ``"open"`` / ``"copy"`` / ``"heart"``; ``paths`` is
    the list of selected asset refs (file paths or URLs).
    """
    if not paths:
        return
    if action == "open":
        _open_paths(paths, emitter)
    elif action == "copy":
        _copy_paths(paths, emitter)
    elif action == "heart":
        _heart_paths(paths, agent, emitter)


def _open_paths(paths: list[str], emitter: EventEmitter) -> None:
    opener = {"darwin": "open", "linux": "xdg-open"}.get(sys.platform)
    if opener is None:
        emitter.emit(SystemNote(f"Don't know how to open files on {sys.platform}.", level="warn"))
        return
    for p in paths:
        try:
            subprocess.Popen([opener, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            emitter.emit(SystemNote(f"open {p!r} failed: {e}", level="warn"))
    emitter.emit(SystemNote(f"Opened {len(paths)} asset{'s' if len(paths) != 1 else ''}."))


def _copy_paths(paths: list[str], emitter: EventEmitter) -> None:
    if copy_to_clipboard("\n".join(paths)):
        emitter.emit(SystemNote(f"Copied {len(paths)} path{'s' if len(paths) != 1 else ''} to clipboard."))
    else:
        emitter.emit(SystemNote("Clipboard copy failed (no pbcopy / xclip / xsel found).", level="warn"))


def _heart_paths(paths: list[str], agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    n = 0
    label = agent.state.info.session_id
    for p in paths:
        artifact = next((a for a in agent.state.artifacts if (a.path or a.url) == p), None)
        ref = (artifact.url or artifact.path) if artifact else p
        artifact_type = (artifact.artifact_type or "") if artifact else ""
        try:
            _favorite_ref(agent, ref or p, artifact_type=artifact_type, symlink_label=label)
            n += 1
        except Exception:
            continue
    emitter.emit(SystemNote(f"Hearted {n} asset{'s' if n != 1 else ''}."))
