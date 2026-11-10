"""Asset favorites and the multi-select picker actions.

One home for everything that touches ``~/.okonomi/favorites/`` and the favorites
table, so the viewer and the asset picker favorite through a single code path —
favorite in the DB under ``session.norm_ref`` *and* symlink local files into
the favorites directory. Browsing favorites goes through ``/find`` (empty query).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .. import session as _session
from ..config import OKONOMI_FAVORITES_DIR
from ..events import AssetAction, EventEmitter, SystemNote
from ..utils import copy_to_clipboard


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


FAVORITES_DIR = OKONOMI_FAVORITES_DIR

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


def _symlink_into_favorites(src: Path, label: str) -> Path:
    """Place a symlink in ``~/.okonomi/favorites/`` pointing at ``src``.

    Returns the symlink path; idempotent — re-favoriting reuses an existing link.
    """
    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    link = FAVORITES_DIR / f"{label}_{src.name}"
    if not link.exists():
        link.symlink_to(src)
    return link


def record_asset_action(session, action: str, refs: list[str]) -> None:
    """Persist a user action on assets (favorite / open / copy) onto the
    session's event stream, so the act sits at its place in the conversation."""
    session.append_stream_events([AssetAction(action=action, refs=[_session.norm_ref(r) for r in refs])])


def set_favorite(ref: str, *, artifact_type: str, favorite: bool, session) -> None:
    """Add or remove one favorite ref (viewer or asset picker). Removing drops every
    favorite row resolving to ``ref`` (internal or external) and its symlinks."""
    target = _session.norm_ref(ref)
    record_asset_action(session, "favorite" if favorite else "unfavorite", [target])
    if favorite:
        _session.favorite(url=target, artifact_type=artifact_type)
        if Path(target).is_file():
            _symlink_into_favorites(Path(target), session.session_id)
        return
    for f in _session.list_favorites():
        if f.get("url") and _session.norm_ref(f["url"]) == target:
            _session.unfavorite(f["id"])
    if FAVORITES_DIR.exists():
        for link in FAVORITES_DIR.iterdir():
            if link.is_symlink() and str(link.resolve()) == target:
                link.unlink()


# --- Asset-picker actions (open / copy / favorite) --------------------------


def handle_asset_action(
    action: str,
    paths: list[str],
    agent: "OkonomiAgent",
    emitter: EventEmitter,
) -> None:
    """Dispatch a multi-asset action from the picker.

    ``action`` is one of ``"open"`` / ``"copy"`` / ``"favorite"`` / ``"view"``;
    ``paths`` is the list of selected asset refs (file paths or URLs). The
    viewer shows the whole session, so ``"view"`` ignores ``paths``.
    """
    if action == "view":
        _open_viewer(agent, emitter)
        return
    if not paths:
        return
    if action == "open":
        _open_paths(paths, agent, emitter)
    elif action == "copy":
        _copy_paths(paths, agent, emitter)
    elif action == "favorite":
        _favorite_paths(paths, agent, emitter)


def _open_paths(paths: list[str], agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    opener = {"darwin": "open", "linux": "xdg-open"}.get(sys.platform)
    if opener is None:
        emitter.emit(SystemNote(f"Don't know how to open files on {sys.platform}.", level="warn"))
        return
    record_asset_action(agent.session, "open", paths)
    for p in paths:
        try:
            subprocess.Popen([opener, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            emitter.emit(SystemNote(f"open {p!r} failed: {e}", level="warn"))
    emitter.emit(SystemNote(f"Opened {len(paths)} asset{'s' if len(paths) != 1 else ''}."))


def _open_viewer(agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    result = agent.viewer.open_viewer()
    if "error" in result:
        emitter.emit(SystemNote(result["error"], level="warn"))
        return
    emitter.emit(SystemNote(f"Viewer open at {result['url']}"))


def _copy_paths(paths: list[str], agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    if copy_to_clipboard("\n".join(paths)):
        record_asset_action(agent.session, "copy", paths)
        emitter.emit(SystemNote(f"Copied {len(paths)} path{'s' if len(paths) != 1 else ''} to clipboard."))
    else:
        emitter.emit(SystemNote("Clipboard copy failed (no pbcopy / xclip / xsel found).", level="warn"))


def _favorite_paths(paths: list[str], agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    n = 0
    for p in paths:
        artifact = next((a for a in agent.state.artifacts if (a.path or a.url) == p), None)
        artifact_type = (artifact.artifact_type if artifact else "") or guess_artifact_type(p)
        try:
            set_favorite(p, artifact_type=artifact_type, favorite=True, session=agent.session)
            n += 1
        except Exception:
            continue
    emitter.emit(SystemNote(f"Added {n} asset{'s' if n != 1 else ''} to favorites."))
