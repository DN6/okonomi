"""Artifact favorites and opening artifacts from the find picker.

One home for everything that touches ``~/.okonomi/favorites/`` and the favorites
table, so the viewer and the find picker favorite through a single code path —
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
from ..events import ArtifactAction, EventEmitter, SystemNote


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


FAVORITES_DIR = OKONOMI_FAVORITES_DIR

def _symlink_into_favorites(src: Path, label: str) -> Path:
    """Place a symlink in ``~/.okonomi/favorites/`` pointing at ``src``.

    Returns the symlink path; idempotent — re-favoriting reuses an existing link.
    """
    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    link = FAVORITES_DIR / f"{label}_{src.name}"
    if not link.exists():
        link.symlink_to(src)
    return link


def record_artifact_action(session, action: str, refs: list[str]) -> None:
    """Persist a user action on artifacts (favorite / open / copy) onto the
    session's event stream, so the act sits at its place in the conversation."""
    session.append_stream_events([ArtifactAction(action=action, refs=[_session.norm_ref(r) for r in refs])])


def set_favorite(ref: str, *, artifact_type: str, favorite: bool, session) -> None:
    """Add or remove one favorite ref (viewer or artifact picker). Removing drops every
    favorite row resolving to ``ref`` (internal or external) and its symlinks."""
    target = _session.norm_ref(ref)
    record_artifact_action(session, "favorite" if favorite else "unfavorite", [target])
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


# --- Opening artifacts from the find picker ---------------------------------


def open_artifacts(paths: list[str], agent: "OkonomiAgent", emitter: EventEmitter) -> None:
    """Open ``paths`` in their OS app and record the action on the session."""
    opener = {"darwin": "open", "linux": "xdg-open"}.get(sys.platform)
    if opener is None:
        emitter.emit(SystemNote(f"Don't know how to open files on {sys.platform}.", level="warn"))
        return
    record_artifact_action(agent.session, "open", paths)
    for p in paths:
        try:
            subprocess.Popen([opener, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            emitter.emit(SystemNote(f"open {p!r} failed: {e}", level="warn"))
    emitter.emit(SystemNote(f"Opened {len(paths)} artifact{'s' if len(paths) != 1 else ''}."))
