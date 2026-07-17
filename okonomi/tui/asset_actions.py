"""Handlers for the multi-select asset-picker actions.

The picker collects a set of asset refs and an action keyword (``open`` /
``copy`` / ``heart``); ``handle_asset_action`` dispatches to the right
side effect and surfaces user-visible notes via the emitter.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from ..events import EventEmitter, SystemNote
from ..session import favorite
from ..utils import copy_to_clipboard


if TYPE_CHECKING:
    from ..agent import OkonomiAgent


def handle_asset_action(
    action: str,
    paths: list[str],
    agent: "OkonomiAgent",
    emitter: EventEmitter,
) -> None:
    """Dispatch a multi-asset action from the picker.

    ``action`` is one of ``"open"`` / ``"copy"`` / ``"heart"``; ``paths``
    is the list of selected asset refs (file paths or URLs).
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
    for p in paths:
        artifact = next(
            (a for a in agent.state.artifacts if (a.path or a.url) == p),
            None,
        )
        if artifact is None:
            continue
        try:
            favorite(
                url=artifact.url or artifact.path or "",
                artifact_type=artifact.artifact_type or "",
            )
            n += 1
        except Exception:
            continue
    emitter.emit(SystemNote(f"Hearted {n} asset{'s' if n != 1 else ''}."))
