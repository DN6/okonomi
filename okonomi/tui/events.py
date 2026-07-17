"""UI-produced chrome events, defined separately from agent events.

These are emitted by the TUI itself for its own scrollback — the agent
never sees or produces them, and they are never persisted. Agent events
(turn lifecycle, streaming text, tool-call display, system notes) live
in ``okonomi.events``; the dependency runs strictly TUI → agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..events import Event


@dataclass
class Banner(Event):
    """Static welcome banner shown at session start."""

    type: ClassVar[str] = "banner"


@dataclass
class Divider(Event):
    """Visual turn separator — a horizontal rule between user turns."""

    type: ClassVar[str] = "divider"


@dataclass
class UserMessage(Event):
    """Echo of the text the user typed, rendered into the scrollback."""

    type: ClassVar[str] = "user_message_display"

    text: str = ""
