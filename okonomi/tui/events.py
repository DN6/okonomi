"""Typed events produced by the agent and rendered by the TUI.

The agent pushes events onto an ``EventEmitter``; the TUI subscribes and
materializes them as widgets. This replaces the raw ``print()`` flow that
the prompt_toolkit CLI used — by making every user-visible artifact a
structured record, we can render it with per-item expansion state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable


_id_counter = itertools.count(1)
_id_lock = Lock()


def _next_id() -> str:
    with _id_lock:
        return f"evt-{next(_id_counter)}"


@dataclass
class Event:
    """Base event. All events carry a unique id for addressing."""

    id: str = field(default_factory=_next_id)


@dataclass
class Banner(Event):
    """Static welcome banner shown at session start."""


@dataclass
class Divider(Event):
    """Visual turn separator — a horizontal rule between user turns."""


@dataclass
class UserMessage(Event):
    text: str = ""


@dataclass
class AssistantStream(Event):
    """Streaming assistant text. `text` grows as tokens arrive.

    The emitter updates the same AssistantStream event across multiple
    emissions; the UI reconciles by id.
    """

    text: str = ""
    done: bool = False


@dataclass
class ToolCall(Event):
    """A collapsible tool call. Starts collapsed.

    args: short one-line summary (shown in collapsed view).
    details: full args/code (shown when expanded — kept as the original
        input so the widget can syntax-highlight just this part).
    output: text appended after the call returns (stdout, summary, etc.).
        Kept separate from details so the UI can render it with plain
        styling below the highlighted code block.
    """

    tool_name: str = ""
    args: str = ""
    details: str = ""
    output: str = ""
    expanded: bool = False
    status: str = "running"  # running | ok | error


@dataclass
class SystemNote(Event):
    """System-side status line (e.g. "Interrupted. History preserved.")."""

    text: str = ""
    level: str = "info"  # info | warn | error


class EventEmitter:
    """Thread-safe event emitter. Agent pushes; UI subscribes.

    Subscribers are called synchronously in the emitting thread. The UI
    subscriber forwards into Textual's message queue via call_from_thread.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: list[Callable[[Event], None]] = []
        self._history: list[Event] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(fn)

        def _unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return _unsubscribe

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers and append to history."""
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 — subscribers must not break the agent
                pass

    def update(self, event: Event) -> None:
        """Notify subscribers that an existing event was mutated (e.g.
        streaming text appended, tool call status changed). The event is
        expected to already be in history — we just fan out the change."""
        with self._lock:
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                pass

    def history(self) -> list[Event]:
        """Snapshot of the event history (for debugging/tests)."""
        with self._lock:
            return list(self._history)

    # --- Convenience helpers ---

    def user(self, text: str) -> UserMessage:
        ev = UserMessage(text=text)
        self.emit(ev)
        return ev

    def divider(self) -> Divider:
        ev = Divider()
        self.emit(ev)
        return ev

    def assistant_start(self) -> AssistantStream:
        ev = AssistantStream()
        self.emit(ev)
        return ev

    def assistant_append(self, ev: AssistantStream, chunk: str) -> None:
        ev.text += chunk
        self.update(ev)

    def assistant_done(self, ev: AssistantStream) -> None:
        ev.done = True
        self.update(ev)

    def tool_call(
        self,
        tool_name: str,
        args: str = "",
        details: str = "",
        expanded: bool = False,
    ) -> ToolCall:
        ev = ToolCall(tool_name=tool_name, args=args, details=details, expanded=expanded)
        self.emit(ev)
        return ev

    def tool_finish(self, ev: ToolCall, output: str = "", error: str | None = None) -> None:
        ev.status = "error" if error else "ok"
        if error:
            ev.output = (ev.output + "\n" if ev.output else "") + f"ERROR: {error}"
        elif output:
            ev.output = (ev.output + "\n" if ev.output else "") + output
        self.update(ev)

    def note(self, text: str, level: str = "info") -> SystemNote:
        ev = SystemNote(text=text, level=level)
        self.emit(ev)
        return ev
