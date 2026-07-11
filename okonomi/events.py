"""Typed events produced by the agent and rendered by the TUI.

The agent pushes events onto an ``EventEmitter``; subscribers render them
(the REPL transcript for both interactive and one-shot modes). By making
every user-visible artifact a structured record, rendering is decoupled
from the agent loop.

Every event carries a ``type`` discriminator (tau-style tagged events):
``TurnStart``/``TurnEnd`` bracket each agent turn and
``ToolCall.tool_call_id`` correlates with the model transcript.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, ClassVar


_id_counter = itertools.count(1)
_id_lock = Lock()


def _next_id() -> str:
    with _id_lock:
        return f"evt-{next(_id_counter)}"


@dataclass
class Event:
    """Base event. All events carry a unique id for addressing and a
    ``type`` tag for serialization/dispatch without isinstance checks.
    ``id`` is keyword-only so subclass fields can be passed positionally
    (e.g. ``SystemNote("saved.")``)."""

    type: ClassVar[str] = "event"

    id: str = field(default_factory=_next_id, kw_only=True)


@dataclass
class Banner(Event):
    """Static welcome banner shown at session start."""

    type: ClassVar[str] = "banner"


@dataclass
class Divider(Event):
    """Visual turn separator — a horizontal rule between user turns."""

    type: ClassVar[str] = "divider"


@dataclass
class TurnStart(Event):
    """An agent turn began. With ``TurnEnd`` this brackets everything the
    agent emits while handling one user message, so a trace can be sliced
    per turn (the visual ``Divider`` is UI chrome, not a reliable marker)."""

    type: ClassVar[str] = "turn_start"

    turn: int = 0


@dataclass
class TurnEnd(Event):
    """The agent turn finished — emitted on every exit path (completed,
    interrupted, timed out, errored)."""

    type: ClassVar[str] = "turn_end"

    turn: int = 0


@dataclass
class UserMessage(Event):
    type: ClassVar[str] = "user_message"

    text: str = ""


@dataclass
class AssistantStream(Event):
    """Streaming assistant text. `text` grows as tokens arrive.

    The emitter updates the same AssistantStream event across multiple
    emissions; the UI reconciles by id.
    """

    type: ClassVar[str] = "assistant_stream"

    text: str = ""
    done: bool = False


@dataclass
class ToolCall(Event):
    """A tool call and its result.

    args: short one-line summary (shown in the compact view).
    details: full args/code — the original input, so the renderer can
        syntax-highlight it in the verbose view.
    output: text appended after the call returns (stdout, summary, etc.).
        Kept separate from details so the UI can render it with plain
        styling below the highlighted code block.
    tool_call_id: the model's tool-call id, correlating this event with
        the pydantic-ai transcript in ``messages.jsonl``.
    """

    type: ClassVar[str] = "tool_call"

    tool_name: str = ""
    args: str = ""
    details: str = ""
    output: str = ""
    status: str = "running"  # running | ok | error
    tool_call_id: str = ""

    def finish(self, output: str = "", error: str | None = None) -> ToolCall:
        """Mark the call complete, folding an error into ``output``.
        Returns self so the transition can be emitted in one expression:
        ``emitter.emit(ev.finish(output=...))``."""
        self.status = "error" if error else "ok"
        if error:
            self.output = (self.output + "\n" if self.output else "") + f"ERROR: {error}"
        elif output:
            self.output = (self.output + "\n" if self.output else "") + output
        return self


@dataclass
class SystemNote(Event):
    """System-side status line (e.g. "Interrupted. History preserved.")."""

    type: ClassVar[str] = "system_note"

    text: str = ""
    level: str = "info"  # info | warn | error


class EventEmitter:
    """Thread-safe event pipe. Producers construct typed events and push
    them through ``emit`` — the single entry point; consumers (the REPL
    transcript, tests) subscribe.

    A first-seen event is appended to history; re-emitting the same event
    after mutating it (streaming text appended, tool status changed) just
    fans the change out to subscribers. Subscribers are called synchronously
    in the emitting thread.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: list[Callable[[Event], None]] = []
        self._history: list[Event] = []
        self._seen_ids: set[str] = set()

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append(fn)

        def _unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return _unsubscribe

    def emit(self, event: Event) -> Event:
        """Push an event (new or mutated) to all subscribers. Returns the
        event so construct-and-emit reads as one expression."""
        with self._lock:
            if event.id not in self._seen_ids:
                self._seen_ids.add(event.id)
                self._history.append(event)
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 — subscribers must not break the agent
                pass
        return event

    def history(self) -> list[Event]:
        """Snapshot of the event history (for debugging/tests)."""
        with self._lock:
            return list(self._history)
