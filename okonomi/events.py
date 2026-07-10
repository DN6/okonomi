"""Typed events produced by the agent and rendered by the TUI.

The agent pushes events onto an ``EventEmitter``; the TUI subscribes and
materializes them as widgets. This replaces the raw ``print()`` flow that
the prompt_toolkit CLI used — by making every user-visible artifact a
structured record, we can render it with per-item expansion state.

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
    """A collapsible tool call. Starts collapsed.

    args: short one-line summary (shown in collapsed view).
    details: full args/code (shown when expanded — kept as the original
        input so the widget can syntax-highlight just this part).
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
    expanded: bool = False
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


_SYNTHETIC_REPLAY_PREFIXES = ("[Resumed", "[system]", "[The user interrupted")


def replay_history(history: list, emitter: EventEmitter) -> None:
    """Re-emit a pydantic-ai message history as visible chat events on
    ``emitter``. Used after a session is resumed so the user sees their
    prior conversation rendered into the chat pane.

    Synthetic injections (resume markers, ``[system]`` notes, interrupt
    framings) are skipped — they're scaffolding, not user turns.
    """
    if not history:
        return
    try:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
    except Exception:
        return

    pending_tool_calls: dict[str, ToolCall] = {}
    any_emitted = False
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = part.content if isinstance(part.content, str) else str(part.content)
                    if text.strip().startswith(_SYNTHETIC_REPLAY_PREFIXES):
                        continue
                    if any_emitted:
                        emitter.emit(Divider())
                    emitter.emit(UserMessage(text))
                    any_emitted = True
                elif isinstance(part, ToolReturnPart):
                    tc = pending_tool_calls.pop(getattr(part, "tool_call_id", ""), None)
                    if tc is None:
                        continue
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    try:
                        emitter.emit(tc.finish(output=content[:2000]))
                    except Exception:
                        pass
        elif isinstance(msg, ModelResponse):
            stream = None
            for part in msg.parts:
                if isinstance(part, TextPart):
                    chunk = part.content
                    if not chunk or not chunk.strip():
                        continue
                    if stream is None:
                        stream = AssistantStream()
                    stream.text += chunk
                    emitter.emit(stream)
                    any_emitted = True
                elif isinstance(part, ToolCallPart):
                    args_repr = repr(part.args) if not isinstance(part.args, str) else part.args
                    tc = emitter.emit(
                        ToolCall(
                            tool_name=part.tool_name,
                            args=args_repr[:80],
                            details=args_repr[:5000],
                            tool_call_id=getattr(part, "tool_call_id", ""),
                        )
                    )
                    pending_tool_calls[getattr(part, "tool_call_id", "")] = tc
                    any_emitted = True
            if stream is not None:
                stream.done = True
                emitter.emit(stream)
