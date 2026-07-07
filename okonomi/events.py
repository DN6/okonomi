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

    def log(self, text: str, level: str = "info") -> SystemNote:
        ev = SystemNote(text=text, level=level)
        self.emit(ev)
        return ev


# ANSI colors mirror okonomi.display so the CLI sink looks the same as
# the historical inline-print path.
_LLM = "\033[38;5;110m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RST = "\033[0m"


class StdoutPrinter:
    """Subscriber that renders events to stdout in the CLI-one-shot style.

    Tracks per-stream char progress so incremental ``AssistantStream``
    updates only emit the new tail, and per-tool status so finished
    tool calls render exactly once. Wire up via::

        emitter.subscribe(StdoutPrinter().handle)
    """

    def __init__(self) -> None:
        self._stream_seen: dict[str, int] = {}
        self._stream_line_buf: dict[str, str] = {}
        self._tool_status_seen: dict[str, str] = {}

    def handle(self, ev: Event) -> None:
        if isinstance(ev, AssistantStream):
            self._render_stream(ev)
        elif isinstance(ev, ToolCall):
            self._render_tool_call(ev)
        elif isinstance(ev, SystemNote):
            color = {"warn": _YELLOW, "error": _RED}.get(ev.level, _DIM)
            print(f"{color}{ev.text}{_RST}")
        elif isinstance(ev, Divider):
            print(f"{_DIM}{'─' * 40}{_RST}")

    def _render_stream(self, ev: AssistantStream) -> None:
        seen = self._stream_seen.get(ev.id, 0)
        new = ev.text[seen:]
        self._stream_seen[ev.id] = len(ev.text)
        buf = self._stream_line_buf.get(ev.id, "") + new
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            print(f"{_LLM}{line}{_RST}")
        self._stream_line_buf[ev.id] = buf
        if ev.done:
            tail = self._stream_line_buf.pop(ev.id, "")
            if tail.strip():
                print(f"{_LLM}{tail}{_RST}")
            if ev.text.strip():
                print()
            self._stream_seen.pop(ev.id, None)

    def _render_tool_call(self, ev: ToolCall) -> None:
        prior = self._tool_status_seen.get(ev.id)
        if prior == ev.status:
            return  # already rendered this state
        self._tool_status_seen[ev.id] = ev.status
        if ev.status == "running":
            label = f"{ev.tool_name}({ev.args})" if ev.args else ev.tool_name
            print(f"{_DIM}▶ {label}{_RST}")
        elif ev.status == "error":
            print(f"{_RED}✗ {ev.tool_name}: {ev.output or 'error'}{_RST}")
        else:  # ok
            if ev.output:
                snippet = ev.output if len(ev.output) <= 200 else ev.output[:200] + "…"
                print(f"{_DIM}  → {snippet}{_RST}")


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
                        emitter.divider()
                    emitter.user(text)
                    any_emitted = True
                elif isinstance(part, ToolReturnPart):
                    tc = pending_tool_calls.pop(getattr(part, "tool_call_id", ""), None)
                    if tc is None:
                        continue
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    try:
                        emitter.tool_finish(tc, output=content[:2000])
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
                        stream = emitter.assistant_start()
                    emitter.assistant_append(stream, chunk)
                    any_emitted = True
                elif isinstance(part, ToolCallPart):
                    args_repr = repr(part.args) if not isinstance(part.args, str) else part.args
                    tc = emitter.tool_call(
                        tool_name=part.tool_name,
                        args=args_repr[:80],
                        details=args_repr[:5000],
                    )
                    pending_tool_calls[getattr(part, "tool_call_id", "")] = tc
                    any_emitted = True
            if stream is not None:
                emitter.assistant_done(stream)
