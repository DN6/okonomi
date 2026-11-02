"""Agent events: the bounded event taxonomy plus the live event pipe.

Two related things live here:

1. **The persistence taxonomy** — ``EventType`` is the closed universe of
   events that can appear in the DB's ``events`` table. The four
   conversation types carry whole pydantic-ai messages as payloads
   (classified by ``classify_message``, indexed via ``extract_content``);
   the rest serialize the typed objects below (``SystemNote``, ``Title``,
   ``Interrupt``) through ``event_to_payload``.

2. **Live-rendering events** the agent emits through ``EventEmitter``
   for the TUI to draw (``TurnStart``/``TurnEnd``, ``AssistantStream``,
   ``ToolCallDisplay``). These are never persisted — their content is
   already in the conversation rows.

UI-produced chrome (banner, divider, the echo of typed input) lives in
``okonomi.tui.events`` — the TUI imports from here, never the reverse.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, ClassVar


class EventType(str, Enum):
    """Closed set of event types persisted to the ``events`` table."""

    # Conversation — payload is one whole pydantic-ai message.
    USER_MESSAGE = "user_message"  # ModelRequest containing a user prompt
    TOOL_RESPONSE = "tool_response"  # ModelRequest containing only tool returns
    ASSISTANT_MESSAGE = "assistant_message"  # ModelResponse, text only
    TOOL_CALL = "tool_call"  # ModelResponse containing tool calls
    # Stream events — payload is the matching typed object below.
    SYSTEM_NOTE = "system_note"
    TITLE = "title"
    INTERRUPT = "interrupt"


CONVERSATION_TYPES = (
    EventType.USER_MESSAGE,
    EventType.TOOL_RESPONSE,
    EventType.ASSISTANT_MESSAGE,
    EventType.TOOL_CALL,
)


def classify_message(message: Any) -> EventType:
    """Semantic event type of one pydantic-ai message, derived from its
    part composition. The message itself is stored unmodified."""
    from pydantic_ai.messages import ModelRequest, ToolCallPart, UserPromptPart

    if isinstance(message, ModelRequest):
        if any(isinstance(p, UserPromptPart) for p in message.parts):
            return EventType.USER_MESSAGE
        return EventType.TOOL_RESPONSE
    if any(isinstance(p, ToolCallPart) for p in message.parts):
        return EventType.TOOL_CALL
    return EventType.ASSISTANT_MESSAGE


# ── content extraction (what FTS indexes) ────────────────────────────────

# Per-event cap on indexed text — a pasted novel shouldn't bloat the index.
CONTENT_CAP_CHARS = 4000

_BASE64_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}")
_HTML_PREFIX_PATTERN = re.compile(r"^\s*(<(!DOCTYPE|html)\b)", re.IGNORECASE)


def _strip_base64(text: str) -> str:
    def _replace(m):
        kb = len(m.group(0)) * 3 // 4 // 1024
        return f"[embedded image, ~{kb}KB]"

    return _BASE64_PATTERN.sub(_replace, text)


def _strip_large_content(text: str) -> str:
    text = _strip_base64(text)
    if len(text) > 500 and _HTML_PREFIX_PATTERN.match(text):
        return f"[HTML content, {len(text)} chars — saved to file]"
    return text


def extract_content(message: Any) -> str:
    """Searchable text of one pydantic-ai message: user-prompt text and
    assistant ``TextPart``s only. System prompts, tool-call args, tool
    returns, base64 blobs, and HTML dumps never reach the index —
    filtering happens here, at write time, not at query time."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    texts: list[str] = []
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                texts.append(content)
    elif isinstance(message, ModelResponse):
        for part in message.parts:
            if isinstance(part, TextPart) and part.content:
                texts.append(part.content)

    text = _strip_large_content("\n".join(t for t in texts if t))
    return text[:CONTENT_CAP_CHARS]


def message_accounting(message: Any) -> dict[str, Any] | None:
    """Per-message token/cost values for a ``ModelResponse`` (the events
    table's accounting columns); ``None`` for requests, which incur none."""
    from pydantic_ai.messages import ModelResponse

    if not isinstance(message, ModelResponse):
        return None
    usage = getattr(message, "usage", None)
    details = getattr(usage, "details", {}) or {}
    cost = None
    provider_details = getattr(message, "provider_details", None)
    if provider_details and provider_details.get("cost") is not None:
        cost = float(provider_details["cost"])
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_write_tokens", 0) or 0,
        "reasoning_tokens": int(details.get("reasoning_tokens") or details.get("output_reasoning_tokens") or 0),
        "cost_usd": cost,
    }


# ── live events ──────────────────────────────────────────────────────────

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
class TurnStart(Event):
    """An agent turn began. With ``TurnEnd`` this brackets everything the
    agent emits while handling one user message, so a trace can be sliced
    per turn."""

    type: ClassVar[str] = "turn_start"
    turn: int = 0


@dataclass
class TurnEnd(Event):
    """The agent turn finished — emitted on every exit path (completed,
    interrupted, timed out, errored)."""

    type: ClassVar[str] = "turn_end"
    turn: int = 0


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
class ToolCallDisplay(Event):
    """A collapsible tool call widget. Starts collapsed. Live rendering
    only — the canonical tool call is the pydantic-ai message that becomes
    an ``EventType.TOOL_CALL`` row.

    args: short one-line summary (shown in collapsed view).
    details: full args/code (shown when expanded — kept as the original
        input so the widget can syntax-highlight just this part).
    output: text appended after the call returns (stdout, summary, etc.).
    tool_call_id: the model's tool-call id, correlating this event with
        the persisted transcript.
    """

    type: ClassVar[str] = "tool_call_display"

    tool_name: str = ""
    args: str = ""
    details: str = ""
    output: str = ""
    status: str = "running"  # running | ok | error
    tool_call_id: str = ""

    def finish(self, output: str = "", error: str | None = None) -> ToolCallDisplay:
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


@dataclass
class Title(Event):
    """A session retitle (out-of-band titler or ``/title``)."""

    type: ClassVar[str] = "title"

    title: str = ""


@dataclass
class Interrupt(Event):
    """The user or a timeout cut a turn short."""

    type: ClassVar[str] = "interrupt"

    reason: str = ""  # cancelled | timeout


@dataclass
class ApprovalRequest(Event):
    """The agent wants permission to run a dangerous command and has
    suspended until someone answers via ``respond``:

    - ``True`` — run it once
    - ``"__session__"`` — run it and approve ``reason``'s patterns for this session
    - ``"__always__"`` — run it and add the patterns to ``approvals.json``
    - ``False`` — deny
    - any other ``str`` — feedback sent back to the model instead of running

    ``respond`` is an in-process resolve handle (thread-safe, one-shot;
    extra calls are ignored). It is excluded from payloads — a wire layer
    maps it to an RPC keyed by the event ``id``.
    """

    type: ClassVar[str] = "approval_request"

    code: str = ""
    reason: str = ""
    respond: Callable[[bool | str], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class StatusChanged(Event):
    """The agent's working-state label changed ("Thinking...",
    "Running bash...", "" when idle). Replaces polling
    ``agent.status`` — subscribers render the latest label."""

    type: ClassVar[str] = "status_changed"

    label: str = ""


@dataclass
class JobStarted(Event):
    """A background job began (term-N shell
    command, task-N sub-agent)."""

    type: ClassVar[str] = "job_started"

    job_id: str = ""
    kind: str = ""
    label: str = ""


@dataclass
class JobFinished(Event):
    """A background job completed. Carries what the job produced so a
    subscriber can act (e.g. inject a follow-up turn) without scanning
    session state; full logs stay pull-based via the job registry."""

    type: ClassVar[str] = "job_finished"

    job_id: str = ""
    kind: str = ""
    status: str = "ok"  # ok | error
    error: str | None = None
    output: str | None = None
    # Lean refs of artifacts this job produced: {artifact_type, path, url}
    artifact_refs: list[dict[str, str]] = field(default_factory=list)


def event_to_payload(event: Event) -> dict[str, Any]:
    """Serialize an event to its JSON-safe payload dict. ``id`` is
    excluded (store identity is (session_id, event_id); wire identity
    travels beside the payload), as is any non-serializable resolve
    handle (``ApprovalRequest.respond``)."""
    data = asdict(event)
    data.pop("id", None)
    return {k: v for k, v in data.items() if not callable(v)}


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
