"""Session storage. SQLite is the single source of truth for everything:
metadata, the event stream (conversation messages + system events), and
artifacts. ``assets/`` (generated media) is the only per-session file
tree on disk; a session can be dumped to JSONL with
``scripts/export_session.py``.

``Session`` is one sessions-table row that knows how to persist itself:
metadata + accounting (an embedded ``TokenTracker``) plus the attached
conversation body (``state``). ``save`` appends new events; resume reads
the active conversation rows back. ``session_dir`` (``assets/``) is
``config.OKONOMI_SESSIONS_DIR/<session_id>``, a pure function of
the id, never renamed.

Cross-session reads span all sessions, so they live as module-level
functions (``list_sessions`` / ``search`` / ``resolve_session`` /
``prune_empty_sessions`` / favorites), all sharing the store's cached
connection via ``db.connection``.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from .agent.agent import OkonomiState


SessionStatus = Literal["in_progress", "completed", "interrupted", "error", "abandoned"]

from . import db
from .config import OKONOMI_SESSIONS_DIR
from .events import (
    CONVERSATION_TYPES,
    Event,
    EventType,
    classify_message,
    event_to_payload,
    extract_content,
    message_accounting,
)


@dataclass
class TokenTracker:
    """Per-session token + cost accumulator.

    Field semantics mirror pydantic-ai's ``RunUsage``: ``input_tokens``
    is the *non-cached* prompt count; cache hits/writes count separately.
    ``input + cache_read + cache_write`` ≈ the full prompt size sent to
    the provider — the columns are non-overlapping and safe to sum.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0  # cumulative actual USD from OpenRouter
    # Estimated size of the prompt the model will see on the NEXT turn:
    # working-set messages + system prompt + tool/skill index. This is
    # "what's in the context window," not cumulative usage.
    context_tokens: int = 0

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        requests: int = 1,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.reasoning_tokens += reasoning_tokens
        self.requests += requests
        self.cost_usd += cost_usd

    def __str__(self) -> str:
        line = f"total: {self.total_tokens:,} (in: {self.input_tokens:,}, out: {self.output_tokens:,})"
        if self.context_tokens:
            line += f" | context: ~{self.context_tokens:,}"
        return line


_ASSET_EXTENSIONS = {
    "images": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "videos": {".mp4", ".webm", ".mov"},
    "html": {".html", ".htm"},
}


def asset_subdir_for_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    for subdir, extensions in _ASSET_EXTENSIONS.items():
        if ext in extensions:
            return subdir
    return "other"


def new_session_id() -> str:
    """Generate a session id: ``YYYY-MM-DD_<8-char hex>``."""
    return f"{datetime.now():%Y-%m-%d}_{uuid.uuid4().hex[:8]}"


@dataclass
class Artifact:
    """Anything the agent or its tools produced during a session.

    Covers both AI-generated media (with a remote ``url`` that gets
    downloaded into ``path``) and locally-written files (where ``path``
    is set up front and ``url`` is empty).
    """

    artifact_type: str  # "image"|"video"|"audio"|"3d"|"research"|"skill"|"download"|"html"|"other"
    path: str = ""
    url: str = ""
    prompt: str = ""
    tool_name: str = ""
    model_id: str = ""
    description: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Session:
    """One sessions-table row that knows how to persist itself.

    Holds identity, display metadata, lifecycle, accounting (an embedded
    :class:`TokenTracker`, flattened to columns only at the DB boundary),
    and the conversation body as an attached :class:`OkonomiState`.
    ``save`` persists the metadata plus ``state``'s persistent fields;
    reads via :meth:`load` / :meth:`_from_row` return metadata with an
    empty state — callers populate it with ``load_messages`` /
    ``load_artifacts`` when they need the body.

    ``title`` is display-only and owned by the title path (the
    out-of-band titler and ``/title`` → :meth:`set_title`); ``save``
    never writes it after the first insert, so a stale live object can't
    clobber a concurrent retitle.
    """

    session_id: str = ""  # generated in __post_init__ when empty
    parent_session_id: str | None = None  # set on sub-agent forks / compression splits
    # Display
    title: str = ""
    task_input: str = ""
    # Lifecycle
    status: SessionStatus = "completed"
    created_at: str = ""
    updated_at: str = ""
    model: str = ""  # primary LLM model id (the agent's brain)
    # Accounting — token counts + actual USD cost, summed across the run
    tokens: TokenTracker = field(default_factory=TokenTracker)
    # The conversation body (messages, artifacts, working set). Forward-referenced so this module never imports the
    # agent package at runtime; the agent attaches it, and metadata-only
    # reads (``load`` / ``_from_row``) leave it None.
    state: "OkonomiState" = None

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = new_session_id()

    @property
    def session_dir(self) -> Path:
        """This session's folder (``assets/``) —
        a pure function of the id, never renamed."""
        return OKONOMI_SESSIONS_DIR / self.session_id

    # ── lookup ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, session_id: str) -> "Session | None":
        """Metadata-only read: one row → :class:`Session`. No artifacts,
        no events — use ``load_artifacts`` / ``load_messages`` for the
        body."""
        row = db.fetch_session(db.connection(), session_id)
        return cls._from_row(row) if row else None

    @classmethod
    def _from_row(cls, row: dict[str, Any]) -> "Session":
        """Build a Session from a DB row dict, folding the flat token
        columns back into an embedded TokenTracker."""
        g = row.get
        in_toks = int(g("input_tokens") or 0)
        out_toks = int(g("output_tokens") or 0)
        return cls(
            session_id=g("session_id") or "",
            parent_session_id=g("parent_session_id"),
            title=g("title") or "",
            task_input=g("task_input") or "",
            status=g("status") or "completed",
            created_at=g("created_at") or "",
            updated_at=g("updated_at") or "",
            model=g("model") or "",
            tokens=TokenTracker(
                input_tokens=in_toks,
                output_tokens=out_toks,
                total_tokens=in_toks + out_toks,
                cache_read_tokens=int(g("cache_read_tokens") or 0),
                cache_write_tokens=int(g("cache_write_tokens") or 0),
                reasoning_tokens=int(g("reasoning_tokens") or 0),
                requests=int(g("requests") or 0),
                cost_usd=float(g("cost_usd") or 0.0),
            ),
        )

    # ── views ────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.title or (self.task_input[:60] if self.task_input else self.session_id[:8])

    def record_usage(self, **kwargs: Any) -> None:
        """Accumulate one request's token + cost usage. Delegates to the
        embedded :class:`TokenTracker`; see ``TokenTracker.add`` for the
        accepted keyword arguments."""
        self.tokens.add(**kwargs)

    def metadata_dict(self) -> dict[str, Any]:
        """Flat, column-shaped dict for the sessions table (tokens unpacked)."""
        t = self.tokens
        return {
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "title": self.title,
            "task_input": self.task_input,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cache_read_tokens": t.cache_read_tokens,
            "cache_write_tokens": t.cache_write_tokens,
            "reasoning_tokens": t.reasoning_tokens,
            "requests": t.requests,
            "cost_usd": t.cost_usd or None,
        }

    # ── persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist this session in one transaction: the metadata row, the
        *new tail* of the conversation as event rows (the log grows by the
        delta — never rewrites), and artifacts. Identity fields
        (title / created_at) are write-once at the DB layer, so no
        read-modify-write is needed."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "assets").mkdir(exist_ok=True)

        conn = db.connection()
        with db.transaction(conn):
            db.upsert_session(conn, self.metadata_dict())
            self._sync_conversation(conn)
            db.replace_artifacts(conn, self.session_id, [a.to_dict() for a in self.state.artifacts])

    def _sync_conversation(self, conn) -> None:
        """Append the new tail of ``state.message_history`` as event rows.
        A live history *shorter* than the persisted active prefix means a
        rewind (``/clear``): the old events are flagged inactive — kept in
        the DAG, invisible to resume and search — then the history is
        re-appended from scratch."""
        history = self.state.message_history
        persisted = db.count_events(conn, self.session_id, types=CONVERSATION_TYPES)
        if persisted > len(history):
            db.deactivate_events(conn, self.session_id)
            persisted = 0
        new = history[persisted:]
        if new:
            db.append_events(conn, self.session_id, [_message_row(m) for m in new])

    def append_stream_events(self, events: list[Event]) -> None:
        """Persist non-conversation events (system notes, titles,
        interrupts) onto this session's stream."""
        rows = [_event_row(ev) for ev in events]
        if not rows:
            return
        conn = db.connection()
        with db.transaction(conn):
            db.append_events(conn, self.session_id, rows)

    def rewind(self) -> int:
        """Flag every event inactive — history preserved in the DAG,
        invisible to resume and search. Returns the number rewound."""
        conn = db.connection()
        with db.transaction(conn):
            return db.deactivate_events(conn, self.session_id)

    def load_messages(self, *, around_event_id: int | None = None, window: int = 5) -> list[Any]:
        """The active conversation, rebuilt from event rows through the
        pydantic-ai adapter. With ``around_event_id``, only the ±``window``
        slice — for quoting a past moment without loading the transcript."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        conn = db.connection()
        if around_event_id is not None:
            rows = db.fetch_events_window(conn, self.session_id, around_event_id, window, types=CONVERSATION_TYPES)
        else:
            rows = db.fetch_events(conn, self.session_id, types=CONVERSATION_TYPES)
        try:
            raw = [json.loads(r["payload"]) for r in rows]
            return list(ModelMessagesTypeAdapter.validate_python(raw))
        except Exception:
            return []

    def load_artifacts(self) -> list[Artifact]:
        return [Artifact.from_dict(r) for r in db.fetch_artifacts(db.connection(), self.session_id)]

    def artifact_id(self, *, url: str = "", path: str = "") -> int | None:
        """Row id of this session's artifact matching ``url`` or ``path``,
        or None if not yet indexed."""
        return db.find_artifact_id(db.connection(), self.session_id, url=url, path=path)

    # ── title / delete ───────────────────────────────────────────────────

    def set_title(self, title: str) -> bool:
        """Set the display title (never touches the session folder) and
        record the retitle as a ``title`` event in the stream."""
        from .events import Title

        conn = db.connection()
        if db.fetch_session(conn, self.session_id) is None:
            return False
        with db.transaction(conn):
            db.update_session_title(conn, self.session_id, title)
            db.append_events(conn, self.session_id, [_event_row(Title(title))])
        self.title = title
        return True

    def delete(self) -> bool:
        conn = db.connection()
        row = db.fetch_session(conn, self.session_id)
        if row is None and not self.session_dir.exists():
            return False
        shutil.rmtree(self.session_dir, ignore_errors=True)
        with db.transaction(conn):
            db.delete_session(conn, self.session_id)
        return True


# ── event-row builders ───────────────────────────────────────────────────


def _message_row(message: Any) -> dict[str, Any]:
    """Event row for one pydantic-ai message: whole message as payload,
    semantic type from its part composition, extracted text for FTS, and
    per-message accounting on model responses."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    payload = json.loads(ModelMessagesTypeAdapter.dump_json([message]))[0]
    ts = getattr(message, "timestamp", None)
    if ts is None and getattr(message, "parts", None):
        ts = getattr(message.parts[0], "timestamp", None)
    row: dict[str, Any] = {
        "type": classify_message(message),
        "payload": json.dumps(payload, ensure_ascii=False),
        "content": extract_content(message),
        "timestamp": ts.isoformat() if ts is not None else datetime.now().isoformat(),
    }
    accounting = message_accounting(message)
    if accounting:
        row.update(accounting)
    return row


def _event_row(event: Event) -> dict[str, Any]:
    """Event row for a persistable stream event (system note / title /
    interrupt): the typed object's dict as payload, its user-facing text
    as searchable content."""
    payload = event_to_payload(event)
    content = payload.get("text") or payload.get("title") or payload.get("reason") or ""
    return {
        "type": EventType(event.type),
        "payload": json.dumps(payload, ensure_ascii=False),
        "content": str(content),
        "timestamp": datetime.now().isoformat(),
    }


# ── cross-session reads ──────────────────────────────────────────────────


def list_sessions(*, limit: int | None = None, status: str | None = None) -> list[Session]:
    rows = db.list_sessions(db.connection(), status=status, limit=limit)
    return [Session._from_row(r) for r in rows]


def search_sessions(query: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """BM25-ranked FTS5 hits as lean dicts (session_id, title, truncated
    task_input, score, …). Drill into one with ``Session.load``."""
    return db.search_sessions_fts(db.connection(), query, limit=limit)


def search_artifacts(query: str, *, artifact_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.search_artifacts_fts(db.connection(), query, artifact_type=artifact_type, limit=limit)


def search_events(
    query: str,
    *,
    types: tuple[EventType, ...] = (EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE),
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Conversation-content hits with highlighted snippets. Defaults to
    what the user and assistant actually said; tool activity and system
    notes are opt-in via ``types``."""
    return db.search_events_fts(db.connection(), query, types=types, limit=limit)


def search(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """One query over everything: conversation content, session
    titles/tasks, and artifacts, interleaved by BM25 (ascending — FTS
    scores are negative-better; LIKE-fallback artifact hits score 0.0 and
    sort last). Every hit carries enough to act on without a second
    lookup: ``source`` (event | session | artifact), ``session_id``,
    ``session_title``, and the event_id / artifact_id / path / url /
    snippet that pins the thing itself."""
    hits: list[dict[str, Any]] = []
    for h in search_events(query, limit=limit):
        hits.append(
            {
                "source": "event",
                "session_id": h["session_id"],
                "session_title": h.get("session_title") or "",
                "event_id": h["event_id"],
                "snippet": h.get("snippet") or "",
                "timestamp": h.get("timestamp") or "",
                "score": h.get("score") or 0.0,
            }
        )
    for h in search_sessions(query, limit=limit):
        hits.append(
            {
                "source": "session",
                "session_id": h["session_id"],
                "session_title": h.get("title") or "",
                "snippet": h.get("task_input") or "",
                "num_artifacts": h.get("num_artifacts") or 0,
                "timestamp": h.get("created_at") or "",
                "score": h.get("score") or 0.0,
            }
        )
    for h in search_artifacts(query, limit=limit):
        hits.append(
            {
                "source": "artifact",
                "session_id": h["session_id"],
                "session_title": h.get("title") or "",
                "artifact_id": h.get("artifact_id"),
                "artifact_type": h.get("artifact_type") or "",
                "path": h.get("path") or "",
                "url": h.get("url") or "",
                "snippet": h.get("prompt") or h.get("description") or "",
                "timestamp": h.get("created_at") or "",
                "score": h.get("score") or 0.0,
            }
        )
    hits.sort(key=lambda h: h["score"])
    return hits[:limit]


def read_window(session_id: str, around_event_id: int, *, window: int = 5) -> list[dict[str, Any]]:
    """Lean conversation rows around one event — for scrolling from a
    search hit without loading payloads."""
    rows = db.fetch_events_window(db.connection(), session_id, around_event_id, window, types=CONVERSATION_TYPES)
    return [_lean_event(r) for r in rows]


def _lean_event(r: dict[str, Any]) -> dict[str, Any]:
    return {"event_id": r["event_id"], "type": r["type"], "content": r["content"], "timestamp": r["timestamp"]}


def session_bookends(
    session_id: str, *, first_n: int = 3, last_n: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """First/last user+assistant exchanges of a session — the goal and
    the resolution, as lean dicts with ``content`` text."""
    first, last = db.fetch_bookends(
        db.connection(),
        session_id,
        first_n=first_n,
        last_n=last_n,
        types=(EventType.USER_MESSAGE, EventType.ASSISTANT_MESSAGE),
    )
    return [_lean_event(r) for r in first], [_lean_event(r) for r in last]


def resolve_session(query: str) -> Session | None:
    """Match free-form input to a session. Tries exact id, id prefix,
    title substring, then BM25 FTS. First hit wins."""
    q = query.strip()
    if not q:
        return None

    conn = db.connection()

    if row := db.fetch_session(conn, q):
        return Session._from_row(row)

    if row := db.resolve_session_loose(conn, q):
        return Session._from_row(row)

    hits = db.search_sessions_fts(conn, q, limit=1)
    return Session.load(hits[0]["session_id"]) if hits else None


def prune_empty_sessions(*, min_age_hours: int = 24) -> int:
    """Delete sessions with no artifacts.

    Truly empty rows (no task_input, no title) go regardless of age —
    they never got past the prompt. Rows with content but no
    artifacts get ``min_age_hours`` of grace so an in-flight
    generation isn't killed.
    """
    conn = db.connection()
    cutoff = (datetime.now() - timedelta(hours=min_age_hours)).isoformat()
    rows = db.find_prunable_sessions(conn, cutoff)
    if not rows:
        return 0

    with db.transaction(conn):
        for r in rows:
            shutil.rmtree(OKONOMI_SESSIONS_DIR / r["session_id"], ignore_errors=True)
            db.delete_session(conn, r["session_id"])
    return len(rows)


# ── favorites ────────────────────────────────────────────────────────────


def favorite(*, artifact_id: int | None = None, url: str = "", artifact_type: str = "") -> int:
    """Heart an asset — internal (``artifact_id``) or external (``url``,
    which accepts URLs or bare local paths). Idempotent; returns the
    favorite row id."""
    conn = db.connection()
    with db.transaction(conn):
        return db.add_favorite(conn, artifact_id=artifact_id, url=url, artifact_type=artifact_type)


def unfavorite(favorite_id: int) -> bool:
    conn = db.connection()
    with db.transaction(conn):
        return db.remove_favorite(conn, favorite_id)


def list_favorites(
    *,
    artifact_type: str | None = None,
    artifact_source: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Newest-first favorites joined with artifact + session context.

    Returns plain dicts (not Artifact) since each row carries
    cross-session fields (``session_id``, ``session_title``,
    ``favorited_at``, ``note``) that don't fit the Artifact dataclass.
    """
    return db.list_favorites(
        db.connection(),
        artifact_type=artifact_type,
        artifact_source=artifact_source,
        limit=limit,
    )
