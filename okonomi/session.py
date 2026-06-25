"""Session storage. SQLite is the single source of truth for metadata
and artifacts; ``messages.jsonl`` (per-session) is the only on-disk
sidecar — it holds the conversation transcript, which is large,
append-friendly, and never queried.

``SessionInfo`` mirrors the sessions table 1:1 — pure metadata +
accounting (an embedded ``TokenTracker``). Index reads
(``list_sessions`` / ``search_sessions`` / ``resolve_session`` /
``get_info``) return ``SessionInfo``. ``load_session`` returns the full
``OkonomiState`` (info + artifacts + message history); ``save_session``
persists one back.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal


SessionStatus = Literal["in_progress", "completed", "interrupted", "error", "abandoned"]

from . import db as _db


_DEFAULT_STORAGE_DIR = Path.home() / ".okonomi" / "sessions"


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


def _slugify(text: str, max_len: int = 50) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].rstrip("-")


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
class SessionInfo:
    """Sessions table row 1:1 — pure metadata + accounting.

    Returned by index reads (``list_sessions`` / ``search_sessions`` /
    ``resolve_session`` / ``get_info``) and composed into
    :class:`OkonomiState` as the "permanent" half. The conversation body
    (``artifacts`` / ``message_history``) lives on ``OkonomiState``, not
    here. Token/cost accounting rides along as an embedded
    :class:`TokenTracker`, flattened to columns only at the DB boundary.
    """

    session_id: str
    parent_session_id: str | None = None  # set on sub-agent forks / compression splits
    # Display
    title: str = ""
    task_input: str = ""
    dir_name: str = ""
    # Lifecycle
    status: SessionStatus = "completed"
    created_at: str = ""
    updated_at: str = ""
    # Aggregates
    num_artifacts: int = 0
    model: str = ""  # primary LLM model id (the agent's brain)
    steps_summary: str = ""
    # Accounting — token counts + actual USD cost, summed across the run
    tokens: TokenTracker = field(default_factory=TokenTracker)

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
            "dir_name": self.dir_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_artifacts": self.num_artifacts,
            "model": self.model,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cache_read_tokens": t.cache_read_tokens,
            "cache_write_tokens": t.cache_write_tokens,
            "reasoning_tokens": t.reasoning_tokens,
            "requests": t.requests,
            "cost_usd": t.cost_usd or None,
            "steps_summary": self.steps_summary,
        }


@dataclass
class OkonomiState:
    """The live, mutable state the agent reads/writes every turn.

    Single source of truth for a running session. Holds permanent data
    (``info`` — identity, task, accounting, persisted metadata; plus the
    ``message_history`` / ``artifacts`` body) alongside ephemeral runtime
    (``context``, ``tools_used``, ``execution_globals``).
    ``SessionManager.save_session`` writes this out;
    ``SessionManager.load_session`` reconstructs it on resume.
    """

    # Permanent metadata + accounting (identity, task, tokens, …).
    # Mirrors one sessions-table row; persisted as-is by SessionManager.
    info: SessionInfo
    # Append-only audit log: every message ever produced this session.
    # Persisted to messages.jsonl. Compression never touches it.
    message_history: list[Any] = field(default_factory=list)
    # Working set sent to the LLM each turn. Sanitized and compressed
    # as needed. Not persisted separately — recomputed from
    # message_history on resume.
    context: list[Any] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    # REPL state across execute_code turns (model-bound locals).
    execution_globals: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Thin façade over the SQLite store + ``messages.jsonl`` sidecar.

    Index reads (``list_sessions`` / ``search_sessions`` /
    ``resolve_session`` / ``get_info``) return :class:`SessionInfo`
    (metadata only). ``load_session`` returns a full :class:`OkonomiState`
    (info + artifacts + message history); ``save_session`` persists one.
    """

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._db = _db.open_db(self.storage_dir / _db.DB_FILENAME)

    # ── creation ─────────────────────────────────────────────────────────

    def create_session(self, task_input: str) -> OkonomiState:
        now = datetime.now().isoformat()
        return OkonomiState(
            info=SessionInfo(
                session_id=new_session_id(),
                task_input=task_input,
                status="in_progress",
                created_at=now,
                updated_at=now,
            )
        )

    # ── reads ────────────────────────────────────────────────────────────

    def list_sessions(self, *, limit: int | None = None, status: str | None = None) -> list[SessionInfo]:
        return [_info_from_row(r) for r in _db.list_sessions(self._db, status=status, limit=limit)]

    def search_sessions(self, query: str, *, limit: int | None = None) -> list[SessionInfo]:
        return [_info_from_row(r) for r in _db.search_sessions_fts(self._db, query, limit=limit or 50)]

    def search_artifacts(
        self, query: str, *, artifact_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return _db.search_artifacts_fts(self._db, query, artifact_type=artifact_type, limit=limit)

    def get_artifacts(self, session_id: str) -> list[Artifact]:
        return [Artifact.from_dict(r) for r in _db.fetch_artifacts(self._db, session_id)]

    def resolve_session(self, query: str) -> SessionInfo | None:
        """Match free-form input to a session. Tries exact id, id prefix,
        title substring, then BM25 FTS. First hit wins."""
        q = query.strip()
        if not q:
            return None

        if row := _db.fetch_session(self._db, q):
            return _info_from_row(row)

        row = self._db.execute(
            "SELECT * FROM sessions WHERE session_id LIKE ? OR LOWER(title) LIKE LOWER(?) "
            "ORDER BY updated_at DESC LIMIT 1",
            (q + "%", f"%{q}%"),
        ).fetchone()
        if row:
            return _info_from_row(dict(row))

        hits = _db.search_sessions_fts(self._db, q, limit=1)
        return _info_from_row(hits[0]) if hits else None

    def get_info(self, session_id: str) -> SessionInfo | None:
        """Metadata-only read: one row → :class:`SessionInfo`. No artifacts,
        no ``messages.jsonl``. For callers that just need title / status /
        token totals without paying for the body."""
        row = _db.fetch_session(self._db, session_id)
        return _info_from_row(row) if row else None

    def load_session(self, session_id: str) -> OkonomiState | None:
        """Full load: metadata row + artifacts + message history → a fresh
        :class:`OkonomiState`. ``context`` and ``execution_globals`` are
        left empty for the agent to populate on resume."""
        row = _db.fetch_session(self._db, session_id)
        if row is None:
            return None
        info = _info_from_row(row)
        artifacts = [Artifact.from_dict(r) for r in _db.fetch_artifacts(self._db, session_id)]
        info.num_artifacts = len(artifacts)

        message_history: list[Any] = []
        dir_name = row.get("dir_name") or session_id
        messages_file = self.storage_dir / dir_name / "messages.jsonl"
        try:
            raw = [json.loads(line) for line in messages_file.read_text().splitlines() if line.strip()]
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            message_history = list(ModelMessagesTypeAdapter.validate_python(raw))
        except Exception:
            message_history = []

        return OkonomiState(info=info, message_history=message_history, artifacts=artifacts)

    # ── writes ───────────────────────────────────────────────────────────

    def save_session(self, state: OkonomiState) -> None:
        """Persist ``state``: metadata + artifacts to SQLite, transcript to
        ``messages.jsonl``.

        Identity-shaping fields are preserved across saves: the
        out-of-band ``title``, the original ``dir_name`` and ``created_at``
        are carried forward from the existing row so mid-session title
        regen never relocates the directory or resets creation time.
        """
        info = state.info

        existing = _db.fetch_session(self._db, info.session_id)
        if existing:
            info.title = existing.get("title") or info.title
            info.created_at = existing.get("created_at") or info.created_at
            if existing.get("dir_name"):
                info.dir_name = existing["dir_name"]
        if not info.created_at:
            info.created_at = datetime.now().isoformat()
        if not info.dir_name:
            info.dir_name = _derive_dir_name(info.session_id, info.title)

        info.updated_at = datetime.now().isoformat()
        info.num_artifacts = len(state.artifacts)

        session_dir = self.storage_dir / info.dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "assets").mkdir(exist_ok=True)

        if state.message_history:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            serialized = json.loads(ModelMessagesTypeAdapter.dump_json(state.message_history))
            with open(session_dir / "messages.jsonl", "w") as f:
                for msg in serialized:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        with _db.transaction(self._db):
            _db.upsert_session(self._db, info.metadata_dict())
            _db.replace_artifacts(self._db, info.session_id, [a.to_dict() for a in state.artifacts])

        # Titles are generated out-of-band by a small model, fire-and-forget,
        # so the agent never blocks on (or owns) titling. Refresh on every
        # completed turn so the title tracks the whole conversation.
        if info.status == "completed" and state.message_history:
            from .utils import _maybe_update_session_title_async

            _maybe_update_session_title_async(self, info.session_id, state.message_history)

    def update_title(self, session_id: str, title: str) -> bool:
        """Update only the stored title, preserving the existing dir_name (no
        directory rename). Used by the out-of-band titler."""
        row = _db.fetch_session(self._db, session_id)
        if row is None:
            return False
        with _db.transaction(self._db):
            _db.update_session_title(self._db, session_id, title, row.get("dir_name") or session_id)
        return True

    def set_title(self, session_id: str, title: str) -> bool:
        """Set a title and rename the session directory accordingly."""
        row = _db.fetch_session(self._db, session_id)
        if row is None:
            return False
        old_dir = self.storage_dir / (row.get("dir_name") or session_id)
        new_dir_name = _derive_dir_name(session_id, title)
        new_dir = self.storage_dir / new_dir_name
        if new_dir.exists() and new_dir != old_dir:
            return False
        if old_dir.exists() and old_dir != new_dir:
            old_dir.rename(new_dir)

        with _db.transaction(self._db):
            _db.update_session_title(self._db, session_id, title, new_dir_name)
        return True

    # ── delete / prune ───────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> bool:
        row = _db.fetch_session(self._db, session_id)
        session_dir = self.storage_dir / ((row or {}).get("dir_name") or session_id)
        if row is None and not session_dir.exists():
            return False
        shutil.rmtree(session_dir, ignore_errors=True)
        with _db.transaction(self._db):
            _db.delete_session(self._db, session_id)
        return True

    def prune_empty_sessions(self, *, min_age_hours: int = 24) -> int:
        """Delete sessions with no artifacts.

        Truly empty rows (no task_input, no title) go regardless of age —
        they never got past the prompt. Rows with content but no
        artifacts get ``min_age_hours`` of grace so an in-flight
        generation isn't killed.
        """
        cutoff = (datetime.now() - timedelta(hours=min_age_hours)).isoformat()
        rows = self._db.execute(
            "SELECT session_id, dir_name FROM sessions WHERE num_artifacts = 0 "
            "AND ((task_input = '' AND title = '') OR created_at < ?)",
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0

        with _db.transaction(self._db):
            for r in rows:
                shutil.rmtree(self.storage_dir / (r["dir_name"] or r["session_id"]), ignore_errors=True)
                _db.delete_session(self._db, r["session_id"])
        return len(rows)

    # ── favorites ────────────────────────────────────────────────────────

    def favorite(
        self,
        *,
        artifact_id: int | None = None,
        url: str = "",
        artifact_type: str = "",
    ) -> int:
        """Heart an asset — internal (``artifact_id``) or external (``url``,
        which accepts URLs or bare local paths). Idempotent; returns the
        favorite row id."""
        with _db.transaction(self._db):
            return _db.add_favorite(
                self._db,
                artifact_id=artifact_id,
                url=url,
                artifact_type=artifact_type,
            )

    def unfavorite(self, favorite_id: int) -> bool:
        with _db.transaction(self._db):
            return _db.remove_favorite(self._db, favorite_id)

    def list_favorites(
        self,
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
        return _db.list_favorites(
            self._db,
            artifact_type=artifact_type,
            artifact_source=artifact_source,
            limit=limit,
        )

    # ── paths ────────────────────────────────────────────────────────────

    def session_dir(self, session_id: str) -> Path:
        row = _db.fetch_session(self._db, session_id)
        name = (row or {}).get("dir_name") or session_id
        return self.storage_dir / name


def _info_from_row(row: dict[str, Any]) -> SessionInfo:
    """Build a SessionInfo from a DB row dict, folding the flat token
    columns back into an embedded TokenTracker."""
    g = row.get
    in_toks = int(g("input_tokens") or 0)
    out_toks = int(g("output_tokens") or 0)
    return SessionInfo(
        session_id=g("session_id") or "",
        parent_session_id=g("parent_session_id"),
        title=g("title") or "",
        task_input=g("task_input") or "",
        dir_name=g("dir_name") or "",
        status=g("status") or "completed",
        created_at=g("created_at") or "",
        updated_at=g("updated_at") or "",
        num_artifacts=int(g("num_artifacts") or 0),
        model=g("model") or "",
        steps_summary=g("steps_summary") or "",
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


def _derive_dir_name(session_id: str, title: str = "") -> str:
    if not title:
        return session_id
    slug = _slugify(title)
    if not slug:
        return session_id
    date_part = session_id[:10] if len(session_id) > 10 else f"{datetime.now():%Y-%m-%d}"
    return f"{date_part}_{slug}"
