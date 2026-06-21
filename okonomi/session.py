"""Session storage. SQLite is the single source of truth for metadata
and artifacts; ``messages.jsonl`` (per-session) is the only on-disk
sidecar — it holds the conversation transcript, which is large,
append-friendly, and never queried.

One dataclass, ``Session``, mirrors the sessions table 1:1 plus carries
optional ``artifacts`` and ``message_history``. Listing operations
(``list_sessions`` / ``search_sessions`` / ``resolve_session``) return
Sessions with the body fields empty — they're index reads.
``load_session`` populates everything.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional


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
class Session:
    """Sessions table row + (optionally) the artifacts list and message history.

    Index reads (list/search/resolve) leave ``artifacts`` empty and
    ``message_history`` ``None``. ``load_session`` populates both.
    """

    session_id: str
    parent_session_id: Optional[str] = None  # set on sub-agent forks / compression splits
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
    # Tokens — non-overlapping; sum = total billed surface
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    # Cost — actual USD reported by OpenRouter (`usage.cost`), summed across the run
    cost_usd: Optional[float] = None
    # Body
    steps_summary: str = ""
    # Optional payloads — populated only by load_session()
    artifacts: list[Artifact] = field(default_factory=list)
    message_history: Optional[list[Any]] = None

    @property
    def display_name(self) -> str:
        return self.title or (self.task_input[:60] if self.task_input else self.session_id[:8])

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.reasoning_tokens
        )

    def metadata_dict(self) -> dict[str, Any]:
        """Row-shaped dict — drops the body payloads (artifacts, message_history)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("artifacts", "message_history")}


@dataclass
class OkonomiState:
    """Per-session runtime state the agent mutates during a run.

    Distinct from :class:`Session` (the persisted snapshot owned by
    SessionManager). ``OkonomiState`` is the live mutable bag — message
    history, working context, artifacts, tokens, REPL globals — that
    the agent reads/writes on every turn. ``save_session`` projects
    OkonomiState into a Session for SQLite + jsonl persistence;
    ``load_session`` materializes a fresh OkonomiState from a Session
    on resume.
    """

    session_id: str
    task: str = ""
    # Append-only audit log: every message ever produced this session.
    # Persisted to messages.jsonl. Compression never touches it.
    message_history: list[Any] = field(default_factory=list)
    # Working set sent to the LLM each turn. Sanitized and compressed
    # as needed. Not persisted separately — recomputed from
    # message_history on resume.
    context: list[Any] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    tokens: TokenTracker = field(default_factory=TokenTracker)
    # REPL state across execute_code turns (model-bound locals).
    execution_globals: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Thin façade over the SQLite store + ``messages.jsonl`` sidecar.

    All reads return :class:`Session`. Index reads (``list_sessions`` /
    ``search_sessions`` / ``resolve_session``) leave ``artifacts`` empty
    and ``message_history`` ``None``; ``load_session`` populates both.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._db = _db.open_db(self.storage_dir / _db.DB_FILENAME)

    # ── creation ─────────────────────────────────────────────────────────

    def create_session(self, task_input: str) -> Session:
        now = datetime.now().isoformat()
        return Session(
            session_id=new_session_id(),
            task_input=task_input,
            status="in_progress",
            created_at=now,
            updated_at=now,
        )

    # ── reads ────────────────────────────────────────────────────────────

    def list_sessions(self, *, limit: Optional[int] = None, status: Optional[str] = None) -> list[Session]:
        return [_session_from_row(r) for r in _db.list_sessions(self._db, status=status, limit=limit)]

    def search_sessions(self, query: str, *, limit: Optional[int] = None) -> list[Session]:
        return [_session_from_row(r) for r in _db.search_sessions_fts(self._db, query, limit=limit or 50)]

    def search_artifacts(
        self, query: str, *, artifact_type: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return _db.search_artifacts_fts(self._db, query, artifact_type=artifact_type, limit=limit)

    def get_artifacts(self, session_id: str) -> list[Artifact]:
        return [Artifact.from_dict(r) for r in _db.fetch_artifacts(self._db, session_id)]

    def resolve_session(self, query: str) -> Optional[Session]:
        """Match free-form input to a session. Tries exact id, id prefix,
        title substring, then BM25 FTS. First hit wins."""
        q = query.strip()
        if not q:
            return None

        if row := _db.fetch_session(self._db, q):
            return _session_from_row(row)

        row = self._db.execute(
            "SELECT * FROM sessions WHERE session_id LIKE ? OR LOWER(title) LIKE LOWER(?) "
            "ORDER BY updated_at DESC LIMIT 1",
            (q + "%", f"%{q}%"),
        ).fetchone()
        if row:
            return _session_from_row(dict(row))

        hits = _db.search_sessions_fts(self._db, q, limit=1)
        return _session_from_row(hits[0]) if hits else None

    def load_session(self, session_id: str, *, with_history: bool = True) -> Optional[Session]:
        """Load full session: metadata row + artifacts + (optional) history.

        Set ``with_history=False`` for callers that just need metadata +
        artifacts — skips the ``messages.jsonl`` read (which can be MBs).
        """
        row = _db.fetch_session(self._db, session_id)
        if row is None:
            return None
        sess = _session_from_row(row)
        sess.artifacts = [Artifact.from_dict(r) for r in _db.fetch_artifacts(self._db, session_id)]
        if with_history:
            dir_name = row.get("dir_name") or session_id
            messages_file = self.storage_dir / dir_name / "messages.jsonl"
            try:
                raw = [json.loads(line) for line in messages_file.read_text().splitlines() if line.strip()]
                from pydantic_ai.messages import ModelMessagesTypeAdapter

                sess.message_history = ModelMessagesTypeAdapter.validate_python(raw)
            except Exception:
                pass
        return sess

    # ── writes ───────────────────────────────────────────────────────────

    def save_session(self, session: Session) -> None:
        """Persist metadata + artifacts to SQLite, transcript to ``messages.jsonl``."""
        if not session.dir_name:
            session.dir_name = _derive_dir_name(session.session_id, session.title)

        session.updated_at = datetime.now().isoformat()
        session.num_artifacts = len(session.artifacts)

        session_dir = self.storage_dir / session.dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "assets").mkdir(exist_ok=True)

        if session.message_history:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            serialized = json.loads(ModelMessagesTypeAdapter.dump_json(session.message_history))
            with open(session_dir / "messages.jsonl", "w") as f:
                for msg in serialized:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        with _db.transaction(self._db):
            _db.upsert_session(self._db, session.metadata_dict())
            _db.replace_artifacts(self._db, session.session_id, [a.to_dict() for a in session.artifacts])

        # Titles are generated out-of-band by a small model, fire-and-forget,
        # so the agent never blocks on (or owns) titling. Refresh on every
        # completed turn so the title tracks the whole conversation.
        if session.status == "completed" and session.message_history:
            from .utils import title_session_async

            title_session_async(self, session.session_id, session.message_history)

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
        artifact_id: Optional[int] = None,
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
        artifact_type: Optional[str] = None,
        artifact_source: Optional[str] = None,
        limit: Optional[int] = None,
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


def _session_from_row(row: dict[str, Any]) -> Session:
    """Build a Session from a DB row dict, dropping unknown keys."""
    known = {f.name for f in fields(Session)}
    return Session(**{k: v for k, v in row.items() if k in known})


def _derive_dir_name(session_id: str, title: str = "") -> str:
    if not title:
        return session_id
    slug = _slugify(title)
    if not slug:
        return session_id
    date_part = session_id[:10] if len(session_id) > 10 else f"{datetime.now():%Y-%m-%d}"
    return f"{date_part}_{slug}"
