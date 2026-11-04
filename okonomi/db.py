"""SQLite store: the single source of truth for sessions, the event
stream (conversation + system events), and artifacts, with FTS5 search
over all of it.

Tables:

  sessions       — one row per session_id, mirrors Session 1:1
  events         — the event stream; conversation messages are rows whose
                   payload is one pydantic-ai message JSON
  artifacts      — one row per artifact, FK to sessions
  favorites      — user-curated artifact bookmarks
  sessions_fts   — FTS5 mirror of (title, task_input)
  events_fts     — FTS5 mirror of events.content (write-time extracted text)
  artifacts_fts  — FTS5 mirror of (prompt, description, tool_name)

Triggers keep the FTS tables synced on insert/delete/update, in the same
transaction as the write — the indexes cannot drift.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .config import OKONOMI_SESSIONS_DIR


DB_FILENAME = "okonomi.db"


# Single source of truth for session columns. Each entry is
# ``(name, kind, decl)`` where ``kind`` drives runtime coercion in
# ``_meta_to_row`` and ``decl`` drives DDL generation. Order is the
# INSERT column order — keep IDs first.
#
# ``kind``: "int" coerces via ``int(... or 0)``; "real" allows None or
# float; "text_nn" is non-null TEXT (None → ""); "text" is nullable TEXT.
SESSION_COLUMNS: list[tuple[str, str, str]] = [
    # Identity
    ("session_id", "text_nn", "TEXT PRIMARY KEY"),
    ("parent_session_id", "text", "TEXT REFERENCES sessions(session_id) ON DELETE SET NULL"),
    # Display
    ("title", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    ("task_input", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    # Lifecycle
    ("status", "text_nn", "TEXT NOT NULL DEFAULT 'completed'"),
    ("created_at", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    ("model", "text_nn", "TEXT NOT NULL DEFAULT ''"),  # primary LLM model id
    # Tokens — non-overlapping; `input_tokens` is the non-cached prompt count.
    ("input_tokens", "int", "INTEGER NOT NULL DEFAULT 0"),
    ("output_tokens", "int", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_read_tokens", "int", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_write_tokens", "int", "INTEGER NOT NULL DEFAULT 0"),
    ("reasoning_tokens", "int", "INTEGER NOT NULL DEFAULT 0"),
    ("requests", "int", "INTEGER NOT NULL DEFAULT 0"),
    # Cost — actual USD reported by OpenRouter
    ("cost_usd", "real", "REAL"),
]


def _build_session_ddl() -> str:
    cols = ",\n    ".join(f"{name:<22} {decl}" for name, _kind, decl in SESSION_COLUMNS)
    return f"CREATE TABLE IF NOT EXISTS sessions (\n    {cols}\n);"


_SCHEMA = (
    _build_session_ddl()
    + """

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status  ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_parent  ON sessions(parent_session_id);

-- The event stream. ``type`` is an okonomi.events.EventType value;
-- ``payload`` is the record (a pydantic-ai message JSON for conversation
-- types, a typed-event dict otherwise); ``content`` is the write-time
-- extracted text the FTS index sees (lossy — payload is the record).
-- ``parent_id``/``active`` give DAG lineage with a materialized active
-- path: resume filters active=1, rewind flips flags instead of deleting.
-- Accounting columns are populated on model-response rows only.
-- NOTE: events_fts maps through this table's implicit rowid; if VACUUM
-- is ever introduced, rebuild the FTS tables afterward (VACUUM may
-- renumber implicit rowids).
CREATE TABLE IF NOT EXISTS events (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_id    INTEGER NOT NULL,
    parent_id   INTEGER,
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    timestamp   TEXT NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,
    reasoning_tokens    INTEGER,
    cost_usd            REAL,
    PRIMARY KEY (session_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session_active ON events(session_id, active, event_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    artifact_type   TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    path            TEXT NOT NULL DEFAULT '',
    prompt          TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    tool_name       TEXT NOT NULL DEFAULT '',
    model_id        TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type    ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_ts      ON artifacts(timestamp DESC);

-- User-curated favorites. Two flavors share one table:
--   internal: artifact_id set → CASCADE-deletes with the artifact
--   external: artifact_id NULL → user bookmarked a bare URL or local path
-- Dedup via partial unique indexes: same artifact / same URL / same path
-- can each be favorited at most once.
CREATE TABLE IF NOT EXISTS favorites (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id     INTEGER REFERENCES artifacts(id) ON DELETE CASCADE,
    artifact_source TEXT NOT NULL DEFAULT 'external',   -- 'internal' | 'external'
    url             TEXT NOT NULL DEFAULT '',           -- remote URL or local filesystem path
    artifact_type   TEXT NOT NULL DEFAULT '',
    favorited_at    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX        IF NOT EXISTS idx_favorites_at       ON favorites(favorited_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_favorites_artifact ON favorites(artifact_id) WHERE artifact_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_favorites_url      ON favorites(url)         WHERE artifact_id IS NULL AND url != '';

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    title, task_input,
    tokenize = 'porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    prompt, description, tool_name,
    tokenize = 'porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    content,
    tokenize = 'porter unicode61'
);

-- Indexed terms, for expanding a typed prefix (see expand_prefixes).
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_vocab  USING fts5vocab(sessions_fts, row);
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_vocab USING fts5vocab(artifacts_fts, row);
CREATE VIRTUAL TABLE IF NOT EXISTS events_vocab    USING fts5vocab(events_fts, row);

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.rowid;
    INSERT INTO events_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    DELETE FROM sessions_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, title, task_input)
    VALUES (new.rowid, new.title, new.task_input);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    DELETE FROM sessions_fts WHERE rowid = old.rowid;
    INSERT INTO sessions_fts(rowid, title, task_input)
    VALUES (new.rowid, new.title, new.task_input);
END;

CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
    INSERT INTO artifacts_fts(rowid, prompt, description, tool_name)
    VALUES (new.id, new.prompt, new.description, new.tool_name);
END;

CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
    DELETE FROM artifacts_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
    DELETE FROM artifacts_fts WHERE rowid = old.id;
    INSERT INTO artifacts_fts(rowid, prompt, description, tool_name)
    VALUES (new.id, new.prompt, new.description, new.tool_name);
END;
"""
)


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create + initialize) the SQLite DB. WAL for concurrent
    reads, foreign keys on, dict-style row access.

    ``check_same_thread=False`` is required because the connection is
    created on the main thread but reads happen on the agent worker
    thread (e.g. when the ``recall`` tool fires mid-turn). WAL mode + serialized writes through
    ``transaction()`` make this safe; the Python-side guard would
    otherwise raise a ``ProgrammingError`` on every cross-thread access.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)

    return conn


_connection: sqlite3.Connection | None = None
_connection_lock = threading.Lock()


def connection() -> sqlite3.Connection:
    """The store's shared connection, opened on first use — every
    caller (agent, tools, pickers, maintenance scripts) hits the same
    handle instead of each opening its own."""
    global _connection
    with _connection_lock:
        if _connection is None:
            _connection = open_db(OKONOMI_SESSIONS_DIR / DB_FILENAME)
        return _connection


# ── sessions ────────────────────────────────────────────────────────────


_COERCE = {
    "int": lambda v: int(v or 0),
    "real": lambda v: None if v is None else float(v),
    "text_nn": lambda v: "" if v is None else str(v),
    "text": lambda v: v,
}


def _meta_to_row(meta: dict[str, Any]) -> tuple:
    """Coerce a Session-shaped dict into the column-ordered tuple.

    Sparse dicts produce valid rows; unknown keys are ignored.
    """
    return tuple(_COERCE[kind](meta.get(name)) for name, kind, _decl in SESSION_COLUMNS)


# Written once at INSERT, never touched by upserts: identity fields plus
# ``title``, which is owned by the title path (out-of-band titler /
# ``set_title``) — a stale live object saving mid-run must not clobber it.
_WRITE_ONCE_COLS = {"session_id", "title", "created_at"}


def upsert_session(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    """Insert or update a session row. ``meta`` mirrors ``Session``.
    Write-once columns are set at insert and excluded from the update."""
    cols = [name for name, _kind, _decl in SESSION_COLUMNS]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in _WRITE_ONCE_COLS)
    conn.execute(
        f"INSERT INTO sessions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
        _meta_to_row(meta),
    )


def fetch_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(conn: sqlite3.Connection, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Newest-first by ``updated_at``."""
    sql = "SELECT * FROM sessions ORDER BY updated_at DESC"
    params: list[Any] = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Cascade-deletes artifacts via FK ON DELETE CASCADE."""
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def resolve_session_loose(conn: sqlite3.Connection, query: str) -> dict[str, Any] | None:
    """Fuzzy single-row lookup: session_id prefix or title substring,
    most recently updated first."""
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id LIKE ? OR LOWER(title) LIKE LOWER(?) "
        "ORDER BY updated_at DESC LIMIT 1",
        (query + "%", f"%{query}%"),
    ).fetchone()
    return dict(row) if row else None


def find_prunable_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Sessions with no artifacts, no task_input and no title."""
    rows = conn.execute(
        "SELECT session_id FROM sessions "
        "WHERE NOT EXISTS (SELECT 1 FROM artifacts a WHERE a.session_id = sessions.session_id) "
        "AND task_input = '' AND title = ''",
    ).fetchall()
    return [dict(r) for r in rows]


def update_session_title(conn: sqlite3.Connection, session_id: str, title: str) -> None:
    """Targeted UPDATE for the title path — the one writer of the title
    column (``upsert_session`` never touches it after insert)."""
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (title, datetime.now().isoformat(), session_id),
    )


# ── artifacts ───────────────────────────────────────────────────────────


def replace_artifacts(conn: sqlite3.Connection, session_id: str, artifacts: Iterable[dict[str, Any]]) -> None:
    """Drop and re-insert all artifacts for a session. Cheap — artifact
    lists are small and append-only within a turn."""
    conn.execute("DELETE FROM artifacts WHERE session_id = ?", (session_id,))
    rows = [
        (
            session_id,
            a.get("artifact_type") or "",
            a.get("url") or "",
            a.get("path") or "",
            a.get("prompt") or "",
            a.get("description") or "",
            a.get("tool_name") or "",
            a.get("model_id") or "",
            a.get("timestamp") or "",
        )
        for a in artifacts
    ]
    if rows:
        conn.executemany(
            "INSERT INTO artifacts (session_id, artifact_type, url, path, prompt, "
            "description, tool_name, model_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def fetch_artifacts(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT artifact_type, url, path, prompt, description, "
        "tool_name, model_id, timestamp FROM artifacts WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── events ──────────────────────────────────────────────────────────────


_EVENT_INSERT_COLS = (
    "session_id", "event_id", "parent_id", "type", "payload", "content", "timestamp",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "cost_usd",
)


def _type_values(types: Iterable[Any]) -> list[str]:
    return [t.value if isinstance(t, Enum) else str(t) for t in types]


def append_events(conn: sqlite3.Connection, session_id: str, rows: Iterable[dict[str, Any]]) -> int:
    """Append events to a session's stream. Each row dict carries ``type``,
    ``payload``, ``content``, ``timestamp`` and optionally the accounting
    values. ``event_id`` continues the session's insertion order (across
    inactive rows too, so ids never collide after a rewind); ``parent_id``
    chains from the last *active* event. Returns the number inserted."""
    rows = list(rows)
    if not rows:
        return 0
    next_id_row = conn.execute(
        "SELECT MAX(event_id) FROM events WHERE session_id = ?", (session_id,)
    ).fetchone()
    next_id = (next_id_row[0] + 1) if next_id_row[0] is not None else 0
    parent_row = conn.execute(
        "SELECT MAX(event_id) FROM events WHERE session_id = ? AND active = 1", (session_id,)
    ).fetchone()
    parent = parent_row[0] if parent_row[0] is not None else None

    placeholders = ", ".join("?" for _ in _EVENT_INSERT_COLS)
    sql = f"INSERT INTO events ({', '.join(_EVENT_INSERT_COLS)}) VALUES ({placeholders})"
    for r in rows:
        conn.execute(sql, (
            session_id, next_id, parent,
            r["type"].value if isinstance(r["type"], Enum) else str(r["type"]),
            r["payload"], r.get("content", ""), r.get("timestamp", ""),
            r.get("input_tokens"), r.get("output_tokens"),
            r.get("cache_read_tokens"), r.get("cache_write_tokens"),
            r.get("reasoning_tokens"), r.get("cost_usd"),
        ))
        parent = next_id
        next_id += 1
    return len(rows)


def fetch_events(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    types: Iterable[Any] | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """A session's events in stream order."""
    sql = "SELECT * FROM events WHERE session_id = ?"
    params: list[Any] = [session_id]
    if active_only:
        sql += " AND active = 1"
    if types:
        values = _type_values(types)
        sql += f" AND type IN ({', '.join('?' for _ in values)})"
        params.extend(values)
    sql += " ORDER BY event_id"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def count_events(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    types: Iterable[Any] | None = None,
    active_only: bool = True,
) -> int:
    sql = "SELECT COUNT(*) FROM events WHERE session_id = ?"
    params: list[Any] = [session_id]
    if active_only:
        sql += " AND active = 1"
    if types:
        values = _type_values(types)
        sql += f" AND type IN ({', '.join('?' for _ in values)})"
        params.extend(values)
    return int(conn.execute(sql, params).fetchone()[0])


def fetch_events_window(
    conn: sqlite3.Connection,
    session_id: str,
    around_event_id: int,
    k: int,
    *,
    types: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """±k events around one event — contiguous per-session numbering makes
    this a plain BETWEEN range. Near a branch point the range may hold
    inactive ids and under-fill; walking ``parent_id`` would fix that."""
    sql = "SELECT * FROM events WHERE session_id = ? AND event_id BETWEEN ? AND ? AND active = 1"
    params: list[Any] = [session_id, around_event_id - k, around_event_id + k]
    if types:
        values = _type_values(types)
        sql += f" AND type IN ({', '.join('?' for _ in values)})"
        params.extend(values)
    sql += " ORDER BY event_id"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def fetch_bookends(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    first_n: int = 3,
    last_n: int = 3,
    types: Iterable[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The session's first and last matching events — the goal and the
    resolution, for search-hit context (and capped whole-session reads)."""
    values = _type_values(types) if types else None
    type_sql = f" AND type IN ({', '.join('?' for _ in values)})" if values else ""
    base = f"SELECT * FROM events WHERE session_id = ? AND active = 1{type_sql}"
    params: list[Any] = [session_id] + (values or [])
    first = [dict(r) for r in conn.execute(base + " ORDER BY event_id LIMIT ?", params + [first_n]).fetchall()]
    last = [dict(r) for r in conn.execute(base + " ORDER BY event_id DESC LIMIT ?", params + [last_n]).fetchall()]
    last.reverse()
    # Drop overlap when the session is shorter than first_n + last_n.
    first_ids = {r["event_id"] for r in first}
    last = [r for r in last if r["event_id"] not in first_ids]
    return first, last


def deactivate_events(conn: sqlite3.Connection, session_id: str, from_event_id: int = 0) -> int:
    """Rewind: flag events inactive from ``from_event_id`` on. They vanish
    from resume and search but stay in the DAG for history tooling."""
    cur = conn.execute(
        "UPDATE events SET active = 0 WHERE session_id = ? AND event_id >= ? AND active = 1",
        (session_id, from_event_id),
    )
    return cur.rowcount


def checkout_events(conn: sqlite3.Connection, session_id: str, target_event_id: int | None) -> int:
    """Materialize root→``target_event_id`` as the session's active path;
    every other event goes inactive. ``None`` empties the path (all
    inactive), matching a full rewind. Returns the active-path length."""
    rows = conn.execute(
        "SELECT event_id, parent_id FROM events WHERE session_id = ?", (session_id,)
    ).fetchall()
    parents = {r["event_id"]: r["parent_id"] for r in rows}
    path: list[int] = []
    node = target_event_id
    while node is not None:
        if node not in parents:
            raise ValueError(f"Event {node} does not exist in session {session_id!r} — cannot check out.")
        path.append(node)
        node = parents[node]
    conn.execute("UPDATE events SET active = 0 WHERE session_id = ? AND active = 1", (session_id,))
    if path:
        placeholders = ", ".join("?" for _ in path)
        conn.execute(
            f"UPDATE events SET active = 1 WHERE session_id = ? AND event_id IN ({placeholders})",
            [session_id, *path],
        )
    return len(path)


# ── search ──────────────────────────────────────────────────────────────


_TASK_INPUT_PREVIEW = 200
_PROMPT_PREVIEW = 300

_FTS_PLAIN_TOKEN = re.compile(r"^[A-Za-z0-9_]+\*?$")
_FTS_BOOLEANS = {"AND", "OR", "NOT", "NEAR"}


def sanitize_fts_query(query: str) -> str:
    """Make free-form input safe for FTS5 ``MATCH``: quoted phrases are
    preserved, boolean operators pass through, and any other token that
    FTS5 syntax would choke on (dots, hyphens, colons — ``my-app.config.ts``)
    is wrapped in quotes so it matches as a phrase instead of parsing as
    syntax."""
    phrases = re.findall(r'"([^"]+)"', query)
    remainder = re.sub(r'"[^"]*"', " ", query)
    tokens = [f'"{p}"' for p in phrases]
    for tok in remainder.split():
        if tok in _FTS_BOOLEANS or _FTS_PLAIN_TOKEN.match(tok):
            tokens.append(tok)
        else:
            cleaned = tok.replace('"', " ").strip()
            if cleaned:
                tokens.append(f'"{cleaned}"')
    return " ".join(tokens)


_FTS_PREFIX = re.compile(r"^(\w+)\*$")
_PREFIX_EXPANSIONS = 64


def expand_prefixes(conn: sqlite3.Connection, vocab: str, q: str) -> str:
    """Rewrite each ``word*`` in a sanitized query as an OR of the indexed
    terms starting with ``word``. FTS5 runs a prefix through the porter
    stemmer before matching it, so a half-typed word gets mangled
    (``theyy*`` → ``theyi*``) and finds nothing; matching against the
    stored terms directly avoids that. The raw ``word*`` stays in the
    group: it covers every term the expansion cap leaves out whenever
    stemming leaves the prefix alone, which it always does for the
    one- and two-letter prefixes most likely to hit the cap. A parenthesised group can't rely on FTS5's
    implicit AND, so adjacent operands get an explicit one."""
    out: list[str] = []
    for tok in re.findall(r'"[^"]*"|\S+', q):
        m = _FTS_PREFIX.match(tok)
        if m:
            prefix = m.group(1).lower()
            terms = [
                r[0]
                for r in conn.execute(
                    f"SELECT term FROM {vocab} WHERE term >= ? AND term < ? ORDER BY doc DESC LIMIT ?",
                    (prefix, prefix + "\uffff", _PREFIX_EXPANSIONS),
                )
            ]
            if terms:
                tok = f"({' OR '.join([tok, *terms])})"
        if out and out[-1] not in _FTS_BOOLEANS and tok not in _FTS_BOOLEANS:
            out.append("AND")
        out.append(tok)
    return " ".join(out)


def _force_phrase(query: str) -> str:
    """Last-resort MATCH argument: the whole query as one quoted phrase."""
    return '"' + query.replace('"', " ").strip() + '"'


def search_events_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    types: Iterable[Any] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """BM25-ranked FTS5 search over event content, with a highlighted
    snippet per hit and the owning session's title joined in. Rewound
    (inactive) events are invisible."""
    q = expand_prefixes(conn, "events_vocab", sanitize_fts_query(query))
    if not q.strip():
        return []

    type_sql = ""
    type_params: list[Any] = []
    if types:
        values = _type_values(types)
        type_sql = f" AND e.type IN ({', '.join('?' for _ in values)})"
        type_params = values

    sql = f"""
        SELECT e.session_id, e.event_id, e.type, e.timestamp,
               snippet(events_fts, 0, '>>>', '<<<', '…', 20) AS snippet,
               COALESCE(NULLIF(s.title, ''), substr(s.task_input, 1, 60)) AS session_title, s.created_at AS session_created_at,
               bm25(events_fts) AS score
        FROM events_fts f
        JOIN events e ON e.rowid = f.rowid
        JOIN sessions s ON s.session_id = e.session_id
        WHERE events_fts MATCH ? AND e.active = 1{type_sql}
        ORDER BY score
        LIMIT ?
    """
    params = [q, *type_params, limit]
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        params[0] = _force_phrase(query)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_sessions_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """BM25-ranked FTS5 search over session titles + task inputs.

    Returns a lean per-hit shape for scanning; truncates ``task_input``
    to ``_TASK_INPUT_PREVIEW`` chars. Drill into a hit with
    ``fetch_session`` for the full row.
    """
    q = expand_prefixes(conn, "sessions_vocab", sanitize_fts_query(query))
    if not q.strip():
        return []
    sql = """
        SELECT s.session_id, s.title, s.task_input, s.created_at,
               (SELECT COUNT(*) FROM artifacts a WHERE a.session_id = s.session_id) AS num_artifacts,
               (s.input_tokens + s.output_tokens + s.cache_read_tokens
                + s.cache_write_tokens + s.reasoning_tokens) AS total_tokens,
               s.cost_usd, s.requests,
               bm25(sessions_fts) AS score
        FROM sessions_fts f
        JOIN sessions s ON s.rowid = f.rowid
        WHERE sessions_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """
    try:
        rows = conn.execute(sql, (q, limit)).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(sql, (_force_phrase(query), limit)).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        ti = d.get("task_input") or ""
        if len(ti) > _TASK_INPUT_PREVIEW:
            d["task_input"] = ti[:_TASK_INPUT_PREVIEW] + "…"
        out.append(d)
    return out


_MIND_TITLE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_MIND_SUMMARY = re.compile(r"^tags:\s*(.+)$", re.MULTILINE)


def _mind_page(path: Path, content: str, **extra: Any) -> dict[str, Any]:
    title = _MIND_TITLE.search(content)
    return {
        "path": str(path),
        "title": title.group(1).strip() if title else path.stem,
        "timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        **extra,
    }


def list_mind_pages(mind_dir: Path) -> list[dict[str, Any]]:
    """Every mind wiki page, most recently edited first, with its
    frontmatter tags as the snippet."""
    if not mind_dir.is_dir():
        return []
    pages = []
    for p in mind_dir.glob("*.md"):
        content = p.read_text(encoding="utf-8", errors="replace")
        summary = _MIND_SUMMARY.search(content)
        pages.append(_mind_page(p, content, snippet=summary.group(1).strip() if summary else "", score=0.0))
    pages.sort(key=lambda h: h["timestamp"], reverse=True)
    return pages


def search_mind_fts(mind_dir: Path, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """BM25-ranked FTS5 search over the mind wiki's markdown pages.

    The pages are plain files written by the agent through any file tool,
    so there is no write hook to keep a persistent index in sync; the
    wiki is small, so it is indexed into a throwaway in-memory table per
    query instead.
    """
    q = sanitize_fts_query(query)
    if not q.strip() or not mind_dir.is_dir():
        return []
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    try:
        mem.execute("CREATE VIRTUAL TABLE mind_fts USING fts5(path UNINDEXED, content, tokenize = 'porter unicode61')")
        mem.execute("CREATE VIRTUAL TABLE mind_vocab USING fts5vocab(mind_fts, row)")
        mem.executemany(
            "INSERT INTO mind_fts(path, content) VALUES (?, ?)",
            [(str(p), p.read_text(encoding="utf-8", errors="replace")) for p in sorted(mind_dir.glob("*.md"))],
        )
        q = expand_prefixes(mem, "mind_vocab", q)
        sql = """
            SELECT path, content, snippet(mind_fts, 1, '>>>', '<<<', '…', 20) AS snippet,
                   bm25(mind_fts) AS score
            FROM mind_fts
            WHERE mind_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """
        try:
            rows = mem.execute(sql, (q, limit)).fetchall()
        except sqlite3.OperationalError:
            rows = mem.execute(sql, (_force_phrase(query), limit)).fetchall()
    finally:
        mem.close()
    return [_mind_page(Path(r["path"]), r["content"], snippet=r["snippet"], score=r["score"]) for r in rows]


def search_artifacts_fts(conn: sqlite3.Connection, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Find artifacts by BM25-ranked FTS5 search over prompt /
    description / tool_name, OR by URL / local-path substring.

    The FTS5 index doesn't tokenize URLs (random hash IDs aren't useful
    search terms), so a URL fragment like ``"gTPx9BTfmSTrVPiInkOrJ"``
    won't FTS-match anything. A ``LIKE %query%`` scan of ``url`` and
    ``path`` fills in after the FTS hits — covers
    the "I have this image URL, where did it come from?" lookup that
    used to require grepping transcripts on disk.

    Returns a lean per-hit shape; truncates ``prompt`` and ``description``
    to ``_PROMPT_PREVIEW`` chars.
    """
    if not query.strip():
        return []

    def _normalize(rows):
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            for k in ("prompt", "description"):
                v = d.get(k) or ""
                if len(v) > _PROMPT_PREVIEW:
                    d[k] = v[:_PROMPT_PREVIEW] + "…"
            out.append(d)
        return out

    # FTS5 path — try first; the BM25 ranking gives sensible ordering
    # for actual word queries.
    try:
        sql = """
            SELECT a.id AS artifact_id, a.session_id, a.artifact_type, a.url, a.path,
                   a.prompt, a.description,
                   COALESCE(NULLIF(s.title, ''), substr(s.task_input, 1, 60)) AS title, s.created_at,
                   bm25(artifacts_fts) AS score
            FROM artifacts_fts f
            JOIN artifacts a ON a.id = f.rowid
            JOIN sessions s ON s.session_id = a.session_id
            WHERE artifacts_fts MATCH ?
            ORDER BY score LIMIT ?
        """
        params: list[Any] = [expand_prefixes(conn, "artifacts_vocab", sanitize_fts_query(query)), limit]
        fts_rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # FTS5 raises on punctuation-heavy queries (URL-like inputs).
        # Skip straight to the URL/path fallback.
        fts_rows = []

    if len(fts_rows) >= limit:
        return _normalize(fts_rows)

    # Then substring matches against url / path, which FTS can't see —
    # appended rather than only-on-empty, so a path fragment that also
    # shares a word with some prompt still finds its file.
    like = f"%{query.rstrip('*')}%"  # live pickers send the word being typed as an FTS prefix
    sql = """
        SELECT a.id AS artifact_id, a.session_id, a.artifact_type, a.url, a.path,
               a.prompt, a.description,
               COALESCE(NULLIF(s.title, ''), substr(s.task_input, 1, 60)) AS title, s.created_at,
               0.0 AS score
        FROM artifacts a
        JOIN sessions s ON s.session_id = a.session_id
        WHERE (a.url LIKE ? OR a.path LIKE ?)
        ORDER BY a.timestamp DESC LIMIT ?
    """
    params = [like, like, limit]
    seen = {r["artifact_id"] for r in fts_rows}
    like_rows = [r for r in conn.execute(sql, params).fetchall() if r["artifact_id"] not in seen]
    return _normalize([*fts_rows, *like_rows][:limit])


# ── favorites ───────────────────────────────────────────────────────────


def add_favorite(
    conn: sqlite3.Connection,
    *,
    artifact_id: int | None = None,
    url: str = "",
    artifact_type: str = "",
) -> int:
    """Favorite an asset. Idempotent — re-favoriting the same target refreshes
    the timestamp and returns the existing row id.

    Identity:
      - ``artifact_id`` set → internal favorite (de-duped per artifact)
      - else ``url`` set    → external favorite (de-duped per URL/path string)

    The ``url`` arg accepts either a remote URL (``http``/``https``/
    ``file``/``data``) or a bare local filesystem path.

    Returns the favorite row's id.
    """
    if artifact_id is None and not url:
        raise ValueError("add_favorite needs artifact_id or url")

    now = datetime.now().isoformat()
    if artifact_id is not None:
        match_sql = "SELECT id FROM favorites WHERE artifact_id = ?"
        match_params = (artifact_id,)
    else:
        match_sql = "SELECT id FROM favorites WHERE artifact_id IS NULL AND url = ?"
        match_params = (url,)

    row = conn.execute(match_sql, match_params).fetchone()
    if row is not None:
        # Refresh the timestamp; preserve artifact_type unless the caller
        # supplied a non-empty one.
        conn.execute(
            "UPDATE favorites SET favorited_at = ?, "
            "artifact_type = COALESCE(NULLIF(?, ''), artifact_type) "
            "WHERE id = ?",
            (now, artifact_type, row["id"]),
        )
        return int(row["id"])

    source = "internal" if artifact_id is not None else "external"
    cur = conn.execute(
        "INSERT INTO favorites (artifact_id, artifact_source, url, artifact_type, "
        "favorited_at) VALUES (?, ?, ?, ?, ?)",
        (artifact_id, source, url, artifact_type, now),
    )
    return int(cur.lastrowid or 0)


def remove_favorite(conn: sqlite3.Connection, favorite_id: int) -> bool:
    """Remove a favorite by favorites.id. Returns True if a row was removed."""
    cur = conn.execute("DELETE FROM favorites WHERE id = ?", (favorite_id,))
    return cur.rowcount > 0


def list_favorites(conn: sqlite3.Connection, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Newest-first favorites with artifact + session context via LEFT JOIN.
    Internal favorites join on ``artifact_id``; external favorites join on the
    newest artifact whose path or URL equals the stored ref, so a favorite
    from the viewer or asset picker still knows its prompt and session.

    Each row carries: ``id``, ``artifact_id`` (or None),
    ``artifact_source``, ``url`` (URL or local path), ``artifact_type``,
    ``favorited_at``, plus ``session_id`` / ``session_title`` / ``prompt``
    / ``tool_name`` / ``model_id`` when the favorite is internal.
    """
    sql = """
        SELECT f.id, f.artifact_id, f.artifact_source, f.favorited_at,
               COALESCE(NULLIF(f.url, ''), a.path, a.url) AS url,
               COALESCE(NULLIF(f.artifact_type, ''), a.artifact_type) AS artifact_type,
               a.session_id, a.prompt, a.description, a.tool_name, a.model_id, a.timestamp,
               COALESCE(NULLIF(s.title, ''), substr(s.task_input, 1, 60)) AS session_title
        FROM favorites f
        LEFT JOIN artifacts a ON a.id = COALESCE(
            f.artifact_id,
            (SELECT x.id FROM artifacts x
             WHERE f.url != '' AND (x.path = f.url OR x.url = f.url)
             ORDER BY x.id DESC LIMIT 1)
        )
        LEFT JOIN sessions  s ON s.session_id = a.session_id
        ORDER BY f.favorited_at DESC
    """
    params: list[Any] = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Wrap multi-statement work in a single transaction."""
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
