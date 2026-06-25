"""SQLite store for sessions + artifacts, with FTS5 search.

This is the *source of truth* for everything except the conversation
transcript (which lives in ``<session_dir>/messages.jsonl`` because it's
large, append-friendly, and never queried).

Tables:

  sessions       — one row per session_id, mirrors SessionInfo 1:1
  artifacts      — one row per artifact, FK to sessions
  favorites      — user-curated artifact bookmarks
  sessions_fts   — FTS5 mirror of (title, task_input)
  artifacts_fts  — FTS5 mirror of (prompt, description, tool_name)

Triggers keep the FTS tables synced on insert/delete/update.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


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
    ("dir_name", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    # Lifecycle
    ("status", "text_nn", "TEXT NOT NULL DEFAULT 'completed'"),
    ("created_at", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "text_nn", "TEXT NOT NULL DEFAULT ''"),
    # Aggregates
    ("num_artifacts", "int", "INTEGER NOT NULL DEFAULT 0"),
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
    # Body
    ("steps_summary", "text_nn", "TEXT NOT NULL DEFAULT ''"),
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

-- User-curated favorites ("hearts"). Two flavors share one table:
--   internal: artifact_id set → CASCADE-deletes with the artifact
--   external: artifact_id NULL → user bookmarked a bare URL or local path
-- Dedup via partial unique indexes: same artifact / same URL / same path
-- can each be hearted at most once.
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
    created on the main thread (when ``SessionManager`` initializes)
    but reads happen on the agent worker thread (e.g. when the
    ``search_sessions`` / ``search_artifacts`` tools fire mid-turn).
    WAL mode + serialized writes through ``transaction()`` make this
    safe; the Python-side guard would otherwise raise a
    ``ProgrammingError`` on every cross-thread access.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)

    return conn


# ── sessions ────────────────────────────────────────────────────────────


_COERCE = {
    "int": lambda v: int(v or 0),
    "real": lambda v: None if v is None else float(v),
    "text_nn": lambda v: "" if v is None else str(v),
    "text": lambda v: v,
}


def _meta_to_row(meta: dict[str, Any]) -> tuple:
    """Coerce a SessionInfo-shaped dict into the column-ordered tuple.

    Sparse dicts produce valid rows; unknown keys are ignored.
    """
    return tuple(_COERCE[kind](meta.get(name)) for name, kind, _decl in SESSION_COLUMNS)


def upsert_session(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    """Insert or replace a session row. ``meta`` mirrors ``SessionInfo``."""
    cols = [name for name, _kind, _decl in SESSION_COLUMNS]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "session_id")
    conn.execute(
        f"INSERT INTO sessions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
        _meta_to_row(meta),
    )


def fetch_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(
    conn: sqlite3.Connection, *, status: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Newest-first by ``updated_at``. Optionally filter by status."""
    sql = "SELECT * FROM sessions"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY updated_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Cascade-deletes artifacts via FK ON DELETE CASCADE."""
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def update_session_title(conn: sqlite3.Connection, session_id: str, title: str, dir_name: str) -> None:
    """Targeted UPDATE for the title-rename path — avoids round-tripping
    every column through `upsert_session`."""
    conn.execute(
        "UPDATE sessions SET title = ?, dir_name = ?, updated_at = ? WHERE session_id = ?",
        (title, dir_name, datetime.now().isoformat(), session_id),
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


# ── search ──────────────────────────────────────────────────────────────


_TASK_INPUT_PREVIEW = 200
_PROMPT_PREVIEW = 300


def search_sessions_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """BM25-ranked FTS5 search over session titles + task inputs.

    Returns a lean per-hit shape for scanning; truncates ``task_input``
    to ``_TASK_INPUT_PREVIEW`` chars. Drill into a hit with
    ``fetch_session`` for the full row.
    """
    if not query.strip():
        return []
    rows = conn.execute(
        """
        SELECT s.session_id, s.title, s.task_input, s.created_at,
               s.num_artifacts, s.dir_name,
               (s.input_tokens + s.output_tokens + s.cache_read_tokens
                + s.cache_write_tokens + s.reasoning_tokens) AS total_tokens,
               s.cost_usd, s.requests,
               bm25(sessions_fts) AS score
        FROM sessions_fts f
        JOIN sessions s ON s.rowid = f.rowid
        WHERE sessions_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        ti = d.get("task_input") or ""
        if len(ti) > _TASK_INPUT_PREVIEW:
            d["task_input"] = ti[:_TASK_INPUT_PREVIEW] + "…"
        out.append(d)
    return out


def search_artifacts_fts(
    conn: sqlite3.Connection,
    query: str,
    *,
    artifact_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find artifacts by BM25-ranked FTS5 search over prompt /
    description / tool_name, OR by URL / local-path substring.

    The FTS5 index doesn't tokenize URLs (random hash IDs aren't useful
    search terms), so a URL fragment like ``"gTPx9BTfmSTrVPiInkOrJ"``
    won't FTS-match anything. We fall back to a ``LIKE %query%`` scan
    of ``url`` and ``path`` when the FTS path returns nothing — covers
    the "I have this image URL, where did it come from?" lookup that
    used to require a bash grep across every messages.jsonl on disk.

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
            SELECT a.session_id, a.artifact_type, a.url, a.path,
                   a.prompt, a.description,
                   s.title, s.created_at,
                   bm25(artifacts_fts) AS score
            FROM artifacts_fts f
            JOIN artifacts a ON a.id = f.rowid
            JOIN sessions s ON s.session_id = a.session_id
            WHERE artifacts_fts MATCH ?
        """
        params: list[Any] = [query]
        if artifact_type:
            sql += " AND a.artifact_type = ?"
            params.append(artifact_type)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        fts_rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # FTS5 raises on punctuation-heavy queries (URL-like inputs).
        # Skip straight to the URL/path fallback.
        fts_rows = []

    if fts_rows:
        return _normalize(fts_rows)

    # Fallback: substring match against url / path.
    like = f"%{query}%"
    sql = """
        SELECT a.session_id, a.artifact_type, a.url, a.path,
               a.prompt, a.description,
               s.title, s.created_at,
               0.0 AS score
        FROM artifacts a
        JOIN sessions s ON s.session_id = a.session_id
        WHERE (a.url LIKE ? OR a.path LIKE ?)
    """
    params = [like, like]
    if artifact_type:
        sql += " AND a.artifact_type = ?"
        params.append(artifact_type)
    sql += " ORDER BY a.timestamp DESC LIMIT ?"
    params.append(limit)
    return _normalize(conn.execute(sql, params).fetchall())


# ── favorites ───────────────────────────────────────────────────────────


def add_favorite(
    conn: sqlite3.Connection,
    *,
    artifact_id: int | None = None,
    url: str = "",
    artifact_type: str = "",
) -> int:
    """Heart an asset. Idempotent — re-hearting the same target refreshes
    the timestamp and returns the existing row id.

    Identity:
      - ``artifact_id`` set → internal heart (de-duped per artifact)
      - else ``url`` set    → external heart (de-duped per URL/path string)

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
    """Unheart by favorites.id. Returns True if a row was removed."""
    cur = conn.execute("DELETE FROM favorites WHERE id = ?", (favorite_id,))
    return cur.rowcount > 0


def list_favorites(
    conn: sqlite3.Connection,
    *,
    artifact_type: str | None = None,
    artifact_source: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Newest-first hearts. Internal hearts get artifact + session context
    via LEFT JOIN; external hearts return their stored url/path/type.

    Each row carries: ``id``, ``artifact_id`` (or None),
    ``artifact_source``, ``url`` (URL or local path), ``artifact_type``,
    ``favorited_at``, plus ``session_id`` / ``session_title`` / ``prompt``
    / ``tool_name`` / ``model_id`` when the heart is internal.

    ``artifact_type`` and ``artifact_source`` are optional filters. The
    type filter matches against the resolved type (favorite's own type
    if set, else the artifact's).
    """
    sql = """
        SELECT f.id, f.artifact_id, f.artifact_source, f.favorited_at,
               COALESCE(NULLIF(f.url, ''), a.path, a.url) AS url,
               COALESCE(NULLIF(f.artifact_type, ''), a.artifact_type) AS artifact_type,
               a.session_id, a.prompt, a.tool_name, a.model_id, a.timestamp,
               s.title AS session_title
        FROM favorites f
        LEFT JOIN artifacts a ON a.id = f.artifact_id
        LEFT JOIN sessions  s ON s.session_id = a.session_id
    """
    where: list[str] = []
    params: list[Any] = []
    if artifact_type:
        where.append("COALESCE(NULLIF(f.artifact_type, ''), a.artifact_type) = ?")
        params.append(artifact_type)
    if artifact_source:
        where.append("f.artifact_source = ?")
        params.append(artifact_source)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY f.favorited_at DESC"
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
