"""Standalone tool: search across past sessions and their artifacts.

Thin wrapper over the SQLite FTS5 index built in ``okonomi.db``. The
model could write the queries itself via ``execute_code`` + ``sqlite3``,
but the wrapper saves the boilerplate (path, connection, BM25 ordering,
shape coercion) every call — which adds up.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, TypedDict

from okonomi.session import SessionManager
from okonomi.tools._metadata import ToolMetadataMixin


class SessionHit(TypedDict, total=False):
    session_id: str
    title: str
    task_input: str
    created_at: str
    updated_at: str
    dir_name: str
    num_artifacts: int
    message_count: int
    total_tokens: int  # sum of input + output + cache_read + cache_write + reasoning
    cost_usd: float | None
    score: float


class ArtifactHit(TypedDict, total=False):
    session_id: str
    title: str  # parent session's title (joined)
    artifact_type: str
    url: str
    path: str
    prompt: str
    description: str
    tool_name: str
    model_id: str
    timestamp: str
    created_at: str  # parent session's created_at
    score: float


class SessionsSearchResult(TypedDict, total=False):
    query: str
    results: list[SessionHit]
    error: str


class ArtifactsSearchResult(TypedDict, total=False):
    query: str
    results: list[ArtifactHit]
    error: str


class SessionsTools(ToolMetadataMixin):
    __namespace__ = "sessions"
    __description__ = "Search across past sessions and their generated artifacts."

    def __init__(self):
        self._sm: SessionManager | None = None

    def _mgr(self) -> SessionManager:
        if self._sm is None:
            self._sm = SessionManager()
        return self._sm

    def search_sessions(self, query: str, limit: int = 8) -> dict:
        """Find past sessions matching a search query (BM25-ranked).

        Returns a *lean* per-hit shape — `session_id`, `title`,
        truncated `task_input`, `created_at`, `num_artifacts`,
        `message_count`, `total_tokens`, `cost_usd`, `score`.
        Use ``get_session(session_id)`` to drill into one match for
        full metadata (model, per-bucket tokens, full task input);
        use ``search_artifacts`` for finer-grained matches on actual
        generation prompts.

        Args:
            query: FTS5 query — phrase ("noir detective"), prefix (cyber*),
                boolean (X OR Y), AND/NOT/NEAR.
            limit: Max results (default 8). Start small; widen only if
                the top hits don't match.
        """
        from okonomi import db as _db

        try:
            hits = _db.search_sessions_fts(self._mgr()._db, query, limit=limit)
            return {"query": query, "results": hits}
        except Exception as e:
            return {"query": query, "error": str(e)}

    def search_artifacts(
        self,
        query: str,
        artifact_type: str | None = None,
        since_days: int | None = None,
        limit: int = 12,
    ) -> dict:
        """Find past artifacts by prompt / description / tool_name OR
        by URL / local-path substring.

        Two modes, transparently dispatched:
          - Word/phrase query → BM25-ranked FTS5 over prompt /
            description / tool_name. Use this for topic searches like
            ``"noir detective"`` or ``"theyyam"``.
          - URL / path fragment → substring scan of url / path columns.
            Use this when you have an asset reference and want to find
            what produced it: ``search_artifacts("gTPx9BTfmSTrV")`` will
            find the artifact whose URL contains that hash. ALWAYS reach
            for this before falling back to ``bash grep`` over session
            files — the DB lookup is orders of magnitude faster.

        Returns a *lean* per-hit shape — `session_id`, `title`,
        `artifact_type`, `url`, `path`, truncated `prompt` and
        `description`, `score`. Use ``get_session(session_id)`` to pull
        full per-session detail for one match.

        Args:
            query: FTS5 phrase / prefix / boolean expression, OR a URL /
                path substring.
            artifact_type: ``image`` / ``video`` / ``audio`` / ``3d``.
                None = any.
            since_days: Restrict to parent sessions created in the last
                N days. None = all time.
            limit: Max results (default 12). Start small.
        """
        try:
            hits = self._mgr().search_artifacts(query, artifact_type=artifact_type, limit=limit)
        except Exception as e:
            return {"query": query, "error": str(e)}

        if since_days is not None:
            cutoff = datetime.now() - timedelta(days=since_days)
            kept: list[dict[str, Any]] = []
            for h in hits:
                ts = h.get("created_at") or ""
                try:
                    if ts and datetime.fromisoformat(ts) >= cutoff:
                        kept.append(h)
                except ValueError:
                    continue
            hits = kept

        return {"query": query, "results": hits}

    def get_session(self, session_id: str) -> dict:
        """Pull one session's metadata + full artifacts list (no message
        history, no truncation). Use this to drill into one match from
        ``search_sessions`` or ``search_artifacts`` instead of
        ``cat session.json`` (which dumps the whole file including
        anything you don't need).

        For the conversation history, read ``messages.jsonl`` directly:
        ``bash(command="head -50 <dir>/messages.jsonl | jq")``.

        Args:
            session_id: Exact session id.

        Returns:
            ``{"session_id", "metadata": {...}, "artifacts": [...]}`` or
            ``{"error": ..., "session_id": ...}`` on miss.
        """
        info = self._mgr().get_info(session_id)
        if info is None:
            return {"error": f"Unknown session_id '{session_id}'", "session_id": session_id}

        return {
            "session_id": session_id,
            "metadata": info.metadata_dict(),
            "artifacts": [a.to_dict() for a in self._mgr().get_artifacts(session_id)],
        }
