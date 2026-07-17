"""Standalone tool: one search/retrieval entry point over past sessions,
conversation content, and media artifacts.

Behavior is inferred from which arguments are present (no mode enum) —
query → discover, session_id + around_event_id → scroll, session_id →
read, nothing → browse. Thin wrapper over ``okonomi.session``'s unified
search; results are lean references (snippets and windows, never full
payloads).
"""

from __future__ import annotations

from okonomi import session as _session
from okonomi.tools._metadata import ToolMetadataMixin


class SessionsTools(ToolMetadataMixin):
    __namespace__ = "sessions"
    __description__ = "Search and recall past sessions, conversation content, and generated media."

    def recall(
        self,
        query: str = "",
        session_id: str = "",
        around_event_id: int | None = None,
        limit: int = 8,
    ) -> dict:
        """Search or recall past work. One tool, four shapes — behavior is
        inferred from the arguments you pass:

        1. ``query`` → **discover**: BM25-ranked hits across conversation
           content, session titles/tasks, and artifact prompts/paths/URLs.
           Each hit carries ``source`` (event | session | artifact),
           ``session_id``, ``session_title``, a highlighted ``snippet`` (or
           prompt/path), and the ``event_id`` / ``artifact_id`` that pins
           it. The best event hit per session includes ``bookends`` — the
           session's opening ask and closing reply — so one call shows
           goal → matched moment → resolution.
        2. ``session_id`` + ``around_event_id`` → **scroll**: the ±5
           conversation window around one event (follow up on a hit).
        3. ``session_id`` alone → **read**: session metadata + artifacts +
           the capped head/tail of the conversation.
        4. no arguments → **browse**: recent sessions, newest first.

        Args:
            query: Words, "quoted phrases", AND/OR/NOT. Also matches URL /
                path fragments for artifacts.
            session_id: Exact session id (from a previous hit or browse).
            around_event_id: Event id from a discover hit, to scroll around.
            limit: Max results for discover/browse (default 8).
        """
        try:
            if session_id and around_event_id is not None:
                return self._scroll(session_id, around_event_id)
            if session_id:
                return self._read(session_id)
            if query:
                return self._discover(query, limit)
            return self._browse(limit)
        except Exception as e:
            return {"error": str(e)}

    def _discover(self, query: str, limit: int) -> dict:
        hits = _session.search(query, limit=limit)
        bookended: set[str] = set()
        for h in hits:
            if h["source"] == "event" and h["session_id"] not in bookended:
                bookended.add(h["session_id"])
                first, last = _session.session_bookends(h["session_id"])
                h["bookends"] = {"start": first, "end": last}
        return {"query": query, "results": hits}

    def _scroll(self, session_id: str, around_event_id: int) -> dict:
        window = _session.read_window(session_id, around_event_id)
        if not window:
            return {"error": f"No events around {around_event_id} in '{session_id}'", "session_id": session_id}
        return {"session_id": session_id, "around_event_id": around_event_id, "events": window}

    def _read(self, session_id: str) -> dict:
        session = _session.Session.load(session_id)
        if session is None:
            return {"error": f"Unknown session_id '{session_id}'", "session_id": session_id}
        head, tail = _session.session_bookends(session_id, first_n=20, last_n=10)
        return {
            "session_id": session_id,
            "metadata": session.metadata_dict(),
            "artifacts": [a.to_dict() for a in session.load_artifacts()],
            "conversation_head": head,
            "conversation_tail": tail,
        }

    def _browse(self, limit: int) -> dict:
        sessions = _session.list_sessions(limit=limit)
        return {
            "results": [
                {
                    "session_id": s.session_id,
                    "title": s.display_name,
                    "status": s.status,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                }
                for s in sessions
            ]
        }
