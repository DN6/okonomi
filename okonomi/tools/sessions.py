"""Standalone tools: one search/retrieval entry point over past sessions,
conversation content, media artifacts, and the mind wiki (``recall``), and
the one way anything is written into the mind wiki (``remember``).

Behavior is inferred from which arguments are present (no mode enum) —
query → discover, session_id + around_event_id → scroll, session_id →
read, nothing → browse. Thin wrapper over ``okonomi.session``'s unified
search; results are lean references (snippets and windows, never full
payloads).
"""

from __future__ import annotations

import re
from datetime import date

from okonomi import session as _session
from okonomi.tools._metadata import ToolMetadataMixin


class SessionsTools(ToolMetadataMixin):
    __namespace__ = "sessions"
    __description__ = "Search and recall past work and the mind wiki; remember things into the mind wiki."

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
           content, session titles/tasks, artifact prompts/paths/URLs, and
           mind wiki pages. Each hit carries ``source`` (event | session |
           artifact | mind), a highlighted ``snippet`` (or prompt/path),
           and what pins it: ``session_id`` / ``session_title`` plus
           ``event_id`` / ``artifact_id``, or ``path`` for a mind page
           (open it with ``read_file``). The best event hit per session includes ``bookends`` — the
           session's opening ask and closing reply — so one call shows
           goal → matched moment → resolution.
        2. ``session_id`` + ``around_event_id`` → **scroll**: the ±5
           conversation window around one event (follow up on a hit).
        3. ``session_id`` alone → **read**: session metadata + artifacts +
           the capped head/tail of the conversation.
        4. no arguments → **browse**: the user's favorite assets (across
           all sessions), then recent sessions, newest first.

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

    def remember(
        self,
        title: str,
        body: str,
        tags: list[str] | None = None,
        related: list[str] | None = None,
        slug: str = "",
        update: bool = False,
    ) -> dict:
        """File a page into the mind wiki — the only way pages get written,
        whether from research, a "remember this" from the user, or `/mind`.

        One page per concept, preference, or synthesis. ``recall`` first: if
        a page on this already exists, ``read_file`` it, merge what's new
        into its body, and call again with its ``slug`` and ``update=True``.
        Writing to an existing slug without ``update`` is refused and returns
        the current page so nothing is clobbered.

        Args:
            title: Human-readable page title.
            body: Markdown body, without frontmatter or the title heading.
                Reference other concepts inline as [[slug]].
            tags: Short lowercase tags.
            related: Slugs of closely related pages (written as [[links]]).
            slug: Filename stem; defaults to the kebab-cased title. Keep it
                short (``sodium-vapour-night``).
            update: Replace an existing page, keeping its ``created`` date.
        """
        slug = slug or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if not slug:
            return {"error": "title must contain letters or digits"}
        mind_dir = _session.OKONOMI_MIND_DIR
        path = mind_dir / f"{slug}.md"
        today = date.today().isoformat()
        created = today
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if not update:
                return {
                    "error": f"'{slug}' already exists. Merge into it and call remember(..., slug='{slug}', update=True).",
                    "path": str(path),
                    "current": current,
                }
            if m := re.search(r"^created:\s*(\S+)", current, re.MULTILINE):
                created = m.group(1)
        heading = "" if body.lstrip().startswith("# ") else f"# {title}\n\n"
        links = ", ".join(f"[[{r}]]" for r in related or [])
        page = (
            f"---\ntitle: {title}\ntags: [{', '.join(tags or [])}]\ncreated: {created}\nupdated: {today}\n"
            f"related: {links}\n---\n\n{heading}{body.strip()}\n"
        )
        mind_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")
        return {"path": str(path), "slug": slug, "updated" if update else "created": True}

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
        return {"results": _session.browse(limit=limit)}
