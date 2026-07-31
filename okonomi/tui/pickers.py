"""On-demand full-screen pickers for the REPL (session + asset + tree).

Each is a transient ``prompt_toolkit`` ``Application`` run on the alternate
screen and awaited; selecting exits with the chosen value, Esc / Ctrl-C exit
with ``None``. The alt-screen leaves the REPL scrollback untouched, so native
scroll + copy are unaffected once the picker closes.

The session picker focuses a live fuzzy-search buffer. The asset picker has
single-letter action verbs (o/c/h/a/x) that would collide with a focused text
buffer, so it omits live search for now and navigates with the arrow keys.
"""

from __future__ import annotations

import json
import sys

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output.defaults import create_output

from .. import db
from ..events import CONVERSATION_TYPES, EventType
from ..session import list_sessions
from ..session import search as content_search
from ..utils import fuzzy_match
from .style import REPL_STYLE


def _ref(a) -> str:
    return getattr(a, "path", "") or getattr(a, "url", "") or ""


async def pick_session() -> str | None:
    """Full-screen fuzzy session picker → chosen session_id or None."""
    rows_all = [
        s
        for s in list_sessions(limit=500)
        if (getattr(s, "title", "") or "").strip() or (getattr(s, "task_input", "") or "").strip()
    ]
    state = {"idx": 0, "rows": rows_all}
    search = Buffer()

    def _hay(s) -> str:
        name = getattr(s, "display_name", "") or ""
        date = (getattr(s, "created_at", "") or "")[:10]
        return f"{name} {date} {getattr(s, 'session_id', '')}".lower()

    def _refilter(_=None) -> None:
        q = search.text.strip().lower()
        state["rows"] = rows_all if not q else [s for s in rows_all if fuzzy_match(q, _hay(s))]
        if q and len(q) >= 3 and not state["rows"]:
            # Title fuzzy-match found nothing — fall through to content
            # search over conversations and artifact prompts.
            try:
                by_id = {s.session_id: s for s in rows_all}
                hit_ids = dict.fromkeys(h["session_id"] for h in content_search(q, limit=20))
                state["rows"] = [by_id[i] for i in hit_ids if i in by_id]
            except Exception:
                pass
        state["idx"] = 0

    search.on_text_changed += _refilter

    def _render():
        rows = state["rows"]
        if not rows:
            return [("class:dim", "  (no matches)")]
        out = []
        for n, s in enumerate(rows):
            date = (getattr(s, "created_at", "") or "")[:10]
            name = (getattr(s, "display_name", "") or "(untitled)")[:90]
            cur = n == state["idx"]
            out.append(("class:picker-highlight" if cur else "", f"{'›' if cur else ' '} {date}  {name}\n"))
        return out

    kb = KeyBindings()

    @kb.add("up", eager=True)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] - 1) % len(state["rows"])

    @kb.add("down", eager=True)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] + 1) % len(state["rows"])

    @kb.add("enter", eager=True)
    def _(e):
        rows = state["rows"]
        e.app.exit(result=getattr(rows[state["idx"]], "session_id", None) if rows else None)

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        e.app.exit(result=None)

    search_win = Window(BufferControl(buffer=search), height=1)
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Resume session — type to filter, ↑↓ Enter, Esc")]), height=1),
            search_win,
            Window(FormattedTextControl(_render)),
        ]
    )
    app: Application = Application(
        layout=Layout(body, focused_element=search_win),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
        style=REPL_STYLE,
        output=create_output(stdout=sys.__stdout__),  # real terminal, bypass patch_stdout
    )
    return await app.run_async()


async def pick_assets(artifacts) -> tuple[str, list[str]] | None:
    """Full-screen multi-select asset picker → (action, paths) or None.

    Space toggles, a=all, x=clear, o=open, c=copy, h=heart, Esc=cancel.
    """
    items = [a for a in artifacts if _ref(a)]
    selected: set[str] = set()
    state = {"idx": 0}

    def _cur() -> str:
        return _ref(items[state["idx"]]) if items else ""

    def _render():
        if not items:
            return [("class:dim", "  (no assets in this session)")]
        out = []
        for n, a in enumerate(items):
            ref = _ref(a)
            box = "[x]" if ref in selected else "[ ]"
            kind = (getattr(a, "artifact_type", "") or "file")[:6]
            line = f"{box} {kind:<6} {ref[-70:]}"
            prompt = getattr(a, "prompt", "") or ""
            if prompt:
                line += f"  — {prompt[:50]}"
            cur = n == state["idx"]
            out.append(("class:picker-highlight" if cur else "", f"{'›' if cur else ' '} {line}\n"))
        return out

    def _footer():
        return [("class:dim", f"  {len(selected)} selected · Space toggle · a all · x clear · o open · c copy · h heart · Esc")]

    def _dispatch(action: str):
        paths = sorted(selected) if selected else ([_cur()] if _cur() else [])
        return (action, [p for p in paths if p])

    kb = KeyBindings()

    @kb.add("up", eager=True)
    def _(e):
        if items:
            state["idx"] = (state["idx"] - 1) % len(items)

    @kb.add("down", eager=True)
    def _(e):
        if items:
            state["idx"] = (state["idx"] + 1) % len(items)

    @kb.add("space", eager=True)
    def _(e):
        ref = _cur()
        if ref:
            selected.discard(ref) if ref in selected else selected.add(ref)

    @kb.add("a")
    def _(e):
        selected.update(r for r in (_ref(a) for a in items) if r)

    @kb.add("x")
    def _(e):
        selected.clear()

    @kb.add("o")
    def _(e):
        e.app.exit(result=_dispatch("open"))

    @kb.add("c")
    def _(e):
        e.app.exit(result=_dispatch("copy"))

    @kb.add("h")
    def _(e):
        e.app.exit(result=_dispatch("heart"))

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        e.app.exit(result=None)

    list_win = Window(FormattedTextControl(_render, focusable=True))
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Assets — ↑↓ to move, Space to select")]), height=1),
            list_win,
            Window(FormattedTextControl(_footer), height=1),
        ]
    )
    app: Application = Application(
        layout=Layout(body, focused_element=list_win),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
        style=REPL_STYLE,
        output=create_output(stdout=sys.__stdout__),  # real terminal, bypass patch_stdout
    )
    return await app.run_async()


def _prompt_text(payload: str) -> str:
    """Exact user text of a persisted USER_MESSAGE payload — the editor
    prefill for edit-and-resubmit. The ``content`` column can't serve here:
    it's capped and stripped at write time."""
    try:
        msg = json.loads(payload)
        texts: list[str] = []
        for part in msg.get("parts", []):
            if part.get("part_kind") != "user-prompt":
                continue
            content = part.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(c for c in content if isinstance(c, str))
        return "\n".join(t for t in texts if t).strip()
    except Exception:
        return ""


def _tree_rows(events: list[dict]) -> list[dict]:
    """Flatten a session's event forest into display rows for ``pick_tree``.

    Only user/assistant messages become rows — tool calls, tool results,
    and stream events stay hidden, which collapses long tool runs with no
    folding logic while keeping every fork point visible (branches start
    at resubmitted user messages). Depth grows below any node with more
    than one child. ``checkout_target`` bakes in the selection semantics:
    a user row checks out to its *parent* (edit and resubmit the prefilled
    text), an assistant row checks out to itself (continue from there).
    """
    by_id = {e["event_id"]: e for e in events}
    children: dict[int | None, list[int]] = {}
    for e in events:
        children.setdefault(e["parent_id"], []).append(e["event_id"])
    for ids in children.values():
        ids.sort()

    current_id = max(
        (e["event_id"] for e in events if e["active"] and e["type"] in (EventType.USER_MESSAGE.value, EventType.ASSISTANT_MESSAGE.value)),
        default=None,
    )

    rows: list[dict] = []
    stack = [(eid, 0) for eid in reversed(children.get(None, []))]
    while stack:
        eid, depth = stack.pop()
        e = by_id[eid]
        if e["type"] == EventType.USER_MESSAGE.value:
            rows.append({
                "event_id": eid,
                "type": e["type"],
                "depth": depth,
                "preview": (e["content"] or "").splitlines()[0] if e["content"] else "(empty)",
                "active": bool(e["active"]),
                "is_current": eid == current_id,
                "checkout_target": e["parent_id"],
                "prefill": _prompt_text(e["payload"]),
            })
        elif e["type"] == EventType.ASSISTANT_MESSAGE.value:
            rows.append({
                "event_id": eid,
                "type": e["type"],
                "depth": depth,
                "preview": (e["content"] or "").splitlines()[0] if e["content"] else "(empty)",
                "active": bool(e["active"]),
                "is_current": eid == current_id,
                "checkout_target": eid,
                "prefill": "",
            })
        kids = children.get(eid, [])
        child_depth = depth + 1 if len(kids) > 1 else depth
        for k in reversed(kids):
            stack.append((k, child_depth))
    return rows


async def pick_tree(session_id: str) -> tuple[int | None, str, bool] | None:
    """Full-screen session-tree navigator → (checkout_target_event_id,
    prefill_text, summarize_abandoned_branch) or None on cancel.

    Selecting a target that abandons active turns first asks whether to
    LLM-summarize them into the new branch's context (y/n, Esc backs out).
    """
    events = db.fetch_events(db.connection(), session_id, active_only=False)
    rows = _tree_rows(events)
    state = {"idx": 0, "confirm": False}
    for n, r in enumerate(rows):
        if r["is_current"]:
            state["idx"] = n

    parents = {e["event_id"]: e["parent_id"] for e in events}
    conv_types = {t.value for t in CONVERSATION_TYPES}

    def _abandoned_count(target: int | None) -> int:
        path: set[int] = set()
        node = target
        while node is not None:
            path.add(node)
            node = parents.get(node)
        return sum(1 for e in events if e["active"] and e["type"] in conv_types and e["event_id"] not in path)

    def _render():
        if not rows:
            return [("class:dim", "  (no messages yet)")]
        try:
            height = max(get_app().output.get_size().rows - 3, 5)
        except Exception:
            height = 30
        start = max(0, min(state["idx"] - height // 2, len(rows) - height))
        out = []
        for n, r in enumerate(rows[start : start + height], start=start):
            glyph = "●" if r["type"] == EventType.USER_MESSAGE.value else "○"
            marker = "  ◀" if r["is_current"] else ""
            cur = n == state["idx"]
            style = "class:picker-highlight" if cur else ("" if r["active"] else "class:dim")
            out.append((style, f"{'›' if cur else ' '} {'  ' * r['depth']}{glyph} {r['preview'][:100]}{marker}\n"))
        return out

    def _footer():
        if state["confirm"]:
            return [("class:prompt", " Summarize the branch you're leaving?  y yes · n no · Esc back")]
        return [("class:dim", "  ↑↓ move · Enter checkout (user msg → edit & resubmit) · Esc")]

    in_confirm = Condition(lambda: state["confirm"])
    kb = KeyBindings()

    @kb.add("up", eager=True, filter=~in_confirm)
    def _(e):
        if rows:
            state["idx"] = (state["idx"] - 1) % len(rows)

    @kb.add("down", eager=True, filter=~in_confirm)
    def _(e):
        if rows:
            state["idx"] = (state["idx"] + 1) % len(rows)

    @kb.add("enter", eager=True, filter=~in_confirm)
    def _(e):
        if not rows:
            e.app.exit(result=None)
            return
        r = rows[state["idx"]]
        if _abandoned_count(r["checkout_target"]):
            state["confirm"] = True
        else:
            e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("y", filter=in_confirm)
    def _(e):
        r = rows[state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], True))

    @kb.add("n", filter=in_confirm)
    def _(e):
        r = rows[state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        if state["confirm"]:
            state["confirm"] = False
        else:
            e.app.exit(result=None)

    list_win = Window(FormattedTextControl(_render, focusable=True))
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Session tree — dim rows are inactive branches, ◀ is the current leaf")]), height=1),
            list_win,
            Window(FormattedTextControl(_footer), height=1),
        ]
    )
    app: Application = Application(
        layout=Layout(body, focused_element=list_win),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
        style=REPL_STYLE,
        output=create_output(stdout=sys.__stdout__),  # real terminal, bypass patch_stdout
    )
    return await app.run_async()
