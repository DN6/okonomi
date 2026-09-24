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


async def pick_assets(artifacts, on_open) -> tuple[str, list[str]] | None:
    """Full-screen multi-select asset picker → (action, paths) or None.

    Space toggles, a=all, x=clear, o=open, c=copy, h=heart, Esc=cancel.
    ``o`` calls ``on_open(paths)`` and keeps the picker up, so assets can be
    viewed and then copied or hearted without reopening it.
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
        on_open(_dispatch("open")[1])

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
    at resubmitted user messages). ``guide`` is the ``tree(1)``-style
    box-drawing prefix: linear runs share their parent's column, and each
    child of a fork opens with ``├──``/``└──`` and indents its descendants
    under ``│``. ``branch`` marks those fork heads for ←/→ hopping.
    ``checkout_target`` bakes in the selection semantics:
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
    stack = [(eid, "", "") for eid in reversed(children.get(None, []))]
    while stack:
        eid, guide, cont = stack.pop()
        e = by_id[eid]
        if e["type"] == EventType.USER_MESSAGE.value:
            rows.append({
                "event_id": eid,
                "type": e["type"],
                "guide": guide,
                "branch": guide != cont,
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
                "guide": guide,
                "branch": guide != cont,
                "preview": (e["content"] or "").splitlines()[0] if e["content"] else "(empty)",
                "active": bool(e["active"]),
                "is_current": eid == current_id,
                "checkout_target": eid,
                "prefill": "",
            })
        kids = children.get(eid, [])
        if len(kids) == 1:
            stack.append((kids[0], cont, cont))
            continue
        for n, k in reversed(list(enumerate(kids))):
            last = n == len(kids) - 1
            stack.append((k, cont + ("└── " if last else "├── "), cont + ("    " if last else "│   ")))
    return rows


def _abandoned_count(events: list[dict], parents: dict, conv_types: set, target: int | None) -> int:
    """Active conversation turns that a checkout to ``target`` would leave
    off the active path. Shared by the tree navigator and history search."""
    path: set[int] = set()
    node = target
    while node is not None:
        path.add(node)
        node = parents.get(node)
    return sum(1 for e in events if e["active"] and e["type"] in conv_types and e["event_id"] not in path)


async def pick_tree(session_id: str) -> tuple[int | None, str, bool] | None:
    """Full-screen session-tree navigator → (checkout_target_event_id,
    prefill_text, summarize_abandoned_branch) or None on cancel.

    Selecting a target that abandons active turns first asks whether to
    LLM-summarize them into the new branch's context (y/n, Esc backs out).
    Typing fuzzy-filters the rows by preview; matches keep their tree indent.
    """
    events = db.fetch_events(db.connection(), session_id, active_only=False)
    all_rows = _tree_rows(events)
    state = {"idx": 0, "confirm": False, "rows": all_rows}
    search = Buffer()

    def _refilter(_=None) -> None:
        q = search.text.strip().lower()
        state["rows"] = all_rows if not q else [r for r in all_rows if fuzzy_match(q, r["preview"].lower())]
        state["idx"] = next((n for n, r in enumerate(state["rows"]) if r["is_current"]), 0)

    _refilter()
    search.on_text_changed += _refilter

    parents = {e["event_id"]: e["parent_id"] for e in events}
    conv_types = {t.value for t in CONVERSATION_TYPES}

    def _render():
        rows = state["rows"]
        if not all_rows:
            return [("class:dim", "  (no messages yet)")]
        if not rows:
            return [("class:dim", "  (no matches)")]
        try:
            height = max(get_app().output.get_size().rows - 4, 5)
        except Exception:
            height = 30
        start = max(0, min(state["idx"] - height // 2, len(rows) - height))
        out = []
        for n, r in enumerate(rows[start : start + height], start=start):
            if r["type"] == EventType.USER_MESSAGE.value:
                role, role_style = "● you  ", "class:tree.user"
            else:
                role, role_style = "○ agent", "class:tree.agent"
            marker = "  ◀" if r["is_current"] else ""
            cur = n == state["idx"]
            style = "class:picker-highlight" if cur else ("" if r["active"] else "class:dim")
            out.append((style, f"{'›' if cur else ' '} "))
            out.append((style if cur else "class:tree.guide", r["guide"]))
            out.append((style if cur or not r["active"] else role_style, role))
            out.append((style, f" {r['preview'][:100]}{marker}\n"))
        return out

    def _footer():
        if state["confirm"]:
            return [("class:prompt", " Summarize the branch you're leaving?  y yes · n no · Esc back")]
        return [("class:dim", "  type to filter · ↑↓ move · ←→ hop branches · Enter checkout (user msg → edit & resubmit) · Esc")]

    in_confirm = Condition(lambda: state["confirm"])
    kb = KeyBindings()

    def _hop(step: int) -> None:
        rows = state["rows"]
        for i in range(1, len(rows)):
            n = (state["idx"] + step * i) % len(rows)
            if rows[n]["branch"]:
                state["idx"] = n
                return

    @kb.add("left", eager=True, filter=~in_confirm)
    def _(e):
        _hop(-1)

    @kb.add("right", eager=True, filter=~in_confirm)
    def _(e):
        _hop(1)

    @kb.add("up", eager=True, filter=~in_confirm)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] - 1) % len(state["rows"])

    @kb.add("down", eager=True, filter=~in_confirm)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] + 1) % len(state["rows"])

    @kb.add("enter", eager=True, filter=~in_confirm)
    def _(e):
        rows = state["rows"]
        if not rows:
            e.app.exit(result=None)
            return
        r = rows[state["idx"]]
        if _abandoned_count(events, parents, conv_types, r["checkout_target"]):
            state["confirm"] = True
        else:
            e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("y", filter=in_confirm)
    def _(e):
        r = state["rows"][state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], True))

    @kb.add("n", filter=in_confirm)
    def _(e):
        r = state["rows"][state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        if state["confirm"]:
            state["confirm"] = False
        else:
            e.app.exit(result=None)

    search_win = Window(BufferControl(buffer=search), height=1)
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Session tree — ├─ └─ open branches, dim rows are inactive, ◀ is the current leaf")]), height=1),
            search_win,
            Window(FormattedTextControl(_render)),
            Window(FormattedTextControl(_footer), height=1),
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


async def pick_history(session_id: str) -> tuple[int | None, str, bool] | None:
    """Full-screen fuzzy conversation search → same checkout result as
    ``pick_tree`` (checkout_target_event_id, prefill_text,
    summarize_abandoned_branch) or None on cancel.

    The search-first counterpart to ``pick_tree``: a live fuzzy buffer
    filters the current session's message previews into a flat ranked
    list, and Enter checks out the chosen message with the identical
    branch semantics (user row → edit & resubmit its parent, assistant
    row → continue from it), including the summarize-abandoned-branch
    confirm. Scope is the current session — branching only has meaning
    within its event graph.
    """
    events = db.fetch_events(db.connection(), session_id, active_only=False)
    all_rows = _tree_rows(events)
    parents = {e["event_id"]: e["parent_id"] for e in events}
    conv_types = {t.value for t in CONVERSATION_TYPES}
    state = {"idx": 0, "confirm": False, "rows": all_rows}
    search = Buffer()

    def _refilter(_=None) -> None:
        q = search.text.strip().lower()
        state["rows"] = all_rows if not q else [r for r in all_rows if fuzzy_match(q, r["preview"].lower())]
        state["idx"] = 0

    search.on_text_changed += _refilter

    def _render():
        rows = state["rows"]
        if not rows:
            return [("class:dim", "  (no matches)")]
        try:
            height = max(get_app().output.get_size().rows - 4, 5)
        except Exception:
            height = 30
        start = max(0, min(state["idx"] - height // 2, len(rows) - height))
        out = []
        for n, r in enumerate(rows[start : start + height], start=start):
            glyph = "●" if r["type"] == EventType.USER_MESSAGE.value else "○"
            marker = "  ◀" if r["is_current"] else ""
            cur = n == state["idx"]
            style = "class:picker-highlight" if cur else ("" if r["active"] else "class:dim")
            out.append((style, f"{'›' if cur else ' '} {glyph} {r['preview'][:100]}{marker}\n"))
        return out

    def _footer():
        if state["confirm"]:
            return [("class:prompt", " Summarize the branch you're leaving?  y yes · n no · Esc back")]
        return [("class:dim", "  type to search · ↑↓ move · Enter checkout (user msg → edit & resubmit) · Esc")]

    in_confirm = Condition(lambda: state["confirm"])
    kb = KeyBindings()

    @kb.add("up", eager=True, filter=~in_confirm)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] - 1) % len(state["rows"])

    @kb.add("down", eager=True, filter=~in_confirm)
    def _(e):
        if state["rows"]:
            state["idx"] = (state["idx"] + 1) % len(state["rows"])

    @kb.add("enter", eager=True, filter=~in_confirm)
    def _(e):
        rows = state["rows"]
        if not rows:
            e.app.exit(result=None)
            return
        r = rows[state["idx"]]
        if _abandoned_count(events, parents, conv_types, r["checkout_target"]):
            state["confirm"] = True
        else:
            e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("y", filter=in_confirm)
    def _(e):
        r = state["rows"][state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], True))

    @kb.add("n", filter=in_confirm)
    def _(e):
        r = state["rows"][state["idx"]]
        e.app.exit(result=(r["checkout_target"], r["prefill"], False))

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        if state["confirm"]:
            state["confirm"] = False
        else:
            e.app.exit(result=None)

    search_win = Window(BufferControl(buffer=search), height=1)
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Search conversation — type to filter, ↑↓ Enter, Esc")]), height=1),
            search_win,
            Window(FormattedTextControl(_render)),
            Window(FormattedTextControl(_footer), height=1),
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
