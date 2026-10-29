"""On-demand full-screen pickers for the REPL (session + asset + tree).

Each is a transient ``prompt_toolkit`` ``Application`` run on the alternate
screen and awaited; selecting exits with the chosen value, Esc / Ctrl-C exit
with ``None``. The alt-screen leaves the REPL scrollback untouched, so native
scroll + copy are unaffected once the picker closes.

The find picker (also ``/session``, as a scoped view) focuses a
live search buffer backed by ``session.search``. The asset picker has
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
from ..session import browse
from ..session import search as content_search
from ..utils import words_match
from .style import REPL_STYLE


def _ref(a) -> str:
    return getattr(a, "path", "") or getattr(a, "url", "") or ""


_SCOPE_TITLES = {
    "": "Find — sessions, conversations, assets, mind wiki",
    "session": "Resume session",
}


def _live_query(q: str) -> str:
    """Treat the word being typed as a prefix, so results keep up with
    the keyboard instead of waiting for a whole word."""
    return q + "*" if q[-1:].isalnum() else q


def _scoped(hits: list[dict], scope: str) -> list[dict]:
    if scope == "session":
        seen: set[str] = set()
        out = []
        for h in hits:
            if h["source"] in ("session", "event") and h["session_id"] not in seen:
                seen.add(h["session_id"])
                out.append({**h, "source": "session"})
        return out
    return hits


async def pick_find(initial: str = "", scope: str = "") -> tuple[str, dict] | None:
    """Full-screen search over everything ``session.search`` covers —
    sessions, conversation content, generated assets, mind wiki pages —
    re-queried on every keystroke. ``scope`` narrows it to one kind
    (``/session``) without changing how matching works.
    Enter → ("enter", hit), Ctrl-O → ("open", hit), Esc → None."""
    state = {"idx": 0, "hits": []}
    search = Buffer()

    def _refilter(_=None) -> None:
        q = search.text.strip()
        try:
            hits = content_search(_live_query(q), limit=60) if q else browse(limit=60)
            state["hits"] = _scoped(hits, scope)
        except Exception:
            state["hits"] = []
        state["idx"] = 0

    search.on_text_changed += _refilter
    search.text = initial
    _refilter()

    def _render():
        hits = state["hits"]
        if not hits:
            return [("class:dim", "  (no matches)")]
        out = []
        for n, h in enumerate(hits):
            label = h["session_title"] or "(untitled)"
            if h["source"] == "artifact":
                label = ("♥ " if h["favorite"] else "") + (h["path"].rsplit("/", 1)[-1] or h["url"] or label)
            snippet = " ".join(h["snippet"].replace(">>>", "").replace("<<<", "").split())
            cur = n == state["idx"]
            line = f"{'›' if cur else ' '} {h['timestamp'][:10]:<10}  {h['source']:<8}  {label[:50]:<50}  {snippet}"
            out.append(("class:picker-highlight" if cur else "", line[:180] + "\n"))
        return out

    kb = KeyBindings()

    @kb.add("up", eager=True)
    def _(e):
        if state["hits"]:
            state["idx"] = (state["idx"] - 1) % len(state["hits"])

    @kb.add("down", eager=True)
    def _(e):
        if state["hits"]:
            state["idx"] = (state["idx"] + 1) % len(state["hits"])

    @kb.add("enter", eager=True)
    def _(e):
        hits = state["hits"]
        e.app.exit(result=("enter", hits[state["idx"]]) if hits else None)

    @kb.add("c-o", eager=True)
    def _(e):
        hits = state["hits"]
        e.app.exit(result=("open", hits[state["idx"]]) if hits else None)

    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        e.app.exit(result=None)

    hint = {
        "session": "Enter resumes",
    }.get(scope, "Enter resumes a session / adds a file to your message, Ctrl-O opens it")
    search_win = Window(BufferControl(buffer=search), height=1)
    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", f"{_SCOPE_TITLES[scope]} — {hint}, Esc")]), height=1),
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

    Space toggles, a=all, x=clear, o=open, c=copy, f=favorite, v=viewer, Esc=cancel.
    ``o`` calls ``on_open(paths)`` and keeps the picker up, so assets can be
    viewed and then copied or favorited without reopening it.
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
        return [("class:dim", f"  {len(selected)} selected · Space toggle · a all · x clear · o open · c copy · f favorite · v viewer · Esc")]

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

    @kb.add("f")
    def _(e):
        e.app.exit(result=_dispatch("favorite"))

    @kb.add("v")
    def _(e):
        e.app.exit(result=("view", []))

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
    at resubmitted user messages). ``guide`` is a ``git log --graph``-style
    box-drawing prefix: at a fork the trunk child (the active one, else the
    newest) keeps its parent's column and is drawn last, and every other
    child hangs off it with ``├──`` and indents its descendants under
    ``│``. Not nesting the trunk keeps a session that re-forks along its
    active path flat instead of staircasing with every fork. ``branch``
    marks the hanging heads and ``heads`` lists the ones enclosing a row,
    outermost first, so the picker can fold a whole branch down to its head.
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
    stack = [(eid, "", "", ()) for eid in reversed(children.get(None, []))]
    while stack:
        eid, guide, cont, heads = stack.pop()
        e = by_id[eid]
        if e["type"] == EventType.USER_MESSAGE.value:
            rows.append({
                "event_id": eid,
                "type": e["type"],
                "guide": guide,
                "branch": guide != cont,
                "heads": heads,
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
                "heads": heads,
                "preview": (e["content"] or "").splitlines()[0] if e["content"] else "(empty)",
                "active": bool(e["active"]),
                "is_current": eid == current_id,
                "checkout_target": eid,
                "prefill": "",
            })
        kids = children.get(eid, [])
        if guide != cont:
            heads = heads + (eid,)
        if not kids:
            continue
        trunk = next((k for k in kids if by_id[k]["active"]), kids[-1])
        stack.append((trunk, cont, cont, heads))
        for k in reversed(kids):
            if k != trunk:
                stack.append((k, cont + "├── ", cont + "│   ", heads))
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
    Branches off the active path start folded to their head row (with a
    hidden-turn count); → unfolds, ← folds and jumps to the head. Typing
    fuzzy-filters every row by preview, folded or not; matches keep their
    tree indent.
    """
    events = db.fetch_events(db.connection(), session_id, active_only=False)
    all_rows = _tree_rows(events)
    folded = {r["event_id"] for r in all_rows if r["branch"] and not r["active"]}
    hidden = {h: sum(1 for r in all_rows if h in r["heads"]) for h in folded}
    state = {"idx": 0, "confirm": False, "rows": all_rows}
    search = Buffer()

    def _rebuild(focus: int | None = None) -> None:
        q = search.text.strip().lower()
        if q:
            state["rows"] = [r for r in all_rows if words_match(q, r["preview"])]
        else:
            state["rows"] = [r for r in all_rows if not folded.intersection(r["heads"])]
        key = "event_id" if focus is not None else "is_current"
        state["idx"] = next((n for n, r in enumerate(state["rows"]) if r[key] == (focus if focus is not None else True)), 0)

    _rebuild()
    search.on_text_changed += _rebuild

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
            if r["event_id"] in folded and not search.text.strip():
                marker += f"  … +{hidden[r['event_id']]}"
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
        return [("class:dim", "  type to filter · ↑↓ move · → unfold · ← fold · Enter checkout (user msg → edit & resubmit) · Esc")]

    in_confirm = Condition(lambda: state["confirm"])
    kb = KeyBindings()

    @kb.add("right", eager=True, filter=~in_confirm)
    def _(e):
        if not state["rows"]:
            return
        r = state["rows"][state["idx"]]
        if r["event_id"] in folded:
            folded.discard(r["event_id"])
            _rebuild(focus=r["event_id"])

    @kb.add("left", eager=True, filter=~in_confirm)
    def _(e):
        if not state["rows"]:
            return
        r = state["rows"][state["idx"]]
        head = r["event_id"] if r["branch"] else (r["heads"][-1] if r["heads"] else None)
        if head is None:
            return
        folded.add(head)
        _rebuild(focus=head)

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
            Window(FormattedTextControl(lambda: [("class:prompt", "Session tree — ├─ hangs a branch off the trunk, … +N is folded, dim rows are inactive, ◀ is the current leaf")]), height=1),
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


def _job_rows(jobs, depth: int = 1) -> list[dict]:
    """Flatten a JobRegistry into indented rows, nesting the jobs a launched
    agent registered on its own registry (``entry["jobs"]``) under it."""
    rows: list[dict] = []
    for job_id, entry in list(jobs.entries.items()):
        rows.append({"id": job_id, "entry": entry, "depth": depth})
        if entry.get("jobs") is not None:
            rows += _job_rows(entry["jobs"], depth + 1)
    return rows


async def pick_jobs(title: str, jobs) -> str | None:
    """Full-screen navigator over the current conversation and every
    background job it launched (term-N, tool-N, /bg tasks, agents and their
    own jobs) → ``"tree"`` to open the session tree, or None on close.

    Enter on a job toggles its status/output (the same snapshot ``/logs``
    shows) below the list; Enter on the conversation row opens ``/tree``.
    Redraws every second so running jobs stay current.
    """
    state = {"idx": 0, "open": None}

    def _rows() -> list[dict]:
        return [{"id": None, "entry": None, "depth": 0}] + _job_rows(jobs)

    def _render():
        import time

        now = time.time()
        out = []
        for n, r in enumerate(_rows()):
            cur = n == state["idx"]
            style = "class:picker-highlight" if cur else ""
            out.append((style, f"{'›' if cur else ' '} "))
            if r["id"] is None:
                out.append((style or "class:tree.user", "● conversation"))
                out.append((style, f"  {title[:80]}\n"))
                continue
            e = r["entry"]
            status = e.get("status")
            glyph = "●" if status == "running" else ("✓" if status == "ok" else "✗")
            secs = (e.get("ended_at") or now) - e["started_at"]
            out.append((style or "class:tree.guide", "   " * (r["depth"] - 1) + "└─ "))
            out.append((style or "class:tree.agent", f"{glyph} {r['id']}"))
            out.append((style, f"  {e.get('kind', '')}  {e.get('label', '')[:70]}  ({secs:.0f}s)\n"))
        return out

    def _detail():
        rows = _rows()
        if state["open"] is None or state["idx"] >= len(rows) or rows[state["idx"]]["id"] != state["open"]:
            return []
        try:
            text = rows[state["idx"]]["entry"]["_reader"]()
        except Exception as e:  # noqa: BLE001
            text = f"(reader error: {e})"
        try:
            height = max(get_app().output.get_size().rows - len(rows) - 4, 5)
        except Exception:
            height = 20
        return [("class:dim", "\n".join(text.splitlines()[-height:]))]

    kb = KeyBindings()

    @kb.add("up", eager=True)
    def _(e):
        state["idx"] = (state["idx"] - 1) % len(_rows())

    @kb.add("down", eager=True)
    def _(e):
        state["idx"] = (state["idx"] + 1) % len(_rows())

    @kb.add("enter", eager=True)
    def _(e):
        row = _rows()[state["idx"]]
        if row["id"] is None:
            e.app.exit(result="tree")
            return
        state["open"] = None if state["open"] == row["id"] else row["id"]

    @kb.add("escape")
    @kb.add("c-c")
    @kb.add("q")
    def _(e):
        e.app.exit(result=None)

    body = HSplit(
        [
            Window(FormattedTextControl(lambda: [("class:prompt", "Conversation & background jobs — ● running, ✓ done, ✗ failed")]), height=1),
            Window(FormattedTextControl(_render), dont_extend_height=True),
            Window(height=1),
            Window(FormattedTextControl(_detail)),
            Window(FormattedTextControl(lambda: [("class:dim", "  ↑↓ move · Enter show/hide output (conversation → session tree) · Esc")]), height=1),
        ]
    )
    app: Application = Application(
        layout=Layout(body),
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
        style=REPL_STYLE,
        refresh_interval=1,
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
        state["rows"] = all_rows if not q else [r for r in all_rows if words_match(q, r["preview"])]
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
