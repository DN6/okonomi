"""On-demand full-screen pickers for the REPL (session + asset).

Each is a transient ``prompt_toolkit`` ``Application`` run on the alternate
screen and awaited; selecting exits with the chosen value, Esc / Ctrl-C exit
with ``None``. The alt-screen leaves the REPL scrollback untouched, so native
scroll + copy are unaffected once the picker closes.

The session picker focuses a live fuzzy-search buffer. The asset picker has
single-letter action verbs (o/c/h/a/x) that would collide with a focused text
buffer, so it omits live search for now and navigates with the arrow keys.
"""

from __future__ import annotations

import sys

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output.defaults import create_output

from ...session import SessionManager
from ...utils import fuzzy_match
from .style import REPL_STYLE


def _ref(a) -> str:
    return getattr(a, "path", "") or getattr(a, "url", "") or ""


async def pick_session() -> str | None:
    """Full-screen fuzzy session picker → chosen session_id or None."""
    rows_all = [
        s
        for s in SessionManager().list_sessions(limit=500)
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
