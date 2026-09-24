"""ReplApp — prompt_toolkit streaming-scrollback front-end.

The interactive front-end driven by ``TuiSession``. The agent's EventEmitter
stream is printed into native terminal scrollback (native scroll + drag-copy +
real OSC 8 links) by ``ReplTranscript``; a bottom inline ``Application`` provides
input with fuzzy slash completion, a status toolbar, and inline approval.

The input chrome is a hand-built, **non-full-screen** ``Application`` (not a
``PromptSession``), so its rendered region is ephemeral — redrawn each frame,
erased before streamed prints, and never auto-committed to scrollback. That lets
later panes (approval, inline tool status) collapse cleanly, and hands us
explicit echo control: real user messages are printed to scrollback by us; the
approval answer is not, so it leaves no residue.

Exposes the same surface ``TuiSession`` calls on the app: ``run_async``,
``exit``, ``call_from_thread``, ``show_approval``, ``show_session_picker``,
``show_asset_picker``, ``show_tree_picker``, ``prefill_input``,
``toggle_verbose``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import FuzzyCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import Frame

from ..events import EventEmitter, SystemNote
from .events import UserMessage
from .completion import SlashCompleter
from .style import REPL_STYLE
from .transcript import ReplTranscript


HISTORY_FILE = Path.home() / ".okonomi" / "input_history"
_SPIN_FRAMES = "✳✦"  # pulses each refresh so the status reads as live


class _ReflowSafeApplication(Application):
    """Application whose resize erase survives terminal reflow.

    On SIGWINCH prompt_toolkit erases its region by moving the cursor up the
    number of rows it *drew*, then clearing downward. But a narrower terminal
    has already re-wrapped the full-width rows (the input Frame's borders) into
    two rows each, so the cursor stops short and the old top border survives
    above the redraw. Recount the rows above the cursor at the new width — rows
    are hard-broken with CR LF, so widening never joins them and needs no fix.
    """

    def _on_resize(self) -> None:
        renderer = self.renderer
        screen, last_size = renderer._last_screen, renderer._last_size
        cols = self.output.get_size().columns
        if screen is not None and last_size is not None and cols < last_size.columns:
            has_style = renderer._style_string_has_style
            x, y = renderer._cursor_pos
            rows = 0
            for r in range(y):
                # Same "used width" rule as the renderer's diff: unstyled
                # trailing spaces are never written, so they don't wrap.
                used = max(
                    (
                        c + (cell.width or 1)
                        for c, cell in screen.data_buffer[r].items()
                        if cell.char != " " or has_style[cell.style]
                    ),
                    default=0,
                )
                rows += max(1, -(-used // cols))
            renderer._cursor_pos = Point(x=x % cols, y=rows + x // cols)
        super()._on_resize()


class ReplApp:
    def __init__(
        self,
        emitter: EventEmitter,
        on_submit: Callable[[str], None],
        on_interrupt: Callable[[], bool] | None = None,
        spinner_label_fn: Callable[[], str] | None = None,
        usage_fn: Callable[[], str] | None = None,
        bg_tasks_fn: Callable[[], list] | None = None,
        on_double_escape: Callable[[], None] | None = None,
        on_history_search: Callable[[], None] | None = None,
    ) -> None:
        self.emitter = emitter
        self.on_submit_cb = on_submit
        # Ctrl+C hook: return True if a running turn was cancelled, False if
        # there was nothing to interrupt (so the app clears the draft instead).
        self.on_interrupt_cb = on_interrupt
        self.on_double_escape = on_double_escape
        self.on_history_search = on_history_search
        self.spinner_label_fn = spinner_label_fn
        self.usage_fn = usage_fn
        self.bg_tasks_fn = bg_tasks_fn

        self.transcript = ReplTranscript()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exit = False
        self._approval_resolve: Callable[[object], None] | None = None
        # A picker requested by a slash command; the app exits so the outer
        # loop can run the (full-screen) modal, then re-enters.
        self._pending_modal: tuple | None = None

        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.input = Buffer(
            completer=FuzzyCompleter(SlashCompleter()),
            complete_while_typing=True,
            history=FileHistory(str(HISTORY_FILE)),
            accept_handler=self._on_accept,
            multiline=False,
        )
        self.app: Application = self._build_app()

    # --- layout ---

    def _build_app(self) -> Application:
        input_window = Window(
            BufferControl(buffer=self.input),
            get_line_prefix=self._input_prefix,
            height=D(min=1),
            wrap_lines=True,
            # Size to content. Without this the box's Frame draws its bottom
            # border at the bottom of the whole region prompt_toolkit reserves
            # below the cursor (all rows-below-cursor, via CPR) — ballooning
            # the input to fill the lower half of the screen on startup.
            dont_extend_height=True,
        )
        body = HSplit(
            [
                # Inline tool-status pane — morphs in place above the box while a
                # tool runs; the committed line still lands in scrollback on done.
                ConditionalContainer(
                    Window(FormattedTextControl(self._tool_status_fragments), height=1),
                    filter=Condition(lambda: bool(self.transcript.active_tool_label())),
                ),
                # Approval question — ephemeral, so it collapses cleanly on
                # answer (the code itself persists in scrollback above).
                ConditionalContainer(
                    Window(FormattedTextControl(self._approval_fragments), height=1),
                    filter=Condition(lambda: self._approval_resolve is not None),
                ),
                # The input, in a box. The box top border is the live delimiter;
                # the persisted turn rule is printed by echo_user_input on accept.
                Frame(input_window),
                # Ambient status line — hidden when idle so no blank line lingers.
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._toolbar),
                        height=1,
                        style="class:bottom-toolbar",
                    ),
                    filter=Condition(lambda: bool(self._toolbar())),
                ),
            ]
        )
        root = FloatContainer(
            content=body,
            floats=[Float(content=CompletionsMenu(max_height=12, scroll_offset=1), xcursor=True, ycursor=True)],
        )
        return _ReflowSafeApplication(
            layout=Layout(root, focused_element=input_window),
            key_bindings=merge_key_bindings([load_key_bindings(), self._key_bindings()]),
            style=REPL_STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            refresh_interval=0.5,
        )

    def _input_prefix(self, line_number: int, wrap_count: int):
        if line_number or wrap_count:
            return [("", "   ")]  # align wrapped lines under the prompt glyph
        if self._approval_resolve is not None:
            return [("class:prompt.approval", " approve> ")]
        return [("class:prompt", " › ")]

    def _glyph(self) -> str:
        return _SPIN_FRAMES[int(time.monotonic() * 2) % len(_SPIN_FRAMES)]

    def _tool_status_fragments(self):
        # Live, in-place tool status above the box, Claude-Code style:
        #   ✳ Running code  (12s)
        label = self.transcript.active_tool_label()
        if not label:
            return []
        segs = [("class:tool.glyph", f" {self._glyph()} "), ("class:tool.label", f"{label}…")]
        elapsed = self.transcript.active_tool_elapsed()
        if elapsed:
            segs.append(("class:tool.meta", f"  ({elapsed})"))
        return segs

    def _approval_fragments(self):
        return [
            ("class:approval.q", " Run this code? "),
            ("class:approval.hint", "[Enter] approve · [a] approve all · [n] skip · or type feedback"),
        ]

    def _toolbar(self):
        # Ambient agent status only — the running tool surfaces in the tool pane
        # above, so the toolbar avoids a duplicate spinner: it shows the agent
        # spinner (when no tool runs), token usage, and background tasks.
        groups: list[list] = []
        if not self.transcript.active_tool_label():
            spin = self._safe_str(self.spinner_label_fn)
            if spin:
                groups.append(
                    [("class:bottom-toolbar.glyph", f" {self._glyph()} "), ("class:bottom-toolbar.label", spin)]
                )
        usage = self._safe_str(self.usage_fn)
        if usage:
            groups.append([("class:bottom-toolbar.meta", f"↓ {usage}")])
        try:
            bg = self.bg_tasks_fn() if self.bg_tasks_fn else []
        except Exception:
            bg = []
        if bg:
            parts = []
            for job_id, label, _kind, elapsed in bg[:3]:
                label = label if len(label) <= 18 else label[:17] + "…"
                parts.append(f"{job_id} {label} {elapsed}".replace("  ", " "))
            text = "bg: " + " · ".join(parts)
            if len(bg) > 3:
                text += f" +{len(bg) - 3}"
            groups.append([("class:bottom-toolbar.bg", text)])
        if not groups:
            return []  # idle → no status bar
        out: list = []
        for i, g in enumerate(groups):
            if i:
                out.append(("class:bottom-toolbar.sep", "  ·  "))
            out.extend(g)
        return out

    @staticmethod
    def _safe_str(fn) -> str:
        if fn is None:
            return ""
        try:
            return str(fn() or "")
        except Exception:
            return ""

    # --- key bindings ---

    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            buff = event.current_buffer
            cs = buff.complete_state
            if cs and cs.current_completion:
                buff.apply_completion(cs.current_completion)  # menu open → pick
            elif cs:
                buff.cancel_completion()
            else:
                buff.validate_and_handle()  # → _on_accept

        @kb.add("c-c", eager=True)
        def _(event):
            self._on_interrupt()

        @kb.add("c-d", eager=True)
        def _(event):
            if not self.input.text:
                self.exit()

        @kb.add("c-r", eager=True)
        def _(event):
            if self._approval_resolve is None and self.on_history_search is not None:
                self.on_history_search()

        @kb.add(
            "escape",
            "escape",
            filter=Condition(
                lambda: not self.input.text
                and self._approval_resolve is None
                and self.input.complete_state is None
            ),
        )
        def _(event):
            if self.on_double_escape is not None:
                self.on_double_escape()

        return kb

    # --- main loop ---

    async def run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        with patch_stdout(raw=True):  # pass rich's ANSI/OSC escapes through unescaped
            unsub = self.emitter.subscribe(self.transcript.on_event)
            self.transcript.replaying = True
            for ev in self.emitter.history():
                self.transcript.on_event(ev)
            self.transcript.replaying = False
            try:
                while not self._exit:
                    try:
                        await self.app.run_async()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if self._exit:
                        break
                    if self._pending_modal is not None:
                        await self._run_modal()
                    else:
                        break  # app ended with no modal pending → stop
            finally:
                unsub()

    def _on_accept(self, buf: Buffer) -> bool:
        text = buf.text
        if self._approval_resolve is not None:
            self._resolve_approval(text)  # no echo — approval is not a turn
            return False
        if text.strip():
            self.transcript.echo_user_input(text)  # persist the turn to scrollback
            self.on_submit_cb(text)
        return False

    def _on_interrupt(self) -> None:
        # 1. answering an approval → skip; 2. turn running → cancel it;
        # 3. otherwise clear the current draft.
        if self._approval_resolve is not None:
            resolver = self._approval_resolve
            self._approval_resolve = None
            self._invalidate()
            self._safe_resolve(resolver, False)
            return
        if self.on_interrupt_cb is not None and self.on_interrupt_cb():
            return  # a running turn was cancelled
        self.input.reset()

    # --- inline approval (runs on the main loop via call_from_thread) ---

    def show_approval(self, code: str, resolve: Callable[[object], None]) -> None:
        self._approval_resolve = resolve
        if code:
            self.transcript.print_code(code)  # persists in scrollback (highlighted)
        # The "Run this code?" prompt renders in the ephemeral approval pane
        # (see _approval_fragments), so it collapses on answer instead of
        # leaving a log line behind.
        self._invalidate()

    def _resolve_approval(self, text: str) -> None:
        stripped = text.strip()
        low = stripped.lower()
        if stripped == "" or low in ("y", "yes"):
            decision: object = True
        elif low in ("a", "all", "approve all"):
            decision = "__auto_approve__"
        elif low in ("n", "no", "skip"):
            decision = False
        else:
            decision = text
            self.emitter.emit(UserMessage(text))  # echo feedback so it isn't lost
        resolver = self._approval_resolve
        self._approval_resolve = None
        self._invalidate()
        self._safe_resolve(resolver, decision)

    @staticmethod
    def _safe_resolve(resolver: Callable[[object], None] | None, decision: object) -> None:
        if resolver is not None:
            try:
                resolver(decision)
            except Exception:
                pass

    def _invalidate(self) -> None:
        try:
            self.app.invalidate()
        except Exception:
            pass

    # --- TuiSession touchpoints ---

    def exit(self) -> None:
        self._exit = True
        self._stop_app()

    def _stop_app(self) -> None:
        try:
            if self.app.is_running:
                self.app.exit()
        except Exception:
            pass

    def call_from_thread(self, fn: Callable, *args) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: fn(*args))

    def show_session_picker(self, on_select) -> None:
        self._pending_modal = ("session", on_select)
        self._stop_app()  # break out so the outer loop runs the modal

    def show_asset_picker(self, artifacts, on_action) -> None:
        self._pending_modal = ("asset", on_action, list(artifacts))
        self._stop_app()

    def show_tree_picker(self, session_id: str, on_select) -> None:
        self._pending_modal = ("tree", on_select, session_id)
        self._stop_app()

    def show_history_picker(self, session_id: str, on_select) -> None:
        self._pending_modal = ("history", on_select, session_id)
        self._stop_app()

    def prefill_input(self, text: str) -> None:
        """Put text in the input box, cursor at the end — the buffer
        survives the modal round-trip, so this lands after re-entry."""
        self.input.document = Document(text, cursor_position=len(text))


    async def _run_modal(self) -> None:
        modal = self._pending_modal
        self._pending_modal = None
        if modal is None:
            return
        try:
            if modal[0] == "session":
                from .pickers import pick_session

                chosen = await pick_session()
                if chosen:
                    modal[1](chosen)
            elif modal[0] == "asset":
                from .pickers import pick_assets

                result = await pick_assets(modal[2], lambda paths: modal[1]("open", paths))
                if result is not None:
                    modal[1](result[0], result[1])  # on_action(action, paths)
            elif modal[0] == "tree":
                from .pickers import pick_tree

                result = await pick_tree(modal[2])
                if result is not None:
                    modal[1](*result)  # on_select(target_event_id, prefill, summarize)
            elif modal[0] == "history":
                from .pickers import pick_history

                result = await pick_history(modal[2])
                if result is not None:
                    modal[1](*result)  # on_select(target_event_id, prefill, summarize)
        except Exception as e:  # noqa: BLE001
            self.emitter.emit(SystemNote(f"Picker error: {e}", level="warn"))

    def toggle_verbose(self) -> None:
        self.transcript.tool_detail = "full" if self.transcript.tool_detail == "compact" else "compact"
        self.emitter.emit(SystemNote(f"Tool detail: {self.transcript.tool_detail}."))
