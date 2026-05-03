"""Textual App: skeleton with input box + scrollable history pane.

Phase 1 of the TUI migration. Renders a stream of events via EventEmitter
into a scrollable VerticalScroll pane. Collapsible tool calls + agent
wiring come in later phases.
"""
from __future__ import annotations

from typing import Callable

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from .events import (
    AssistantStream,
    Banner,
    Divider,
    Event,
    EventEmitter,
    SystemNote,
    ToolCall,
    UserMessage,
)
from .widgets import (
    AssistantStreamWidget,
    BgTasksWidget,
    DividerWidget,
    StatusLineWidget,
    SystemNoteWidget,
    ToolCallWidget,
    UserMessageWidget,
    WelcomeBannerWidget,
)


class SessionPicker(ModalScreen[str]):
    """Modal listing past sessions. Filter via the top Input, navigate
    with arrow keys, Enter resumes. Esc closes.
    """

    DEFAULT_CSS = """
    SessionPicker {
        align: center middle;
    }
    #picker-dialog {
        width: 80%;
        height: 70%;
        background: #1a1a2e;
        border: round #6f4355;
        padding: 1 2;
        layout: vertical;
    }
    #picker-search {
        height: 3;
        border: round #6f4355;
        background: transparent;
        color: #e8e8e8;
        margin: 0 0 1 0;
    }
    #picker-search:focus {
        border: round #8f5a75;
    }
    #picker-list {
        height: 1fr;
        background: transparent;
        border: none;
        padding: 0;
        scrollbar-size: 0 0;
    }
    #picker-list > .option-list--option {
        padding: 0 1;
        color: #b4b4c3;
    }
    #picker-list > .option-list--option-highlighted {
        background: #333355;
        color: #d787af;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("down", "to_list", "Move to list", show=False),
    ]

    def __init__(self, on_select: Callable[[str], None]) -> None:
        super().__init__()
        self.on_select = on_select
        self._all_sessions: list = []

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        from ..session import SessionManager
        all_sessions = SessionManager().list_sessions(limit=500)
        # Drop sessions that have no usable display content. These are
        # interrupted/empty rows that just show date-prefixed session_id
        # placeholders and aren't meaningfully resumable.
        self._all_sessions = [
            s for s in all_sessions
            if (getattr(s, "title", "") or "").strip()
            or (getattr(s, "task_input", "") or "").strip()
        ]

        with Vertical(id="picker-dialog"):
            yield Input(placeholder="Filter sessions…", id="picker-search")
            yield OptionList(*self._build_options(""), id="picker-list")

    def on_mount(self) -> None:
        self.query_one("#picker-search", Input).focus()

    def _build_options(self, query: str) -> list[Option]:
        from rich.text import Text

        q = query.strip().lower()
        rows = []
        for s in self._all_sessions:
            name = s.display_name or "(untitled)"
            date = (s.created_at or "")[:10]
            haystack = f"{name} {date} {s.session_id}".lower()
            if q and not _fuzzy_match(q, haystack):
                continue
            line = Text()
            line.append(f"{date}  ", style="#888888")
            line.append(name[:90], style="#b4b4c3")
            rows.append(Option(line, id=s.session_id))
        if not rows:
            rows = [Option(Text("(no matches)", style="#888888"), disabled=True)]
        return rows

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "picker-search":
            return
        ol = self.query_one("#picker-list", OptionList)
        ol.clear_options()
        for opt in self._build_options(event.value):
            ol.add_option(opt)
        if ol.option_count > 0:
            ol.highlighted = 0

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the search field accepts the highlighted session.
        if event.input.id != "picker-search":
            return
        ol = self.query_one("#picker-list", OptionList)
        idx = ol.highlighted if ol.highlighted is not None else 0
        opt = ol.get_option_at_index(idx) if ol.option_count > 0 else None
        if opt is None or opt.id is None:
            return
        self.on_select(opt.id)
        self.dismiss(opt.id)

    def action_to_list(self) -> None:
        """Down-arrow from the search field jumps focus to the list."""
        if isinstance(self.focused, Input) and self.focused.id == "picker-search":
            self.query_one("#picker-list", OptionList).focus()

    def on_key(self, event) -> None:
        """Up-arrow at the top of the list bounces focus back to search.

        Without this, once the user descends into the list with Down, there
        is no way to return to the filter input — Up only moves within the
        list and stops at index 0.
        """
        if event.key != "up":
            return
        if not isinstance(self.focused, OptionList):
            return
        ol = self.focused
        if ol.id != "picker-list":
            return
        if ol.highlighted in (0, None):
            self.query_one("#picker-search", Input).focus()
            event.stop()
            event.prevent_default()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.on_select(event.option.id)
        self.dismiss(event.option.id or "")

    def action_close(self) -> None:
        self.dismiss("")


def _fuzzy_match(query: str, haystack: str) -> bool:
    """Simple subsequence match: every char of `query` appears in order in
    `haystack`. Cheap, dependency-free, and matches what cmd-K pickers do."""
    i = 0
    for ch in haystack:
        if i < len(query) and ch == query[i]:
            i += 1
        if i == len(query):
            return True
    return i == len(query)




class OkonomiApp(App):
    """Main TUI application."""

    # Palette mirrors the prompt_toolkit CLI theme:
    #   accent pink   #d787af   user bullet, input border, prompt
    #   amber         #ffaf00   approval state
    #   session green #00cc66   resume-session state
    #   LLM text gray #b4b4c3   assistant streaming text
    #   dim gray      #888888   dividers, tool-call headers
    #   light fg      #e8e8e8   code preview, input text
    #   menu hi bg    #333355   focused selection
    # Backgrounds use ansi_default so the terminal's own theme shows through
    # (we run in inline mode — no alt-screen takeover).
    CSS = """
    Screen {
        background: transparent;
        color: #b4b4c3;
        layout: vertical;
        /* inline mode sizes the app from Screen height. Keep a generous
           viewport so multi-line messages and tool-call details are not
           clipped down to the last couple of rows. */
        min-height: 24;
        height: 90%;
    }
    #history {
        height: 1fr;
        padding: 0 1;
        background: transparent;
        /* Hide the scrollbar — users can still scroll with arrow keys,
           Page Up / Page Down, or the mouse wheel. */
        scrollbar-size: 0 0;
        scrollbar-gutter: auto;
    }
    #input {
        height: auto;
        border: round #6f4355;
        padding: 0 1;
        margin: 0 1;
        background: transparent;
        color: #e8e8e8;
    }
    #input:focus {
        border: round #8f5a75;
    }
    #input.-approval {
        border: round #ffaf00;
        color: #ffaf00;
    }
    #slash-dropdown {
        height: auto;
        max-height: 8;
        margin: 0 1;
        padding: 0;
        background: #1a1a2e;
        border: none;
        color: #e8e8e8;
        display: none;
    }
    #slash-dropdown.-visible {
        display: block;
    }
    #slash-dropdown > .option-list--option-highlighted {
        background: #333355;
        color: #d787af;
        text-style: bold;
    }
    #slash-dropdown > .option-list--option {
        padding: 0 1;
    }
    /* Gap between the bottom-most metadata row and the terminal edge. */
    #bottom-spacer {
        height: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+d", "quit", "Quit", show=False),
        # Ctrl+C still needs priority so it interrupts a running turn even if
        # the focus is on a tool call or the dropdown.
        Binding("ctrl+c", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+o", "toggle_latest", "Expand latest tool call", show=True),
        # Tab/Up/Down are NOT priority — let modal screens (SessionPicker,
        # OptionList) handle them when focused. They bubble back to the
        # App when no widget consumes them.
        Binding("tab", "focus_history", "Focus history", show=True),
        Binding("escape", "focus_input", "Focus input", show=True),
        Binding("up", "history_prev", "Previous input", show=False),
        Binding("down", "history_next", "Next input", show=False),
    ]

    HISTORY_FILE = Path.home() / ".okonomi" / "input_history"
    HISTORY_LIMIT = 500

    def __init__(
        self,
        emitter: EventEmitter,
        on_submit: Callable[[str], None] | None = None,
        spinner_label_fn: Callable[[], str] | None = None,
        usage_fn: Callable[[], str] | None = None,
        bg_tasks_fn: Callable[[], list[tuple[str, str]]] | None = None,
    ) -> None:
        super().__init__()
        self.emitter = emitter
        self.on_submit_cb = on_submit
        self.spinner_label_fn = spinner_label_fn or (lambda: "")
        self.usage_fn = usage_fn or (lambda: "")
        self.bg_tasks_fn = bg_tasks_fn or (lambda: [])
        self._unsubscribe: Callable[[], None] | None = None
        self._widgets_by_id: dict[str, object] = {}
        # Inline approval state. When set, the next input submission is
        # routed through this callback instead of as a user message.
        self._approval_resolve: Callable[[object], None] | None = None
        # Tracks whether the user has submitted at least one message.
        # Used to drop the "What's on your mind?" placeholder after the
        # first turn — by then it's just visual noise.
        self._has_submitted: bool = False
        # The SystemNote event currently asking for approval. We mutate it
        # to a one-line summary on resolve so the verbose prompt doesn't
        # litter scrollback after every approval.
        self._approval_prompt_event = None
        # Persistent input history — same file as the old prompt_toolkit CLI.
        self._history: list[str] = self._load_history()
        self._history_cursor: int | None = None  # None = typing fresh
        self._pending_input: str = ""  # saved draft when cursor steps up

    def compose(self) -> ComposeResult:
        from textual.widgets import Static as _Static

        # History fills the top; input and metadata flow after it in order.
        # can_focus=False so Tab skips the scroll pane and reaches the
        # App-level `tab` binding which focuses the last ToolCallWidget.
        scroll = VerticalScroll(id="history")
        scroll.can_focus = False
        yield scroll

        # Slash-command completions appear just above the input.
        dropdown = OptionList(id="slash-dropdown")
        dropdown.can_focus = False  # keep focus on Input; dropdown is a hint
        yield dropdown

        yield Input(
            id="input",
            placeholder="What's on your mind?",
            suggester=_SlashSuggester(),
        )

        # Metadata rows below the input: spinner/tokens on top, then bg tasks.
        yield StatusLineWidget(self.spinner_label_fn, self.usage_fn)
        yield BgTasksWidget(self.bg_tasks_fn)

        # Spacer pushes the metadata clear of the terminal bottom edge.
        yield _Static("", id="bottom-spacer")

    async def on_mount(self) -> None:
        self._unsubscribe = self.emitter.subscribe(self._on_event)
        # Replay any history that was emitted before the app mounted.
        for ev in self.emitter.history():
            self._render_event(ev)
        self.query_one(Input).focus()

    async def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()

    def _on_event(self, event: Event) -> None:
        """Called from the emitting thread. Marshal into Textual's loop.

        When the emit originates on the UI thread itself (e.g. input submit
        -> emitter.user()), call_from_thread() raises — so branch on thread
        identity and dispatch directly in that case.
        """
        import threading
        if threading.current_thread() is threading.main_thread():
            self._render_event(event)
        else:
            self.call_from_thread(self._render_event, event)

    def _render_event(self, event: Event) -> None:
        history = self.query_one("#history", VerticalScroll)

        existing = self._widgets_by_id.get(event.id)
        if existing is not None:
            # Event was mutated (streaming append, tool finished). Re-sync.
            if hasattr(existing, "sync_from_event"):
                existing.sync_from_event()
            history.scroll_end(animate=False)
            return

        widget = self._build_widget(event)
        if widget is None:
            return
        self._widgets_by_id[event.id] = widget
        history.mount(widget)
        history.scroll_end(animate=False)

    def _build_widget(self, event: Event):
        if isinstance(event, Banner):
            return WelcomeBannerWidget()
        if isinstance(event, Divider):
            return DividerWidget(event)
        if isinstance(event, UserMessage):
            return UserMessageWidget(event)
        if isinstance(event, AssistantStream):
            return AssistantStreamWidget(event)
        if isinstance(event, ToolCall):
            return ToolCallWidget(event)
        if isinstance(event, SystemNote):
            return SystemNoteWidget(event)
        return None

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Show/update the slash-command dropdown as the user types."""
        if event.input.id != "input":
            return
        self._refresh_slash_dropdown(event.value)

    def _refresh_slash_dropdown(self, value: str) -> None:
        from . import commands

        dropdown = self.query_one("#slash-dropdown", OptionList)
        # Don't pop the dropdown while user is responding to an approval.
        if self._approval_resolve is not None or not value.startswith("/"):
            dropdown.clear_options()
            dropdown.remove_class("-visible")
            return

        low = value.lower()
        matches = [
            (cmd, desc) for cmd, desc in commands.SLASH_COMMANDS
            if cmd.startswith(low) and cmd != value
        ]
        dropdown.clear_options()
        if not matches:
            dropdown.remove_class("-visible")
            return
        from rich.text import Text
        for cmd, desc in matches:
            line = Text()
            line.append(cmd, style="bold")
            line.append("  ")
            line.append(desc, style="dim")
            dropdown.add_option(Option(line, id=cmd))
        dropdown.add_class("-visible")
        if dropdown.option_count > 0:
            dropdown.highlighted = 0

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter while the slash dropdown is visible accepts the highlighted
        # option instead of submitting the partial command. Matches the
        # Slack/VSCode/Cmd-palette convention. User then presses Enter
        # again to submit. Skipped during approval mode.
        dropdown = self.query_one("#slash-dropdown", OptionList)
        if (
            self._approval_resolve is None
            and dropdown.has_class("-visible")
            and dropdown.option_count > 0
        ):
            self._accept_dropdown()
            return

        text = event.value.strip()
        inp = self.query_one(Input)
        inp.value = ""

        # Inline approval mode: route the submission to the approval resolver
        # rather than treating it as a user message.
        if self._approval_resolve is not None:
            decision: object
            low = text.lower()
            if text == "" or low in ("y", "yes"):
                decision = True
            elif low in ("a", "all", "approve all"):
                decision = "__auto_approve__"
            elif low in ("n", "no", "skip"):
                decision = False
            else:
                decision = text  # feedback for the agent
            resolver = self._approval_resolve
            self._approval_resolve = None
            self._exit_approval_mode()
            resolver(decision)
            return

        if not text:
            return

        # Record to persistent history and reset cursor state.
        self._append_history(text)
        self._history_cursor = None
        self._pending_input = ""

        # First submit drops the welcome placeholder permanently.
        if not self._has_submitted:
            self._has_submitted = True
            inp.placeholder = ""

        # Dismiss slash dropdown on submit.
        dropdown = self.query_one("#slash-dropdown", OptionList)
        dropdown.clear_options()
        dropdown.remove_class("-visible")

        if self.on_submit_cb is not None:
            self.on_submit_cb(text)
        else:
            # No agent wired — standalone/demo path.
            self.emitter.divider()
            self.emitter.user(text)

    # --- Focus / navigation actions ---

    def action_focus_history(self) -> None:
        """Tab: complete dropdown option if visible, otherwise focus last tool call."""
        dropdown = self.query_one("#slash-dropdown", OptionList)
        if dropdown.has_class("-visible") and dropdown.option_count > 0:
            self._accept_dropdown()
            return
        tool_calls = list(self.query(ToolCallWidget))
        if not tool_calls:
            return
        tool_calls[-1].focus()

    def _accept_dropdown(self) -> None:
        dropdown = self.query_one("#slash-dropdown", OptionList)
        idx = dropdown.highlighted
        if idx is None:
            idx = 0
        opt = dropdown.get_option_at_index(idx)
        if opt is None or opt.id is None:
            return
        inp = self.query_one(Input)
        inp.value = opt.id + " "
        inp.cursor_position = len(inp.value)
        dropdown.clear_options()
        dropdown.remove_class("-visible")

    def action_focus_input(self) -> None:
        # Also dismiss any open slash dropdown.
        dropdown = self.query_one("#slash-dropdown", OptionList)
        dropdown.clear_options()
        dropdown.remove_class("-visible")
        self.query_one(Input).focus()

    def action_toggle_latest(self) -> None:
        """Ctrl+O — toggle the most recent tool call from anywhere."""
        tool_calls = list(self.query(ToolCallWidget))
        if not tool_calls:
            return
        tool_calls[-1].action_toggle()

    def action_interrupt(self) -> None:
        """Ctrl+C — cancel current agent turn if one is running."""
        session = getattr(self, "session", None)
        if session is not None and hasattr(session, "cancel_current_turn"):
            session.cancel_current_turn()

    # --- Session picker ---

    def show_session_picker(self, on_select: Callable[[str], None]) -> None:
        """Open a modal listing past sessions to resume."""
        self.push_screen(SessionPicker(on_select))

    # --- Inline approval ---

    def show_approval(self, code: str, resolve: Callable[[object], None]) -> None:
        """Enter inline approval mode.

        Emits the prompt into the history stream and retints the input box
        in amber until the user responds. The next submission is passed to
        `resolve` as True / False / "__auto_approve__" / feedback string.
        On resolve the prompt event is mutated to a short summary so the
        full menu doesn't stay in scrollback forever.
        """
        self._approval_prompt_event = self.emitter.note(
            "Run this code? [Enter] approve  |  [a] approve all  |  "
            "[n] skip  |  any other text: feedback",
            level="warn",
        )
        self._approval_resolve = resolve
        inp = self.query_one(Input)
        inp.add_class("-approval")
        inp.placeholder = "approve>"
        inp.focus()

    def _exit_approval_mode(self) -> None:
        inp = self.query_one(Input)
        inp.remove_class("-approval")
        # Restore the prompt-style placeholder only on a fresh session;
        # after the first submit we keep it empty so the input stays
        # visually clean.
        inp.placeholder = "" if self._has_submitted else "What's on your mind?"
        # Remove the verbose "Run this code?" prompt from the pane — once
        # the user has answered, leaving the menu hint in scrollback is
        # noise. The Event itself stays in emitter.history() for any other
        # subscribers that might care.
        ev = self._approval_prompt_event
        self._approval_prompt_event = None
        if ev is not None:
            self._remove_event_widget(ev)

    def _remove_event_widget(self, event) -> None:
        """Unmount the widget mapped to the given event id, if present."""
        widget = self._widgets_by_id.pop(getattr(event, "id", None), None)
        if widget is not None:
            try:
                widget.remove()
            except Exception:
                pass

    # --- Input history (Up/Down) ---

    def _load_history(self) -> list[str]:
        try:
            if self.HISTORY_FILE.exists():
                lines = self.HISTORY_FILE.read_text().splitlines()
                return [ln for ln in lines if ln.strip()][-self.HISTORY_LIMIT:]
        except Exception:
            pass
        return []

    def _append_history(self, text: str) -> None:
        if not text.strip():
            return
        if self._history and self._history[-1] == text:
            return  # collapse immediate dupes
        self._history.append(text)
        self._history = self._history[-self.HISTORY_LIMIT:]
        try:
            self.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.HISTORY_FILE.write_text("\n".join(self._history) + "\n")
        except Exception:
            pass

    def action_history_prev(self) -> None:
        """↑ — navigate dropdown if visible, else step back through history."""
        inp = self.query_one(Input)
        if not isinstance(self.focused, Input):
            return
        dropdown = self.query_one("#slash-dropdown", OptionList)
        if dropdown.has_class("-visible") and dropdown.option_count > 0:
            idx = dropdown.highlighted or 0
            dropdown.highlighted = max(0, idx - 1)
            return
        if not self._history:
            return
        if self._history_cursor is None:
            self._pending_input = inp.value
            self._history_cursor = len(self._history) - 1
        elif self._history_cursor > 0:
            self._history_cursor -= 1
        inp.value = self._history[self._history_cursor]
        inp.cursor_position = len(inp.value)

    def action_history_next(self) -> None:
        """↓ — navigate dropdown if visible, else forward through history."""
        inp = self.query_one(Input)
        if not isinstance(self.focused, Input):
            return
        dropdown = self.query_one("#slash-dropdown", OptionList)
        if dropdown.has_class("-visible") and dropdown.option_count > 0:
            idx = dropdown.highlighted or 0
            dropdown.highlighted = min(dropdown.option_count - 1, idx + 1)
            return
        if self._history_cursor is None:
            return
        if self._history_cursor < len(self._history) - 1:
            self._history_cursor += 1
            inp.value = self._history[self._history_cursor]
        else:
            self._history_cursor = None
            inp.value = self._pending_input
            self._pending_input = ""
        inp.cursor_position = len(inp.value)


class _SlashSuggester(Suggester):
    """Tab-completion for slash commands. Matches the static SLASH_COMMANDS
    table from commands.py so users see completions without typing full names.
    """

    def __init__(self) -> None:
        super().__init__(use_cache=False, case_sensitive=False)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        from . import commands

        low = value.lower()
        for cmd, _desc in commands.SLASH_COMMANDS:
            if cmd.startswith(low) and cmd != value:
                return cmd
        return None


def run_standalone(demo: bool = True) -> None:
    """Run the TUI with no agent wiring — for dev/testing.

    When demo=True, emits a handful of seed events so the layout is
    visible without typing anything.
    """
    emitter = EventEmitter()
    if demo:
        emitter.divider()
        emitter.user("What were my last 6 sessions?")
        t = emitter.tool_call(
            "run",
            args="sessions.list(limit=6)",
            details="sessions.list(limit=6)\n# returns List[SessionSummary]",
        )
        emitter.tool_finish(
            t,
            output=(
                "session-042  2026-04-20  image generation\n"
                "session-041  2026-04-19  video stitch\n"
                "session-040  2026-04-18  browser render\n"
                "session-039  2026-04-17  image edit\n"
                "session-038  2026-04-16  reference image\n"
                "session-037  2026-04-15  text-to-image"
            ),
        )
        emitter.note("Interrupted. History preserved.", level="warn")
    OkonomiApp(emitter).run()


if __name__ == "__main__":
    run_standalone()
