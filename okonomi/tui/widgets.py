"""Per-event widget classes rendered in the history pane."""
from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from .events import AssistantStream, Divider, SystemNote, ToolCall, UserMessage


class WelcomeBannerWidget(Static):
    """Sakura wave banner shown at session start."""

    DEFAULT_CSS = """
    WelcomeBannerWidget {
        height: auto;
        margin: 0 0 1 0;
        text-align: center;
        width: 100%;
    }
    """

    def __init__(self) -> None:
        super().__init__(render_welcome_banner())


class DividerWidget(Static):
    """Thin horizontal rule between turns."""

    DEFAULT_CSS = """
    DividerWidget {
        height: 1;
        color: #444455;
    }
    """

    def __init__(self, event: Divider) -> None:
        super().__init__("─" * 40)
        self.event = event


class UserMessageWidget(Static):
    """User turn line — pink bullet, light text, matches prompt color."""

    DEFAULT_CSS = """
    UserMessageWidget {
        height: auto;
        color: #e8e8e8;
    }
    """

    def __init__(self, event: UserMessage) -> None:
        # Compose as Rich Text so user content containing `[...]` isn't
        # eaten as Textual markup tags. The bullet keeps its pink style.
        text = Text()
        text.append("● ", style="bold #d787af")
        text.append(event.text)
        super().__init__(text)
        self.event = event


class AssistantStreamWidget(Static):
    """Streaming assistant text — muted LLM gray, matches the prompt_toolkit palette.

    Decouples the emit rate from the render rate: the emitter mutates
    ``event.text`` as fast as tokens arrive (could be 100+/sec), but the
    widget redraws on a fixed-cadence timer (~30 Hz) and only when the
    event has actually grown. That gives a smooth typewriter feel without
    a re-render storm during bursty streams. When the event is marked
    done, the timer stops and a final sync flushes whatever is left.
    """

    REFRESH_INTERVAL_S = 0.033  # ~30 Hz

    DEFAULT_CSS = """
    AssistantStreamWidget {
        height: auto;
        color: #b4b4c3;
    }
    """

    def __init__(self, event: AssistantStream) -> None:
        super().__init__(Text(event.text))
        self.event = event
        self._last_rendered_len: int = 0
        self._refresh_timer = None

    def on_mount(self) -> None:
        self._render_now()
        self._refresh_timer = self.set_interval(
            self.REFRESH_INTERVAL_S, self._tick
        )

    def _render_now(self) -> None:
        if not self.is_mounted:
            return
        self.update(Text(self.event.text))
        self._last_rendered_len = len(self.event.text)

    def _tick(self) -> None:
        # Cheap length compare — skip the full update() if nothing arrived.
        if len(self.event.text) != self._last_rendered_len:
            self._render_now()

    def sync_from_event(self) -> None:
        """Mid-stream: do nothing (the timer covers it). On done: flush
        the final state immediately and stop the timer."""
        if not self.is_mounted:
            return
        if self.event.done:
            self._render_now()
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
                self._refresh_timer = None


class SystemNoteWidget(Static):
    """Status line — dim gray for info, amber for warnings, red for errors."""

    DEFAULT_CSS = """
    SystemNoteWidget {
        height: auto;
        color: #888888;
    }
    SystemNoteWidget.-warn { color: #ffaf00; }
    SystemNoteWidget.-error { color: #ff5f5f; }
    """

    def __init__(self, event: SystemNote) -> None:
        # Plain Text, not markup — the approval prompt uses [Enter]/[a]/[n]
        # literally and we need them to render.
        super().__init__(Text(event.text))
        self.event = event
        if event.level == "warn":
            self.add_class("-warn")
        elif event.level == "error":
            self.add_class("-error")


_HEARTBEAT_FRAMES = ("·", "•", "●", "•")
_TRAIL_DOTS = ("●", "●", "•", "·")


class StatusLineWidget(Static):
    """Thin status line showing the agent's activity (spinner_text) and
    token usage. Polls callables each tick; collapses to height 0 when
    idle and no usage is available.

    Mirrors the pink heartbeat/trailing-dots animation from display.py.
    """

    DEFAULT_CSS = """
    StatusLineWidget {
        height: 1;
        color: #d787af;
        padding: 0 1;
    }
    """

    tick = reactive(0, init=False)

    def __init__(self, label_fn, usage_fn=None) -> None:
        super().__init__("")
        self.label_fn = label_fn
        self.usage_fn = usage_fn or (lambda: "")
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.12, self._advance)

    def _advance(self) -> None:
        self.tick += 1
        label = self.label_fn() or ""
        usage = self.usage_fn() or ""
        text = Text()
        if label:
            glyph = self._render_glyph(label, self.tick)
            text.append_text(glyph)
            text.append("  ")
            text.append(label, style="bold #d787af")
        if usage:
            if label:
                text.append("    ")
            text.append(usage, style="#888888")
        self.update(text)

    def _render_glyph(self, label: str, tick: int):
        out = Text()
        if "thinking" in label.lower():
            idx = tick % len(_HEARTBEAT_FRAMES)
            out.append(_HEARTBEAT_FRAMES[idx], style="#d787af")
            return out
        # Trailing-dots animation: a dot sweeps leaving a fading trail.
        width = 4
        cycle = width + 2
        for i in range(width):
            pos = (tick - i) % cycle
            if pos < len(_TRAIL_DOTS):
                out.append(_TRAIL_DOTS[pos], style="#d787af")
            else:
                out.append(" ")
        return out


_WAVE_COLORS = ["#f087af", "#d25a87", "#a5375f"]


def render_welcome_banner(label: str = "okonomi") -> Text:
    """Build the static sakura-wave banner as a Rich Text.

    Each line is emitted *without* leading whitespace; the widget centers
    them via CSS `text-align: center` so the banner sits in the middle of
    the actual terminal width regardless of size.

    Mirrors the prompt_toolkit `print_loading_art`: three rows of wave
    glyphs above and below a centered pink label, in graduated pink tones.
    """

    def _wave_line(content_w: int, offset: int = 0) -> str:
        wave = "~^~"
        shifted = wave[offset % 3:] + wave[: offset % 3]
        return (shifted * (content_w // 3 + 2))[offset % 3 : offset % 3 + content_w]

    text = Text()
    n = len(_WAVE_COLORS)
    # Above (outermost first → innermost)
    for i in range(n - 1, -1, -1):
        content_w = len(label) + (i + 1) * 6
        text.append(_wave_line(content_w, i) + "\n", style=_WAVE_COLORS[i])
    # Label
    text.append(label + "\n", style="bold #d787af")
    # Below (innermost → outermost)
    for i in range(n):
        content_w = len(label) + (i + 1) * 6
        text.append(_wave_line(content_w, i + 1) + "\n", style=_WAVE_COLORS[i])
    return text


class BgTasksWidget(Static):
    """Dim status row listing in-flight background tasks.

    Polls a supplier each tick that returns ``list[(id, label)]``. When the
    list is empty the widget collapses to height 0. An animated pink dot
    precedes each entry so the user can tell at a glance that real work is
    happening in the background.
    """

    DEFAULT_CSS = """
    BgTasksWidget {
        height: 0;
        color: #888888;
        padding: 0 1;
    }
    BgTasksWidget.-active {
        height: auto;
    }
    """

    def __init__(self, supplier) -> None:
        super().__init__("")
        self.supplier = supplier
        self._tick = 0

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh)

    def _refresh(self) -> None:
        self._tick += 1
        try:
            items = self.supplier() or []
        except Exception:
            items = []
        if not items:
            self.remove_class("-active")
            self.update("")
            return
        self.add_class("-active")
        glyph = _HEARTBEAT_FRAMES[self._tick % len(_HEARTBEAT_FRAMES)]
        text = Text()
        text.append("background: ", style="#888888")
        for i, (task_id, label) in enumerate(items):
            if i > 0:
                text.append("  ·  ", style="#444455")
            text.append(glyph, style="#d787af")
            text.append(f" {task_id}", style="#d787af bold")
            if label:
                text.append(f" ({label})", style="#888888")
        self.update(text)


class ToolCallWidget(Widget):
    """Collapsible tool call.

    Shows `▶ tool_name` when collapsed, `▼ tool_name` when expanded with
    a details block below. Highlightable for keyboard navigation.
    """

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
    }
    ToolCallWidget > .header {
        height: 1;
        color: #888888;
    }
    ToolCallWidget:focus > .header {
        color: #d787af;
        text-style: bold;
    }
    ToolCallWidget > .details {
        height: auto;
        color: #888888;
        padding: 0 0 0 2;
        display: none;
    }
    ToolCallWidget.-expanded > .details {
        display: block;
    }
    ToolCallWidget.-error > .header {
        color: #ff5f5f;
    }
    """

    can_focus = True

    BINDINGS = [
        Binding("enter", "toggle", "Expand/collapse", show=True),
        Binding("space", "toggle", "Expand/collapse", show=False),
    ]

    expanded = reactive(False, init=False)

    def action_toggle(self) -> None:
        self.expanded = not self.expanded
        # Ensure the newly-revealed details land in the visible area. Without
        # this, Ctrl+O on a long history silently expands content below the
        # viewport and the user sees no change.
        if self.expanded:
            self.call_after_refresh(self.scroll_visible, animate=False)

    def watch_expanded(self, value: bool) -> None:
        self.set_class(value, "-expanded")
        if not self.is_mounted:
            return
        try:
            header = self.query_one("#header", Static)
        except Exception:
            return
        header.update(self._header_text())

    def __init__(self, event: ToolCall) -> None:
        super().__init__()
        self.event = event
        # Sync the reactive from the event before compose so the glyph
        # (▼/▶) matches the details visibility on first render. The
        # reactive's watcher safely no-ops the header update pre-mount,
        # but set_class still applies — which is what controls whether
        # the details block is visible.
        if event.expanded:
            self.expanded = True

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), classes="header", id="header")
        yield Static(self._details_text(), classes="details", id="details")

    def _glyph(self) -> str:
        return "▼" if self.expanded else "▶"

    def _header_text(self):
        """Build the collapsed/expanded header as a Rich Text.

        Rich Text avoids markup-parsing of tool args that may contain
        `[...]` (dicts, indexing, type hints, etc.). The "Ctrl+O to view"
        hint makes the expand affordance discoverable — matches the old
        prompt_toolkit CLI's `▶ execute_code (N lines) — Ctrl+O to view`.
        A status mark (✓ / ✗) shows success/failure at a glance even when
        collapsed.
        """
        text = Text()
        text.append(f"{self._glyph()} ")
        text.append(self.event.tool_name, style="bold")
        if self.event.args:
            text.append(f"  ({self.event.args})", style="dim")
        status = self.event.status
        if status == "ok":
            text.append("  ✓", style="#00cc66")
        elif status == "error":
            text.append("  ✗", style="#ff5f5f")
        hint = "  — Ctrl+O to collapse" if self.expanded else "  — Ctrl+O to view"
        text.append(hint, style="dim")
        return text

    def _details_text(self):
        details = self.event.details or ""
        output = self.event.output or ""
        from rich.console import Group

        blocks = []
        if details:
            if self.event.tool_name == "execute_code":
                try:
                    blocks.append(
                        Syntax(
                            details,
                            "python",
                            theme="monokai",
                            line_numbers=False,
                            background_color="default",
                            word_wrap=True,
                        )
                    )
                except Exception:
                    blocks.append(Text(details))
            else:
                blocks.append(Text(details))

        if output:
            label = Text("\noutput\n", style="bold #888888")
            blocks.append(label)
            # Output stays plain so the model's print() lines render exactly
            # as the user expects, without Pygments mis-coloring strings.
            blocks.append(Text(output, style="#b4b4c3"))

        if not blocks:
            return Text("(no details)")
        if len(blocks) == 1:
            return blocks[0]
        return Group(*blocks)

    def sync_from_event(self) -> None:
        """Called when the underlying event was mutated."""
        if self.event.status == "error":
            self.add_class("-error")
            # Auto-expand errors so the user sees what went wrong without
            # having to hunt with Ctrl+O.
            if not self.expanded:
                self.expanded = True
        else:
            self.remove_class("-error")
        header = self.query_one("#header", Static)
        header.update(self._header_text())
        details = self.query_one("#details", Static)
        details.update(self._details_text())

    def toggle(self) -> None:
        self.expanded = not self.expanded
