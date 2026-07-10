"""Streaming-scrollback transcript renderer for the REPL front-end.

Subscribe ``ReplTranscript.on_event`` to the agent's EventEmitter; every event
is printed to stdout through a real ``rich.console.Console`` so the transcript
lands in native terminal scrollback (native scroll + drag-copy) and on-disk
paths / image URLs become genuine OSC 8 hyperlinks.

The single event renderer for both front-ends — the interactive REPL and the
one-shot ``--task`` CLI (``cli._run_single_task``). Handles Banner / UserMessage
/ AssistantStream / ToolCall / SystemNote / Divider, with inline
non-collapsible tool rendering (compact/verbose toggle) and OSC 8 link emission.
"""

from __future__ import annotations

import ast
import json
import time

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from ...events import (
    AssistantStream,
    Banner,
    Divider,
    Event,
    SystemNote,
    ToolCall,
    TurnEnd,
    UserMessage,
)
from ..rich_helpers import render_welcome_banner, text_with_path_links


_ASSISTANT_STYLE = "color(110)"  # LLM gray
_ACCENT = "#d787af"  # pink bullet
_DIM = "dim"
_WARN = "#ffaf00"
_ERROR = "#ff5f5f"
_RULE = "#6c6c80"  # muted rule — matches style.py class:rule

# Lines of tool output shown inline in compact mode before clamping.
_OUTPUT_CLAMP = 10

# Tool name → (gerund while running, past tense when done) for the action
# header. Unmapped tools fall back to the raw name. Extend as needed.
_TOOL_VERBS = {
    "execute_code": ("Running code", "Ran code"),
    "load_skill": ("Loading skill", "Loaded skill"),
}


def _verb(tool_name: str, *, running: bool) -> str:
    if tool_name in _TOOL_VERBS:
        gerund, past = _TOOL_VERBS[tool_name]
        return gerund if running else past
    # Fallback: a capitalized tool name (bash → Bash), Claude-Code style.
    return tool_name[:1].upper() + tool_name[1:] if tool_name else tool_name


def _head_detail(ev) -> str:
    """A short one-line header descriptor — never the full args/code body
    (that lives in the ⎿ output / verbose view). For execute_code the args are
    the whole code blob, so we show nothing; other tools get a bounded snippet.
    """
    if ev.tool_name == "execute_code" or not ev.args:
        return ""
    val = ev.args.strip()
    if val[:1] in "{[":  # dict/JSON blob → show the value(s), not the wrapper
        parsed = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(val)
                break
            except Exception:
                continue
        if isinstance(parsed, dict):
            val = " ".join(str(v) for v in parsed.values())
        elif isinstance(parsed, (list, tuple)):
            val = " ".join(str(v) for v in parsed)
    s = " ".join(val.split())  # collapse newlines/whitespace to one line
    return s if len(s) <= 48 else s[:47].rstrip() + "…"


class ReplTranscript:
    """EventEmitter subscriber that renders to stdout for the streaming REPL.

    ``tool_detail`` controls tool-call verbosity (toggled at runtime via
    ``/verbose``): ``"compact"`` prints a summary line + output clamped to
    ``_OUTPUT_CLAMP`` lines; ``"full"`` prints the code body (syntax-highlighted
    for ``execute_code``) and the complete output.
    """

    def __init__(self, *, tool_detail: str = "compact") -> None:
        self.tool_detail = tool_detail
        # No explicit file= so rich resolves sys.stdout lazily on each write —
        # picking up prompt_toolkit's patch_stdout proxy, so streamed prints
        # land above the live prompt instead of corrupting it.
        self.console = Console(force_terminal=True, highlight=False)
        self._stream_seen: dict[str, int] = {}
        self._stream_line_buf: dict[str, str] = {}
        self._tool_status_seen: dict[str, str] = {}
        self._oneshot_seen: set[str] = set()
        # id → gerund label for tools currently running (drives the live
        # status toolbar; entries are committed to scrollback on completion).
        self._running: dict[str, str] = {}
        self._running_since: dict[str, float] = {}  # id → monotonic start (live elapsed)
        # True only while replaying a resumed session's history. Live user
        # input is already echoed by the prompt, so we skip re-rendering the
        # ● user line + turn divider for it (avoids a double echo).
        self.replaying = False

    def on_event(self, ev: Event) -> None:
        if isinstance(ev, AssistantStream):
            self._render_stream(ev)
            return
        if isinstance(ev, ToolCall):
            self._render_tool_call(ev)
            return
        # One-shot events: render exactly once even if a history replay and a
        # live emit overlap (e.g. an initial message racing startup).
        if ev.id in self._oneshot_seen:
            return
        self._oneshot_seen.add(ev.id)
        if isinstance(ev, UserMessage):
            # Live input is already echoed by the prompt line; only render the
            # ● user message when replaying a resumed session.
            if not self.replaying:
                return
            t = Text()
            t.append("● ", style=_ACCENT)
            t.append(ev.text)
            self.console.print(t)
        elif isinstance(ev, SystemNote):
            style = {"warn": _WARN, "error": _ERROR}.get(ev.level, _DIM)
            self.console.print(Text(ev.text, style=style))
        elif isinstance(ev, TurnEnd):
            # Fires on every turn exit path (incl. cancel) — drop any tool
            # left dangling so the status toolbar can't show a stale gerund.
            self._running.clear()
            self._running_since.clear()
        elif isinstance(ev, Divider):
            if self.replaying:  # live turns are separated by the prompt, not a rule
                self.console.print(Text("─" * 40, style=_DIM))
        elif isinstance(ev, Banner):
            self.console.print(render_welcome_banner(), justify="center")

    def _render_stream(self, ev: AssistantStream) -> None:
        seen = self._stream_seen.get(ev.id, 0)
        new = ev.text[seen:]
        self._stream_seen[ev.id] = len(ev.text)
        buf = self._stream_line_buf.get(ev.id, "") + new
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            self._print_line(line)
        self._stream_line_buf[ev.id] = buf
        if ev.done:
            tail = self._stream_line_buf.pop(ev.id, "")
            if tail.strip():
                self._print_line(tail)
            if ev.text.strip():
                self.console.print()
            self._stream_seen.pop(ev.id, None)

    def _print_line(self, line: str) -> None:
        # Real OSC 8 links via the Console; LLM-gray base, links ride within.
        self.console.print(text_with_path_links(line), style=_ASSISTANT_STYLE)

    def _render_tool_call(self, ev: ToolCall) -> None:
        prior = self._tool_status_seen.get(ev.id)
        if prior == ev.status:
            return  # already rendered this state
        self._tool_status_seen[ev.id] = ev.status
        if ev.status == "running":
            # Defer the scrollback line until done; the live gerund surfaces in
            # the status toolbar meanwhile (via active_tool_label()).
            self._running[ev.id] = _verb(ev.tool_name, running=True)
            self._running_since[ev.id] = time.monotonic()
            return
        self._running.pop(ev.id, None)
        self._running_since.pop(ev.id, None)
        if ev.status == "error":
            self.console.print(Text(f"⏺ {_verb(ev.tool_name, running=False)} — error", style=_ERROR))
            if ev.output:
                self._print_output(ev.output, style=_ERROR)
            return
        # ok — commit the past-tense action header + its result. The header
        # shows only a short descriptor, never the full args/code body.
        head = _verb(ev.tool_name, running=False)
        detail = _head_detail(ev)
        if detail:
            head += f"  {detail}"
        self.console.print(Text(f"⏺ {head}", style=_ACCENT))
        if self.tool_detail == "full" and ev.details:
            self._print_details(ev)
        if ev.output:
            self._print_output(ev.output)

    def _print_details(self, ev: ToolCall) -> None:
        if ev.tool_name == "execute_code":
            self.console.print(
                Syntax(ev.details, "python", theme="monokai", background_color="default", word_wrap=True)
            )
        else:
            self.console.print(text_with_path_links(ev.details), style=_DIM)

    def _print_output(self, output: str, style: str = _DIM) -> None:
        # Tree-branch render: first line under a ⎿, the rest aligned beneath.
        lines = output.split("\n")
        clamp = 0
        if self.tool_detail != "full" and len(lines) > _OUTPUT_CLAMP:
            clamp = len(lines) - _OUTPUT_CLAMP
            lines = lines[:_OUTPUT_CLAMP]
        for i, line in enumerate(lines):
            prefix = "  ⎿ " if i == 0 else "    "
            self.console.print(text_with_path_links(prefix + line), style=style)
        if clamp:
            self.console.print(Text(f"    (+{clamp} more lines)", style=_DIM))

    def echo_user_input(self, text: str) -> None:
        """Persist a submitted input line to scrollback (rule + ``● text``).

        The bare-Application input region is ephemeral, so — unlike a
        PromptSession — nothing is auto-committed; the app calls this on accept
        to keep the turn in history, matching the live prompt chrome.
        """
        self.console.print(Text("─" * 40, style=_RULE))
        t = Text()
        t.append("● ", style=_ACCENT)
        t.append(text)
        self.console.print(t)

    def print_code(self, code: str, lang: str = "python") -> None:
        """Render a standalone syntax-highlighted code block — used by the
        approval prompt so the user can see what they're authorizing."""
        self.console.print(
            Syntax(code, lang, theme="monokai", background_color="default", word_wrap=True)
        )

    def active_tool_label(self) -> str:
        """Gerund of the most-recently-started running tool, for the live
        status toolbar; '' when nothing is running."""
        if not self._running:
            return ""
        return next(reversed(self._running.values()))

    def active_tool_elapsed(self) -> str:
        """Elapsed time of the most-recently-started running tool ('' if none).
        Ticks because the toolbar re-renders on refresh_interval."""
        if not self._running_since:
            return ""
        secs = int(time.monotonic() - next(reversed(self._running_since.values())))
        m, s = divmod(secs, 60)
        return f"{m}m {s}s" if m else f"{s}s"
