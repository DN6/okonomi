"""Streaming-scrollback transcript renderer for the REPL front-end.

Subscribe ``ReplTranscript.on_event`` to the agent's EventEmitter; every event
is printed to stdout through a real ``rich.console.Console`` so the transcript
lands in native terminal scrollback (native scroll + drag-copy) and on-disk
paths / image URLs become genuine OSC 8 hyperlinks.

The single event renderer for both front-ends — the interactive REPL and the
one-shot ``--task`` CLI (``cli._run_single_task``). Handles Banner / UserMessage
/ AssistantStream / ToolCallDisplay / SystemNote / Divider, with inline
non-collapsible tool rendering (compact/verbose toggle) and OSC 8 link emission.

``replay_history`` re-emits a resumed session's pydantic-ai message history as
chat events; the ``text_with_path_links`` / ``render_welcome_banner`` helpers
build the Rich renderables the transcript prints.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from ..events import (
    AssistantStream,
    Event,
    EventEmitter,
    SystemNote,
    ToolCallDisplay,
    TurnEnd,
)
from ..utils import format_elapsed
from .events import Banner, Divider, UserMessage
from .palette import ACCENT as _ACCENT
from .palette import AMBER as _WARN
from .palette import ERROR as _ERROR
from .palette import MUTED as _RULE


_ASSISTANT_STYLE = "color(110)"  # LLM gray
_DIM = "dim"  # rich style keyword

# Lines of tool output shown inline in compact mode before clamping,
# and the per-line char cap (a single long line would otherwise wrap
# into many terminal rows and defeat the clamp).
_OUTPUT_CLAMP = 4
_OUTPUT_LINE_CAP = 160

# Tool name → (gerund while running, past tense when done) for the action
# header. Unmapped tools fall back to the raw name. Extend as needed.
_TOOL_VERBS = {
    "bash": ("Running shell command", "Ran shell command"),
    "web_search": ("Searching the web", "Searched the web"),
    "web_fetch": ("Fetching page", "Fetched page"),
    "read_file": ("Reading file", "Read file"),
    "write_file": ("Writing file", "Wrote file"),
    "edit_file": ("Editing file", "Edited file"),
    "search": ("Searching files", "Searched files"),
    "recall": ("Recalling past work", "Recalled past work"),
    "remember": ("Writing to the mind wiki", "Wrote to the mind wiki"),
    "launch_agent": ("Launching agent", "Launched agent"),
    "load_skill": ("Loading skill", "Loaded skill"),
}


def _verb(tool_name: str, *, running: bool) -> str:
    if tool_name in _TOOL_VERBS:
        gerund, past = _TOOL_VERBS[tool_name]
        return gerund if running else past
    # Fallback: the raw tool name, lowercase (bash, launch_agent).
    return tool_name.lower()


def tool_log_name(call_id: str) -> str:
    """File stem for a tool call's log. Some providers hand out ids like
    ``bash:11``; a colon breaks the ``file://`` link and macOS file names."""
    return re.sub(r"[^\w.-]+", "_", call_id)


def _head_detail(ev) -> str:
    """A short one-line header descriptor — never the full args/code body
    (that lives in the ⎿ output / verbose view). Tools get a bounded snippet.
    """
    if not ev.args:
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
    ``/verbose``): ``"compact"`` prints one summary line per call, linked
    to a log file holding the full call and output (``log_dir_fn`` says
    where; without it the output is printed inline, clamped to
    ``_OUTPUT_CLAMP`` lines); ``"full"`` prints full details and the
    complete output.
    """

    def __init__(self, *, tool_detail: str = "compact", log_dir_fn: Callable[[], Path] | None = None) -> None:
        self.tool_detail = tool_detail
        # Scrollback can't expand in place, so a compact line links (OSC 8)
        # to a per-call log file.
        self.log_dir_fn = log_dir_fn
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
        if isinstance(ev, ToolCallDisplay):
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
            self.console.clear()
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

    def _render_tool_call(self, ev: ToolCallDisplay) -> None:
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
        log = self._write_log(ev) if self.tool_detail != "full" else None
        if log is not None:
            self.console.print(Text(f"⏺ {head}", style=f"{_ACCENT} link {log.as_uri()}"))
            return
        self.console.print(Text(f"⏺ {head}", style=_ACCENT))
        if self.tool_detail == "full" and ev.details:
            self.console.print(text_with_path_links(ev.details), style=_DIM)
        if ev.output:
            self._print_output(ev.output)

    def _write_log(self, ev: ToolCallDisplay) -> Path | None:
        """The full call and its output as a markdown file the compact line
        links to. ``None`` when there is nowhere to write or nothing to show."""
        if self.log_dir_fn is None or not (ev.details or ev.output):
            return None
        try:
            log_dir = self.log_dir_fn()
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{tool_log_name(ev.tool_call_id or ev.id)}.md"
            body = f"# {_verb(ev.tool_name, running=False)}\n\n"
            if ev.details:
                body += f"## Call\n\n```\n{ev.details}\n```\n\n"
            if ev.output:
                body += f"## Output\n\n```\n{ev.output}\n```\n"
            path.write_text(body, encoding="utf-8")
            return path
        except OSError:
            return None

    def _print_output(self, output: str, style: str = _DIM) -> None:
        # Tree-branch render: first line under a ⎿, the rest aligned beneath.
        lines = output.split("\n")
        clamp = 0
        if self.tool_detail != "full":
            if len(lines) > _OUTPUT_CLAMP:
                clamp = len(lines) - _OUTPUT_CLAMP
                lines = lines[:_OUTPUT_CLAMP]
            lines = [ln if len(ln) <= _OUTPUT_LINE_CAP else ln[: _OUTPUT_LINE_CAP - 1] + "…" for ln in lines]
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
        return format_elapsed(time.monotonic() - next(reversed(self._running_since.values())))


# =============================================================================
# Rich renderables — path/URL links and the welcome banner
# =============================================================================

# One pass matches either an http(s) URL or an on-disk path. URL first so the
# `://` is consumed there; the path branch's negative lookbehind then can't
# re-match the slashes inside a URL.
_LINK_RE = re.compile(
    r"(?P<url>https?://[^\s`\"'<>(){}\[\]]+)"
    r"|(?P<path>(?<![\w:/])(?:~/|/)[^\s`\"'<>(){}\[\]]+)"
)
# Trailing punctuation that's almost never part of a real path/URL — strip
# before resolving so "see /foo/bar.png." or "...png)." resolves.
_PATH_TRAILING_PUNCT = ".,;:!?)"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")


@lru_cache(maxsize=512)
def _path_exists(expanded: str) -> bool:
    return os.path.exists(expanded)


def _is_image_url(url: str) -> bool:
    path = url.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(_IMAGE_EXTS)


def text_with_path_links(s: str) -> Text:
    """Return a Rich ``Text`` with on-disk paths and http(s) image URLs
    wrapped as ``link`` styles. Only paths that exist on disk, and URLs that
    look like images, are linked — to avoid false positives. Printed through a
    real ``rich.console.Console``, these become genuine OSC 8 hyperlinks.
    """
    out = Text()
    last = 0
    for m in _LINK_RE.finditer(s):
        raw = m.group(0)
        trim = 0
        while trim < len(raw) and raw[-1 - trim] in _PATH_TRAILING_PUNCT:
            trim += 1
        candidate = raw[: len(raw) - trim] if trim else raw

        if m.lastgroup == "url":
            if not _is_image_url(candidate):
                continue
            target = candidate
        else:
            expanded = os.path.expanduser(candidate)
            if not _path_exists(expanded):
                continue
            target = f"file://{expanded}"

        out.append(s[last : m.start()])
        out.append(candidate, style=f"link {target}")
        if trim:
            out.append(raw[len(raw) - trim :])
        last = m.end()
    out.append(s[last:])
    return out


# Radial banner gradient: full-block glyphs fade from a bright pink centre out
# to a dim plum edge. The label sits punched into the middle row.
_BANNER_CENTER = "#ffa8cf"
_BANNER_EDGE = "#3a1420"
_BANNER_LABEL = "bold #fff0f6"
_BANNER_HEIGHT = 5
_BANNER_MARGIN = 10  # block cells on each side of the label
_BANNER_GLYPH = "█"


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linear-interpolate between two ``#rrggbb`` colours (t in [0, 1])."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[j] + (b[j] - a[j]) * t):02x}" for j in range(3))


def render_welcome_banner(label: str = "okonomi") -> Text:
    """Build the welcome banner as a Rich ``Text``: a block of full-block
    glyphs whose colour fades outward from a bright centre to a dim edge, with
    the label punched into the middle row. Lines carry no leading whitespace —
    the caller centers them.
    """
    width = len(label) + _BANNER_MARGIN * 2
    cx, cy = (width - 1) / 2, (_BANNER_HEIGHT - 1) / 2
    norm = math.hypot(cx, cy) or 1.0
    label_start = (width - len(label)) // 2
    mid = _BANNER_HEIGHT // 2

    text = Text()
    for y in range(_BANNER_HEIGHT):
        for x in range(width):
            if y == mid and label_start <= x < label_start + len(label):
                text.append(label[x - label_start], style=_BANNER_LABEL)
                continue
            # ** 0.85 eases the falloff so the fade stays gentle near the centre
            t = (math.hypot(x - cx, y - cy) / norm) ** 0.85
            text.append(_BANNER_GLYPH, style=_lerp_hex(_BANNER_CENTER, _BANNER_EDGE, t))
        text.append("\n")
    return text


# =============================================================================
# Session-resume replay
# =============================================================================

_SYNTHETIC_REPLAY_PREFIXES = ("[Resumed", "[system]", "[The user interrupted", "[Branch summary")


def replay_history(history: list, emitter: EventEmitter) -> None:
    """Re-emit a pydantic-ai message history as visible chat events on
    ``emitter``. Used after a session is resumed so the user sees their
    prior conversation rendered into the chat pane.

    Synthetic injections (resume markers, ``[system]`` notes, interrupt
    framings) are skipped — they're scaffolding, not user turns.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    pending_tool_calls: dict[str, ToolCallDisplay] = {}
    any_emitted = False
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = part.content if isinstance(part.content, str) else str(part.content)
                    if text.strip().startswith(_SYNTHETIC_REPLAY_PREFIXES):
                        continue
                    if any_emitted:
                        emitter.emit(Divider())
                    emitter.emit(UserMessage(text))
                    any_emitted = True
                elif isinstance(part, ToolReturnPart):
                    tc = pending_tool_calls.pop(part.tool_call_id, None)
                    if tc is None:
                        continue
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    emitter.emit(tc.finish(output=content[:2000]))
        elif isinstance(msg, ModelResponse):
            stream = None
            for part in msg.parts:
                if isinstance(part, TextPart):
                    chunk = part.content
                    if not chunk or not chunk.strip():
                        continue
                    if stream is None:
                        stream = AssistantStream()
                    stream.text += chunk
                    emitter.emit(stream)
                    any_emitted = True
                elif isinstance(part, ToolCallPart):
                    args_repr = repr(part.args) if not isinstance(part.args, str) else part.args
                    tc = emitter.emit(
                        ToolCallDisplay(
                            tool_name=part.tool_name,
                            args=args_repr[:80],
                            details=args_repr[:5000],
                            tool_call_id=part.tool_call_id,
                        )
                    )
                    pending_tool_calls[part.tool_call_id] = tc
                    any_emitted = True
            if stream is not None:
                stream.done = True
                emitter.emit(stream)
    # Sessions saved before cancelled turns were stripped can end on a call
    # with no return; close it here or it shows as running forever.
    for tc in pending_tool_calls.values():
        emitter.emit(tc.finish(error="interrupted — no result"))
