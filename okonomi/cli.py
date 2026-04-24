"""Command-line interface for Okonomi.

Provides an interactive chat loop where the user can interrupt the agent
at any time:
  - Escape key: bare interrupt (stops agent, keeps history)
  - Ctrl+C: interrupt (press twice to exit)
  - Type + Enter while agent is working: interrupt and send new message

The agent runs in a background thread while prompt_toolkit's
``patch_stdout(raw=True)`` keeps the input prompt pinned at the bottom
of the terminal — agent output scrolls above it.
"""

import argparse
import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path
import time
from queue import Queue

from prompt_toolkit.application import Application
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea

from .agent import OkonomiAgent
from .config import config
from .display import (
    BOLD, CYAN, DIM, GREEN, PINK, RED, RST, YELLOW,
    get_task_input, highlight_code, print_interrupt, print_loading_art, print_user_message,
    spinner_formatted_text,
)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


import re

_MENTION_RE = re.compile(r"(?:^|(?<=\s))@(\S+)")


def _expand_mentions(text: str) -> str:
    """Expand @path mentions into inline file contents."""
    mentions = _MENTION_RE.findall(text)
    if not mentions:
        return text

    attachments = []
    for path_str in mentions:
        expanded = os.path.expanduser(path_str)
        p = Path(expanded)
        if not p.exists():
            continue
        if p.is_dir():
            entries = sorted(str(e.relative_to(p)) for e in p.iterdir() if not e.name.startswith("."))
            content = "\n".join(entries)
            attachments.append(f"<file path=\"{path_str}\" type=\"directory\">\n{content}\n</file>")
        else:
            try:
                content = p.read_text(errors="replace")
                if len(content) > 50_000:
                    content = content[:50_000] + "\n... (truncated)"
                attachments.append(f"<file path=\"{path_str}\">\n{content}\n</file>")
            except Exception:
                continue

    if not attachments:
        return text

    return text + "\n\n" + "\n\n".join(attachments)


def _print_welcome() -> None:
    print_loading_art()
    print()


# =============================================================================
# Setup
# =============================================================================


def setup_observability() -> None:
    if os.getenv("LOG_CREATE_AGENT_TRACES", "false").lower() not in ["true", "1"]:
        return


# =============================================================================
# Session Management
# =============================================================================


def handle_list_sessions() -> None:
    from .session import SessionManager

    manager = SessionManager()
    sessions = manager.list_sessions(limit=20)

    if not sessions:
        print(f"{YELLOW}No sessions found.{RST}")
        return

    print(f"\n{BOLD}{CYAN}Past Sessions:{RST}\n")
    for session in sessions:
        print(f"{BOLD}{session.session_id}{RST}")
        print(f"  Task: {session.task_input}")
        print(f"  Time: {session.created_at}")
        print(f"  Status: {session.status}")
        print(f"  Outputs: {session.num_outputs}")
        print()


def handle_search_outputs(query: str) -> None:
    from .session import SessionManager

    manager = SessionManager()
    results = manager.search_outputs(query=query, limit=20)

    if not results:
        print(f"{YELLOW}No outputs found matching '{query}'.{RST}")
        return

    print(f"\n{BOLD}{CYAN}Outputs matching '{query}':{RST}\n")
    for result in results:
        output = result["output"]
        print(f"{BOLD}Session:{RST} {result['session_id'][:8]}...")
        print(f"  Task: {result['task_input']}")
        print(f"  Type: {output['asset_type']}")
        print(f"  URL: {output['asset_url']}")
        print()


def resolve_session(args: argparse.Namespace) -> str | None:
    if not (args.resume or args.session):
        return None

    from .session import SessionManager

    manager = SessionManager()

    if args.resume:
        latest = manager.get_latest_session()
        if latest:
            print(f"{BOLD}{GREEN}Resuming: {latest.metadata.display_name}{RST}\n")
            return latest.metadata.session_id
        print(f"{YELLOW}No previous sessions found. Starting new session.{RST}\n")
        return None

    session = manager.load_session(args.session)
    if session:
        print(f"{BOLD}{GREEN}Loaded: {session.metadata.display_name}{RST}\n")
        return args.session

    print(f"{RED}Session {args.session} not found. Starting new session.{RST}\n")
    return None


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Okonomi - AI-powered image generation agent with creative strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  okonomi                                    # Interactive mode
  okonomi "Create a cyberpunk city scene"    # Direct task
  okonomi --task "Design a futuristic logo"  # Using --task flag
  okonomi --resume                           # Resume from latest session
  okonomi --session <id>                     # Load specific session
  okonomi --list-sessions                    # List all past sessions
  okonomi --search-outputs "portrait"         # Search for outputs
        """,
    )

    parser.add_argument("task", nargs="?", help="Task description for the agent")
    parser.add_argument("--task", "-t", dest="task_flag", help="Task description (alternative)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest session")
    parser.add_argument("--session", "-s", type=str, help="Load specific session by ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all past sessions")
    parser.add_argument("--search-outputs", type=str, metavar="QUERY", help="Search outputs across sessions")
    parser.add_argument("--discord", action="store_true", help="Run as a Discord bot")
    parser.add_argument("--verbose", "-V", action="store_true", help="Show executed code and output")
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# Command Handling
# =============================================================================


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/assets", "List generated assets in this session"),
    ("/bg", "Run a task in the background"),
    ("/clear", "Clear conversation history"),
    ("/reset", "Clear conversation history"),
    ("/session", "Browse and resume a past session"),
    ("/last", "Resume the most recent session"),
    ("/title", "Set/show session title"),
    ("/fav", "Favourite the last generated media asset"),
    ("/favs", "List all favourited assets"),
    ("/prune", "Delete empty sessions older than 24h"),
    ("/save", "Save current session"),
    ("/exit", "End session"),
    ("/quit", "End session"),
]


class SlashCommandCompleter(Completer):
    """Completer that shows a dropdown for slash commands and @ file mentions."""

    def __init__(self, skill_provider=None):
        self._skill_provider = skill_provider

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor

        # @ file mention: find the latest @ token and complete file paths
        at_idx = text_before.rfind("@")
        if at_idx >= 0:
            before_at = text_before[:at_idx]
            after_at = text_before[at_idx + 1:]
            # Only trigger if @ is at start or preceded by whitespace,
            # and there's no whitespace in the partial path
            if (not before_at or before_at[-1].isspace()) and " " not in after_at and "\t" not in after_at:
                yield from self._file_completions(after_at)
                return

        text = text_before.lstrip()
        if not text.startswith("/"):
            return

        # Match against registered commands
        for cmd, description in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=description,
                )

        # Also complete skill names for inline skill loading
        if self._skill_provider:
            try:
                skills = self._skill_provider()
                for skill in skills:
                    skill_cmd = f"/skill {skill['name']}"
                    if skill_cmd.startswith(text) or (text == "/skill" and True):
                        yield Completion(
                            skill_cmd,
                            start_position=-len(text),
                            display=f"/skill {skill['name']}",
                            display_meta=skill["description"][:60],
                        )
            except Exception:
                pass

    def _file_completions(self, partial: str):
        """Yield file path completions relative to cwd for @ mentions."""
        import os

        partial_expanded = os.path.expanduser(partial)
        if os.path.isabs(partial_expanded):
            base_dir, prefix = os.path.split(partial_expanded)
            if not base_dir:
                base_dir = "/"
        elif "/" in partial:
            base_dir, prefix = os.path.split(partial)
        else:
            base_dir = "."
            prefix = partial

        try:
            entries = sorted(os.listdir(base_dir or "."))
        except (OSError, PermissionError):
            return

        for entry in entries:
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if not entry.lower().startswith(prefix.lower()):
                continue
            full = os.path.join(base_dir, entry) if base_dir and base_dir != "." else entry
            is_dir = os.path.isdir(os.path.join(base_dir or ".", entry))
            completion_text = full + ("/" if is_dir else "")
            display = f"{entry}/" if is_dir else entry
            meta = "dir" if is_dir else "file"
            yield Completion(
                completion_text,
                start_position=-len(partial),
                display=display,
                display_meta=meta,
            )


COMPLETION_STYLE = PTStyle.from_dict(
    {
        "completion-menu": "bg:#1a1a2e #e8e8e8",
        "completion-menu.completion": "bg:#1a1a2e #e8e8e8",
        "completion-menu.completion.current": "bg:#333355 #d787af",
        "completion-menu.meta.completion": "bg:#1a1a2e #888888",
        "completion-menu.meta.completion.current": "bg:#333355 #d787af",
        "input-frame frame.border": "#d787af",
        "input-frame frame.label": "#d787af bold",
    }
)



def _handle_fav(agent: OkonomiAgent) -> bool:
    outputs = agent.generated_outputs
    if not outputs:
        print(f"{DIM}No generated media in this session.{RST}")
        return True

    last = outputs[-1]
    local = last.local_path
    if not local or not os.path.exists(local):
        if last.asset_url and os.path.exists(last.asset_url):
            local = last.asset_url
        else:
            print(f"{DIM}Last output has no local file yet. Try /save first.{RST}")
            return True

    session_dir = agent.session_manager.get_session_dir(agent.session_id)
    favs = session_dir / "favourites"
    favs.mkdir(parents=True, exist_ok=True)

    src = Path(local)
    dest = favs / src.name

    if dest.exists() or dest.is_symlink():
        print(f"{DIM}Already favourited: {src.name}{RST}")
        return True

    dest.symlink_to(src.resolve())
    print(f"{PINK}★{RST} {DIM}Favourited: {src.name}{RST}")
    return True


def _handle_favs(agent: OkonomiAgent) -> bool:
    session_dir = agent.session_manager.get_session_dir(agent.session_id)
    favs = session_dir / "favourites"

    if not favs.exists():
        print(f"{DIM}No favourites in this session. Use /fav to favourite the last generated asset.{RST}")
        return True

    links = sorted(favs.iterdir())
    if not links:
        print(f"{DIM}No favourites in this session. Use /fav to favourite the last generated asset.{RST}")
        return True

    print(f"\n{DIM}Favourites ({len(links)}):{RST}")
    for link in links:
        print(f"  {PINK}★{RST} {link.name}")
    print()
    return True


def _handle_command(command: str, agent: OkonomiAgent) -> str | bool:
    """Handle slash commands.

    Returns True if command was handled (no agent input needed),
    a string if the command should be sent to the agent as input,
    or False if unrecognized.
    """
    cmd = command.lower().strip()
    if cmd == "/assets":
        outputs = agent.generated_outputs
        if not outputs:
            print(f"{DIM}No assets generated yet.{RST}")
        else:
            print(f"\n{DIM}Assets ({len(outputs)}):{RST}")
            for i, o in enumerate(outputs, 1):
                icon = "🖼" if o.asset_type == "image" else "🎬" if o.asset_type == "video" else "📄"
                path = o.local_path or o.asset_url or ""
                prompt = o.prompt[:60] + "..." if o.prompt and len(o.prompt) > 60 else o.prompt or ""
                print(f"  {icon} {BOLD}{i}.{RST} {path}")
                if prompt:
                    print(f"     {DIM}{prompt}{RST}")
            print()
        return True
    elif cmd in ["/clear", "/reset"]:
        print(f"{DIM}Conversation cleared.{RST}")
        agent.message_history = []
        return True
    elif cmd == "/last":
        sessions = agent.session_manager.list_sessions(limit=1)
        if sessions:
            if agent.load_session(sessions[0].session_id):
                print(f"{GREEN}Resumed: {sessions[0].display_name}{RST}")
            else:
                print(f"{RED}Failed to load last session.{RST}")
        else:
            print(f"{DIM}No previous sessions.{RST}")
        return True
    elif cmd.startswith("/title"):
        title_text = command[6:].strip()
        if title_text:
            if agent.session_manager.set_title(agent.session_id, title_text):
                print(f"{DIM}Title set: {title_text}{RST}")
            else:
                print(f"{RED}Failed to set title (conflict or session not saved yet).{RST}")
        else:
            index = agent.session_manager._read_index()
            for entry in index:
                if entry["session_id"] == agent.session_id:
                    title = entry.get("title", "")
                    print(f"{DIM}Title: {title or '(none)'}{RST}")
                    break
            else:
                print(f"{DIM}Title: (none — save session first){RST}")
        return True
    elif cmd == "/fav":
        return _handle_fav(agent)
    elif cmd == "/favs":
        return _handle_favs(agent)
    elif cmd == "/prune":
        count = agent.session_manager.prune_empty_sessions()
        print(f"{DIM}Pruned {count} empty session{'s' if count != 1 else ''}.{RST}")
        return True
    elif cmd == "/save":
        agent.save_session(status="saved")
        print(f"{DIM}Session saved.{RST}")
        return True
    elif cmd == "/help":
        print(f"\n{DIM}Commands:{RST}")
        for slash_cmd, desc in SLASH_COMMANDS:
            print(f"{DIM}  {slash_cmd:<14}{desc}{RST}")
        print(f"{DIM}  {'Escape':<14}Interrupt agent{RST}")
        print(f"{DIM}  {'Ctrl+C':<14}Interrupt (2x to exit){RST}")
        print(f"{DIM}  {'Type+Enter':<14}Interrupt & send new message{RST}\n")
        return True
    elif cmd.startswith("/skill "):
        skill_name = command[7:].strip()
        result = agent.skills.load_skill(skill_name)
        if "instructions" in result:
            return f"Follow this skill: {result['instructions']}"
        else:
            print(f"{RED}{result.get('error', 'Skill not found')}{RST}")
            return True
    return False


# =============================================================================
# Interactive Session
# =============================================================================


def _shutdown_loop(loop: asyncio.AbstractEventLoop, timeout: float = 2.0) -> None:
    """Cancel remaining tasks and close the event loop.

    Uses a timeout to avoid blocking indefinitely when threads are stuck
    in blocking calls (e.g. FAL API requests via asyncio.to_thread).
    """
    try:
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            async def _wait():
                await asyncio.wait(pending, timeout=timeout)
            loop.run_until_complete(_wait())
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    finally:
        try:
            loop.close()
        except Exception:
            pass


class InteractiveSession:
    """Manages the interactive chat loop with a persistent floating input.

    Architecture (matching hermes-agent):
      - prompt_toolkit ``Application(full_screen=False)`` with ``HSplit`` layout
        keeps a TextArea pinned at the bottom of the terminal
      - ``patch_stdout()`` routes all ``print()`` / ``sys.stdout.write()`` output
        above the layout so the input area is never overwritten
      - A worker thread pulls messages from ``_input_queue`` and runs the agent
      - Key bindings handle submit, interrupt, and exit
    """

    def __init__(self, agent: OkonomiAgent):
        self.agent = agent
        self._agent_running = threading.Event()

        self._agent_lock = threading.Lock()
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._agent_task: asyncio.Task | None = None

        self._input_queue: Queue[str | None] = Queue()
        self._next_reset = False

        self._session_candidates: list | None = None
        self._prompt_style = ("#d787af bold", "> ")

        self._approval_queue: Queue[str] = Queue()
        self._awaiting_approval = False
        self._show_code_panel = False
        self.agent.approval_callback = self._approval_callback
        self.agent.on_bg_complete = self._on_bg_exec_complete

        self._bg_tasks: dict[str, threading.Thread] = {}
        self._bg_counter = 0

    def _on_bg_exec_complete(
        self,
        *,
        bg_id: str,
        error: str | None,
        output: str | None,
        new_outputs: int,
        new_artifacts: int,
    ) -> None:
        """Called from the agent's bg-exec coroutine when a background code block finishes.

        Injects a synthetic user turn so the agent can react (e.g. surface the
        media to the user). Mirrors hermes-agent's notify_on_complete.
        """
        if error:
            msg = (
                f"[system] Background task {bg_id} failed. Review the error and "
                f"decide whether to retry or inform the user:\n{error}"
            )
        else:
            parts = []
            if new_outputs:
                parts.append(f"{new_outputs} output{'s' if new_outputs != 1 else ''}")
            if new_artifacts:
                parts.append(f"{new_artifacts} artifact{'s' if new_artifacts != 1 else ''}")
            tail = f" ({', '.join(parts)})" if parts else ""
            msg = (
                f"[system] Background task {bg_id} completed{tail}. The generated "
                "media is now in the session. Present it to the user concisely."
            )
        self._input_queue.put(msg)

    async def _approval_callback(self, code: str) -> bool | str:
        """Called by the agent before running expensive code. Blocks until user responds.

        Returns True to run, False to skip, or a feedback string for the agent to revise.
        """
        print(f"\n{YELLOW}Run this code? [Enter] approve / [a] approve all / [n] skip / or type feedback{RST}")
        self._awaiting_approval = True
        self._prompt_style = ("#ffaf00 bold", "approve> ")

        decision = await asyncio.to_thread(self._approval_queue.get)

        self._awaiting_approval = False
        self._prompt_style = ("#d787af bold", "> ")

        d = decision.strip().lower()
        if d in ("y", "yes", ""):
            return True
        if d in ("a", "all", "approve all"):
            return "__auto_approve__"
        if d in ("n", "no", "skip"):
            return False
        return decision.strip()

    def _enter_session_mode(self) -> None:
        from .session import SessionManager

        manager = SessionManager()
        sessions = manager.list_sessions(limit=20)
        if not sessions:
            print(f"{YELLOW}No past sessions found.{RST}")
            return

        self._session_candidates = sessions
        self._prompt_style = ("#00cc66 bold", "session> ")
        self._print_session_list(sessions)
        print(f"{DIM}Type to filter, number to select, Enter to cancel.{RST}\n")

    def _print_session_list(self, sessions: list) -> None:
        print()
        for i, s in enumerate(sessions, 1):
            name = s.display_name
            date = s.created_at[:10] if s.created_at else ""
            outputs = f"{s.num_outputs} outputs" if s.num_outputs else "no outputs"
            print(f"  {BOLD}{i}.{RST} {name}")
            print(f"     {DIM}{date}  {outputs}  [{s.session_id[:12]}]{RST}")
        print()

    def _handle_session_input(self, text: str) -> None:
        candidates = self._session_candidates or []

        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(candidates):
                self._select_session(candidates[idx])
                return
            print(f"{RED}Invalid number. Pick 1-{len(candidates)}.{RST}")
            return

        from .session import SessionManager
        manager = SessionManager()

        direct = manager.load_session(text)
        if direct:
            self._session_candidates = None
            self._prompt_style = ("#d787af bold", "> ")
            if self.agent.load_session(text):
                self._next_reset = False
                print(f"{GREEN}Resumed: {direct.metadata.display_name}{RST}")
                print(f"{DIM}Task: {direct.metadata.task_input}{RST}")
            else:
                print(f"{RED}Failed to load session.{RST}")
            return

        filtered = manager.search_sessions(text, limit=20)
        if not filtered:
            print(f"{YELLOW}No matches for '{text}'. Keep typing or Enter to cancel.{RST}")
            return

        if len(filtered) == 1:
            self._select_session(filtered[0])
            return

        self._session_candidates = filtered
        self._print_session_list(filtered)

    def _select_session(self, session_meta) -> None:
        self._session_candidates = None
        self._prompt_style = ("#d787af bold", "> ")
        if self.agent.load_session(session_meta.session_id):
            self._next_reset = False
            print(f"{GREEN}Resumed: {session_meta.display_name}{RST}")
            print(f"{DIM}{session_meta.created_at[:10]}  [{session_meta.session_id[:12]}]{RST}")
        else:
            print(f"{RED}Failed to load session.{RST}")

    def _build_app(self) -> Application:
        session = self

        def _get_skills():
            result = self.agent.skills.list_skills()
            return result.get("skills", [])

        completer = SlashCommandCompleter(skill_provider=_get_skills)

        history_path = Path.home() / ".okonomi" / "input_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history = FileHistory(str(history_path))

        input_area = TextArea(
            multiline=True,
            prompt=lambda: [session._prompt_style],
            completer=completer,
            complete_while_typing=True,
            style="class:input-area",
            height=Dimension(min=1, max=2),
            history=self._history,
            scrollbar=False,
        )
        self._input_area = input_area

        rule = Window(height=0)

        # Collapsible code preview panel (toggled via Ctrl+O)
        def _code_panel_text():
            code = session.agent._last_code
            if not code:
                return [("#888888", "  (no code executed yet)\n")]
            lines = code.split("\n")
            result = [("#888888", " code preview "), ("#888888", "─" * 30 + "\n")]
            for line in lines:
                result.append(("#e8e8e8", f" {line}\n"))
            result.append(("#888888", "─" * 40 + " Ctrl+O to close\n"))
            return result

        def _code_panel_height():
            if not session._show_code_panel:
                return 0
            code = session.agent._last_code
            if not code:
                return 3
            return min(code.count("\n") + 4, 20)

        code_panel = Window(
            content=FormattedTextControl(_code_panel_text),
            height=_code_panel_height,
        )

        def _spinner_text():
            txt = session.agent.spinner_text
            if not txt:
                return []
            tick = int(time.monotonic() * 5)
            return spinner_formatted_text(txt, tick)

        def _spinner_height():
            return 1 if session.agent.spinner_text else 0

        spinner_widget = Window(
            content=FormattedTextControl(_spinner_text),
            height=_spinner_height,
        )

        def _status_text():
            usage = session.agent.token_usage
            if usage.total_tokens:
                return [("#888888", f" {usage} ")]
            return [("#888888", " ")]

        status_bar = Window(FormattedTextControl(_status_text), height=1)

        def _border(char: str, **kw):
            return Window(char=char, style="class:input-frame,frame.border", **kw)

        def _pad(width: int = 1):
            return Window(char=" ", style="class:input-frame", width=width)

        input_frame = HSplit(
            [
                VSplit(
                    [
                        _border("┌", width=1, height=1),
                        _border("─", height=1),
                        _border("┐", width=1, height=1),
                    ],
                    height=1,
                ),
                VSplit(
                    [
                        _border("│", width=1),
                        _pad(1),
                        input_area,
                        _pad(1),
                        _border("│", width=1),
                    ],
                ),
                VSplit(
                    [
                        _border("└", width=1, height=1),
                        _border("─", height=1),
                        _border("┘", width=1, height=1),
                    ],
                    height=1,
                ),
            ]
        )

        layout = Layout(
            HSplit(
                [
                    spinner_widget,
                    code_panel,
                    rule,
                    input_frame,
                    CompletionsMenu(max_height=8),
                    status_bar,
                ]
            ),
            focused_element=input_area,
        )

        kb = KeyBindings()

        @kb.add("c-o")
        def _toggle_code(event):
            session._show_code_panel = not session._show_code_panel

        @kb.add("escape", "enter")
        def _newline(event):
            input_area.buffer.insert_text("\n")

        @kb.add("up", filter=~has_completions)
        def _history_prev(event):
            buf = input_area.buffer
            if "\n" in buf.text:
                buf.cursor_up()
            else:
                buf.history_backward()

        @kb.add("down", filter=~has_completions)
        def _history_next(event):
            buf = input_area.buffer
            if "\n" in buf.text:
                buf.cursor_down()
            else:
                buf.history_forward()

        @kb.add("escape")
        def _escape(event):
            if session._awaiting_approval:
                session._approval_queue.put("n")
                return
            if session._session_candidates is not None:
                session._session_candidates = None
                session._prompt_style = ("#d787af bold", "> ")
                input_area.text = ""
                print(f"{DIM}Cancelled.{RST}")
                return
            if session._agent_running.is_set():
                session._cancel_agent()
                print_interrupt()

        @kb.add("enter", filter=has_completions)
        def _accept_completion(event):
            buf = event.current_buffer
            if buf.complete_state and buf.complete_state.current_completion:
                buf.apply_completion(buf.complete_state.current_completion)

        @kb.add("enter", filter=~has_completions)
        def _submit(event):
            text = input_area.text.strip()

            if session._awaiting_approval:
                input_area.text = ""
                session._approval_queue.put(text if text else "y")
                return

            if session._session_candidates is not None:
                input_area.text = ""
                if not text:
                    session._session_candidates = None
                    session._prompt_style = ("#d787af bold", "> ")
                    print(f"{DIM}Cancelled.{RST}")
                    return
                session._handle_session_input(text)
                return

            if not text:
                if session._agent_running.is_set():
                    session._cancel_agent()
                    print_interrupt()
                return

            input_area.buffer.reset(append_to_history=True)

            if text.lower() in ["/exit", "/quit", "exit", "quit"]:
                event.app.exit()
                return

            if text.lower().startswith("/bg "):
                bg_input = text[4:].strip()
                if bg_input:
                    print_user_message(text)
                    session._run_bg_task(bg_input)
                else:
                    print(f"{DIM}Usage: /bg <task description>{RST}")
                return

            if text.lower() == "/session" or text.lower().startswith("/session "):
                query = text[8:].strip() if len(text) > 8 else ""
                if query:
                    session._handle_session_input(query)
                else:
                    session._enter_session_mode()
                return

            if text.startswith("/"):
                cmd_result = _handle_command(text, session.agent)
                if cmd_result is True:
                    return
                if isinstance(cmd_result, str):
                    text = cmd_result

            expanded = _expand_mentions(text)

            if session._agent_running.is_set():
                session._cancel_agent()
                print_interrupt()
                print_user_message(text)
                print(f"{DIM}(queued — waiting for current operation to stop){RST}")
                session._input_queue.put(expanded)
                return

            print_user_message(text)
            session._input_queue.put(expanded)

        @kb.add("c-c")
        def _ctrl_c(event):
            if session._agent_running.is_set():
                session._cancel_agent()
                print_interrupt()
            else:
                event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event):
            event.app.exit()

        return Application(
            layout=layout,
            key_bindings=kb,
            style=COMPLETION_STYLE,
            full_screen=False,
            mouse_support=False,
        )

    def _run_agent(self, user_message: str, reset: bool = False) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._agent_lock:
            self._agent_loop = loop
        try:
            task = loop.create_task(self.agent.run_async(user_message, reset=reset))
            with self._agent_lock:
                self._agent_task = task
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")
        finally:
            with self._agent_lock:
                self._agent_task = None
                self._agent_loop = None
            try:
                _shutdown_loop(loop)
            except Exception:
                pass
            self._agent_running.clear()

    def _run_bg_task(self, task_input: str) -> None:
        """Run a task in the background with a separate agent, same session."""
        from .agent import OkonomiAgent
        from .utils import generate_title

        self._bg_counter += 1
        task_id = f"bg-{self._bg_counter}"
        title = generate_title(task_input) or task_id
        parent_agent = self.agent

        print(f"{DIM}Background task started: {title}{RST}")

        approval_callback = parent_agent.approval_callback

        def _run():
            try:
                bg_agent = OkonomiAgent(
                    parent_agent.config,
                    session_id=parent_agent.session_id,
                )
                bg_agent.approval_callback = approval_callback

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(bg_agent.run_async(task_input, reset=True))
                    parent_agent.generated_outputs.extend(bg_agent.generated_outputs)
                    parent_agent.artifacts.extend(bg_agent.artifacts)
                    parent_agent.tools_used.update(bg_agent.tools_used)
                    n = len(bg_agent.generated_outputs)
                    print(f"\n{GREEN}✓ Background done: {title} — {n} output{'s' if n != 1 else ''}{RST}")
                except Exception as e:
                    print(f"\n{RED}✗ Background failed: {title} — {e}{RST}")
                finally:
                    _shutdown_loop(loop)
                    self._bg_tasks.pop(task_id, None)
            except Exception as e:
                print(f"\n{RED}✗ Background error: {title} — {e}{RST}")
                self._bg_tasks.pop(task_id, None)

        t = threading.Thread(target=_run, daemon=True, name=f"okonomi-{task_id}")
        self._bg_tasks[task_id] = t
        t.start()

    def _cancel_agent(self) -> None:
        with self._agent_lock:
            loop = self._agent_loop
            task = self._agent_task
        if loop and task and not task.done():
            loop.call_soon_threadsafe(task.cancel)
            # Don't block — the worker thread will pick up CancelledError
            # and move to the next queued message immediately.

    def _worker(self) -> None:
        """Background thread: pull messages from the queue and run the agent."""
        echo_first = self._echo_first
        while True:
            try:
                msg = self._input_queue.get()
                if msg is None:
                    break

                if echo_first:
                    print_user_message(msg)
                    echo_first = False

                reset = self._next_reset
                self._next_reset = False

                self._agent_running.set()
                self.agent.spinner_text = "Thinking..."
                self._run_agent(msg, reset=reset)
            except Exception as e:
                print(f"{RED}Worker error: {e}{RST}")
                self._agent_running.clear()

    def _refresh_loop(self, app: Application) -> None:
        """Periodically invalidate the app to animate the spinner."""
        while not self._stop_refresh.is_set():
            if self.agent.spinner_text:
                app.invalidate()
            self._stop_refresh.wait(0.08)

    async def run(self, initial_message: str = "") -> None:
        """Run the interactive session with a persistent floating input."""
        print("\033[2J\033[H", end="", flush=True)

        app = self._build_app()
        self._stop_refresh = threading.Event()

        with pt_patch_stdout(raw=True):
            if not initial_message:
                _print_welcome()

            pruned = self.agent.session_manager.prune_empty_sessions()
            if pruned:
                print(f"{DIM}Pruned {pruned} empty session{'s' if pruned != 1 else ''}.{RST}")

            self._next_reset = True
            self._echo_first = bool(initial_message)
            if initial_message:
                self._input_queue.put(initial_message)

            worker = threading.Thread(target=self._worker, daemon=True, name="okonomi-agent")
            worker.start()

            refresher = threading.Thread(target=self._refresh_loop, args=(app,), daemon=True)
            refresher.start()

            try:
                await app.run_async()
            except (EOFError, KeyboardInterrupt):
                pass

            self._stop_refresh.set()

        if self._agent_running.is_set():
            self._cancel_agent()

        self._input_queue.put(None)

        try:
            self.agent.save_session(status="completed")
            print(f"\033[2mSession saved.\033[0m")
        except Exception as e:
            print(f"\033[31mError saving session: {e}\033[0m")



# =============================================================================
# Non-Interactive Runner
# =============================================================================


async def _run_single_task(agent: OkonomiAgent, user_message: str) -> None:
    """Run a single task without interactive input."""
    import signal

    loop = asyncio.get_running_loop()
    agent_task = asyncio.create_task(agent.run_async(user_message, reset=True))
    interrupted = False

    def on_sigint():
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        if not agent_task.done():
            interrupted = True
            agent_task.cancel()

    loop.add_signal_handler(signal.SIGINT, on_sigint)

    try:
        await agent_task
    except asyncio.CancelledError:
        pass
    finally:
        loop.remove_signal_handler(signal.SIGINT)

    if interrupted:
        print(f"\n{PINK}Interrupted.{RST}")

    try:
        agent.save_session(status="completed")
    except Exception:
        pass


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_agent_async(args: argparse.Namespace, user_request: str) -> None:
    session_id = resolve_session(args)
    interactive_mode = not (args.task or args.task_flag)

    agent = OkonomiAgent(
        config=config,
        session_id=session_id,
        verbose=args.verbose,
    )

    if session_id:
        agent.load_session(session_id)

    if interactive_mode:
        session = InteractiveSession(agent)
        await session.run(user_request)
    else:
        print()
        await _run_single_task(agent, user_request)


def main() -> None:
    args = parse_arguments()

    if args.discord:
        try:
            from .integrations.discord_bot import run_discord_bot
        except ImportError:
            print(f"{RED}discord.py not installed. Run: pip install okonomi[discord]{RST}")
            return
        run_discord_bot()
        return

    if args.list_sessions:
        handle_list_sessions()
        return

    if args.search_outputs:
        handle_search_outputs(args.search_outputs)
        return

    interactive_mode = not (args.task or args.task_flag)

    if interactive_mode:
        asyncio.run(run_agent_async(args, ""))
    else:
        user_request = get_task_input(args)
        if not user_request.strip():
            print(f"{DIM}No task provided.{RST}")
            return
        asyncio.run(run_agent_async(args, user_request))


if __name__ == "__main__":
    main()
