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
import time
from queue import Queue

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea

from .agent import OkonomiAgent
from .config import config
from .display import (
    BOLD, CYAN, DIM, GREEN, PINK, RED, RST, SPINNER_FRAMES, YELLOW,
    get_task_input, print_interrupt, print_loading_art, print_user_message,
)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _print_welcome() -> None:
    w = shutil.get_terminal_size().columns

    print_loading_art()
    print()
    label = "What's on your mind?"
    pad = (w - len(label)) // 2
    print(f"\033[38;2;215;135;175;1m{' ' * pad}{label}{RST}\n")


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
            print(f"{BOLD}{GREEN}Resuming from session: {latest.metadata.session_id}{RST}")
            print(f"Previous task: {latest.metadata.task_input}\n")
            return latest.metadata.session_id
        print(f"{YELLOW}No previous sessions found. Starting new session.{RST}\n")
        return None

    session = manager.load_session(args.session)
    if session:
        print(f"{BOLD}{GREEN}Loaded session: {args.session}{RST}")
        print(f"Task: {session.metadata.task_input}\n")
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
    parser.add_argument("--verbose", "-V", action="store_true", help="Show executed code and output")
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# Command Handling
# =============================================================================


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/clear", "Clear conversation history"),
    ("/reset", "Clear conversation history"),
    ("/resume", "Resume a past session (or /resume <query>)"),
    ("/save", "Save current session"),
    ("/exit", "End session"),
    ("/quit", "End session"),
]


class SlashCommandCompleter(Completer):
    """Completer that shows a dropdown for slash commands."""

    def __init__(self, skill_provider=None):
        self._skill_provider = skill_provider

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()

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


COMPLETION_STYLE = PTStyle.from_dict(
    {
        "completion-menu": "bg:#1a1a2e #e8e8e8",
        "completion-menu.completion": "bg:#1a1a2e #e8e8e8",
        "completion-menu.completion.current": "bg:#333355 #d787af",
        "completion-menu.meta.completion": "bg:#1a1a2e #888888",
        "completion-menu.meta.completion.current": "bg:#333355 #d787af",
    }
)


def _handle_resume(command: str, agent: OkonomiAgent) -> bool:
    """Handle /resume with optional search query."""
    from .session import SessionManager

    manager = SessionManager()
    query = command[7:].strip() if len(command) > 7 else ""

    if query:
        direct = manager.load_session(query)
        if direct:
            if agent.load_session(query):
                print(f"{GREEN}Resumed session: {query[:8]}...{RST}")
                print(f"{DIM}Task: {direct.metadata.task_input}{RST}")
            else:
                print(f"{RED}Failed to load session.{RST}")
            return True

        sessions = manager.search_sessions(query, limit=10)
        if not sessions:
            print(f"{YELLOW}No sessions matching '{query}'.{RST}")
            return True
    else:
        sessions = manager.list_sessions(limit=10)
        if not sessions:
            print(f"{YELLOW}No past sessions found.{RST}")
            return True

    if len(sessions) == 1:
        s = sessions[0]
        if agent.load_session(s.session_id):
            print(f"{GREEN}Resumed session: {s.session_id[:8]}...{RST}")
            print(f"{DIM}Task: {s.task_input}{RST}")
        else:
            print(f"{RED}Failed to load session.{RST}")
        return True

    print(f"\n{BOLD}Sessions:{RST}")
    for i, s in enumerate(sessions, 1):
        task_preview = s.task_input[:60] + ("..." if len(s.task_input) > 60 else "")
        print(f"{DIM}  {i}. [{s.session_id[:8]}] {task_preview}{RST}")
        print(f"{DIM}     {s.created_at}  {s.status}  {s.num_outputs} outputs{RST}")
    print(f"\n{DIM}Type /resume <query> to narrow results, or copy a session ID.{RST}\n")
    return True


def _handle_command(command: str, agent: OkonomiAgent) -> str | bool:
    """Handle slash commands.

    Returns True if command was handled (no agent input needed),
    a string if the command should be sent to the agent as input,
    or False if unrecognized.
    """
    cmd = command.lower().strip()
    if cmd in ["/clear", "/reset"]:
        print(f"{DIM}Conversation cleared.{RST}")
        agent.message_history = []
        return True
    elif cmd == "/resume" or cmd.startswith("/resume "):
        return _handle_resume(command, agent)
    elif cmd == "/save":
        agent.save_session(status="saved")
        print(f"{DIM}Session saved: {agent.session_id}{RST}")
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


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel remaining tasks and close the event loop cleanly."""
    try:
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    finally:
        loop.close()


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

    def _build_app(self) -> Application:
        session = self

        def _get_skills():
            result = self.agent.skills.list_skills()
            return result.get("skills", [])

        completer = SlashCommandCompleter(skill_provider=_get_skills)

        input_area = TextArea(
            multiline=False,
            prompt=[("#d787af bold", "> ")],
            completer=completer,
            complete_while_typing=True,
            style="class:input-area",
            height=Dimension(min=1, max=1),
        )
        self._input_area = input_area

        def _rule_text():
            w = shutil.get_terminal_size().columns
            return [("#d787af", "\u2500" * w)]

        rule = Window(FormattedTextControl(_rule_text), height=1)
        spacer = Window(height=Dimension(weight=1))

        def _spinner_text():
            txt = session.agent.spinner_text
            if not txt:
                return []
            frame = SPINNER_FRAMES[int(time.monotonic() * 10) % len(SPINNER_FRAMES)]
            return [("#00cc66", f"  {frame} {txt}")]

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
            return [("#888888", f" session: {session.agent.session_id[:8]} ")]

        status_bar = Window(FormattedTextControl(_status_text), height=3)

        layout = Layout(
            HSplit(
                [
                    spacer,
                    spinner_widget,
                    rule,
                    input_area,
                    CompletionsMenu(max_height=8),
                    status_bar,
                ]
            ),
            focused_element=input_area,
        )

        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _newline(event):
            input_area.buffer.insert_text("\n")

        @kb.add("escape")
        def _escape(event):
            if session._agent_running.is_set():
                session._cancel_agent()
                print_interrupt()

        @kb.add("enter", filter=has_completions)
        def _accept_completion(event):
            buf = event.current_buffer
            if buf.complete_state:
                buf.apply_completion(buf.complete_state.current_completion)

        @kb.add("enter", filter=~has_completions)
        def _submit(event):
            text = input_area.text.strip()
            if not text:
                if session._agent_running.is_set():
                    session._cancel_agent()
                    print_interrupt()
                return

            input_area.text = ""

            if text.lower() in ["/exit", "/quit", "exit", "quit"]:
                event.app.exit()
                return

            if text.startswith("/"):
                cmd_result = _handle_command(text, session.agent)
                if cmd_result is True:
                    return
                if isinstance(cmd_result, str):
                    text = cmd_result

            if session._agent_running.is_set():
                session._cancel_agent()
                print_interrupt()

            print_user_message(text)
            session._input_queue.put(text)

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
            _shutdown_loop(loop)
            self._agent_running.clear()

    def _cancel_agent(self) -> None:
        with self._agent_lock:
            loop = self._agent_loop
            task = self._agent_task
        if loop and task and not task.done():
            loop.call_soon_threadsafe(task.cancel)

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

        if not initial_message:
            _print_welcome()

        app = self._build_app()
        self._stop_refresh = threading.Event()

        with pt_patch_stdout(raw=True):
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
            print(f"\033[2mSession saved: {self.agent.session_id}\033[0m")
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
