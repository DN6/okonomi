"""Command-line entry point for okonomi.

The interactive UI lives in ``okonomi.tui`` (a prompt_toolkit streaming-
scrollback REPL). This module is thin: argparse + one-shot task runner +
session resolution.
"""

import argparse
import asyncio
import sys

from .agent import OkonomiAgent
from .agent.jobs import REPORTED_KINDS, job_notice
from .config import load_env_or_raise
from .events import JobFinished, SystemNote
from .tools._command_guard import load_approval_rules
from .tools._sandbox import wrap_command


# ANSI colors for the startup chrome (session resolution, one-shot notes) —
# printed before any TUI exists, so raw escapes rather than the REPL style.
DIM = "\033[2m"
RST = "\033[0m"
RED = "\033[31m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


# =============================================================================
# Session Management
# =============================================================================


def resolve_session(args: argparse.Namespace) -> str | None:
    if not (args.resume or args.session):
        return None

    from .session import list_sessions
    from .session import resolve_session as find_session

    if args.resume:
        recent = list_sessions(limit=1)
        if recent:
            print(f"{BOLD}{GREEN}Resuming: {recent[0].display_name}{RST}\n")
            return recent[0].session_id
        print(f"{YELLOW}No previous sessions found. Starting new session.{RST}\n")
        return None

    session = find_session(args.session)
    if session:
        print(f"{BOLD}{GREEN}Loaded: {session.display_name}{RST}\n")
        return session.session_id

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
  okonomi --task "Design a futuristic logo"  # One-off task
  okonomi --resume                           # Resume from latest session
  okonomi --session <id|title|words>         # Load specific session
        """,
    )

    parser.add_argument("--task", "-t", help="Task description for the agent (one-off task)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest session")
    parser.add_argument("--session", "-s", type=str, help="Load a session by ID, ID prefix, title, or content words")
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# One-shot task runner (non-interactive: `okonomi "do X"`)
# =============================================================================


async def _run_single_task(agent: OkonomiAgent, user_message: str) -> None:
    """Run a single task without interactive input."""
    import signal

    from .tui.transcript import ReplTranscript

    transcript = ReplTranscript(tool_detail="compact")
    agent.emitter.subscribe(transcript.on_event)
    loop = asyncio.get_running_loop()

    # A one-shot run has no input loop to deliver completion notices later,
    # so it takes one more turn per launched agent; exiting after the first
    # turn would drop their results.
    async def _run_until_agents_report() -> None:
        notices: asyncio.Queue[JobFinished] = asyncio.Queue()

        def _collect(ev) -> None:
            if isinstance(ev, JobFinished) and ev.kind in REPORTED_KINDS:
                loop.call_soon_threadsafe(notices.put_nowait, ev)

        agent.emitter.subscribe(_collect)
        await agent.run_async(user_message, reset=True)
        # Counted by job id rather than by "still running": a job is marked
        # done just before its notice is emitted.
        reported: set[str] = set()
        while True:
            launched = {job_id for job_id, entry in agent.jobs.entries.items() if entry["kind"] in REPORTED_KINDS}
            if launched <= reported:
                return
            ev = await notices.get()
            reported.add(ev.job_id)
            await agent.run_async(job_notice(ev))

    agent_task = asyncio.create_task(_run_until_agents_report())
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
        agent.emitter.emit(SystemNote("Interrupted.", level="warn"))

    try:
        agent.save_session(status="completed")
    except Exception:
        pass


# =============================================================================
# Main entry
# =============================================================================


async def run_agent_async(args: argparse.Namespace, user_request: str) -> None:
    session_id = resolve_session(args)
    interactive_mode = not args.task

    agent = OkonomiAgent(session_id=session_id)

    if session_id:
        agent.load_session(session_id)

    if interactive_mode:
        from .tui.session import TuiSession

        session = TuiSession(agent)
        await session.run(user_request)
    else:
        print()
        await _run_single_task(agent, user_request)


def main() -> None:
    try:
        load_env_or_raise()
        load_approval_rules()
        wrap_command(["true"])
    except (EnvironmentError, ValueError) as e:
        print(f"{RED}{BOLD}Error:{RST} {e}", file=sys.stderr)
        sys.exit(1)
    args = parse_arguments()

    if not args.task:
        asyncio.run(run_agent_async(args, ""))
    else:
        user_request = args.task.strip()
        if not user_request:
            print(f"{DIM}No task provided.{RST}")
            return

        asyncio.run(run_agent_async(args, user_request))


if __name__ == "__main__":
    main()
