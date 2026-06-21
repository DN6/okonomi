"""Command-line entry point for okonomi.

The interactive UI lives in ``okonomi.tui`` (Textual). This module is thin:
argparse + one-shot task runner + session resolution.
"""

import argparse
import asyncio

from .agent import OkonomiAgent
from .config import load_env_or_raise
from .display import BOLD, DIM, GREEN, PINK, RED, RST, YELLOW, get_task_input


# =============================================================================
# Session Management
# =============================================================================


def resolve_session(args: argparse.Namespace) -> str | None:
    if not (args.resume or args.session):
        return None

    from .session import SessionManager

    manager = SessionManager()

    if args.resume:
        recent = manager.list_sessions(limit=1)
        if recent:
            print(f"{BOLD}{GREEN}Resuming: {recent[0].display_name}{RST}\n")
            return recent[0].session_id
        print(f"{YELLOW}No previous sessions found. Starting new session.{RST}\n")
        return None

    session = manager.load_session(args.session)
    if session:
        print(f"{BOLD}{GREEN}Loaded: {session.display_name}{RST}\n")
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
  okonomi --task "Design a futuristic logo"  # One-off task
  okonomi --resume                           # Resume from latest session
  okonomi --session <id>                     # Load specific session
        """,
    )

    parser.add_argument("--task", "-t", help="Task description for the agent (one-off task)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest session")
    parser.add_argument("--session", "-s", type=str, help="Load specific session by ID")
    parser.add_argument("--discord", action="store_true", help="Run as a Discord bot")
    parser.add_argument("--verbose", "-V", action="store_true", help="Show executed code and output")
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# One-shot task runner (non-interactive: `okonomi "do X"`)
# =============================================================================


async def _run_single_task(agent: OkonomiAgent, user_message: str) -> None:
    """Run a single task without interactive input."""
    import signal

    from .events import StdoutPrinter

    agent.emitter.subscribe(StdoutPrinter().handle)
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
# Main entry
# =============================================================================


async def run_agent_async(args: argparse.Namespace, user_request: str) -> None:
    session_id = resolve_session(args)
    interactive_mode = not args.task

    agent = OkonomiAgent(
        session_id=session_id,
        verbose=args.verbose,
    )

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
    load_env_or_raise()
    args = parse_arguments()

    if args.discord:
        try:
            from .integrations.discord_bot import run_discord_bot
        except ImportError:
            print(f"{RED}discord.py not installed. Run: pip install okonomi[discord]{RST}")
            return
        run_discord_bot()
        return

    is_interactive = not args.task
    if is_interactive:
        asyncio.run(run_agent_async(args, ""))
    else:
        user_request = get_task_input(args)
        if not user_request.strip():
            print(f"{DIM}No task provided.{RST}")
            return

        asyncio.run(run_agent_async(args, user_request))


if __name__ == "__main__":
    main()
