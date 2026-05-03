"""Command-line entry point for okonomi.

The interactive UI lives in ``okonomi.tui`` (Textual). This module is thin:
argparse + one-shot task runner + session resolution.
"""

import argparse
import asyncio
import logging
import os

from .agent import OkonomiAgent
from .config import config
from .display import BOLD, CYAN, DIM, GREEN, PINK, RED, RST, YELLOW, get_task_input


# =============================================================================
# Observability
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
# One-shot task runner (non-interactive: `okonomi "do X"`)
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
# Main entry
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
        from .tui.session import TuiSession

        session = TuiSession(agent)
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
