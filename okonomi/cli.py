"""Command-line interface for Okonomi."""

import argparse
import logging
import os

from rich.console import Console
from rich.panel import Panel

from .config import config
from .interface import get_multiline_input, get_task_input
from .agent import OkonomiAgent

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# =============================================================================
# Setup Functions
# =============================================================================


def setup_observability() -> None:
    """Initialize observability instrumentation if enabled."""
    if os.getenv("LOG_CREATE_AGENT_TRACES", "false").lower() not in ["true", "1"]:
        return

    # Observability setup can be added here for Pydantic AI if needed
    pass


# =============================================================================
# Session Management
# =============================================================================


def handle_list_sessions(console: Console) -> None:
    """Display list of past sessions."""
    from .session import SessionManager

    manager = SessionManager()
    sessions = manager.list_sessions(limit=20)

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    console.print("\n[bold cyan]Past Sessions:[/bold cyan]\n")
    for session in sessions:
        console.print(f"[bold]{session.session_id}[/bold]")
        console.print(f"  Task: {session.task_input}")
        console.print(f"  Time: {session.created_at}")
        console.print(f"  Status: {session.status}")
        console.print(f"  Outputs: {session.num_outputs}")
        console.print()


def handle_search_outputs(console: Console, query: str) -> None:
    """Search and display outputs matching query."""
    from .session import SessionManager

    manager = SessionManager()
    results = manager.search_outputs(query=query, limit=20)

    if not results:
        console.print(f"[yellow]No outputs found matching '{query}'.[/yellow]")
        return

    console.print(f"\n[bold cyan]Outputs matching '{query}':[/bold cyan]\n")
    for result in results:
        output = result["output"]
        console.print(f"[bold]Session:[/bold] {result['session_id'][:8]}...")
        console.print(f"  Task: {result['task_input']}")
        console.print(f"  Type: {output['asset_type']}")
        console.print(f"  URL: {output['asset_url']}")
        console.print()


def resolve_session(console: Console, args: argparse.Namespace) -> str | None:
    """Resolve session ID from resume or session arguments."""
    if not (args.resume or args.session):
        return None

    from .session import SessionManager

    manager = SessionManager()

    if args.resume:
        latest = manager.get_latest_session()
        if latest:
            console.print(f"[bold green]Resuming from session: {latest.metadata.session_id}[/bold green]")
            console.print(f"Previous task: {latest.metadata.task_input}\n")
            return latest.metadata.session_id
        console.print("[yellow]No previous sessions found. Starting new session.[/yellow]\n")
        return None

    # args.session specified
    session = manager.load_session(args.session)
    if session:
        console.print(f"[bold green]Loaded session: {args.session}[/bold green]")
        console.print(f"Task: {session.metadata.task_input}\n")
        return args.session

    console.print(f"[red]Session {args.session} not found. Starting new session.[/red]\n")
    return None


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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

    # Task input
    parser.add_argument("task", nargs="?", help="Task description for the agent")
    parser.add_argument("--task", "-t", dest="task_flag", help="Task description (alternative)")

    # Session management
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest session")
    parser.add_argument("--session", "-s", type=str, help="Load specific session by ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all past sessions")
    parser.add_argument("--search-outputs", type=str, metavar="QUERY", help="Search outputs across sessions")

    # LoRA management
    parser.add_argument("--update-loras", action="store_true", help="Update LoRA catalog from Hugging Face")

    # Output control
    parser.add_argument("--verbose", "-V", action="store_true", help="Show executed code and output")

    # Meta
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================


def get_chat_input(console: Console) -> str | None:
    """Get chat input from user. Returns None to exit."""
    try:
        user_input = get_multiline_input(console)
        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            return None
        return user_input
    except (EOFError, KeyboardInterrupt):
        return None


def run_agent(console: Console, args: argparse.Namespace) -> None:
    """Initialize and run the agent."""
    session_id = resolve_session(console, args)

    # Check if task was provided via CLI args
    cli_task = args.task or args.task_flag
    interactive_mode = cli_task is None

    # Get initial task from args or prompt
    user_request = get_task_input(args)
    if not user_request.strip():
        console.print("[dim]No task provided.[/dim]")
        return

    # Create the agent
    agent = OkonomiAgent(
        config=config,
        session_id=session_id,
        console=console,
        verbose=args.verbose,
    )

    if session_id:
        agent.load_session(session_id)

    # Run the initial task
    console.print()
    agent.run(user_request, reset=True)

    # If task was provided via CLI, exit after running
    if not interactive_mode:
        try:
            agent.save_session(status="completed")
        except Exception:
            pass
        return

    # Interactive chat loop
    while True:
        try:
            console.print()
            user_request = get_chat_input(console)

            if user_request is None:
                break

            if not user_request.strip():
                continue

            # Handle commands
            if user_request.startswith("/"):
                if user_request.lower() in ["/clear", "/reset"]:
                    console.print("[dim]Conversation cleared.[/dim]")
                    agent.message_history = []
                    continue
                elif user_request.lower() == "/save":
                    agent.save_session(status="saved")
                    console.print(f"[dim]Session saved: {agent.session_id}[/dim]")
                    continue
                elif user_request.lower() == "/help":
                    console.print("\n[dim]Commands:[/dim]")
                    console.print("[dim]  /exit, /quit  End session[/dim]")
                    console.print("[dim]  /clear        Clear history[/dim]")
                    console.print("[dim]  /save         Save session[/dim]")
                    console.print("[dim]  /help         Show help[/dim]\n")
                    continue

            console.print()
            agent.run(user_request, reset=False)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. /exit to quit[/dim]")
            continue

    # Save session on exit
    try:
        agent.save_session(status="completed")
        console.print(f"[dim]Session saved: {agent.session_id}[/dim]")
    except Exception as e:
        console.print(f"[dim red]Error saving session: {e}[/dim red]")


def handle_update_loras(console: Console) -> None:
    """Update the LoRA catalog from Hugging Face."""
    from .utils import update_loras_catalog

    console.print("[yellow]Updating LoRA catalog from Hugging Face...[/yellow]")
    try:
        path = update_loras_catalog()
        console.print("[green]LoRA catalog updated successfully![/green]")
        console.print(f"[dim]Saved to: {path}[/dim]")
    except Exception as e:
        console.print(f"[red]Error updating LoRA catalog: {e}[/red]")


def main() -> None:
    """Main entry point for okonomi CLI."""
    args = parse_arguments()
    setup_observability()
    console = Console()

    # Handle standalone commands
    if args.list_sessions:
        handle_list_sessions(console)
        return

    if args.search_outputs:
        handle_search_outputs(console, args.search_outputs)
        return

    if args.update_loras:
        handle_update_loras(console)
        return

    # Run the agent
    run_agent(console, args)


if __name__ == "__main__":
    main()
