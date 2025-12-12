"""Command-line interface for Okonomi."""

import argparse
import logging
import os
from pathlib import Path

import yaml
from mcp import StdioServerParameters
from rich.console import Console
from smolagents import InferenceClientModel, LiteLLMModel, MCPClient

from .code_agent import OkonomiCodeAgent
from .config import config
from .interface import get_task_input
from .tools.server import okonomi_server

# Constants
TEMPLATE_FILE = "prompt_templates/system_prompt.yml"
ENV_KEYS = ["HF_TOKEN", "FAL_KEY", "TAVILY_API_KEY"]

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)


# =============================================================================
# Setup Functions
# =============================================================================


def setup_observability() -> None:
    """Initialize observability instrumentation if enabled."""
    if os.getenv("LOG_CREATE_AGENT_TRACES", "false").lower() not in ["true", "1"]:
        return

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    register()
    SmolagentsInstrumentor().instrument()


def initialize_model() -> InferenceClientModel:
    """Create and configure the language model."""
    model_cls_map = {
        "inference_client": InferenceClientModel,
        "litellm": LiteLLMModel,
    }
    model_cls = model_cls_map.get(config.agent.model_type, InferenceClientModel)

    return model_cls(
        model_id=config.agent.model_id,
        temperature=config.agent.temperature,
        provider=config.agent.model_provider,
        top_p=config.agent.top_p,
        max_tokens=config.agent.max_tokens,
    )


def load_prompt_templates() -> dict:
    """Load prompt templates from package or project root."""
    package_path = Path(__file__).parent / TEMPLATE_FILE
    project_path = Path(__file__).parent.parent / TEMPLATE_FILE

    templates_path = package_path if package_path.exists() else project_path

    with open(templates_path) as f:
        return yaml.safe_load(f)


def setup_mcp_server() -> StdioServerParameters:
    """Configure MCP server parameters with environment variables."""
    env = {key: os.getenv(key) for key in ENV_KEYS if os.getenv(key)}

    return StdioServerParameters(
        command="python",
        args=["-m", "okonomi.tools.server"],
        env=env,
    )


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
        console.print(f"  Task: {session.task}")
        console.print(f"  Time: {session.timestamp}")
        console.print(f"  Status: {session.status}")
        console.print(f"  Outputs: {session.num_outputs}")
        console.print()


def handle_search_images(console: Console, query: str) -> None:
    """Search and display images matching query."""
    from .session import SessionManager

    manager = SessionManager()
    images = manager.search_images(query=query, limit=20)

    if not images:
        console.print(f"[yellow]No images found matching '{query}'.[/yellow]")
        return

    console.print(f"\n[bold cyan]Images matching '{query}':[/bold cyan]\n")
    for img in images:
        console.print(f"[bold]Session:[/bold] {img['session_id'][:8]}...")
        console.print(f"  Task: {img['session_task']}")
        console.print(f"  URL: {img['image_url']}")
        console.print(f"  Score: {img.get('visual_critique_score', 'N/A')}")
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
            console.print(f"Previous task: {latest.metadata.task}\n")
            return latest.metadata.session_id
        console.print("[yellow]No previous sessions found. Starting new session.[/yellow]\n")
        return None

    # args.session specified
    session = manager.load_session(args.session)
    if session:
        console.print(f"[bold green]Loaded session: {args.session}[/bold green]")
        console.print(f"Task: {session.metadata.task}\n")
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
  okonomi --eval "Generate artwork"          # Disable automatic evaluation
  okonomi --plan "Create a logo"             # Disable automatic planning
  okonomi --resume                           # Resume from latest session
  okonomi --session <id>                     # Load specific session
  okonomi --list-sessions                    # List all past sessions
  okonomi --search-images "portrait"         # Search for images
        """,
    )

    # Task input
    parser.add_argument("task", nargs="?", help="Task description for the agent")
    parser.add_argument("--task", "-t", dest="task_flag", help="Task description (alternative)")

    # Agent behavior
    parser.add_argument("--eval", "-e", action="store_false", help="Disable automatic evaluation")
    parser.add_argument("--plan", "-p", action="store_false", help="Disable automatic planning")

    # Session management
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest session")
    parser.add_argument("--session", "-s", type=str, help="Load specific session by ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all past sessions")
    parser.add_argument("--search-images", type=str, metavar="QUERY", help="Search images across sessions")

    # Meta
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================


def get_chat_input(console: Console) -> str | None:
    """Get chat input from user. Returns None to exit."""
    try:
        user_input = console.input("[green]>>> [/green]").strip()
        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            return None
        return user_input
    except (EOFError, KeyboardInterrupt):
        return None


def run_agent(console: Console, args: argparse.Namespace) -> None:
    """Initialize and run the agent in interactive chat mode."""
    model = initialize_model()
    templates = load_prompt_templates()
    server_params = setup_mcp_server()
    session_id = resolve_session(console, args)

    # Get initial task from args or prompt
    user_request = get_task_input(args)
    if not user_request.strip():
        console.print("[yellow]No task provided. Exiting.[/yellow]")
        return

    with MCPClient([server_params]) as mcp_tools:
        agent = OkonomiCodeAgent(
            model=model,
            tools=mcp_tools,
            server_tools=okonomi_server.tools,
            planning_interval=config.agent.planning_interval,
            prompt_templates=templates,
            return_full_result=True,
            max_steps=config.agent.max_steps,
            verbosity_level=config.agent.verbosity_level,
            session_id=session_id,
        )

        if session_id:
            agent.load_from_session(session_id)

        is_first_run = True

        # Interactive chat loop
        while True:
            try:
                console.print("\n[bold yellow]Processing your request...[/bold yellow]\n")
                agent.run(user_request, reset=is_first_run)
                is_first_run = False

                # Prompt for next message
                console.print("\n[dim]Type your follow-up or /exit to quit[/dim]")
                user_request = get_chat_input(console)

                if user_request is None:
                    console.print("\n[cyan]Ending chat session...[/cyan]")
                    break

                if not user_request.strip():
                    continue

                # Handle commands
                if user_request.startswith("/"):
                    if user_request.lower() in ["/clear", "/reset"]:
                        is_first_run = True
                        console.print("[yellow]Conversation cleared.[/yellow]")
                        user_request = get_chat_input(console)
                        if user_request is None:
                            break
                        continue
                    elif user_request.lower() == "/save":
                        agent._save_session(status="saved", merged_outputs=agent.generated_outputs)
                        console.print(f"[green]Session saved: {agent.session_id}[/green]")
                        continue
                    elif user_request.lower() == "/help":
                        console.print("\n[bold cyan]Commands:[/bold cyan]")
                        console.print("  /exit, /quit - End chat session")
                        console.print("  /clear, /reset - Clear conversation history")
                        console.print("  /save - Save current session")
                        console.print("  /help - Show this help\n")
                        continue

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /exit to quit or continue chatting.[/yellow]")
                user_request = get_chat_input(console)
                if user_request is None:
                    break
                continue

        # Save session on exit
        try:
            agent._save_session(status="completed", merged_outputs=agent.generated_outputs)
            console.print(f"[green]Session saved: {agent.session_id}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving session: {e}[/red]")


def main() -> None:
    """Main entry point for okonomi CLI."""
    args = parse_arguments()
    setup_observability()
    console = Console()

    # Handle standalone commands
    if args.list_sessions:
        handle_list_sessions(console)
        return

    if args.search_images:
        handle_search_images(console, args.search_images)
        return

    # Run the agent
    run_agent(console, args)


if __name__ == "__main__":
    main()
