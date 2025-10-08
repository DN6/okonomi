import argparse
import os
from functools import partial
from pathlib import Path
import logging
from typing import List

import yaml
from mcp import StdioServerParameters
from rich.console import Console
from smolagents import InferenceClientModel, LiteLLMModel, MCPClient, PlanningStep

from .agent import OkonomiAgent
from .callbacks import eval_image, eval_prompt, interrupt_after_plan
from .config import config
from .interface import get_task_input
from .utils import update_available_loras


# Load prompt templates from package
TEMPLATE_FILE = "prompt_templates/system_prompt.yml"

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


def setup_observability() -> None:
    if os.getenv("LOG_CREATE_AGENT_TRACES", "false").lower() not in ["true", "1"]:
        return

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    register()
    SmolagentsInstrumentor().instrument()


def initialize_model() -> InferenceClientModel:
    model_cls_map = {"inference_client": InferenceClientModel, "litellm": LiteLLMModel}
    model_cls = model_cls_map.get(config.agent.model_type, InferenceClientModel)

    return model_cls(
        model_id=config.agent.model_id,
        temperature=config.agent.temperature,
        provider=config.agent.model_provider,
        top_p=config.agent.top_p,
        max_tokens=config.agent.max_tokens,
    )


def setup_mcp_tools() -> List[StdioServerParameters]:
    # Define tool module names
    tool_modules = ["prompt", "image", "memory", "strategies", "sessions"]

    tools = []

    env = {}
    if os.getenv("HF_TOKEN"):
        env["HF_TOKEN"] = os.getenv("HF_TOKEN")
    if os.getenv("FAL_KEY"):
        env["FAL_KEY"] = os.getenv("FAL_KEY")
    if os.getenv("CHROMA_API_KEY"):
        env["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")

    for module_name in tool_modules:
        params = StdioServerParameters(
            command="python",
            args=[
                "-m",
                f"okonomi.tools.{module_name}",
            ],
            env=env,
        )
        tools.append(params)

    return tools


def setup_tool_callbacks() -> dict:
    autoeval = config.tools.callbacks.autoeval

    return {
        "create_prompt": partial(eval_prompt, autoeval=autoeval),
        "text_to_image": partial(eval_image, autoeval=autoeval),
        "image_to_image": partial(eval_image, autoeval=autoeval),
        "edit_image": partial(eval_image, autoeval=autoeval),
        "reference_image": partial(eval_image, autoeval=autoeval),
    }


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
  okonomi --eval "Generate artwork"          # Disable automatic evaluation (manual review)
  okonomi --plan "Create a logo"             # Disable automatic planning (plan approval)
  okonomi --eval --plan "Create artwork"     # Disable both auto features
  okonomi --update-loras                     # Update LoRA collections and exit
  okonomi --resume                           # Resume from latest session
  okonomi --session <id>                     # Load specific session
  okonomi --list-sessions                    # List all past sessions
  okonomi --search-images "portrait"         # Search for images across sessions
        """,
    )

    parser.add_argument("task", nargs="?", help="Task description for the agent to process")
    parser.add_argument("--task", "-t", dest="task_flag", help="Task description (alternative to positional argument)")
    parser.add_argument("--version", "-v", action="version", version="okonomi 0.1.0")
    parser.add_argument(
        "--eval", "-e", action="store_false", help="Disable automatic evaluation (enable manual review)"
    )
    parser.add_argument(
        "--plan", "-p", action="store_false", help="Disable automatic planning (enable plan approval prompts)"
    )
    parser.add_argument("--update-loras", action="store_true", help="Update available LoRA collections and exit")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from the latest session")
    parser.add_argument("--session", "-s", type=str, help="Load a specific session by ID")
    parser.add_argument("--list-sessions", action="store_true", help="List all past sessions and exit")
    parser.add_argument(
        "--search-images", type=str, metavar="QUERY", help="Search for images across all sessions and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point for okonomi CLI."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Setup observability
        setup_observability()

        console = Console()

        # Handle session management commands
        if args.list_sessions:
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
            return

        if args.search_images:
            from .session import SessionManager

            manager = SessionManager()
            images = manager.search_images(query=args.search_images, limit=20)

            if not images:
                console.print(f"[yellow]No images found matching '{args.search_images}'.[/yellow]")
                return

            console.print(f"\n[bold cyan]Images matching '{args.search_images}':[/bold cyan]\n")
            for img in images:
                console.print(f"[bold]Session:[/bold] {img['session_id'][:8]}...")
                console.print(f"  Task: {img['session_task']}")
                console.print(f"  URL: {img['image_url']}")
                console.print(f"  Score: {img.get('visual_critique_score', 'N/A')}")
                console.print()
            return

        # Handle update-loras command
        if args.update_loras:
            console.print("[bold yellow]Updating available LoRA collections...[/bold yellow]")
            try:
                update_available_loras()
                console.print("[bold green]✅ LoRA collections updated successfully![/bold green]")
            except Exception as e:
                console.print(f"[bold red]❌ Failed to update LoRA collections: {str(e)}[/bold red]")
                exit(1)
            return

        # Initialize model
        model = initialize_model()

        # Try to load from package location first (for installed packages)
        package_template_path = Path(__file__).parent / TEMPLATE_FILE
        if package_template_path.exists():
            templates_path = package_template_path
        else:
            # Fallback to project root (for development)
            project_root = Path(__file__).parent.parent
            templates_path = project_root / TEMPLATE_FILE

        templates = yaml.safe_load(open(templates_path, "r").read())

        # Setup MCP tools
        tools = setup_mcp_tools()

        # Override config with CLI arguments
        # Note: args.eval and args.plan are store_false, so they start as True and become False when flags are used
        autoeval = args.eval and config.tools.callbacks.autoeval
        autoplan = args.plan and config.agent.autoplan

        # Setup callbacks with potentially overridden autoeval
        config["tools"]["callbacks"]["autoeval"] = autoeval
        tool_callback_map = setup_tool_callbacks()

        # Handle session loading
        session_id = None
        if args.resume or args.session:
            from .session import SessionManager

            manager = SessionManager()

            if args.resume:
                latest = manager.get_latest_session()
                if latest:
                    session_id = latest.metadata.session_id
                    console.print(f"[bold green]Resuming from session: {session_id}[/bold green]")
                    console.print(f"Previous task: {latest.metadata.task}\n")
                else:
                    console.print("[yellow]No previous sessions found. Starting new session.[/yellow]\n")
            elif args.session:
                session = manager.load_session(args.session)
                if session:
                    session_id = args.session
                    console.print(f"[bold green]Loaded session: {session_id}[/bold green]")
                    console.print(f"Task: {session.metadata.task}\n")
                else:
                    console.print(f"[red]Session {args.session} not found. Starting new session.[/red]\n")

        # Get task input (from CLI args or interactive)
        user_request = get_task_input(args)

        if not user_request.strip():
            console.print("[yellow]No task provided. Exiting.[/yellow]")
            return

        console.print("\n[bold yellow]Processing your request...[/bold yellow]\n")

        with MCPClient(tools) as tools:
            agent = OkonomiAgent(
                tools=tools,
                model=model,
                planning_interval=config.agent.planning_interval,
                prompt_templates=templates,
                step_callbacks={
                    PlanningStep: partial(interrupt_after_plan, autoplan=autoplan),
                },
                tool_callbacks=tool_callback_map,
                return_full_result=True,
                max_steps=config.agent.max_steps,
                verbosity_level=config.agent.verbosity_level,
                session_id=session_id,
            )

            # Load from session if specified
            if session_id:
                agent.load_from_session(session_id)

            agent.run(user_request, reset=True)

    except KeyboardInterrupt:
        console = Console()
        console.print("\n[red]Operation cancelled by user[/red]")
        try:
            merged_outputs = agent._write_to_permanent_memory()
            agent._save_session(status="interrupted", merged_outputs=merged_outputs)
            console.print(f"[yellow]Session saved: {agent.session_id}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error saving session: {e}[/red]")
        exit(0)


if __name__ == "__main__":
    main()
