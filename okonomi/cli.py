import argparse
import os
from functools import partial
from pathlib import Path
from typing import List

import yaml
from mcp import StdioServerParameters
from rich.console import Console
from smolagents import InferenceClientModel, LiteLLMModel, MCPClient, PlanningStep

from .agent import CreateAgent
from .callbacks import eval_image, eval_prompt, interrupt_after_plan
from .config import config
from .interface import get_task_input
from .utils import update_available_loras


def setup_observability() -> None:
    if os.getenv("LOG_CREATE_AGENT_TRACES", "false").lower() not in ["true", "1"]:
        return

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    register()
    SmolagentsInstrumentor().instrument()


def initialize_model() -> InferenceClientModel:
    model_cls_map = {"inference_client": InferenceClientModel, "litellm": LiteLLMModel}
    model_cls = model_cls_map.get(config["agent"]["model_type"], InferenceClientModel)

    return model_cls(
        model_id=config["agent"]["model_id"],
        temperature=config["agent"]["temperature"],
        provider=config["agent"]["model_provider"],
        top_p=config["agent"]["top_p"],
        max_tokens=config["agent"]["max_tokens"],
    )


def setup_mcp_tools() -> List[StdioServerParameters]:
    # Define tool module names
    tool_modules = ["prompt", "image", "memory", "strategies"]

    tools = []
    for module_name in tool_modules:
        env = {}
        if os.getenv("HF_TOKEN"):
            env["HF_TOKEN"] = os.getenv("HF_TOKEN")
        if os.getenv("FAL_KEY"):
            env["FAL_KEY"] = os.getenv("FAL_KEY")
        if os.getenv("CHROMA_API_KEY"):
            env["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")

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
    autoeval = config["tools"]["callbacks"]["autoeval"]

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
  okonomi --config custom-config.json        # Use custom config file
  okonomi --update-loras                     # Update LoRA collections and exit

Environment Variables:
  OKONOMI_CONFIG=/path/to/config.json       # Set default config file path
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
    parser.add_argument(
        "--config", "-c", dest="config_file", help="Path to custom configuration file (default: config.json)"
    )
    parser.add_argument("--update-loras", action="store_true", help="Update available LoRA collections and exit")

    return parser.parse_args()


def main():
    """Main entry point for okonomi CLI."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Setup observability
        setup_observability()

        # Determine config file path (priority: CLI arg > env var > default)
        config_path = args.config_file or os.getenv("OKONOMI_CONFIG", "config.json")

        # Set configuration path
        config.set_config_path(config_path)

        # Handle update-loras command
        if args.update_loras:
            console = Console()
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

        # Load prompt templates from package
        template_file = "prompt_templates/system_prompt.yml"

        # Try to load from package location first (for installed packages)
        package_template_path = Path(__file__).parent / template_file
        if package_template_path.exists():
            templates_path = package_template_path
        else:
            # Fallback to project root (for development)
            project_root = Path(__file__).parent.parent
            templates_path = project_root / template_file

        templates = yaml.safe_load(open(templates_path, "r").read())

        # Setup MCP tools
        tools = setup_mcp_tools()

        # Override config with CLI arguments
        # Note: args.eval and args.plan are store_false, so they start as True and become False when flags are used
        autoeval = args.eval and config["tools"]["callbacks"]["autoeval"]
        autoplan = args.plan and config["agent"]["autoplan"]

        # Setup callbacks with potentially overridden autoeval
        config["tools"]["callbacks"]["autoeval"] = autoeval
        tool_callback_map = setup_tool_callbacks()

        # Get task input (from CLI args or interactive)
        user_request = get_task_input(args)

        if not user_request.strip():
            console = Console()
            console.print("[yellow]No task provided. Exiting.[/yellow]")
            return

        console = Console()
        console.print("\n[bold yellow]Processing your request...[/bold yellow]\n")

        with MCPClient(tools) as tools:
            agent = CreateAgent(
                tools=tools,
                model=model,
                planning_interval=config["agent"]["planning_interval"],
                prompt_templates=templates,
                step_callbacks={
                    PlanningStep: partial(interrupt_after_plan, autoplan=autoplan),
                },
                tool_callbacks=tool_callback_map,
                return_full_result=True,
                max_steps=config["agent"]["max_steps"],
                verbosity_level=config["agent"]["verbosity_level"],
            )
            agent.run(user_request, reset=True)

    except KeyboardInterrupt:
        console = Console()
        console.print("\n[red]Operation cancelled by user[/red]")
        agent._write_to_permanent_memory()
        exit(0)


if __name__ == "__main__":
    main()
