"""Interactive input handling for Okonomi CLI."""

import argparse

from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console


def create_key_bindings() -> KeyBindings:
    """Create key bindings for multiline input."""
    bindings = KeyBindings()

    @bindings.add("escape", "enter")  # Alt+Enter or Escape then Enter
    def _(event):
        """Insert newline on Alt+Enter."""
        event.current_buffer.insert_text("\n")

    return bindings


def get_multiline_input(console: Console) -> str:
    """Get input with multiline support (Alt+Enter for newline)."""
    try:
        bindings = create_key_bindings()
        user_input = prompt(
            "> ",
            key_bindings=bindings,
            multiline=False,  # Enter submits, special keys for newline
        ).strip()
        return user_input
    except EOFError:
        return ""
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled[/dim]")
        raise


def get_task_input(args: argparse.Namespace) -> str:
    """Get task input either from CLI arguments or interactive prompt.

    Args:
        args: Parsed command line arguments

    Returns:
        str: The task description from user input
    """
    # Use positional argument first, then --task flag
    task = args.task or args.task_flag

    if task:
        return task

    # Interactive mode
    console = Console()
    console.print()
    console.print("[#d787af]What's on your mind?[/#d787af]", justify="center")
    console.print()

    try:
        return get_multiline_input(console)
    except KeyboardInterrupt:
        exit(0)
