"""Interactive input handling for Okonomi CLI."""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


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

    # Interactive mode with better in-terminal editing
    console = Console()
    console.print(
        Panel(
            Text("What's on your mind?", style="bold cyan", justify="center"),
            border_style="bright_blue",
            expand=True,
        )
    )

    try:
        import readline

        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set editing-mode vi")  # or vi

        console.print(
            "[dim]Enter your request. Use arrow keys to edit. Multi-line paste supported. Press Enter twice to finish:[/dim]\n"
        )

        lines = []
        empty_line_count = 0

        while empty_line_count < 2:
            try:
                if len(lines) == 0:
                    prompt = "\x1b[92m>>> \x1b[0m"  # Bright green
                else:
                    prompt = "\x1b[92m... \x1b[0m"  # Bright green

                line = input(prompt)

                if line.strip() == "":
                    empty_line_count += 1
                else:
                    empty_line_count = 0

                lines.append(line)

            except EOFError:
                if lines:
                    console.print("\n[yellow]Input finished with Ctrl+D, using what was entered so far...[/yellow]")
                    break
                else:
                    console.print("\n[yellow]No input provided[/yellow]")
                    return ""
            except KeyboardInterrupt:
                console.print("\n[red]Cancelled by user[/red]")
                exit(0)

        result = "\n".join(lines[:-empty_line_count] if empty_line_count > 0 else lines)
        return result.strip()

    except ImportError:
        # Fallback if readline not available
        console.print("[dim]Enter your request (type 'END' on a new line to finish):[/dim]\n")

        lines = []
        try:
            while True:
                if len(lines) == 0:
                    line = console.input("[bright_green]>>> [/bright_green]")
                else:
                    line = console.input("[bright_green]... [/bright_green]")

                if line.strip() == "END":
                    break

                lines.append(line)

        except EOFError:
            console.print("\n[yellow]Input finished with Ctrl+D[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled by user[/red]")
            exit(0)

        return "\n".join(lines)
