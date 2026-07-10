"""ANSI color constants and CLI input helper for Okonomi.

Color constants for the CLI startup chrome (session resolution, one-shot
messages) plus the ``--task`` input accessor. All agent and tool output now
flows through the EventEmitter and its single transcript renderer.
"""

import argparse


DIM = "\033[2m"
RST = "\033[0m"
RED = "\033[31m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


def get_task_input(args: argparse.Namespace) -> str:
    """Return the task string provided via the --task flag.

    The interactive REPL handles interactive input; this helper is only used
    by the one-shot non-interactive code path.
    """
    return (args.task or "").strip()
