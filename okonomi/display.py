"""Display and input utilities for Okonomi.

Centralizes ANSI color constants, box-drawing helpers,
user-facing print functions, and input prompts used across the CLI and agent.
"""

import argparse
import shutil
import textwrap

from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

from pygments import highlight as _highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name

_CODE_FORMATTER = TerminalTrueColorFormatter(style="monokai")
_PYTHON_LEXER = PythonLexer()

DIM = "\033[2m"
RST = "\033[0m"
RED = "\033[31m"
PINK = "\033[38;2;215;135;175m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

MAX_DISPLAY_LINES = 50
MAX_DISPLAY_CHARS = 4000


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate text that exceeds display limits.

    Returns (truncated_text, was_truncated).
    Keeps 40% head and 60% tail when character-truncating.
    """
    if len(text) > MAX_DISPLAY_CHARS:
        head_size = int(MAX_DISPLAY_CHARS * 0.4)
        tail_size = MAX_DISPLAY_CHARS - head_size
        text = text[:head_size] + f"\n\n... ({len(text):,} chars, truncated) ...\n\n" + text[-tail_size:]
        return text, True

    lines = text.splitlines()
    if len(lines) > MAX_DISPLAY_LINES:
        return "\n".join(lines[:MAX_DISPLAY_LINES]), True

    return text, False


def print_ansi_box(text: str, label: str = "", color: str = "2") -> None:
    """Print text in a bordered box using ANSI box-drawing characters.

    Uses ``print()`` so output routes through ``patch_stdout`` correctly.
    ``color`` is an ANSI SGR code: "2" for dim, "31" for red, etc.
    """
    w = shutil.get_terminal_size().columns
    c = f"\033[{color}m"
    h = "\u2500"

    text, truncated = _truncate_output(text)
    inner_width = max(w - 4, 20)
    wrapped_lines = []
    for raw_line in text.splitlines():
        if len(raw_line) > inner_width:
            wrapped_lines.extend(
                textwrap.wrap(raw_line, width=inner_width, break_long_words=True, break_on_hyphens=False)
            )
        else:
            wrapped_lines.append(raw_line)

    tag = f" {label} " if label else ""
    top_fill = w - 2 - len(tag)
    print(f"{c}\u256d{h}{tag}{h * max(top_fill, 0)}\u256e{RST}")

    for line in wrapped_lines:
        print(f"{c}\u2502{RST} {line}")

    if truncated:
        print(f"{c}\u2502{RST} {DIM}... truncated ...{RST}")

    print(f"{c}\u2570{h * (w - 2)}\u256f{RST}")


def highlight_code(code: str, language: str = "python") -> str:
    """Syntax-highlight code and return ANSI-colored string."""
    try:
        lexer = _PYTHON_LEXER if language == "python" else get_lexer_by_name(language)
        return _highlight(code, lexer, _CODE_FORMATTER).rstrip("\n")
    except Exception:
        return code


def print_code_box(code: str, label: str = "code", language: str = "python") -> None:
    """Print syntax-highlighted code in a bordered box."""
    highlighted = highlight_code(code, language)
    w = shutil.get_terminal_size().columns
    c = f"\033[2m"
    h = "\u2500"

    tag = f" {label} " if label else ""
    top_fill = w - 2 - len(tag)
    print(f"{c}\u256d{h}{tag}{h * max(top_fill, 0)}\u256e{RST}")

    for line in highlighted.splitlines():
        print(f"{c}\u2502{RST} {line}")

    print(f"{c}\u2570{h * (w - 2)}\u256f{RST}")


def print_user_message(text: str) -> None:
    """Echo the user's message into the output area."""
    w = shutil.get_terminal_size().columns
    print(f"\n{PINK}{'━' * w}{RST}")
    if "\n" in text:
        first_line = text.split("\n")[0]
        line_count = text.count("\n") + 1
        print(f"{PINK}{BOLD}●{RST} {BOLD}{first_line}{RST} {DIM}(+{line_count - 1} lines){RST}")
    else:
        print(f"{PINK}{BOLD}●{RST} {BOLD}{text}{RST}")


def print_interrupt() -> None:
    print(f"{PINK}Interrupted.{RST} {DIM}History preserved.{RST}")


_SUNSET_ART = [
    ("240;135;175",  r"~^~^~^~^~^~^~^~^~^~^~"),
    ("225;110;155",  r" ~^~^~^~^~^~^~^~^~^~ "),
    ("210;90;135",   r"  ~^~  ~^~  ~^~  ~^~ "),
    ("195;75;120",   r"    ~^~  ~^~  ~^~    "),
    ("165;55;95",    r"      ~^~    ~^~     "),
]


def print_loading_art() -> None:
    """Print sakura wave reflection art, centered."""
    w = shutil.get_terminal_size().columns
    for color, line in _SUNSET_ART:
        pad = (w - len(line)) // 2
        print(f"\033[38;2;{color}m{' ' * pad}{line}{RST}")


def print_rule() -> None:
    w = shutil.get_terminal_size().columns
    h = "\u2500"
    print(f"{PINK}{h * w}{RST}")


def _basic_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return bindings


def get_multiline_input() -> str:
    """Get input with bordered prompt (Alt+Enter for newline)."""
    try:
        print_rule()
        user_input = prompt(
            [("#d787af bold", "> ")],
            key_bindings=_basic_key_bindings(),
            multiline=False,
        ).strip()
        if user_input:
            print_rule()
        return user_input
    except EOFError:
        return ""
    except KeyboardInterrupt:
        print(f"\n{DIM}Cancelled{RST}")
        raise


def get_task_input(args: argparse.Namespace) -> str:
    """Get task input either from CLI arguments or interactive prompt."""
    task = args.task or args.task_flag

    if task:
        return task

    print()
    w = shutil.get_terminal_size().columns
    label = "What's on your mind?"
    pad = (w - len(label)) // 2
    print(f"{PINK}{' ' * pad}{label}{RST}")
    print()

    try:
        return get_multiline_input()
    except KeyboardInterrupt:
        exit(0)
