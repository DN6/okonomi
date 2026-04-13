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
    print(f"\n{PINK}{'─' * 20}{RST}")
    if "\n" in text:
        first_line = text.split("\n")[0]
        line_count = text.count("\n") + 1
        print(f"{PINK}{BOLD}●{RST} {BOLD}{first_line}{RST} {DIM}(+{line_count - 1} lines){RST}")
    else:
        print(f"{PINK}{BOLD}●{RST} {BOLD}{text}{RST}")
    print()


def print_interrupt() -> None:
    print(f"{PINK}Interrupted.{RST} {DIM}History preserved.{RST}")


WAVE_COLORS = [
    ("240;135;175", "#f087af"),
    ("210;90;135",  "#d25a87"),
    ("165;55;95",   "#a5375f"),
]

_GRADIENT_STEPS = [
    (215, 135, 175),
    (195, 120, 158),
    (175, 105, 141),
    (155, 90, 124),
    (135, 75, 107),
    (115, 60, 90),
    (95, 45, 73),
    (75, 35, 58),
]

_BLOCK_CHARS = "█▓▒░"


import math

_HEARTBEAT_FRAMES = ["·", "•", "●", "◉", "●", "•", "·", " "]
_HEARTBEAT_BRIGHTNESS = [0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.3]
_EQ_BARS = "▁▂▃▄▅▆▇█"


def _heartbeat(tick: int) -> list[tuple[str, str]]:
    """Heartbeat — single dot expands and contracts with pulsing color."""
    idx = tick % len(_HEARTBEAT_FRAMES)
    br = _HEARTBEAT_BRIGHTNESS[idx]
    r, g, b = int(215 * br), int(135 * br), int(175 * br)
    return [(f"#{r:02x}{g:02x}{b:02x}", _HEARTBEAT_FRAMES[idx])]


_TRAIL_DOTS = ["●", "●", "•", "·"]
_TRAIL_BRIGHTNESS = [1.0, 0.7, 0.45, 0.25]


def _trailing_dots(width: int, tick: int) -> list[tuple[str, str]]:
    """Trailing dots — a dot sweeps across leaving a fading trail."""
    cycle = width + 2
    result = []
    for i in range(width):
        pos = (tick - i) % cycle
        if pos < len(_TRAIL_DOTS):
            br = _TRAIL_BRIGHTNESS[pos]
            r, g, b = int(215 * br), int(135 * br), int(175 * br)
            ch = _TRAIL_DOTS[pos]
        else:
            r, g, b = 25, 20, 23
            ch = " "
        result.append((f"#{r:02x}{g:02x}{b:02x}", ch))
    return result


def spinner_formatted_text(label: str, tick: int) -> list[tuple[str, str]]:
    """Generate spinner animation.

    Uses heartbeat for 'Thinking', trailing dots for everything else.
    """
    if "thinking" in label.lower():
        blocks = _heartbeat(tick)
    else:
        blocks = _trailing_dots(4, tick)

    result = [("", "  ")]
    result.extend(blocks)
    result.append(("", " "))
    result.append(("bold #d787af", label))
    return result


def _wave_line(content_w: int, offset: int = 0) -> str:
    """Generate a wave string of given width with animation offset."""
    wave = "~^~"
    shifted = wave[offset % 3:] + wave[:offset % 3]
    return (shifted * (content_w // 3 + 2))[offset % 3: offset % 3 + content_w]


def wave_formatted_text(label: str, w: int, tick: int = 0, mode: str = "diamond") -> list[tuple[str, str]]:
    """Generate animated wave text as prompt_toolkit formatted text tuples.

    Args:
        label: Center text.
        w: Terminal width.
        tick: Animation tick (increment for movement).
        mode: "diamond" for expanding waves above/below, "inline" for waves flanking text.

    Returns:
        List of (style, text) tuples for FormattedTextControl.
    """
    if mode == "inline":
        wave_w = max((w - len(label) - 4) // 2, 6)
        left = _wave_line(wave_w, tick)
        right = _wave_line(wave_w, tick + 1)
        pad = max((w - wave_w * 2 - len(label) - 2) // 2, 0)
        return [
            (WAVE_COLORS[0][1], " " * pad + left + " "),
            ("bold #d787af", label),
            (WAVE_COLORS[0][1], " " + right + "\n"),
        ]

    result = []

    # Waves above (outermost first)
    for i in range(len(WAVE_COLORS) - 1, -1, -1):
        color_hex = WAVE_COLORS[i][1]
        content_w = len(label) + (i + 1) * 6
        line = _wave_line(content_w, tick + i)
        pad = max((w - content_w) // 2, 0)
        result.append((color_hex, " " * pad + line + "\n"))

    # Label
    pad = max((w - len(label)) // 2, 0)
    result.append(("bold #d787af", " " * pad + label + "\n"))

    # Waves below (innermost first)
    for i in range(len(WAVE_COLORS)):
        color_hex = WAVE_COLORS[i][1]
        content_w = len(label) + (i + 1) * 6
        line = _wave_line(content_w, tick + i + 1)
        pad = max((w - content_w) // 2, 0)
        result.append((color_hex, " " * pad + line + "\n"))

    return result


def _render_ft_ansi(formatted_text: list[tuple[str, str]]) -> str:
    """Convert prompt_toolkit formatted text tuples to ANSI string."""
    out = []
    for style, text in formatted_text:
        if "bold" in style:
            out.append(f"\033[1m\033[38;2;215;135;175m{text}{RST}")
        else:
            hex_c = style.lstrip("#")
            if len(hex_c) == 6:
                r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                out.append(f"\033[38;2;{r};{g};{b}m{text}{RST}")
            else:
                out.append(text)
    return "".join(out)


def print_loading_art(label: str = "What's on your mind?") -> None:
    """Print static sakura waves expanding outward from centered label."""
    import sys

    w = shutil.get_terminal_size().columns
    sys.stdout.write(_render_ft_ansi(wave_formatted_text(label, w, 0)))
    sys.stdout.flush()


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
