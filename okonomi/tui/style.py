"""prompt_toolkit Style for the REPL — the okonomi palette.

One source of truth for the chrome colors. The accent (pink) and amber are
shared with the transcript's rich constants so the live region and scrollback
read as one theme.
"""

from __future__ import annotations

from prompt_toolkit.styles import Style


# Core palette — keep in sync with transcript.py's rich constants.
_ACCENT = "#d787af"  # pink — prompt, spinner, selection
_AMBER = "#ffaf00"  # approval / warnings
_SURFACE = "#1a1a2e"  # toolbar / menu background (dark navy)
_SURFACE_HI = "#2a2a45"  # highlighted row background
_TEXT = "#c6c6d0"  # primary toolbar text
_MUTED = "#6c6c80"  # metadata / separators
_FAINT = "#3a3a4a"  # de-emphasized (background jobs)

REPL_STYLE = Style.from_dict(
    {
        "prompt": _ACCENT,
        "prompt.approval": f"{_AMBER} bold",
        "rule": _MUTED,  # full-width delimiter above the input
        "frame.border": _MUTED,  # the input box
        # Inline tool-status pane (above the input box).
        "tool.glyph": _ACCENT,
        "tool.label": f"{_TEXT} bold",
        "tool.meta": _MUTED,
        # Approval question pane (above the input box, collapses on answer).
        "approval.q": f"{_AMBER} bold",
        "approval.hint": _MUTED,
        # Bottom live region: no fill — inherit the terminal background.
        # `noreverse` cancels prompt_toolkit's default reverse-video toolbar.
        "bottom-toolbar": f"noreverse {_MUTED}",
        "bottom-toolbar.glyph": f"noreverse {_ACCENT}",
        "bottom-toolbar.label": f"noreverse {_TEXT} bold",
        "bottom-toolbar.meta": f"noreverse {_MUTED}",
        "bottom-toolbar.sep": f"noreverse {_MUTED}",
        "bottom-toolbar.bg": f"noreverse {_FAINT}",  # background jobs — darkest
        # Slash-command completion menu.
        "completion-menu": f"bg:{_SURFACE} {_TEXT}",
        "completion-menu.completion.current": f"bg:{_SURFACE_HI} {_ACCENT} bold",
        "completion-menu.meta.completion": f"bg:{_SURFACE} {_MUTED}",
        "completion-menu.meta.completion.current": f"bg:{_SURFACE_HI} {_ACCENT}",
        # Picker rows (session / asset modals) — these were referenced but
        # never defined, so the selected row had no highlight.
        "picker-highlight": f"bg:{_SURFACE_HI} {_ACCENT} bold",
        "dim": _MUTED,
    }
)
