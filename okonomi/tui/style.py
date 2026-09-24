"""prompt_toolkit Style for the REPL, built from the shared palette.

Maps the okonomi palette (:mod:`okonomi.tui.palette`) onto prompt_toolkit
style classes so the live input region reads as one theme with the rich
transcript in scrollback.
"""

from __future__ import annotations

from prompt_toolkit.styles import Style

from .palette import (
    ACCENT as _ACCENT,
    AMBER as _AMBER,
    FAINT as _FAINT,
    MUTED as _MUTED,
    SURFACE as _SURFACE,
    SURFACE_HI as _SURFACE_HI,
    TEXT as _TEXT,
)


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
        # Session-tree role tags: who wrote the row.
        "tree.user": f"{_ACCENT} bold",
        "tree.agent": f"{_AMBER} bold",
        "tree.guide": _MUTED,
        "dim": _MUTED,
    }
)
