"""The okonomi color palette — single source of truth for the TUI chrome.

Plain hex strings, no imports, so both the transcript renderer (rich, and the
one-shot ``--task`` path) and the REPL style (prompt_toolkit) can share them
without either pulling in the other's dependency.
"""

from __future__ import annotations


ACCENT = "#d787af"  # pink — prompt, spinner, selection, bullets
AMBER = "#ffaf00"  # approval prompts / warnings
ERROR = "#ff5f5f"  # errors
MUTED = "#6c6c80"  # metadata, separators, rules
SURFACE = "#1a1a2e"  # toolbar / menu background (dark navy)
SURFACE_HI = "#2a2a45"  # highlighted row background
TEXT = "#c6c6d0"  # primary toolbar text
FAINT = "#3a3a4a"  # de-emphasized (background jobs)
