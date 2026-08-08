"""Fuzzy slash-command completer for the REPL input."""

from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion

from ..utils import fuzzy_match
from .commands import SLASH_COMMANDS, discover_skill_commands


class SlashCompleter(Completer):
    """Completes ``/command`` names with subsequence-fuzzy filtering.

    Only fires on a leading ``/`` with no space yet — so free prose and command
    arguments are never completed. Reuses okonomi's ``fuzzy_match`` so the
    ranking matches the rest of the app.
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        query = text[1:].lower()
        for cmd, desc in [*SLASH_COMMANDS, *discover_skill_commands()]:
            if not query or fuzzy_match(query, cmd[1:].lower()):
                yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)
