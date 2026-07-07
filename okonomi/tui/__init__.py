"""Streaming-scrollback REPL TUI for okonomi.

The interactive chat front-end (:mod:`okonomi.tui.repl`) plus the event
subscribers that render :mod:`okonomi.events` into native terminal
scrollback. Headless callers (CLI one-shot, Discord) don't need to import
anything from this package.
"""
