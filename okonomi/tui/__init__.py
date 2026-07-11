"""Streaming-scrollback REPL TUI for okonomi.

``TuiSession`` drives the agent; ``ReplApp`` (:mod:`okonomi.tui.app`) is the
prompt_toolkit front-end; ``ReplTranscript`` (:mod:`okonomi.tui.transcript`)
renders the :mod:`okonomi.events` stream into native terminal scrollback.
Headless callers (CLI one-shot, Discord) don't need to import anything from
this package.
"""
