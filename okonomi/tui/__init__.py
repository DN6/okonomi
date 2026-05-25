"""Textual-based TUI for okonomi.

Replaces the prompt_toolkit-based InteractiveSession in cli.py. Built around
an event-driven output model so tool calls can be rendered as collapsible
sections instead of raw streamed text.
"""

from .events import (
    AssistantStream,
    Banner,
    Divider,
    Event,
    EventEmitter,
    SystemNote,
    ToolCall,
    UserMessage,
)


__all__ = [
    "AssistantStream",
    "Banner",
    "Divider",
    "Event",
    "EventEmitter",
    "SystemNote",
    "ToolCall",
    "UserMessage",
]
