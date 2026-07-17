"""UI-produced chrome events, defined separately from agent events.

These are emitted by the TUI itself for its own scrollback — the agent
never sees or produces them, and they are never persisted. Agent events
(turn lifecycle, streaming text, tool-call display, system notes) live
in ``okonomi.events``; the dependency runs strictly TUI → agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..events import AssistantStream, Event, EventEmitter, ToolCallDisplay


@dataclass
class Banner(Event):
    """Static welcome banner shown at session start."""

    type: ClassVar[str] = "banner"


@dataclass
class Divider(Event):
    """Visual turn separator — a horizontal rule between user turns."""

    type: ClassVar[str] = "divider"


@dataclass
class UserMessage(Event):
    """Echo of the text the user typed, rendered into the scrollback."""

    type: ClassVar[str] = "user_message_display"

    text: str = ""


_SYNTHETIC_REPLAY_PREFIXES = ("[Resumed", "[system]", "[The user interrupted")


def replay_history(history: list, emitter: EventEmitter) -> None:
    """Re-emit a pydantic-ai message history as visible chat events on
    ``emitter``. Used after a session is resumed so the user sees their
    prior conversation rendered into the chat pane.

    Synthetic injections (resume markers, ``[system]`` notes, interrupt
    framings) are skipped — they're scaffolding, not user turns.
    """
    if not history:
        return
    try:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
    except Exception:
        return

    pending_tool_calls: dict[str, ToolCallDisplay] = {}
    any_emitted = False
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = part.content if isinstance(part.content, str) else str(part.content)
                    if text.strip().startswith(_SYNTHETIC_REPLAY_PREFIXES):
                        continue
                    if any_emitted:
                        emitter.emit(Divider())
                    emitter.emit(UserMessage(text))
                    any_emitted = True
                elif isinstance(part, ToolReturnPart):
                    tc = pending_tool_calls.pop(getattr(part, "tool_call_id", ""), None)
                    if tc is None:
                        continue
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    try:
                        emitter.emit(tc.finish(output=content[:2000]))
                    except Exception:
                        pass
        elif isinstance(msg, ModelResponse):
            stream = None
            for part in msg.parts:
                if isinstance(part, TextPart):
                    chunk = part.content
                    if not chunk or not chunk.strip():
                        continue
                    if stream is None:
                        stream = AssistantStream()
                    stream.text += chunk
                    emitter.emit(stream)
                    any_emitted = True
                elif isinstance(part, ToolCallPart):
                    args_repr = repr(part.args) if not isinstance(part.args, str) else part.args
                    tc = emitter.emit(
                        ToolCallDisplay(
                            tool_name=part.tool_name,
                            args=args_repr[:80],
                            details=args_repr[:5000],
                            tool_call_id=getattr(part, "tool_call_id", ""),
                        )
                    )
                    pending_tool_calls[getattr(part, "tool_call_id", "")] = tc
                    any_emitted = True
            if stream is not None:
                stream.done = True
                emitter.emit(stream)
