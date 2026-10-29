"""Per-model-call traces for training.

``TracingModel`` wraps the provider model and appends one JSON line per
completed model call to ``<session_dir>/traces.jsonl``. Each line holds
exactly what the model was sent — after context pruning/compression,
with the system prompt and tool schemas as they were at that call — and
what it returned, in OpenAI chat format so it renders through any
open-weight chat template.

The ``events`` table is not enough for this: it logs the raw
conversation, while the model sees a compressed working set and a
system prompt/tool list rebuilt every turn.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from .session import Session


logger = logging.getLogger(__name__)

TRACES_FILENAME = "traces.jsonl"


def _harness_version() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


HARNESS_VERSION = _harness_version()


def _content_part(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if isinstance(item, ImageUrl):
        return {"type": "image_url", "image_url": {"url": item.url}}
    if isinstance(item, BinaryContent) and item.is_image:
        return {"type": "image_url", "image_url": {"url": item.data_uri}}
    raise ValueError(f"Unsupported user content for traces: {type(item).__name__}")


def _user_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_content_part(item) for item in content]


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    text = "".join(p.content for p in response.parts if isinstance(p, TextPart))
    reasoning = "".join(p.content for p in response.parts if isinstance(p, ThinkingPart))
    tool_calls = [
        {
            "id": p.tool_call_id,
            "type": "function",
            "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
        }
        for p in response.parts
        if isinstance(p, ToolCallPart)
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def to_openai_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Map pydantic-ai messages to OpenAI chat messages, mirroring how
    ``OpenAIChatModel`` puts them on the wire."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            out.append(_assistant_message(message))
            continue
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                out.append({"role": "system", "content": part.content})
            elif isinstance(part, UserPromptPart):
                out.append({"role": "user", "content": _user_content(part.content)})
            elif isinstance(part, ToolReturnPart):
                text, extra = part.model_response_str_and_user_content()
                out.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": text})
                if extra:
                    out.append({"role": "user", "content": _user_content(extra)})
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    out.append({"role": "user", "content": part.model_response()})
                else:
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": part.model_response()})

    # Like OpenAIChatModel: the latest request's instructions lead as a system message.
    instructions = next(
        (m.instructions for m in reversed(messages) if isinstance(m, ModelRequest) and m.instructions), None
    )
    if instructions:
        out.insert(0, {"role": "system", "content": instructions})
    return out


def _tool_schemas(params: ModelRequestParameters) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters_json_schema,
            },
        }
        for tool in [*params.function_tools, *params.output_tools]
    ]


class TracingModel(WrapperModel):
    """Appends every completed model call to the live session's
    ``traces.jsonl``. ``session_fn`` is read per call because the agent
    swaps sessions on resume."""

    def __init__(self, wrapped: Model, session_fn: Callable[[], Session]):
        super().__init__(wrapped)
        self._session_fn = session_fn

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        self._write(messages, model_settings, model_request_parameters, response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as stream:
            yield stream
        # Not reached on cancel/error: a partial completion is not a sample.
        self._write(messages, model_settings, model_request_parameters, stream.get())

    def _write(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        params: ModelRequestParameters,
        response: ModelResponse,
    ) -> None:
        try:
            self._append(messages, model_settings, params, response)
        except Exception as e:
            # A trace failure must not fail the user's turn.
            logger.warning("Trace write failed: %s", e)

    def _append(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        params: ModelRequestParameters,
        response: ModelResponse,
    ) -> None:
        session = self._session_fn()
        record = {
            "session_id": session.session_id,
            "parent_session_id": session.parent_session_id,
            "timestamp": datetime.now().isoformat(),
            "harness_version": HARNESS_VERSION,
            "model": response.model_name or self.model_name,
            "settings": model_settings or {},
            "tools": _tool_schemas(params),
            "messages": to_openai_messages(messages),
            "completion": _assistant_message(response),
            "finish_reason": response.finish_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
        session.session_dir.mkdir(parents=True, exist_ok=True)
        with open(session.session_dir / TRACES_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
