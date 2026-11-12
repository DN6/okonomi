"""Shared test doubles: the slice of ``OkonomiAgent`` that tool wrappers,
``AgentTools.launch`` and the task registry touch, a stub sub-agent class,
and small pydantic-ai model builders."""

from __future__ import annotations

import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from okonomi.agent.agent import OkonomiAgent
from okonomi.agent.tasks import TaskRegistry
from okonomi.events import EventEmitter


WAIT_SECONDS = 5.0

_MCP_DEMO_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP
    m = FastMCP("demo")

    @m.tool()
    def shout(text: str) -> str:
        return text.upper()

    @m.tool()
    async def slow(seconds: float) -> str:
        import asyncio
        await asyncio.sleep(seconds)
        return f"slept {seconds}"

    m.run()
    """
)


def write_mcp_server(path: Path) -> Path:
    """A stdio MCP server with a ``shout`` tool and a ``slow`` tool."""
    path.write_text(_MCP_DEMO_SERVER)
    return path


def fake_agent(**overrides) -> SimpleNamespace:
    """The attributes of ``OkonomiAgent`` that ``_register_tool``,
    ``_detach_tool_call``, ``_task_finished`` and ``AgentTools.launch`` read,
    without a provider or model. Real methods are bound onto it so the
    code under test runs unchanged; ``overrides`` replace any of them."""
    emitter = EventEmitter()
    fake = SimpleNamespace(
        model_id="test-model",
        timeout=0,
        compress_at_tokens=0,
        session=SimpleNamespace(session_id="fake-session"),
        state=SimpleNamespace(artifacts=[]),
        emitter=emitter,
        tasks=TaskRegistry(emitter),
        _tool_pool=ThreadPoolExecutor(),
        _sync_session_artifacts=lambda deps, tool_call_id="": 0,
        _get_dependencies=lambda: None,
        _turn_cancelled=threading.Event(),
        _detach_now=threading.Event(),
        _task_tool_call_ids={},
        _session_approvals=set(),
        can_prompt=False,
    )
    bound = ("_detach_tool_call", "_task_finished", "_register_bg_log", "_mark_bg_log_done", "_unapproved_patterns", "_ask_approval")
    for name in bound:
        setattr(fake, name, partial(getattr(OkonomiAgent, name), fake))
    fake.__dict__.update(overrides)
    return fake


def child_agent(run_async, built: list[str] | None = None):
    """A stand-in for the ``OkonomiAgent`` class that ``AgentTools.launch``
    constructs. ``run_async(self, message, reset=False)`` is the child's
    turn; ``built`` collects the model id each child was created with."""

    class _Child:
        def __init__(self, **kwargs):
            self.model_id = kwargs["model_id"]
            if built is not None:
                built.append(self.model_id)
            self.state = SimpleNamespace(artifacts=[])
            self.last_turn_error = None
            self.emitter = EventEmitter()
            self.tasks = TaskRegistry(self.emitter)

        async def run_async(self, message, reset=False):
            return await run_async(self, message, reset=reset)

    return _Child


def users(messages) -> list:
    """Every user prompt's content in a pydantic-ai message list, in order."""
    return [
        p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
    ]


def streaming_model(stream) -> FunctionModel:
    """A ``FunctionModel`` driven only by ``stream(messages, info)``."""
    return FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=stream)


def model_calling_web_fetch(seen_prompts: list[list[str]]) -> FunctionModel:
    """Calls ``web_fetch`` once, then replies to the latest user prompt."""

    async def _stream(messages, info: AgentInfo):
        seen_prompts.append(users(messages))
        if len(messages) == 1:
            yield {1: DeltaToolCall(name="web_fetch", json_args='{"url": "u"}', tool_call_id="w")}
        else:
            yield f"reply to {users(messages)[-1]!r}"

    return streaming_model(_stream)


def patch_web_fetch(monkeypatch, agent: OkonomiAgent, body) -> None:
    """Replace the agent's ``web_fetch`` tool with ``body()`` and rebuild."""

    def web_fetch(self, url: str) -> dict:
        """Patched."""
        return body()

    monkeypatch.setattr(type(agent.web), "web_fetch", web_fetch)
    agent.reload()
