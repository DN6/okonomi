"""Messages sent while a turn is running are delivered into that turn at its
next model request (pydantic-ai ``AgentRun.enqueue``) instead of waiting for
the turn to end. Anything the turn never picked up becomes the next turn."""

from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from okonomi import db, mcp
from okonomi import session as session_mod
from okonomi.agent.agent import OkonomiAgent
from okonomi.events import SystemNote


@pytest.fixture
def okonomi_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OKONOMI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(db, "_connection", db.open_db(tmp_path / "test.db"))
    monkeypatch.setattr(session_mod, "OKONOMI_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mcp, "OKONOMI_MCP_CONFIG", tmp_path / "mcp.json")
    return OkonomiAgent()


def _users(messages) -> list[str]:
    return [
        p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
    ]


def _model_calling_web_fetch(seen_prompts: list[list[str]]) -> FunctionModel:
    """Calls ``web_fetch`` once, then replies to the latest user prompt."""

    async def _stream(messages, info: AgentInfo):
        seen_prompts.append(_users(messages))
        if len(messages) == 1:
            yield {1: DeltaToolCall(name="web_fetch", json_args='{"url": "u"}', tool_call_id="w")}
        else:
            yield f"reply to {_users(messages)[-1]!r}"

    return FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=_stream)


def _patch_web_fetch(monkeypatch, agent: OkonomiAgent, body) -> None:
    def web_fetch(self, url: str) -> dict:
        """Patched."""
        return body()

    monkeypatch.setattr(type(agent.web), "web_fetch", web_fetch)
    agent.reload()


def test_message_sent_mid_turn_reaches_the_running_turn(okonomi_agent, monkeypatch):
    tool_started = threading.Event()
    release_tool = threading.Event()

    def _body():
        tool_started.set()
        release_tool.wait(5)
        return {"ok": True}

    _patch_web_fetch(monkeypatch, okonomi_agent, _body)
    notes: list[str] = []
    okonomi_agent.emitter.subscribe(lambda ev: isinstance(ev, SystemNote) and notes.append(ev.text))
    seen_prompts: list[list[str]] = []

    async def _turn():
        task = asyncio.create_task(okonomi_agent.run_async("render the intro"))
        while not tool_started.is_set():
            await asyncio.sleep(0.01)
        assert okonomi_agent.enqueue("also make it 4k"), "a running turn must accept the message"
        release_tool.set()
        return await task

    with okonomi_agent.agent.override(model=_model_calling_web_fetch(seen_prompts)):
        reply = asyncio.run(_turn())

    assert seen_prompts[-1] == ["render the intro", "also make it 4k"], f"model never saw the message: {seen_prompts}"
    assert reply == "reply to 'also make it 4k'", reply
    assert okonomi_agent.take_undelivered() == [], "a delivered message must not be requeued"
    assert "also make it 4k" in _users(okonomi_agent.state.message_history), "delivered message must be in history"
    assert any("running turn" in n for n in notes), f"pickup not shown in transcript: {notes}"


def test_message_not_picked_up_before_cancel_is_returned(okonomi_agent, monkeypatch):
    tool_started = threading.Event()
    release_tool = threading.Event()

    def _body():
        tool_started.set()
        release_tool.wait(5)
        return {"ok": True}

    _patch_web_fetch(monkeypatch, okonomi_agent, _body)

    async def _turn_then_cancel():
        task = asyncio.create_task(okonomi_agent.run_async("render the intro"))
        while not tool_started.is_set():
            await asyncio.sleep(0.01)
        assert okonomi_agent.enqueue("also make it 4k")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_tool.set()

    with okonomi_agent.agent.override(model=_model_calling_web_fetch([])):
        asyncio.run(_turn_then_cancel())

    assert okonomi_agent.take_undelivered() == ["also make it 4k"], "undelivered message must be handed back"
    assert okonomi_agent.take_undelivered() == [], "take_undelivered must clear"
