"""Messages sent while a turn is running are delivered into that turn at its
next model request (pydantic-ai ``AgentRun.enqueue``) instead of waiting for
the turn to end. Anything the turn never picked up becomes the next turn."""

from __future__ import annotations

import asyncio
import threading

import pytest
from _helpers import model_calling_web_fetch, patch_web_fetch, users

from okonomi.events import SystemNote


def test_message_sent_mid_turn_reaches_the_running_turn(okonomi_agent, monkeypatch):
    tool_started = threading.Event()
    release_tool = threading.Event()

    def _body():
        tool_started.set()
        release_tool.wait(5)
        return {"ok": True}

    patch_web_fetch(monkeypatch, okonomi_agent, _body)
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

    with okonomi_agent.agent.override(model=model_calling_web_fetch(seen_prompts)):
        reply = asyncio.run(_turn())

    assert seen_prompts[-1] == ["render the intro", "also make it 4k"], f"model never saw the message: {seen_prompts}"
    assert reply == "reply to 'also make it 4k'", reply
    assert okonomi_agent.take_undelivered() == [], "a delivered message must not be requeued"
    assert "also make it 4k" in users(okonomi_agent.state.message_history), "delivered message must be in history"
    assert any("running turn" in n for n in notes), f"pickup not shown in transcript: {notes}"


def test_message_not_picked_up_before_cancel_is_returned(okonomi_agent, monkeypatch):
    tool_started = threading.Event()
    release_tool = threading.Event()

    def _body():
        tool_started.set()
        release_tool.wait(5)
        return {"ok": True}

    patch_web_fetch(monkeypatch, okonomi_agent, _body)

    async def _turn_then_cancel():
        task = asyncio.create_task(okonomi_agent.run_async("render the intro"))
        while not tool_started.is_set():
            await asyncio.sleep(0.01)
        assert okonomi_agent.enqueue("also make it 4k")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_tool.set()

    with okonomi_agent.agent.override(model=model_calling_web_fetch([])):
        asyncio.run(_turn_then_cancel())

    assert okonomi_agent.take_undelivered() == ["also make it 4k"], "undelivered message must be handed back"
    assert okonomi_agent.take_undelivered() == [], "take_undelivered must clear"
