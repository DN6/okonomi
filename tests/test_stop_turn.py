"""Stopping a turn (Esc / Ctrl+C) makes the agent go quiet: tool calls still
attached to the turn are stopped rather than detached into jobs that would
wake the agent later, the agent's bash command is killed, and messages the
turn never picked up go back to the input box instead of starting a turn."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from test_mid_turn_messages import _model_calling_web_fetch, _patch_web_fetch

from okonomi.agent import agent as agent_mod
from okonomi.events import EventEmitter, JobFinished
from okonomi.tools import terminal as terminal_mod
from okonomi.tui.session import TuiSession


def _cancel_once(agent, started: threading.Event) -> None:
    async def _turn_then_cancel():
        task = asyncio.create_task(agent.run_async("segment the rocks"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_turn_then_cancel())


def test_stopped_tool_call_does_not_become_a_job(okonomi_agent, monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.3)
    started = threading.Event()

    def _body():
        started.set()
        time.sleep(1.0)
        return {"ok": True}

    _patch_web_fetch(monkeypatch, okonomi_agent, _body)
    finished: list[JobFinished] = []
    okonomi_agent.emitter.subscribe(lambda ev: isinstance(ev, JobFinished) and finished.append(ev))

    with okonomi_agent.agent.override(model=_model_calling_web_fetch([])):
        _cancel_once(okonomi_agent, started)
    time.sleep(1.5)

    assert not okonomi_agent.jobs.entries, f"a stopped call must not detach into a job: {okonomi_agent.jobs.entries}"
    assert not finished, f"a stopped call must not report back and wake the agent: {finished}"


def test_stop_kills_the_agents_bash_command(okonomi_agent, monkeypatch):

    async def _stream(messages, info: AgentInfo):
        if len(messages) == 1:
            yield {1: DeltaToolCall(name="bash", json_args='{"command": "sleep 20"}', tool_call_id="b")}
        else:
            yield "done"

    model = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=_stream)
    started = threading.Event()

    class _Procs(set):
        def add(self, proc):
            super().add(proc)
            started.set()

    monkeypatch.setattr(terminal_mod, "_live_procs", _Procs())

    with okonomi_agent.agent.override(model=model):
        _cancel_once(okonomi_agent, started)

    deadline = time.monotonic() + 3
    while terminal_mod._live_procs and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not terminal_mod._live_procs, "the stopped turn's bash subprocess is still running"


class _App:
    def __init__(self):
        self.input = SimpleNamespace(text="")
        self.prefilled: list[str] = []

    def call_from_thread(self, fn, *args):
        fn(*args)

    def prefill_input(self, text):
        self.prefilled.append(text)


class _Queue(list):
    def put_nowait(self, item):
        self.append(item)


def _session_after_turn() -> SimpleNamespace:
    async def run_async(msg, reset=False):
        raise asyncio.CancelledError

    fake = SimpleNamespace(
        agent=SimpleNamespace(run_async=run_async, take_undelivered=lambda: ["use the SAM3 space instead"]),
        emitter=EventEmitter(),
        _app=_App(),
        _input_queue=_Queue(),
        _user_cancel=True,
        _steer_texts=[],
        _quiet=False,
        _safe_save_session=lambda **kw: None,
    )
    fake._restore_draft = lambda texts: TuiSession._restore_draft(fake, texts)
    asyncio.run(TuiSession._run_agent_turn(fake, "segment the rocks", reset=False))
    return fake


def test_user_stop_returns_unsent_message_to_the_input_box():
    fake = _session_after_turn()

    assert fake._input_queue == [], f"a stop must not start another turn: {fake._input_queue}"
    assert fake._app.prefilled == ["use the SAM3 space instead"], f"unsent message lost: {fake._app.prefilled}"
