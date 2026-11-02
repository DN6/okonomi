"""Steering: a message typed while the model is mid-reply cuts in at once
(the reply is cut off and the model sees what it had said), job results
wait out a user stop."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from test_mid_turn_messages import _users

from okonomi.events import EventEmitter, SystemNote
from okonomi.tui.session import TuiSession


def _wait(cond, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_message_typed_mid_reply_cuts_in_with_the_partial_reply(okonomi_agent):
    requests: list[tuple[float, list[str]]] = []

    async def _stream(messages, info: AgentInfo):
        requests.append((time.monotonic(), _users(messages)))
        if len(requests) == 1:
            yield "Once upon a time, "
            for _ in range(100):
                await asyncio.sleep(0.05)
                yield "and then "
        else:
            yield "Rocks are aggregates of minerals."

    okonomi_agent.agent.model = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=_stream)
    session = TuiSession(okonomi_agent)
    try:
        session._on_submit("tell me a story")
        assert _wait(lambda: okonomi_agent.streaming_text), "model never started streaming"
        steered_at = time.monotonic()
        session._on_submit("actually, tell me about rocks")
        assert _wait(lambda: len(requests) >= 2), "the steer never reached the model"
        assert _wait(lambda: not session._agent_running and session._input_queue.empty())
    finally:
        session._submit(None)
        session._worker_loop_future.result(timeout=2)
        session._loop.call_soon_threadsafe(session._loop.stop)

    at, prompt = requests[1]
    assert at - steered_at < 1.0, f"steer waited {at - steered_at:.1f}s for the reply to finish"
    assert "actually, tell me about rocks" in prompt[-1], prompt
    assert "Once upon a time" in prompt[-1], f"the model must see where it was cut off: {prompt[-1]!r}"
    assert len(requests) == 2, f"the story must not resume after the steer: {len(requests)} requests"


def _loop_session(**kw) -> SimpleNamespace:
    queued: list[str] = []
    notes: list[str] = []
    emitter = EventEmitter()
    emitter.subscribe(lambda ev: isinstance(ev, SystemNote) and notes.append(ev.text))
    fake = SimpleNamespace(
        _quiet=False,
        _held_notices=[],
        _steer_texts=[],
        _agent_running=False,
        agent=SimpleNamespace(enqueue=lambda t: False, streaming_text=False, status=""),
        _input_queue=SimpleNamespace(put_nowait=queued.append),
        emitter=emitter,
        queued=queued,
        notes=notes,
    )
    fake.__dict__.update(kw)
    return fake


def test_job_result_after_a_stop_waits_for_the_users_next_message():
    fake = _loop_session(_quiet=True)

    TuiSession._deliver_notice(fake, "[system] Background tool call tool-1 finished.")
    assert fake.queued == [], f"a stopped agent must not be woken by a job: {fake.queued}"
    assert fake.notes, "the user must be told a result is waiting"

    TuiSession._deliver_on_loop(fake, "now crop the video")
    assert fake.queued == ["now crop the video\n\n[system] Background tool call tool-1 finished."], fake.queued
    assert not fake._quiet and not fake._held_notices


def test_message_typed_mid_tool_reaches_the_model_without_waiting(okonomi_agent, monkeypatch):
    from test_mid_turn_messages import _patch_web_fetch

    started = threading.Event()

    def _body():
        started.set()
        time.sleep(3)
        return {"ok": True}

    _patch_web_fetch(monkeypatch, okonomi_agent, _body)
    requests: list[tuple[float, list[str]]] = []

    async def _stream(messages, info: AgentInfo):
        requests.append((time.monotonic(), _users(messages)))
        if len(requests) == 1:
            yield {1: DeltaToolCall(name="web_fetch", json_args='{"url": "u"}', tool_call_id="w")}
        else:
            yield "ok"

    okonomi_agent.agent.model = FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=_stream)
    session = TuiSession(okonomi_agent)
    try:
        session._on_submit("fetch the page")
        assert started.wait(5), "tool never started"
        steered_at = time.monotonic()
        session._on_submit("use the mirror instead")
        assert _wait(lambda: len(requests) >= 2, 2), "the message waited on the tool"
    finally:
        session._submit(None)
        session._worker_loop_future.result(timeout=5)
        session._loop.call_soon_threadsafe(session._loop.stop)

    at, prompt = requests[1]
    assert at - steered_at < 1.0, f"message reached the model {at - steered_at:.1f}s later"
    assert "use the mirror instead" in prompt, prompt
    kinds = [j["kind"] for j in okonomi_agent.jobs.entries.values()]
    assert kinds == ["tool"], f"the tool must carry on as a background job: {kinds}"
