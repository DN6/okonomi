"""Steering: a message typed while the model is mid-reply cuts in at once
(the reply is cut off and the model sees what it had said), job results
wait out a user stop."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from _helpers import patch_web_fetch, streaming_model, users
from pydantic_ai.models.function import AgentInfo, DeltaToolCall

from okonomi.events import EventEmitter, SystemNote
from okonomi.tui.session import TuiSession


def _wait(cond, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _drive(session: TuiSession, first: str, steer: str, started) -> float:
    """Submit ``first``, wait for ``started()``, submit ``steer``; returns
    the time of the steer. Always tears the session's loop down."""
    try:
        session._on_submit(first)
        assert _wait(started), "the first turn never got going"
        steered_at = time.monotonic()
        session._on_submit(steer)
        assert _wait(lambda: not session._agent_running and session._input_queue.empty())
        return steered_at
    finally:
        session._submit(None)
        session._worker_loop_future.result(timeout=5)
        session._loop.call_soon_threadsafe(session._loop.stop)


def test_message_typed_mid_reply_cuts_in_with_the_partial_reply(okonomi_agent):
    requests: list[tuple[float, list[str]]] = []

    async def _stream(messages, info: AgentInfo):
        requests.append((time.monotonic(), users(messages)))
        if len(requests) == 1:
            yield "Once upon a time, "
            for _ in range(100):
                await asyncio.sleep(0.05)
                yield "and then "
        else:
            yield "Rocks are aggregates of minerals."

    okonomi_agent.agent.model = streaming_model(_stream)
    session = TuiSession(okonomi_agent)

    steered_at = _drive(
        session, "tell me a story", "actually, tell me about rocks", lambda: okonomi_agent.streaming_text
    )

    at, prompt = requests[1]
    assert at - steered_at < 1.0, f"steer waited {at - steered_at:.1f}s for the reply to finish"
    assert "actually, tell me about rocks" in prompt[-1], prompt
    assert "Once upon a time" in prompt[-1], f"the model must see where it was cut off: {prompt[-1]!r}"
    assert len(requests) == 2, f"the story must not resume after the steer: {len(requests)} requests"


def test_message_typed_mid_tool_reaches_the_model_without_waiting(okonomi_agent, monkeypatch):
    started = threading.Event()

    def _body():
        started.set()
        time.sleep(3)
        return {"ok": True}

    patch_web_fetch(monkeypatch, okonomi_agent, _body)
    requests: list[tuple[float, list[str]]] = []

    async def _stream(messages, info: AgentInfo):
        requests.append((time.monotonic(), users(messages)))
        if len(requests) == 1:
            yield {1: DeltaToolCall(name="web_fetch", json_args='{"url": "u"}', tool_call_id="w")}
        else:
            yield "ok"

    okonomi_agent.agent.model = streaming_model(_stream)
    session = TuiSession(okonomi_agent)

    steered_at = _drive(session, "fetch the page", "use the mirror instead", started.is_set)

    at, prompt = requests[1]
    assert at - steered_at < 1.0, f"message reached the model {at - steered_at:.1f}s later"
    assert "use the mirror instead" in prompt, prompt
    kinds = [j["kind"] for j in okonomi_agent.jobs.entries.values()]
    assert kinds == ["tool"], f"the tool must carry on as a background job: {kinds}"


def test_job_result_after_a_stop_waits_for_the_users_next_message():
    queued: list[str] = []
    notes: list[str] = []
    emitter = EventEmitter()
    emitter.subscribe(lambda ev: isinstance(ev, SystemNote) and notes.append(ev.text))
    fake = SimpleNamespace(
        _quiet=True,
        _held_notices=[],
        _steer_texts=[],
        _agent_running=False,
        agent=SimpleNamespace(enqueue=lambda t: False, streaming_text=False, status=""),
        _input_queue=SimpleNamespace(put_nowait=queued.append),
        emitter=emitter,
    )

    TuiSession._deliver_notice(fake, "[system] Background tool call tool-1 finished.")
    assert queued == [], f"a stopped agent must not be woken by a job: {queued}"
    assert notes, "the user must be told a result is waiting"

    TuiSession._deliver_on_loop(fake, "now crop the video")
    assert queued == ["now crop the video\n\n[system] Background tool call tool-1 finished."], queued
    assert not fake._quiet and not fake._held_notices
