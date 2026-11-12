"""The terminal UI: slash commands, skill invocation, the transcript,
steering and stopping a turn, and the tasks navigator."""

from __future__ import annotations

import asyncio
import io
import threading
import time
from types import SimpleNamespace

import pytest
from _helpers import (
    WAIT_SECONDS,
    child_agent,
    fake_agent,
    model_calling_web_fetch,
    patch_web_fetch,
    streaming_model,
    users,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall
from rich.console import Console

import okonomi.agent as agent_pkg
from okonomi.agent import agent as agent_mod
from okonomi.events import EventEmitter, SystemNote, TaskFinished, ToolCallDisplay
from okonomi.tools import skills
from okonomi.tools import terminal as terminal_mod
from okonomi.tools import viewer as viewer_mod
from okonomi.tools.agents import AgentTools
from okonomi.tui import commands
from okonomi.tui.commands import Action, handle
from okonomi.tui.pickers import _task_rows
from okonomi.tui.session import TuiSession
from okonomi.tui.transcript import ReplTranscript


def test_resume_opens_the_session_picker_with_the_query(okonomi_agent):
    result = handle("/resume", okonomi_agent, EventEmitter())
    assert result is not None and result.action is Action.FIND_PICKER, result
    assert (result.payload, result.scope) == ("", "session"), result

    result = handle("/resume fox logo", okonomi_agent, EventEmitter())
    assert result is not None and (result.payload, result.scope) == ("fox logo", "session"), result


def test_viewer_opens_the_viewer(okonomi_agent, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(viewer_mod.webbrowser, "open", lambda url: opened.append(url) or True)
    notes: list[str] = []
    emitter = EventEmitter()

    def _on_event(ev):
        if isinstance(ev, SystemNote):
            notes.append(ev.text)

    emitter.subscribe(_on_event)
    try:
        handle("/viewer", okonomi_agent, emitter)
    finally:
        if okonomi_agent.viewer._server is not None:
            okonomi_agent.viewer._server.shutdown()

    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:"), opened
    assert notes == [f"Viewer: {opened[0]}"], notes


class _StubSkills:
    def __init__(self):
        self.loaded = None

    def _load_full(self, name):
        self.loaded = name
        return {"content": f"BODY OF {name}", "name": name}


def _install_user_skill(name: str) -> tuple[str, str]:
    """Write a user skill and return its ``(slug, real_name)`` command entry."""
    skill_dir = skills.USER_SKILLS_DIR / name.lower().replace(" ", "-")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\nintro\n")
    slug, real, _desc = next(c for c in commands._skill_commands() if c[1] == name)
    return slug, real


def test_skill_slug_dispatches_with_or_without_prefix_and_forwards_the_real_name(store):
    emitter = SimpleNamespace(emit=lambda *a, **k: None)
    slug, real = _install_user_skill("FLUX LoRA Recipes")
    agent = SimpleNamespace(skills=_StubSkills())

    res = handle(f"/{slug} make it warmer", agent, emitter)

    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "loader must receive the real (spaced) name"
    assert f"BODY OF {real}" in res.payload
    assert "make it warmer" in res.payload, "trailing text must pass through as instruction"

    slug, real = _install_user_skill("demo")
    agent = SimpleNamespace(skills=_StubSkills())

    res = handle(f"/{slug.split(':', 1)[1]}", agent, emitter)

    assert res is not None and res.action is Action.FORWARD
    assert agent.skills.loaded == real, "bare slug must reach the same skill"


def _print(transcript: ReplTranscript) -> str:
    buf = io.StringIO()
    transcript.console = Console(file=buf, force_terminal=True, width=200)
    call = ToolCallDisplay(tool_name="bash", args="ls artifacts", details="ls artifacts", tool_call_id="bash:7")
    transcript.on_event(call.finish(output="fox_v1_1.png\nfox_v1_2.png\n"))
    return buf.getvalue()


def test_compact_line_links_to_a_log_or_prints_the_output_inline(tmp_path):
    printed = _print(ReplTranscript(log_dir_fn=lambda: tmp_path / "logs"))

    log = tmp_path / "logs" / "bash_7.md"
    assert "```\nls artifacts\n```" in log.read_text() and "fox_v1_2.png" in log.read_text(), log.read_text()
    assert "⏺ Ran shell command  ls artifacts" in printed, printed
    assert f";{log.as_uri()}\x1b\\" in printed, "the summary line must be an OSC 8 link to the log"
    assert "fox_v1_1.png" not in printed, "compact mode keeps the output out of the scrollback"

    assert "⎿ fox_v1_1.png" in _print(ReplTranscript()), "without a log dir the output prints inline"


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
    kinds = [j["kind"] for j in okonomi_agent.tasks.entries.values()]
    assert kinds == ["tool"], f"the tool must carry on as a background task: {kinds}"


def test_task_result_after_a_stop_waits_for_the_users_next_message():
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
    assert queued == [], f"a stopped agent must not be woken by a task: {queued}"
    assert notes, "the user must be told a result is waiting"

    TuiSession._deliver_on_loop(fake, "now crop the video")
    assert queued == ["now crop the video\n\n[system] Background tool call tool-1 finished."], queued
    assert not fake._quiet and not fake._held_notices


def _cancel_once(agent, started: threading.Event) -> None:
    async def _turn_then_cancel():
        task = asyncio.create_task(agent.run_async("segment the rocks"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_turn_then_cancel())


def test_stopped_tool_call_does_not_become_a_task(okonomi_agent, monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.3)
    started = threading.Event()

    def _body():
        started.set()
        time.sleep(1.0)
        return {"ok": True}

    patch_web_fetch(monkeypatch, okonomi_agent, _body)
    finished: list[TaskFinished] = []
    okonomi_agent.emitter.subscribe(lambda ev: isinstance(ev, TaskFinished) and finished.append(ev))

    with okonomi_agent.agent.override(model=model_calling_web_fetch([])):
        _cancel_once(okonomi_agent, started)
    time.sleep(1.5)

    assert not okonomi_agent.tasks.entries, f"a stopped call must not detach into a task: {okonomi_agent.tasks.entries}"
    assert not finished, f"a stopped call must not report back and wake the agent: {finished}"


def test_stop_kills_the_agents_bash_command(okonomi_agent, monkeypatch):
    async def _stream(messages, info: AgentInfo):
        if len(messages) == 1:
            yield {1: DeltaToolCall(name="bash", json_args='{"command": "sleep 20"}', tool_call_id="b")}
        else:
            yield "done"

    started = threading.Event()

    class _Procs(set):
        def add(self, proc):
            super().add(proc)
            started.set()

    monkeypatch.setattr(terminal_mod, "_live_procs", _Procs())

    with okonomi_agent.agent.override(model=streaming_model(_stream)):
        _cancel_once(okonomi_agent, started)

    deadline = time.monotonic() + 3
    while terminal_mod._live_procs and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not terminal_mod._live_procs, "the stopped turn's bash subprocess is still running"


def test_user_stop_returns_unsent_message_to_the_input_box():
    async def run_async(msg, reset=False):
        raise asyncio.CancelledError

    prefilled: list[str] = []
    queue: list = []
    fake = SimpleNamespace(
        agent=SimpleNamespace(run_async=run_async, take_undelivered=lambda: ["use the SAM3 space instead"]),
        emitter=EventEmitter(),
        _app=SimpleNamespace(
            input=SimpleNamespace(text=""),
            call_from_thread=lambda fn, *args: fn(*args),
            prefill_input=prefilled.append,
        ),
        _input_queue=SimpleNamespace(put_nowait=queue.append),
        _user_cancel=True,
        _steer_texts=[],
        _quiet=False,
        _safe_save_session=lambda **kw: None,
    )
    fake._restore_draft = lambda texts: TuiSession._restore_draft(fake, texts)

    asyncio.run(TuiSession._run_agent_turn(fake, "segment the rocks", reset=False))

    assert queue == [], f"a stop must not start another turn: {queue}"
    assert prefilled == ["use the SAM3 space instead"], f"unsent message lost: {prefilled}"


def test_launched_agent_exposes_its_tasks_and_closing_message(monkeypatch):
    async def run_async(self, task, reset=False):
        self.tasks.register(task_id="term-1", label="pip install", reader=lambda: "")
        self.tasks.mark_done(task_id="term-1", status="ok")
        return "segmented 3 images"

    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async))
    parent = fake_agent()
    done = threading.Event()
    parent.emitter.subscribe(lambda ev: isinstance(ev, TaskFinished) and done.set())
    tools = AgentTools()
    tools._bind(parent)
    tools.launch("segment images", description="sam3")
    assert done.wait(WAIT_SECONDS), f"no TaskFinished within {WAIT_SECONDS}s of launch"

    rows = [(r["id"], r["depth"]) for r in _task_rows(parent.tasks)]
    assert rows == [("agent-1", 1), ("term-1", 2)], f"sub-agent's task not nested: {rows}"
    detail = parent.tasks.entries["agent-1"]["_reader"]()
    assert "segmented 3 images" in detail, f"closing message missing from detail: {detail!r}"
