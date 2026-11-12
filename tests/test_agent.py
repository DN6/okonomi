"""The agent loop: tool-call widgets, detached calls, launched sub-agents
and their notices, and messages that arrive while a turn runs."""

from __future__ import annotations

import asyncio
import json
import sys
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
    write_mcp_server,
)
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

import okonomi.agent as agent_pkg
from okonomi import cli, mcp
from okonomi.agent import agent as agent_mod
from okonomi.agent.agent import OkonomiAgent
from okonomi.agent.tasks import TaskRegistry
from okonomi.events import AssistantStream, EventEmitter, SystemNote, TaskFinished, ToolCallDisplay
from okonomi.tools import terminal as terminal_mod
from okonomi.tools.agents import AgentTools
from okonomi.tui.session import TuiSession
from okonomi.tui.transcript import replay_history


@pytest.fixture
def mcp_agent(okonomi_agent, tmp_path):
    """The agent with the demo MCP server's tools registered."""
    assert mcp.add_server("demo", sys.executable, [str(write_mcp_server(tmp_path / "srv.py"))], {})["success"]
    okonomi_agent.reload()
    return okonomi_agent


def _scripted_model(calls: list[ToolCallPart]) -> FunctionModel:
    """First response: some text plus ``calls``; second response: 'done'."""

    async def _stream(messages, info: AgentInfo):
        if len(messages) == 1:
            yield "working"
            yield {
                i: DeltaToolCall(name=c.tool_name, json_args=c.args_as_json_str(), tool_call_id=c.tool_call_id)
                for i, c in enumerate(calls, start=1)
            }
        else:
            yield "done"

    return streaming_model(_stream)


def _capture(agent: OkonomiAgent) -> list:
    seen: list = []
    agent.emitter.subscribe(seen.append)
    return seen


def _widgets(seen: list) -> dict[str, ToolCallDisplay]:
    return {ev.tool_call_id: ev for ev in seen if isinstance(ev, ToolCallDisplay)}


def test_function_and_mcp_tools_finish_by_id(mcp_agent):
    seen = _capture(mcp_agent)
    calls = [
        ToolCallPart("demo_shout", {"text": "hi"}, tool_call_id="mcp-1"),
        ToolCallPart("load_skill", {"name": "no-such-skill"}, tool_call_id="fn-1"),
        ToolCallPart("list_skills", {}, tool_call_id="fn-2"),
    ]
    with mcp_agent.agent.override(model=_scripted_model(calls)):
        mcp_agent.run("go")

    widgets = _widgets(seen)
    assert set(widgets) == {"mcp-1", "fn-1", "fn-2"}, f"expected one widget per call, got {list(widgets)}"
    for call_id, w in widgets.items():
        assert w.status != "running", f"{call_id} left running: {w!r}"
    assert "HI" in widgets["mcp-1"].output, f"MCP result not shown: {widgets['mcp-1'].output!r}"
    assert widgets["fn-1"].status == "error" and "no-such-skill" in widgets["fn-1"].output, (
        "error dict must surface as an error"
    )
    assert widgets["fn-1"].args == "no-such-skill", "load_skill shows just the skill name"
    assert "skills" in widgets["fn-2"].output

    text = "".join(ev.text for ev in seen if isinstance(ev, AssistantStream) and ev.done)
    assert "working" in text and "done" in text, f"streamed text missing: {text!r}"


def test_cancelled_turn_closes_running_widget(okonomi_agent, monkeypatch):
    def slow(self, url: str) -> dict:
        """Sleeps."""
        time.sleep(1.0)
        return {"ok": True}

    monkeypatch.setattr(type(okonomi_agent.web), "web_fetch", slow)
    okonomi_agent.reload()
    seen = _capture(okonomi_agent)

    async def _run_and_cancel():
        task = asyncio.create_task(okonomi_agent.run_async("go"))
        while not _widgets(seen):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with okonomi_agent.agent.override(
        model=_scripted_model([ToolCallPart("web_fetch", {"url": "u"}, tool_call_id="c")])
    ):
        asyncio.run(_run_and_cancel())

    w = _widgets(seen)["c"]
    assert w.status == "error" and "interrupted" in w.output, f"cancelled widget left as {w.status!r}: {w.output!r}"
    assert not okonomi_agent._open_tool_calls, "open widgets must not leak into the next turn"


def _model_calling_each_turn(tool_name: str, args: dict) -> FunctionModel:
    """Calls ``tool_name`` once per user turn, then replies 'done'."""
    counter = iter(range(1000))

    async def _stream(messages, info: AgentInfo):
        if any(isinstance(p, UserPromptPart) for p in messages[-1].parts):
            yield {1: DeltaToolCall(name=tool_name, json_args=json.dumps(args), tool_call_id=f"t{next(counter)}")}
        else:
            yield "done"

    return streaming_model(_stream)


def test_slow_mcp_call_moves_to_background(mcp_agent, monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.3)
    seen = _capture(mcp_agent)

    async def _turn_then_wait_for_notice() -> TaskFinished:
        loop = asyncio.get_running_loop()
        finished: asyncio.Future[TaskFinished] = loop.create_future()
        mcp_agent.emitter.subscribe(
            lambda ev: isinstance(ev, TaskFinished) and loop.call_soon_threadsafe(finished.set_result, ev)
        )
        await mcp_agent.run_async("go")
        try:
            return await asyncio.wait_for(finished, 5)
        finally:
            await mcp_agent.close_mcp()

    with mcp_agent.agent.override(model=_model_calling_each_turn("demo_slow", {"seconds": 1.0})):
        started = time.monotonic()
        ev = asyncio.run(_turn_then_wait_for_notice())

    w = next(iter(_widgets(seen).values()))
    assert "moved to background as tool-" in w.output, f"turn must not wait on the MCP call: {w.output!r}"
    assert ev.kind == "tool" and ev.status == "ok", f"unexpected notice {ev!r}"
    assert "slept 1.0" in (ev.output or ""), f"MCP result must ride on the notice: {ev.output!r}"
    assert time.monotonic() - started < 5


class _Tools:
    def slow(self, seconds: float) -> dict:
        """Sleep then return."""
        time.sleep(seconds)
        return {"slept": seconds}


def _model_calling(tool_name: str, args: dict):
    """A FunctionModel that calls ``tool_name`` once, then replies 'done'."""

    def _fn(messages, info: AgentInfo):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(_fn)


def _tool_return(messages) -> ToolReturnPart:
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    return p
    raise AssertionError("no ToolReturnPart in run messages")


def test_slow_tool_becomes_a_task_and_reports_via_task_finished(monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.2)
    fake = fake_agent()
    finished: list[TaskFinished] = []
    done = threading.Event()

    def _on_event(ev):
        if isinstance(ev, TaskFinished):
            finished.append(ev)
            done.set()

    fake.emitter.subscribe(_on_event)
    agent = Agent(_model_calling("slow", {"seconds": 0.6}))
    OkonomiAgent._register_tool(fake, agent, _Tools())

    part = _tool_return(agent.run_sync("go").all_messages())
    assert done.wait(3), "detached tool never reported"
    (ev,) = finished

    assert isinstance(part.content, dict) and part.content.get("pending") == ev.task_id, (
        f"placeholder naming the task expected, got {part.content!r}"
    )
    assert ev.kind == "tool" and ev.status == "ok", f"unexpected notice {ev!r}"
    assert "slept" in (ev.output or ""), f"tool result must ride on the notice, got {ev.output!r}"
    assert fake.tasks.entries[ev.task_id]["status"] == "ok", "task must be marked done before the notice fires"


def test_kill_all_processes_stops_background_commands():
    """Shutdown kills every live bash subprocess — foreground-in-grace,
    detached, and term-N alike — so none outlive the REPL."""
    t = terminal_mod.TerminalTools()
    results: list[dict] = []
    worker = threading.Thread(target=lambda: results.append(t.bash("sleep 30", timeout=60)), daemon=True)
    worker.start()
    t._on_bg_complete = lambda **kw: None
    t.bash("sleep 30", background=True)

    deadline = time.monotonic() + 3
    while len(terminal_mod._live_procs) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(terminal_mod._live_procs) == 2, f"expected 2 live procs, got {len(terminal_mod._live_procs)}"

    killed = terminal_mod.kill_all_processes()
    assert killed == 2, f"killed {killed}"
    worker.join(timeout=5)
    assert results and results[0]["exit_code"] != 0, f"foreground command should have died: {results}"
    deadline = time.monotonic() + 3
    while terminal_mod._live_procs and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not terminal_mod._live_procs, "procs still registered after kill"


def test_replay_closes_dangling_tool_call():
    """A persisted ToolCallPart with no return (cancelled turn) must not
    replay as a tool that is still running."""
    history = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[ToolCallPart(tool_name="bash", args={"command": "sleep 120"}, tool_call_id="dangling")]),
    ]
    seen: list = []
    em = EventEmitter()
    em.subscribe(seen.append)
    replay_history(history, em)
    calls = [ev for ev in seen if isinstance(ev, ToolCallDisplay)]
    assert calls, "tool call was not replayed"
    assert calls[-1].status != "running", f"dangling call left running: {calls[-1].status!r}"


def test_message_sent_mid_turn_reaches_the_turn_or_comes_back_if_cancelled(okonomi_agent, monkeypatch):
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

    tool_started.clear()
    release_tool.clear()

    async def _turn_then_cancel():
        task = asyncio.create_task(okonomi_agent.run_async("render the intro", reset=True))
        while not tool_started.is_set():
            await asyncio.sleep(0.01)
        assert okonomi_agent.enqueue("now in 8k")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_tool.set()

    with okonomi_agent.agent.override(model=model_calling_web_fetch([])):
        asyncio.run(_turn_then_cancel())

    assert okonomi_agent.take_undelivered() == ["now in 8k"], "undelivered message must be handed back"
    assert okonomi_agent.take_undelivered() == [], "take_undelivered must clear"


def _launch_and_wait(monkeypatch, run_async) -> TaskFinished:
    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async))
    parent = fake_agent()
    finished: list[TaskFinished] = []
    done = threading.Event()

    def _on_event(ev):
        if isinstance(ev, TaskFinished):
            finished.append(ev)
            done.set()

    parent.emitter.subscribe(_on_event)
    tools = AgentTools()
    tools._bind(parent)
    tools.launch("research noir films", description="noir/films")

    assert done.wait(WAIT_SECONDS), f"no TaskFinished within {WAIT_SECONDS}s of launch"
    assert parent.tasks.entries["agent-1"]["status"] != "running", "task must be marked done before the notice fires"
    return finished[0]


def _launch_two_and_finish(tasks: TaskRegistry) -> None:
    """Stand-in for the model launching two agents mid-turn."""
    for task_id in ("agent-1", "agent-2"):
        tasks.register(task_id=task_id, label=task_id, reader=lambda: "")

    def _finish():
        for task_id in ("agent-1", "agent-2"):
            # Same order as AgentTools.launch: marked done, then announced.
            tasks.mark_done(task_id=task_id, status="ok")
            tasks.finished(bg_id=task_id, output=f"{task_id} brief")

    threading.Thread(target=_finish, daemon=True).start()


def test_launch_pushes_closing_message_or_failure_after_its_own_children(monkeypatch):
    turns: list[str] = []

    async def run_async(self, message, reset=False):
        turns.append(message)
        if len(turns) == 1:
            _launch_two_and_finish(self.tasks)
        return f"reply {len(turns)}"

    ev = _launch_and_wait(monkeypatch, run_async)

    assert len(turns) == 3, f"expected the first turn plus one per agent, got {turns!r}"
    assert "agent-1 brief" in turns[1] and "agent-2 brief" in turns[2], turns
    assert (ev.task_id, ev.kind, ev.status) == ("agent-1", "agent", "ok"), ev
    assert ev.output == "reply 3", f"parent must get the reply to the last notice, got {ev.output!r}"

    async def failing(self, task, reset=False):
        raise RuntimeError("provider down")

    ev = _launch_and_wait(monkeypatch, failing)

    assert (ev.kind, ev.status, ev.error) == ("agent", "error", "provider down"), ev


def _submitted(ev: TaskFinished) -> str:
    submitted: list[str] = []
    loop = SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))
    fake = SimpleNamespace(_loop=loop, _quiet=False, _agent_running=False, _input_queue=SimpleNamespace(put_nowait=submitted.append))
    fake._deliver_notice = lambda notice: TuiSession._deliver_notice(fake, notice)
    TuiSession._on_task_finished(fake, ev)
    assert len(submitted) == 1, f"expected one injected turn, got {len(submitted)}"
    return submitted[0]


def test_tui_injects_the_notice_with_its_output():
    msg = _submitted(TaskFinished(task_id="agent-1", kind="agent", output="brief saved"))
    assert "Background agent agent-1 finished" in msg, msg
    assert "Its closing message:\nbrief saved" in msg, msg

    msg = _submitted(TaskFinished(task_id="term-1", kind="term", output="frame 120/120\nwrote out.mp4"))
    assert "frame 120/120\nwrote out.mp4" in msg, f"the command's output must reach the model: {msg}"


def _run_one_shot(on_first_turn) -> list[str]:
    """Drive ``cli._run_single_task`` over the slice of OkonomiAgent it touches."""
    emitter = EventEmitter()
    turns: list[str] = []
    fake = SimpleNamespace(emitter=emitter, tasks=TaskRegistry(emitter), save_session=lambda **kw: None)

    async def run_async(message, reset=False):
        turns.append(message)
        if len(turns) == 1:
            on_first_turn(fake.tasks)
        return f"reply {len(turns)}"

    fake.run_async = run_async
    asyncio.run(asyncio.wait_for(cli._run_single_task(fake, "go"), WAIT_SECONDS))
    return turns


def test_one_shot_takes_a_turn_per_launched_agent_but_not_per_shell_task():
    turns = _run_one_shot(_launch_two_and_finish)
    assert len(turns) == 3, f"expected the first turn plus one per agent, got {turns!r}"
    assert "agent-1 brief" in turns[1] and "agent-2 brief" in turns[2], turns

    turns = _run_one_shot(lambda tasks: tasks.register(task_id="term-1", label="sleep", reader=lambda: ""))
    assert turns == ["go"], f"a running shell task must not hold the run open, got {turns!r}"


PARENT_MODEL = "big-model"


def test_sub_agent_model_defaults_to_parent_and_must_be_listed(monkeypatch):
    built: list[str] = []
    done = threading.Event()

    async def run_async(self, message, reset=False):
        return "done"

    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async, built))
    tools = AgentTools()
    tools._bind(fake_agent(model_id=PARENT_MODEL, _task_finished=lambda **kw: done.set()))
    monkeypatch.setenv("OKONOMI_SUBAGENT_MODELS", "small-model, other-model")

    tools.launch("task")
    assert done.wait(WAIT_SECONDS), f"sub-agent did not finish within {WAIT_SECONDS}s"
    assert built == [PARENT_MODEL], f"expected the parent's model, got {built!r}"

    done.clear()
    reply = tools.launch("task", model="other-model")
    assert done.wait(WAIT_SECONDS), f"sub-agent did not finish within {WAIT_SECONDS}s"
    assert built == [PARENT_MODEL, "other-model"], f"expected the requested model, got {built!r}"
    assert "on other-model" in reply, reply

    reply = tools.launch("task", model="made-up-model")
    assert reply.startswith("Error: model 'made-up-model' is not available."), reply
    assert "small-model" in reply, f"error must list the valid models: {reply!r}"
    assert len(tools._tasks) == 2 and len(built) == 2, "a refused launch must neither register nor build a sub-agent"
