"""Detached tool calls: a tool that outlives ``TOOL_DETACH_AFTER`` becomes a
``tool-N`` background job. The model gets a short placeholder so the turn
can end, and the real result arrives later as a ``JobFinished`` notice, the
same path as every other background job. History is never rewritten."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from okonomi.agent import agent as agent_mod
from okonomi.agent.agent import OkonomiAgent
from okonomi.agent.jobs import JobRegistry
from okonomi.events import EventEmitter, JobFinished


class _Tools:
    def slow(self, seconds: float) -> dict:
        """Sleep then return."""
        time.sleep(seconds)
        return {"slept": seconds}

    def slow_fail(self, seconds: float) -> dict:
        """Sleep then raise."""
        time.sleep(seconds)
        raise RuntimeError("render crashed")


def _fake_agent_self() -> SimpleNamespace:
    """The slice of OkonomiAgent that ``_register_tool`` and
    ``_detach_tool_call`` touch, without a provider or model."""
    emitter = EventEmitter()
    fake = SimpleNamespace(
        _tool_pool=ThreadPoolExecutor(),
        emitter=emitter,
        jobs=JobRegistry(emitter),
        _sync_session_assets=lambda deps: 0,
        _get_dependencies=lambda: None,
        _turn_cancelled=threading.Event(),
        _detach_now=threading.Event(),
    )
    fake._detach_tool_call = lambda *a, **k: OkonomiAgent._detach_tool_call(fake, *a, **k)
    fake._job_finished = lambda **kw: OkonomiAgent._job_finished(fake, **kw)
    return fake


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


def _run_and_wait_for_notice(fake, tool_name: str, args: dict) -> tuple[ToolReturnPart, JobFinished]:
    finished: list[JobFinished] = []
    done = threading.Event()

    def _on_event(ev):
        if isinstance(ev, JobFinished):
            finished.append(ev)
            done.set()

    fake.emitter.subscribe(_on_event)
    agent = Agent(_model_calling(tool_name, args))
    OkonomiAgent._register_tool(fake, agent, _Tools())
    part = _tool_return(agent.run_sync("go").all_messages())
    assert done.wait(3), "detached tool never reported"
    return part, finished[0]


def test_slow_tool_becomes_a_job_and_reports_via_job_finished(monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.2)
    fake = _fake_agent_self()

    part, ev = _run_and_wait_for_notice(fake, "slow", {"seconds": 0.6})

    assert isinstance(part.content, dict) and part.content.get("pending") == ev.job_id, (
        f"placeholder naming the job expected, got {part.content!r}"
    )
    assert ev.kind == "tool" and ev.status == "ok", f"unexpected notice {ev!r}"
    assert "slept" in (ev.output or ""), f"tool result must ride on the notice, got {ev.output!r}"
    assert fake.jobs.entries[ev.job_id]["status"] == "ok", "job must be marked done before the notice fires"
    assert part.content.get("pending") == ev.job_id, "the placeholder in history must not be rewritten"


def test_kill_all_processes_stops_background_commands():
    """Shutdown kills every live bash subprocess — foreground-in-grace,
    detached, and term-N alike — so none outlive the REPL."""
    from okonomi.tools import terminal as term_mod

    t = term_mod.TerminalTools()
    results: list[dict] = []
    worker = __import__("threading").Thread(target=lambda: results.append(t.bash("sleep 30", timeout=60)), daemon=True)
    worker.start()
    t._on_bg_complete = lambda **kw: None
    t.bash("sleep 30", background=True)

    deadline = time.monotonic() + 3
    while len(term_mod._live_procs) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(term_mod._live_procs) == 2, f"expected 2 live procs, got {len(term_mod._live_procs)}"

    killed = term_mod.kill_all_processes()
    assert killed == 2, f"killed {killed}"
    worker.join(timeout=5)
    assert results and results[0]["exit_code"] != 0, f"foreground command should have died: {results}"
    deadline = time.monotonic() + 3
    while term_mod._live_procs and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not term_mod._live_procs, "procs still registered after kill"


def test_replay_closes_dangling_tool_call():
    """A persisted ToolCallPart with no return (cancelled turn) must not
    replay as a tool that is still running."""
    from okonomi.events import ToolCallDisplay
    from okonomi.tui.transcript import replay_history

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
