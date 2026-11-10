"""Launched agents push their result: when a ``launch_agent`` child finishes,
the parent emitter gets a ``JobFinished`` carrying the child's closing
message, and the TUI turns it into a follow-up turn for the parent."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from _helpers import WAIT_SECONDS, child_agent, fake_agent

import okonomi.agent as agent_pkg
from okonomi import cli
from okonomi.agent.jobs import JobRegistry
from okonomi.events import EventEmitter, JobFinished
from okonomi.tools.agents import AgentTools
from okonomi.tui.session import TuiSession


def _launch_and_wait(monkeypatch, run_async) -> JobFinished:
    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async))
    parent = fake_agent()
    finished: list[JobFinished] = []
    done = threading.Event()

    def _on_event(ev):
        if isinstance(ev, JobFinished):
            finished.append(ev)
            done.set()

    parent.emitter.subscribe(_on_event)
    tools = AgentTools()
    tools._bind(parent)
    tools.launch("research noir films", description="noir/films")

    assert done.wait(WAIT_SECONDS), f"no JobFinished within {WAIT_SECONDS}s of launch"
    assert parent.jobs.entries["agent-1"]["status"] != "running", "job must be marked done before the notice fires"
    return finished[0]


def test_launch_pushes_closing_message(monkeypatch):
    async def run_async(self, task, reset=False):
        return "brief saved to research_strand_films.md"

    ev = _launch_and_wait(monkeypatch, run_async)

    assert ev.job_id == "agent-1", f"unexpected job id {ev.job_id!r}"
    assert ev.kind == "agent", f"unexpected kind {ev.kind!r}"
    assert ev.status == "ok", f"unexpected status {ev.status!r}"
    assert ev.output == "brief saved to research_strand_films.md", f"closing message not carried: {ev.output!r}"


def test_launch_pushes_failure(monkeypatch):
    async def run_async(self, task, reset=False):
        raise RuntimeError("provider down")

    ev = _launch_and_wait(monkeypatch, run_async)

    assert ev.kind == "agent", f"unexpected kind {ev.kind!r}"
    assert ev.status == "error", f"unexpected status {ev.status!r}"
    assert ev.error == "provider down", f"error not carried: {ev.error!r}"


def _submitted(ev: JobFinished) -> str:
    submitted: list[str] = []
    loop = SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))
    fake = SimpleNamespace(_loop=loop, _quiet=False, _agent_running=False, _input_queue=SimpleNamespace(put_nowait=submitted.append))
    fake._deliver_notice = lambda notice: TuiSession._deliver_notice(fake, notice)
    TuiSession._on_job_finished(fake, ev)
    assert len(submitted) == 1, f"expected one injected turn, got {len(submitted)}"
    return submitted[0]


def test_tui_injects_closing_message():
    msg = _submitted(JobFinished(job_id="agent-1", kind="agent", output="brief saved"))

    assert "Background agent agent-1 finished" in msg, msg
    assert "Its closing message:\nbrief saved" in msg, msg


def test_tui_terminal_notice_carries_output():
    msg = _submitted(JobFinished(job_id="term-1", kind="term", output="frame 120/120\nwrote out.mp4"))

    assert "frame 120/120\nwrote out.mp4" in msg, f"the command's output must reach the model: {msg}"


def _launch_two_and_finish(jobs: JobRegistry) -> None:
    """Stand-in for the model launching two agents mid-turn."""
    for job_id in ("agent-1", "agent-2"):
        jobs.register(task_id=job_id, label=job_id, reader=lambda: "")

    def _finish():
        for job_id in ("agent-1", "agent-2"):
            # Same order as AgentTools.launch: marked done, then announced.
            jobs.mark_done(task_id=job_id, status="ok")
            jobs.finished(bg_id=job_id, output=f"{job_id} brief")

    threading.Thread(target=_finish, daemon=True).start()


def test_sub_agent_takes_a_turn_per_agent_it_launched(monkeypatch):
    turns: list[str] = []

    async def run_async(self, message, reset=False):
        turns.append(message)
        if len(turns) == 1:
            _launch_two_and_finish(self.jobs)
        return f"reply {len(turns)}"

    ev = _launch_and_wait(monkeypatch, run_async)

    assert len(turns) == 3, f"expected the first turn plus one per agent, got {turns!r}"
    assert "agent-1 brief" in turns[1], turns[1]
    assert "agent-2 brief" in turns[2], turns[2]
    assert ev.output == "reply 3", f"parent must get the reply to the last notice, got {ev.output!r}"


def _one_shot_agent(on_first_turn) -> tuple[SimpleNamespace, list[str]]:
    """The slice of OkonomiAgent that ``cli._run_single_task`` touches."""
    emitter = EventEmitter()
    turns: list[str] = []
    fake = SimpleNamespace(emitter=emitter, jobs=JobRegistry(emitter), save_session=lambda **kw: None)

    async def run_async(message, reset=False):
        turns.append(message)
        if len(turns) == 1:
            on_first_turn(fake.jobs)
        return f"reply {len(turns)}"

    fake.run_async = run_async
    return fake, turns


def _run_one_shot(fake) -> None:
    asyncio.run(asyncio.wait_for(cli._run_single_task(fake, "go"), WAIT_SECONDS))


def test_one_shot_takes_a_turn_per_launched_agent():
    fake, turns = _one_shot_agent(_launch_two_and_finish)

    _run_one_shot(fake)

    assert len(turns) == 3, f"expected the first turn plus one per agent, got {turns!r}"
    assert "agent-1 brief" in turns[1], turns[1]
    assert "agent-2 brief" in turns[2], turns[2]


def test_one_shot_ignores_shell_jobs():
    fake, turns = _one_shot_agent(lambda jobs: jobs.register(task_id="term-1", label="sleep", reader=lambda: ""))

    _run_one_shot(fake)

    assert turns == ["go"], f"a running shell job must not hold the run open, got {turns!r}"
