"""The ↓ jobs navigator lists the conversation's background jobs with each
launched agent's own jobs nested under it."""

from __future__ import annotations

import threading

from _helpers import WAIT_SECONDS, child_agent, fake_agent

import okonomi.agent as agent_pkg
from okonomi.events import JobFinished
from okonomi.tools.agents import AgentTools
from okonomi.tui.pickers import _job_rows


def test_launched_agent_exposes_its_jobs_and_closing_message(monkeypatch):
    async def run_async(self, task, reset=False):
        self.jobs.register(task_id="term-1", label="pip install", reader=lambda: "")
        self.jobs.mark_done(task_id="term-1", status="ok")
        return "segmented 3 images"

    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async))
    parent = fake_agent()
    done = threading.Event()
    parent.emitter.subscribe(lambda ev: isinstance(ev, JobFinished) and done.set())
    tools = AgentTools()
    tools._bind(parent)
    tools.launch("segment images", description="sam3")
    assert done.wait(WAIT_SECONDS), f"no JobFinished within {WAIT_SECONDS}s of launch"

    rows = [(r["id"], r["depth"]) for r in _job_rows(parent.jobs)]
    assert rows == [("agent-1", 1), ("term-1", 2)], f"sub-agent's job not nested: {rows}"
    detail = parent.jobs.entries["agent-1"]["_reader"]()
    assert "segmented 3 images" in detail, f"closing message missing from detail: {detail!r}"
