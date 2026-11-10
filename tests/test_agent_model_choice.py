"""``launch_agent(model=...)``: the parent may put a sub-agent on another
model, but only one listed in ``OKONOMI_SUBAGENT_MODELS``."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from _helpers import WAIT_SECONDS, child_agent, fake_agent

import okonomi.agent as agent_pkg
from okonomi.tools.agents import AgentTools


PARENT_MODEL = "big-model"


@pytest.fixture
def launched(monkeypatch):
    """Bound ``AgentTools`` whose children record the model they were built
    with instead of calling a provider."""
    built: list[str] = []
    done = threading.Event()

    async def run_async(self, message, reset=False):
        return "done"

    monkeypatch.setattr(agent_pkg, "OkonomiAgent", child_agent(run_async, built))
    parent = fake_agent(model_id=PARENT_MODEL, _job_finished=lambda **kw: done.set())
    tools = AgentTools()
    tools._bind(parent)
    return SimpleNamespace(tools=tools, built=built, done=done)


def test_default_is_parent_model(monkeypatch, launched):
    monkeypatch.setenv("OKONOMI_SUBAGENT_MODELS", "small-model")

    launched.tools.launch("task")

    assert launched.done.wait(WAIT_SECONDS), f"sub-agent did not finish within {WAIT_SECONDS}s"
    assert launched.built == [PARENT_MODEL], f"expected the parent's model, got {launched.built!r}"


def test_listed_model_is_used(monkeypatch, launched):
    monkeypatch.setenv("OKONOMI_SUBAGENT_MODELS", "small-model, other-model")

    reply = launched.tools.launch("task", model="other-model")

    assert launched.done.wait(WAIT_SECONDS), f"sub-agent did not finish within {WAIT_SECONDS}s"
    assert launched.built == ["other-model"], f"expected the requested model, got {launched.built!r}"
    assert "on other-model" in reply, reply


def test_unlisted_model_is_refused(monkeypatch, launched):
    monkeypatch.setenv("OKONOMI_SUBAGENT_MODELS", "small-model")

    reply = launched.tools.launch("task", model="made-up-model")

    assert reply.startswith("Error: model 'made-up-model' is not available."), reply
    assert "small-model" in reply, f"error must list the valid models: {reply!r}"
    assert launched.tools._tasks == {}, "a refused launch must not register an agent"
    assert launched.built == [], "a refused launch must not build a sub-agent"
