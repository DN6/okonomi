"""Tool-call widgets are driven by pydantic-ai's event stream: every
ToolCallDisplay a turn opens is finished by tool_call_id, whatever kind of
tool ran (function tool, MCP tool, one that raises)."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from okonomi import db, mcp
from okonomi import session as session_mod
from okonomi.agent import agent as agent_mod
from okonomi.agent.agent import OkonomiAgent
from okonomi.events import AssistantStream, JobFinished, ToolCallDisplay


@pytest.fixture
def okonomi_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OKONOMI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(db, "_connection", db.open_db(tmp_path / "test.db"))
    monkeypatch.setattr(session_mod, "OKONOMI_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mcp, "OKONOMI_MCP_CONFIG", tmp_path / "mcp.json")

    server = tmp_path / "srv.py"
    server.write_text(
        textwrap.dedent(
            """
            from mcp.server.fastmcp import FastMCP
            m = FastMCP("demo")

            @m.tool()
            def shout(text: str) -> str:
                return text.upper()

            @m.tool()
            async def slow(seconds: float) -> str:
                import asyncio
                await asyncio.sleep(seconds)
                return f"slept {seconds}"

            m.run()
            """
        )
    )
    assert mcp.add_server("demo", sys.executable, [str(server)], {})["success"]
    agent = OkonomiAgent()
    agent.reload()
    return agent


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

    def _fn(messages, info):
        return ModelResponse(parts=[TextPart("unused")])

    return FunctionModel(_fn, stream_function=_stream)


def _capture(agent: OkonomiAgent) -> list:
    seen: list = []
    agent.emitter.subscribe(seen.append)
    return seen


def _widgets(seen: list) -> dict[str, ToolCallDisplay]:
    return {ev.tool_call_id: ev for ev in seen if isinstance(ev, ToolCallDisplay)}


def test_function_and_mcp_tools_finish_by_id(okonomi_agent):
    seen = _capture(okonomi_agent)
    calls = [
        ToolCallPart("demo_shout", {"text": "hi"}, tool_call_id="mcp-1"),
        ToolCallPart("load_skill", {"name": "no-such-skill"}, tool_call_id="fn-1"),
        ToolCallPart("list_skills", {}, tool_call_id="fn-2"),
    ]
    with okonomi_agent.agent.override(model=_scripted_model(calls)):
        okonomi_agent.run("go")

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


def _slow_web_fetch(monkeypatch, agent: OkonomiAgent, seconds: float) -> None:
    def slow(self, url: str) -> dict:
        """Sleeps."""
        time.sleep(seconds)
        return {"ok": True}

    monkeypatch.setattr(type(agent.web), "web_fetch", slow)
    agent.reload()


def test_cancelled_turn_closes_running_widget(okonomi_agent, monkeypatch):
    _slow_web_fetch(monkeypatch, okonomi_agent, 1.0)
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

    return FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("unused")]), stream_function=_stream)


def test_slow_mcp_call_moves_to_background(okonomi_agent, monkeypatch):
    monkeypatch.setattr(agent_mod, "TOOL_DETACH_AFTER", 0.3)
    seen = _capture(okonomi_agent)

    async def _turn_then_wait_for_notice() -> JobFinished:
        loop = asyncio.get_running_loop()
        finished: asyncio.Future[JobFinished] = loop.create_future()
        okonomi_agent.emitter.subscribe(
            lambda ev: isinstance(ev, JobFinished) and loop.call_soon_threadsafe(finished.set_result, ev)
        )
        await okonomi_agent.run_async("go")
        try:
            return await asyncio.wait_for(finished, 5)
        finally:
            await okonomi_agent.close_mcp()

    with okonomi_agent.agent.override(model=_model_calling_each_turn("demo_slow", {"seconds": 1.0})):
        started = time.monotonic()
        ev = asyncio.run(_turn_then_wait_for_notice())

    w = next(iter(_widgets(seen).values()))
    assert "moved to background as tool-" in w.output, f"turn must not wait on the MCP call: {w.output!r}"
    assert ev.kind == "tool" and ev.status == "ok", f"unexpected notice {ev!r}"
    assert "slept 1.0" in (ev.output or ""), f"MCP result must ride on the notice: {ev.output!r}"
    assert time.monotonic() - started < 5
