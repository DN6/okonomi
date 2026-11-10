"""``TracingModel``: one OpenAI-format JSON line per completed model call."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from okonomi.traces import TRACES_FILENAME, TracingModel


def _read_traces(session) -> list[dict]:
    lines = (session.session_dir / TRACES_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_tool_round_trip_records_each_call(tmp_path):
    session = SimpleNamespace(session_id="s1", parent_session_id=None, session_dir=tmp_path)

    def respond(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("add", {"a": 1, "b": 2}, tool_call_id="c1")], provider_response_id="pr1")
        return ModelResponse(parts=[TextPart("3")])

    agent = Agent(TracingModel(FunctionModel(respond), lambda: session), system_prompt="sys")

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    agent.run_sync("1+2?")

    first, second = _read_traces(session)
    assert first["session_id"] == "s1", first
    assert first["provider_response_id"] == "pr1" and "run_id" in first, "trace lines must join to their event row"
    assert first["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1+2?"},
    ], first["messages"]
    assert [t["function"]["name"] for t in first["tools"]] == ["add"], first["tools"]
    assert first["completion"]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}}
    ], first["completion"]
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "3"}, second["messages"]
    assert second["completion"] == {"role": "assistant", "content": "3"}, second["completion"]


def test_streamed_call_is_recorded(tmp_path):
    session = SimpleNamespace(session_id="s2", parent_session_id="s1", session_dir=tmp_path)

    async def stream(messages, info):
        yield "hel"
        yield "lo"

    agent = Agent(TracingModel(FunctionModel(stream_function=stream), lambda: session))

    async def run():
        async with agent.run_stream("hi") as result:
            await result.get_output()

    asyncio.run(run())

    (trace,) = _read_traces(session)
    assert trace["parent_session_id"] == "s1", trace
    assert trace["messages"] == [{"role": "user", "content": "hi"}], trace["messages"]
    assert trace["completion"] == {"role": "assistant", "content": "hello"}, trace["completion"]
