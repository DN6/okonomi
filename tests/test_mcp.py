"""MCP server registry: ~/.okonomi/mcp.json round-trips through /mcp add /
remove, and configured servers surface as prefixed tools on the agent."""

import json
import textwrap

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from okonomi import mcp


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    monkeypatch.setattr(mcp, "OKONOMI_MCP_CONFIG", path)
    return path


def test_add_remove_round_trip(config_path):
    assert mcp.add_server("fs", "npx", ["-y", "server-fs", "."], {"TOKEN": "x"})["success"]
    assert mcp.add_server("docs", "https://example.com/mcp", [], {"Authorization": "Bearer y"})["success"]
    data = json.loads(config_path.read_text())
    assert data == {
        "mcpServers": {
            "fs": {"command": "npx", "args": ["-y", "server-fs", "."], "env": {"TOKEN": "x"}},
            "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer y"}},
        }
    }, "config must use the standard mcpServers shape"
    assert mcp.remove_server("fs")["success"]
    assert list(mcp.load_servers()) == ["docs"]


def test_stdio_server_tools_are_prefixed_and_callable(config_path, tmp_path, monkeypatch):
    server = tmp_path / "srv.py"
    server.write_text(
        textwrap.dedent(
            """
            from mcp.server.fastmcp import FastMCP
            m = FastMCP("demo")

            @m.tool()
            def shout(text: str) -> str:
                return text.upper()

            m.run()
            """
        )
    )
    monkeypatch.setenv("SRV_PATH", str(server))
    import sys

    assert mcp.add_server("demo", sys.executable, ["${SRV_PATH}"], {})["success"]

    seen: list[str] = []

    async def hook(ctx, call_tool, name, args):
        seen.append(name)
        return await call_tool(name, args)

    agent = Agent(TestModel(), toolsets=mcp.build_toolsets(process_tool_call=hook))
    result = agent.run_sync("hi")
    tool_names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == "tool-call"]
    assert tool_names == ["demo_shout"], f"expected the prefixed MCP tool, got {tool_names}"
    assert seen == ["shout"], "process_tool_call hook must see the unprefixed name"
