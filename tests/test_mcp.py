"""MCP server registry: ~/.okonomi/mcp.json round-trips through /mcp add /
remove, and configured servers surface as prefixed tools on the agent."""

import json
import sys

from _helpers import write_mcp_server
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from okonomi import mcp


def test_servers_round_trip_through_the_config_and_surface_as_prefixed_tools(store, tmp_path, monkeypatch):
    assert mcp.add_server("fs", "npx", ["-y", "server-fs", "."], {"TOKEN": "x"})["success"]
    assert mcp.add_server("docs", "https://example.com/mcp", [], {"Authorization": "Bearer y"})["success"]
    data = json.loads((tmp_path / "mcp.json").read_text())
    assert data == {
        "mcpServers": {
            "fs": {"command": "npx", "args": ["-y", "server-fs", "."], "env": {"TOKEN": "x"}},
            "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer y"}},
        }
    }, "config must use the standard mcpServers shape"
    assert mcp.remove_server("fs")["success"] and mcp.remove_server("docs")["success"]
    assert list(mcp.load_servers()) == []

    monkeypatch.setenv("SRV_PATH", str(write_mcp_server(tmp_path / "srv.py")))
    assert mcp.add_server("demo", sys.executable, ["${SRV_PATH}"], {})["success"]
    seen: list[str] = []

    async def hook(ctx, call_tool, name, args):
        seen.append(name)
        return await call_tool(name, args)

    agent = Agent(TestModel(), toolsets=mcp.build_toolsets(process_tool_call=hook))
    result = agent.run_sync("hi")
    tool_names = [p.tool_name for m in result.all_messages() for p in m.parts if p.part_kind == "tool-call"]
    assert tool_names == ["demo_shout", "demo_slow"], f"expected the prefixed MCP tools, got {tool_names}"
    assert seen == ["shout", "slow"], "process_tool_call hook must see the unprefixed names"
