"""MCP server registry.

Servers live under ``mcpServers`` in ``~/.okonomi/settings.json``, in the shape
shared by Claude Desktop, Cursor and Codex, so entries can be pasted between
tools:

    {"mcpServers": {
        "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}}
    }}

``$VAR`` / ``${VAR}`` in string values expand from the environment. Each
server becomes one pydantic-ai ``MCPToolset`` whose tools are exposed to the
model as ``<name>_<tool>``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset, ProcessToolCallback
from pydantic_ai.toolsets import AbstractToolset

from okonomi import config


VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Per-request timeout. FastMCP's 5-minute default would kill a call that was
# moved to the background; match the cap on background bash tasks instead.
MCP_CALL_TIMEOUT_S = 3600


def load_servers() -> dict[str, dict]:
    return config.load_settings().get("mcpServers", {})


def _save_servers(servers: dict[str, dict]) -> None:
    data = config.load_settings()
    data["mcpServers"] = servers
    config.save_settings(data)


def add_server(name: str, target: str, args: list[str], env: dict[str, str]) -> dict:
    """Register ``target`` as an HTTP server (``http(s)://`` URL) or a stdio
    command. Returns ``{"success", "error"?}``."""
    if not VALID_NAME_RE.match(name):
        return {
            "success": False,
            "error": f"Invalid server name '{name}'. Use lowercase letters, numbers, '-' or '_' (it prefixes tool names).",
        }
    servers = load_servers()
    if name in servers:
        return {"success": False, "error": f"MCP server '{name}' already exists. Remove it first: /mcp remove {name}"}
    if target.startswith(("http://", "https://")):
        if args:
            return {"success": False, "error": "A URL server takes no extra arguments."}
        entry: dict[str, Any] = {"url": target}
    else:
        entry = {"command": target, "args": args}
    if env:
        entry["env" if "command" in entry else "headers"] = env
    servers[name] = entry
    _save_servers(servers)
    return {"success": True}


def remove_server(name: str) -> dict:
    servers = load_servers()
    if name not in servers:
        return {
            "success": False,
            "error": f"No MCP server named '{name}'. Configured: {', '.join(servers) or '(none)'}",
        }
    del servers[name]
    _save_servers(servers)
    return {"success": True}


def describe(server: dict) -> str:
    if "url" in server:
        return server["url"]
    return " ".join([server.get("command", ""), *server.get("args", [])])


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def build_toolsets(process_tool_call: ProcessToolCallback | None = None) -> list[AbstractToolset[Any]]:
    """One prefixed ``MCPToolset`` per configured server. Raises ``ValueError``
    naming the offending entry when the config is malformed."""
    toolsets: list[AbstractToolset[Any]] = []
    for name, server in load_servers().items():
        server = _expand_env(server)
        if "command" in server:
            client: Any = StdioTransport(
                command=server["command"],
                args=list(server.get("args") or []),
                env=server.get("env"),
                cwd=server.get("cwd"),
            )
            toolset = MCPToolset(
                client, id=name, process_tool_call=process_tool_call, read_timeout=MCP_CALL_TIMEOUT_S
            )
        elif "url" in server:
            toolset = MCPToolset(
                server["url"],
                id=name,
                headers=server.get("headers"),
                process_tool_call=process_tool_call,
                read_timeout=MCP_CALL_TIMEOUT_S,
            )
        else:
            raise ValueError(f"MCP server '{name}' in {config.OKONOMI_SETTINGS_FILE} needs a 'command' or a 'url'.")
        toolsets.append(toolset.prefixed(name))
    return toolsets
