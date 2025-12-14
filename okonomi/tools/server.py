"""
Tools are simple Python classes. The MCP server injects them directly into
the code execution environment, so agents can use them naturally without
any code generation or proxies.
"""

import sys
from pathlib import Path


# Add parent directory to path so imports work when running with mcp dev
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource

from okonomi.config import OkonomiConfig
from okonomi.tools._image import ImageTools
from okonomi.tools._image_generation import ImageGenerationTools
from okonomi.tools._memory import MemoryTools
from okonomi.tools._prompt import PromptTools
from okonomi.tools._research import ResearchTools
from okonomi.tools._sessions import SessionTools
from okonomi.tools._strategies import StrategiesTools
from okonomi.tools._video import VideoTools


class OkonomiServer:
    """
    The MCP client should inject these tool instances directly into the
    code execution environment as global variables, allowing agents to
    use them naturally:

        result = await image_generation.text_to_image(prompt="...")
    """

    def __init__(self, config: OkonomiConfig, server_name: str = "okonomi-tools"):
        self.server = Server(server_name)
        self.tools: dict[str, object] = {}
        self._setup_handlers()

    def register_tool(self, name: str, instance: object):
        """
        Register a tool instance that will be available in code execution.

        Args:
            name: Variable name for the code execution environment (e.g., "image_generation")
            instance: Tool class instance with async methods
        """
        self.tools[name] = instance

    def _get_readme(self):
        readme = "# Available Tools\n\n"
        readme += "**IMPORTANT:** Before using any tool method, you MUST first call `get_tool_method_documentation(method_path='tool_name.method_name')` to get the complete documentation, including all parameters, their types, descriptions, and usage examples. This ensures you use the tool correctly.\n\n"
        readme += "## Tool Discovery Workflow\n\n"
        readme += "1. Use `get_tools_readme()` to see all available tools and methods\n"
        readme += "2. Use `get_tool_method_documentation(method_path='tool.method')` to get detailed docs for the specific method you want to use\n"
        readme += "3. Execute the tool method with the correct parameters based on the documentation\n\n"
        readme += "---\n\n"

        for namespace in sorted(self.tools.keys()):
            tool = self.tools[namespace]
            metadata = tool.__get_metadata__()

            readme += f"## {namespace}\n"
            for method_name, method_meta in sorted(metadata["methods"].items()):
                full_name = f"{namespace}.{method_name}"
                readme += f"- `{full_name}` - {method_meta['description']}\n"
            readme += "\n"

        return readme

    def _get_tool_method_documentation(self, method_path: str) -> str:
        """Get full documentation for a specific tool method.

        Args:
            method_path: Full method path in format 'tool_name.method_name'
                        (e.g., 'image_generation.edit_image', 'prompt.create_prompt')

        Returns:
            Full docstring of the specified method
        """
        parts = method_path.split(".", 1)
        if len(parts) != 2:
            return f"Error: Invalid method path '{method_path}'. Expected format: 'tool_name.method_name'"

        tool_name, method_name = parts

        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Unknown tool '{tool_name}'. Available tools: {available}"

        tool = self.tools[tool_name]
        metadata = tool.__get_metadata__()

        if method_name not in metadata["methods"]:
            available = ", ".join(metadata["methods"].keys())
            return f"Unknown method '{method_name}' in tool '{tool_name}'. Available methods: {available}"

        method_meta = metadata["methods"][method_name]

        # Return the full docstring and signature
        doc = f"# {method_path}\n\n"
        doc += f"**Signature:** `{method_meta['signature']}`\n\n"

        if method_meta.get("docstring"):
            doc += f"{method_meta['docstring']}\n"
        else:
            doc += "No documentation available.\n"

        return doc

    def _setup_handlers(self):
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            resources = []
            resources.append(
                Resource(
                    uri="doc://README",
                    name="Available Tools",
                    mimeType="text/markdown",
                    description="Directory of all available tools",
                )
            )
            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            uri_str = str(uri)
            if uri_str == "doc://README":
                readme = self._get_readme()
                return readme

            raise ValueError(f"Unknown resource URI: {uri_str}")

        @self.server.list_tools()
        async def list_tools() -> list:
            """List available tools.

            Returns tools for accessing documentation about the available tool instances.
            Actual tool execution happens directly in the Python executor.
            """
            from mcp.types import Tool

            return [
                Tool(
                    name="get_tools_readme",
                    description="Get a summary of all available tools and their methods",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="get_tool_method_documentation",
                    description="Get detailed documentation for a specific tool method including full docstring, parameters, and return types. IMPORTANT: Call with keyword argument: get_tool_method_documentation(method_path='tool_name.method_name')",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "method_path": {
                                "type": "string",
                                "description": "Full method path in format 'tool_name.method_name' (e.g., 'image_generation.edit_image', 'prompt.create_prompt')",
                            }
                        },
                        "required": ["method_path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list:
            """Handle tool calls.

            Implements:
            - get_tools_readme: Returns summary of all tools
            - get_tool_method_documentation: Returns detailed documentation for a specific method
            """
            from mcp.types import TextContent

            if name == "get_tools_readme":
                readme = self._get_readme()
                return [TextContent(type="text", text=readme)]

            elif name == "get_tool_method_documentation":
                method_path = arguments.get("method_path")
                if not method_path:
                    return [TextContent(type="text", text="Error: method_path parameter is required")]

                documentation = self._get_tool_method_documentation(method_path)
                return [TextContent(type="text", text=documentation)]

            raise ValueError(f"Unknown tool: {name}")

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


from okonomi.config import config


# Create global server instance
okonomi_server = OkonomiServer(config)
okonomi_server.register_tool("image", ImageTools(config))
okonomi_server.register_tool("image_generation", ImageGenerationTools(config))
okonomi_server.register_tool("prompt", PromptTools(config))
okonomi_server.register_tool("memory", MemoryTools(config))
okonomi_server.register_tool("research", ResearchTools(config))
okonomi_server.register_tool("sessions", SessionTools(config))
okonomi_server.register_tool("strategies", StrategiesTools(config))
okonomi_server.register_tool("video", VideoTools(config))

# Expose the MCP server object globally
server = okonomi_server.server  # <-- This is what mcp dev looks for


async def main():
    await okonomi_server.run()


if __name__ == "__main__":
    asyncio.run(main())
