"""
Tools are simple Python classes. The MCP server injects them directly into
the code execution environment, so agents can use them naturally without
any code generation or proxies.
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool
import asyncio
import json
import inspect

from okonomi.config import OkonomiConfig

from .image import ImageGenerationTools


class OkonomiServer:
    """
    The MCP client should inject these tool instances directly into the
    code execution environment as global variables, allowing agents to
    use them naturally:

        result = await image_generation.text_to_image(prompt="...", concept_id="...")
    """

    def __init__(self, config: OkonomiConfig, server_name: str = "okonomi-tools"):
        self.server = Server(server_name)
        self.tool_instances: dict[str, Any] = {}
        self._setup_handlers()

    def register_tool(self, name: str, instance: Any):
        """
        Register a tool instance that will be available in code execution.

        Args:
            name: Variable name for the code execution environment (e.g., "image_generation")
            instance: Tool class instance with async methods
        """
        self.tool_instances[name] = instance

    def _get_tool_documentation(self, name: str, instance: Any) -> dict:
        methods = {}

        for method_name, method in inspect.getmembers(instance, predicate=inspect.iscoroutinefunction):
            if method_name.startswith("_"):
                continue

            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or f"Method: {method_name}"

            methods[method_name] = {
                "signature": f"async def {method_name}{sig}",
                "docstring": doc,
                "parameters": {
                    param_name: {
                        "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    }
                    for param_name, param in sig.parameters.items()
                    if param_name != "self"
                },
            }

        return {
            "name": name,
            "class": instance.__class__.__name__,
            "methods": methods,
            "description": inspect.getdoc(instance.__class__) or f"Tool: {name}",
        }

    def _setup_handlers(self):
        return

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


async def main():
    server = OkonomiServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
