import inspect
from typing import Any


class ToolMetadataMixin:
    """Base class for tool classes with self-describing metadata."""

    # Subclasses should set this
    __namespace__: str = ""
    __description__: str = ""

    def __get_metadata__(self) -> dict[str, Any]:
        """Get metadata for all public methods in this tool class."""
        metadata = {"namespace": self.__namespace__, "description": self.__description__, "methods": {}}

        for name, method in inspect.getmembers(self, predicate=inspect.iscoroutinefunction):
            if name.startswith("_"):
                continue

            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or ""

            # Extract short description
            doc_lines = [line.strip() for line in doc.split("\n") if line.strip()]
            short_desc = doc_lines[0] if doc_lines else f"Method: {name}"

            # Extract parameters
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_info = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "required": param.default == inspect.Parameter.empty,
                }

                if param.default != inspect.Parameter.empty:
                    if isinstance(param.default, list):
                        param_info["default"] = "[]"
                    elif isinstance(param.default, dict):
                        param_info["default"] = "{}"
                    else:
                        param_info["default"] = str(param.default)

                parameters[param_name] = param_info

            # Build signature
            params = []
            for param_name, param_info in parameters.items():
                param_str = f"{param_name}: {param_info['type']}"
                if not param_info["required"]:
                    param_str += f" = {param_info.get('default', '...')}"
                params.append(param_str)

            params_str = ", ".join(params)
            return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"

            metadata["methods"][name] = {
                "description": short_desc,
                "signature": f"async def {name}({params_str}) -> {return_type}",
                "parameters": parameters,
                "returns": return_type,
                "docstring": doc,
            }

        return metadata
