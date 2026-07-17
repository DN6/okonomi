"""ToolWrapper and helpers for exposing tool instances inside execute_code."""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..session import asset_subdir_for_file


if TYPE_CHECKING:
    from .agent import AgentDependencies


class ToolResult(dict):
    """Dict subclass that shows available keys on KeyError."""

    def __missing__(self, key):
        available = ", ".join(repr(k) for k in self.keys())
        raise KeyError(f"Key {key!r} not found. Available keys: {available}")


def _extract_bg_label(code: str) -> str:
    """First identifiable expensive call in the code, used as a human label.

    Priority:
      1. ``genmedia.run("fal-ai/...")`` → label is the model id
      2. ``genmedia run fal-ai/...`` (legacy CLI form) → same
      3. Any other namespace.method match
      4. Fallback: "code block"
    """
    gm_prog = re.search(r"\bgenmedia\.run\(\s*['\"]([^'\"]+)['\"]", code)
    if gm_prog:
        return gm_prog.group(1)

    gm_cli = re.search(r"\bgenmedia\s+run\s+(\S+)", code)
    if gm_cli:
        return gm_cli.group(1)

    match = re.search(r"\b(browser|prompt|research|genmedia)\.(\w+)", code)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return "code block"


def _short_repr(obj: Any, limit: int = 2000) -> str:
    """Compact repr for tool return values shown in the UI.

    Tools commonly return dicts with URLs, IDs, and sometimes large payloads.
    Keep the dict shape readable but avoid multi-megabyte strings landing in
    the history pane.
    """
    try:
        if isinstance(obj, (dict, list, tuple)):
            s = json.dumps(obj, indent=2, default=str)
        else:
            s = str(obj)
    except Exception:
        s = repr(obj)
    if len(s) > limit:
        head = int(limit * 0.7)
        tail = limit - head
        s = f"{s[:head]}\n...[truncated {len(s):,} chars]...\n{s[-tail:]}"
    return s


class ToolWrapper:
    """Wrapper that adds tracking to tool method calls."""

    def __init__(self, tool_instance: Any, deps: AgentDependencies, namespace: str):
        self._tool_instance = tool_instance
        self._deps = deps
        self._namespace = namespace
        self._build_doc()
        # Optional hook: tools that need deps (e.g. genmedia, which
        # auto-tracks generated URLs via deps.add_artifact) implement
        # `_set_deps`. Wrappers that don't need deps just don't define it.
        if hasattr(tool_instance, "_set_deps"):
            try:
                tool_instance._set_deps(deps)
            except Exception:
                pass

    def _build_doc(self) -> None:
        """Build documentation string from tool metadata."""
        if hasattr(self._tool_instance, "__get_metadata__"):
            meta = self._tool_instance.__get_metadata__()
            lines = [f"{self._namespace}: {meta.get('description', '')}", ""]
            lines.append("Methods:")
            for method_name, method_info in meta.get("methods", {}).items():
                sig = method_info.get("signature", method_name)
                desc = method_info.get("description", "")
                lines.append(f"  {sig}")
                lines.append(f"      {desc}")
            lines.append("")
            lines.append("Properties:")
            lines.append("  session_dir: str")
            lines.append("      Path to the current session directory for saving files.")
            self.__doc__ = "\n".join(lines)
        else:
            self.__doc__ = self._tool_instance.__doc__ or ""

    def __dir__(self) -> list[str]:
        """List available methods and properties."""
        tool_attrs = [
            name
            for name in dir(self._tool_instance)
            if not name.startswith("_") and callable(getattr(self._tool_instance, name, None))
        ]
        return tool_attrs + ["session_dir"]

    def __repr__(self) -> str:
        return f"<{self._namespace} tool - use help({self._namespace}) for details>"

    @property
    def session_dir(self) -> str:
        """Get the current session directory path, creating it if needed."""
        session_dir = self._deps.session.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _generate_output_path(self, method_name: str, extension: str = ".mp4") -> str:
        """Generate an output path in the session's asset subdirectory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method_name}_{timestamp}{extension}"
        subdir = asset_subdir_for_file(filename)
        asset_dir = Path(self.session_dir) / "assets" / subdir
        asset_dir.mkdir(parents=True, exist_ok=True)
        return str(asset_dir / filename)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tool_instance, name)

        if callable(attr) and not name.startswith("_"):

            def wrapper(*args, **kwargs):
                tool_name = f"{self._namespace}.{name}"

                if "output_path" in kwargs and kwargs["output_path"] is None:
                    kwargs["output_path"] = self._generate_output_path(name)

                try:
                    result = attr(*args, **kwargs)
                except TypeError as e:
                    sig = inspect.signature(attr)
                    raise TypeError(f"{tool_name}() {e}. Expected signature: {name}{sig}") from None

                if isinstance(result, dict):
                    result = ToolResult(result)

                return result

            wrapper.__name__ = name
            wrapper.__doc__ = attr.__doc__
            try:
                wrapper.__signature__ = inspect.signature(attr)
            except (ValueError, TypeError):
                pass
            return wrapper
        return attr
