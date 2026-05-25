"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM interactions
with a local Python code executor that has access to all tool instances.
"""

import asyncio
import contextlib
import inspect
import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openrouter import OpenRouterModel


# Patch: OpenRouter returns service_tier="standard" which pydantic-ai doesn't accept.
# Widen the field to accept any string until upstream fixes it.
try:
    from pydantic_ai.models.openrouter import _OpenRouterChatCompletionChunk

    _OpenRouterChatCompletionChunk.model_fields["service_tier"].annotation = str | None  # type: ignore
    _OpenRouterChatCompletionChunk.model_rebuild()
except Exception:
    pass

from .config import OkonomiConfig
from .context import ContextManager
from .display import DIM, GREEN, LLM, RED, RST, YELLOW, print_ansi_box, print_code_box
from .session import Artifact, Session, SessionManager, SessionStatus, new_session_id
from .tools._browser_renderer import BrowserRendererTools
from .tools._genmedia import GenmediaTools
from .tools.agents import AgentTools
from .tools.skills import SkillTools
from .tools.terminal import TerminalTools
from .tools.web import WebTools


logger = logging.getLogger(__name__)


class CachedOpenRouterModel(OpenRouterModel):
    """OpenRouterModel with Anthropic prompt caching.

    Uses OpenRouter's top-level cache_control to automatically cache
    the conversation prefix for Claude models, reducing input token
    costs by ~75% on multi-turn conversations.
    """

    def __init__(self, model_name: str, enable_caching: bool = True, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self._enable_caching = enable_caching and "claude" in model_name.lower()
        if self._enable_caching:
            logger.info("Prompt caching enabled for %s", model_name)

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        if self._enable_caching:
            model_settings = args[2] if len(args) > 2 else kwargs.get("model_settings", {})
            extra_body = dict(model_settings.get("extra_body", {}) or {})
            extra_body["cache_control"] = {"type": "ephemeral"}
            if isinstance(model_settings, dict):
                model_settings = {**model_settings, "extra_body": extra_body}
            else:
                model_settings["extra_body"] = extra_body
            if len(args) > 2:
                args = (*args[:2], model_settings, *args[3:])
            else:
                kwargs["model_settings"] = model_settings
        return await super()._completions_create(*args, **kwargs)


@dataclass
class TokenUsage:
    """Tracks token usage across agent runs.

    Field semantics mirror pydantic-ai's ``RunUsage``: ``input_tokens``
    is the *non-cached* prompt count, with cache hits/writes counted
    separately in their own fields. So
    ``input + cache_read + cache_write`` ≈ the full prompt size sent to
    the provider — the columns are non-overlapping and safe to sum.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0  # cumulative actual USD from OpenRouter
    last_turn_input: int = 0
    history_tokens: int = 0  # estimated size of the persistent message_history

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        requests: int = 1,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Add usage from a single run. All counts are deltas to add."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.reasoning_tokens += reasoning_tokens
        self.requests += requests
        self.cost_usd += cost_usd
        self.last_turn_input = input_tokens

    def __str__(self) -> str:
        parts = [
            f"tokens: {self.total_tokens:,}",
            f"in: {self.input_tokens:,}",
            f"out: {self.output_tokens:,}",
            f"requests: {self.requests}",
        ]
        line = parts[0] + f" ({parts[1]}, {parts[2]}) | " + parts[3]
        if self.history_tokens:
            line += f" | history: ~{self.history_tokens:,}"
        if self.last_turn_input:
            line += f" | last turn: {self.last_turn_input:,}"
        return line


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent and tools at runtime."""

    session_id: str
    session_manager: SessionManager
    config: OkonomiConfig
    tools: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    models_used: set[str] = field(default_factory=set)
    verbose: bool = False
    # Persistent state for code execution across multiple calls
    execution_globals: dict[str, Any] = field(default_factory=dict)
    # Optional hook fired from add_artifact so the agent can kick off an
    # eager background download for AI-generated artifacts (those with a
    # remote ``url``). Set by OkonomiAgent in _get_dependencies.
    # Signature: (artifact: Artifact) -> None, must not raise.
    on_artifact_added: Optional[Any] = None

    def add_artifact(self, artifact: Artifact) -> None:
        """Track an artifact (AI-generated media OR locally-written file).

        Stamps a timestamp if missing, records ``tool_name`` / ``model_id``
        in the session metadata sets, and fires the eager-download hook
        for artifacts with a remote ``url``.
        """
        if not artifact.timestamp:
            artifact.timestamp = datetime.now().isoformat()
        self.artifacts.append(artifact)

        if artifact.tool_name:
            self.tools_used.add(artifact.tool_name)

        if artifact.model_id:
            self.models_used.add(artifact.model_id)

        if self.on_artifact_added is not None:
            try:
                self.on_artifact_added(artifact)
            except Exception:
                pass


class ToolResult(dict):
    """Dict subclass that shows available keys on KeyError."""

    def __missing__(self, key):
        available = ", ".join(repr(k) for k in self.keys())
        raise KeyError(f"Key {key!r} not found. Available keys: {available}")


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")
# URLs (http/https/file/data:image), bounded by whitespace or quotes.
_IMAGE_REF_RE = __import__("re").compile(
    r"""(?:https?://|file://|data:image/[a-zA-Z]+;base64,)[^\s<>"'`]+""",
    __import__("re").IGNORECASE,
)
# Absolute local paths ending in an image extension. Anchored to whitespace
# or string-start to avoid matching mid-word.
_LOCAL_IMAGE_RE = __import__("re").compile(
    r"(?:^|(?<=\s))(/[^\s'\"`]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff))(?=\s|$|[,;.])",
    __import__("re").IGNORECASE,
)


def _looks_like_image_ref(ref: str) -> bool:
    if ref.startswith("data:image/"):
        return True
    # Strip query/fragment for extension check
    path = ref.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(_IMAGE_EXTS)


def _make_image_part(ref: str):
    """Convert an image reference (URL, file://, data:, local path) into
    a pydantic-ai content part. Returns None for refs that don't resolve
    (dead local paths) so the caller can fall back to passing the string."""
    from pathlib import Path as _Path

    from pydantic_ai.messages import BinaryContent, ImageUrl

    if ref.startswith(("http://", "https://", "data:")):
        return ImageUrl(url=ref)

    # Local file path (with or without `file://` prefix)
    path_str = ref[7:] if ref.startswith("file://") else ref
    p = _Path(path_str).expanduser()
    if not p.exists() or not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except Exception:
        return None
    ext = p.suffix.lstrip(".").lower()
    media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    return BinaryContent(data=data, media_type=media_type)


def _build_multimodal_prompt(text: str):
    """If ``text`` mentions any image URLs / data URIs / local paths,
    return a list interleaving the text and the resolved image parts so
    the model receives the actual pixels. Otherwise return the original
    string unchanged (keeping the cheap path cheap)."""
    refs: list[tuple[int, int, str]] = []
    for m in _IMAGE_REF_RE.finditer(text):
        url = m.group(0).rstrip(".,;:)\"'")
        if _looks_like_image_ref(url):
            refs.append((m.start(), m.start() + len(url), url))
    for m in _LOCAL_IMAGE_RE.finditer(text):
        refs.append((m.start(1), m.end(1), m.group(1)))
    refs.sort()

    if not refs:
        return text

    parts: list = []
    cursor = 0
    for start, end, ref in refs:
        if start > cursor:
            chunk = text[cursor:start]
            if chunk.strip():
                parts.append(chunk)
        part = _make_image_part(ref)
        if part is not None:
            parts.append(part)
        else:
            # Failed to resolve — keep the literal text
            parts.append(text[start:end])
        cursor = end
    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            parts.append(tail)

    # Collapse to a string if we found refs but couldn't resolve any.
    if not any(not isinstance(p, str) for p in parts):
        return text
    return parts


def _extract_bg_label(code: str) -> str:
    """First identifiable expensive call in the code, used as a human label.

    Priority:
      1. ``genmedia.run("fal-ai/...")`` → label is the model id
      2. ``genmedia run fal-ai/...`` (legacy CLI form) → same
      3. Any other namespace.method match
      4. Fallback: "code block"
    """
    import re

    # Programmatic wrapper: genmedia.run("fal-ai/flux/dev", …)
    gm_prog = re.search(r"\bgenmedia\.run\(\s*['\"]([^'\"]+)['\"]", code)
    if gm_prog:
        return gm_prog.group(1)

    # Legacy CLI form: genmedia run fal-ai/flux/dev …
    gm_cli = re.search(r"\bgenmedia\s+run\s+(\S+)", code)
    if gm_cli:
        return gm_cli.group(1)

    match = re.search(
        r"\b(browser|prompt|research|genmedia)\.(\w+)",
        code,
    )
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
        import json

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
        session_dir = self._deps.session_manager.session_dir(self._deps.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _generate_output_path(self, method_name: str, extension: str = ".mp4") -> str:
        """Generate an output path in the session's asset subdirectory."""
        from .session import asset_subdir_for_file

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

                self._deps.tools_used.add(tool_name)

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


class OkonomiAgent:
    """Pydantic AI-based agent with local code execution.

    This agent provides:
    - Code execution with tool instances injected into the environment
    - Session management for continuity
    - Message history for multi-turn conversations
    """

    def __init__(
        self,
        config: OkonomiConfig,
        session_id: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.session_manager = SessionManager()
        self.session_id = session_id or new_session_id()
        self.message_history: list[ModelMessage] = []
        self.task: str = ""
        self.verbose = verbose
        self._setup_tools()

        # Background pool for eager asset downloads. Each generated output
        # is downloaded to the session dir as soon as it's tracked, so
        # save_session is fast even with many large videos. Daemon
        # threads — they don't block process exit. Cap concurrency at 4
        # so a burst of generations doesn't saturate bandwidth.
        from concurrent.futures import Future, ThreadPoolExecutor

        self._asset_dl_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="okonomi-dl")
        # Tracked so quit can bounded-wait for in-flight downloads instead
        # of either dropping them silently or blocking forever.
        self._asset_dl_futures: list[Future] = []

        # Persistent REPL state across turns
        self.execution_globals: dict[str, Any] = {}

        # Interrupt state
        self._was_interrupted: bool = False
        self._partial_output: str = ""

        # Last executed code block (for expand-on-demand)
        self._last_code: str = ""

        # Session tracking
        self.artifacts: list[Artifact] = []
        self.tools_used: set[str] = set()
        self.models_used: set[str] = set()

        self.token_usage = TokenUsage()
        self.context = ContextManager(config)
        self.spinner_text: str = ""

        # Approval callback for expensive operations.
        # Signature: async (code: str) -> bool | str
        # Returns True to proceed, False to skip, or a string with revised code.
        # If None, all code runs without approval.
        self.approval_callback: Any = None
        self.auto_approve: bool = False

        self._expensive_patterns = [
            # Local Playwright capture is non-trivial; treat as expensive.
            "browser.render",
            # Any genmedia namespace call (run / search / schema). Cheap
            # methods get gated too — acceptable cost for the simpler rule.
            "genmedia.",
            # Legacy direct CLI usage: run("genmedia run ...").
            "genmedia run",
            # Direct fal_client calls in raw user code (rare, but still
            # flagged so the user sees what's running).
            "fal_client",
        ]

        self._bg_exec_counter = 0
        # Background execute_code jobs keyed by bg_id. Plain threads (not asyncio
        # tasks) so they survive the per-turn event loop teardown in the CLI.
        import threading as _threading_mod

        self._bg_execs: dict[str, _threading_mod.Thread] = {}
        # Per-job log registry covering bg-exec-N (Python execute_code jobs)
        # and term-N (terminal.bash background jobs). Survives completion so
        # /logs <id> can read after a job finishes. Each entry:
        #   {"kind", "label", "started_at", "ended_at", "status", "_reader"}
        # `_reader` is a zero-arg callable returning the current snapshot.
        self._bg_logs: dict[str, dict[str, Any]] = {}

        # Optional event emitter. When set (e.g. by the Textual TUI), the agent
        # emits typed events instead of printing to stdout. None → legacy
        # prompt_toolkit CLI path which relies on print() + patch_stdout.
        self.emitter: Any = None
        self._current_stream: Any = None
        # Tracks the ToolCall event for the in-flight execute_code so we can
        # attach its output/error back to the UI when the tool returns.
        self._pending_tool_call: Any = None
        # Optional notification hook invoked from the bg-exec coroutine when a
        # background code block finishes. Signature:
        #   on_bg_complete(*, bg_id, error, output,
        #                  new_outputs, new_artifacts,
        #                  produced_outputs, produced_artifacts)
        # produced_outputs/artifacts are the *actual* objects appended by
        # this bg run (sliced from ctx.deps.* between before/after) so the
        # callback can reference them by URL/path instead of asking the
        # agent to scan session-wide state.
        self.on_bg_complete: Any = None

        # Skill creation nudge: after N tool-calling turns without skill
        # management, append a reminder to consider saving as a skill.
        self._turns_since_skill = 0
        self._skill_nudge_interval = 10

        self.agent = self._create_agent()

    def _setup_tools(self) -> None:
        """Create all tool instances.

        Tools are split into two categories:
        - **Programmatic tools**: Available inside ``execute_code`` via ToolWrapper.
          The model writes Python that calls these in loops, filters results, etc.
        - **Standalone tools**: Registered directly on the agent. Each call is a
          separate model round-trip.
        """
        # Programmatic tools — called from inside execute_code.
        # Image / video / audio / 3D generation flows through `genmedia`
        # (any fal endpoint, single namespace). Local image / frame
        # loading is no longer wrapped — the model uses PIL / imageio
        # directly via Python imports inside `execute_code`.
        self.programmatic_tools: dict[str, Any] = {
            "browser": BrowserRendererTools(self.config),
            "genmedia": GenmediaTools(self.config),
        }

        # Standalone tools — direct agent tools (one call per round-trip)
        from .tools.sessions import SessionsTools

        self.terminal = TerminalTools(self.config)
        self.terminal._on_bg_register = self._register_bg_log
        self.terminal._on_bg_finish = self._mark_bg_log_done
        self.web = WebTools(self.config)
        self.sessions = SessionsTools(self.config)
        self.agents = AgentTools(self.config)
        self.skills = SkillTools(
            self.config,
            available_tools=set(self.programmatic_tools.keys()) | {"terminal", "web", "sessions", "skills"},
        )

    def _create_agent(self) -> Agent[AgentDependencies, str]:
        """Create and configure the Pydantic AI agent."""
        model = self._create_model()
        agent = Agent(
            model,
            deps_type=AgentDependencies,
            system_prompt=self._get_system_prompt(),
        )

        # Standalone tools
        self._register_tool(agent, self.skills)
        self._register_tool(agent, self.terminal)
        self._register_tool(agent, self.web)
        self._register_tool(agent, self.sessions)
        self.agents._bind(self)
        self._register_tool(agent, self.agents, rename={"launch": "launch_agent", "list_agents": "list_agents"})

        # Programmatic tools (available inside execute_code)
        self._register_code_execution_tool(agent)

        return agent

    def _create_model(self) -> CachedOpenRouterModel:
        from pydantic_ai.models.openrouter import OpenRouterModelSettings

        # `openrouter_usage={"include": True}` makes OpenRouter return
        # `usage.cost` (actual USD spent on the call) in each response.
        # We pull that out per-turn in `_accumulate_usage` and persist
        # it as `actual_cost_usd` — authoritative billing without
        # maintaining a local pricing table.
        return CachedOpenRouterModel(
            self.config.agent.model_id,
            settings=OpenRouterModelSettings(
                timeout=self.config.timeouts.chat_completion,
                openrouter_usage={"include": True},
            ),
        )

    # Skills inlined into the system prompt instead of being load-on-demand.
    # Use this for skills the agent ALWAYS needs in scope (e.g. genmedia is
    # the only path to media generation now that the fal wrappers are
    # being retired). Listed by skill name; resolution goes through the
    # SkillTools loader so user-overridden versions win over bundled ones.
    _CORE_SKILLS = ("genmedia",)

    def _build_skills_index(self) -> str:
        """Build a skills index from discovered skills for the system prompt.

        Core skills (always-active, content inlined elsewhere via
        ``_build_core_skill_blocks``) are excluded from the index — listing
        them as load-on-demand would mislead the model into round-tripping
        through `load_skill` for content that's already in scope.
        """
        result = self.skills.list_skills()
        skills = result.get("skills", [])
        if not skills:
            return ""

        by_category: dict[str, list[dict]] = {}
        for skill in skills:
            if skill.get("name") in self._CORE_SKILLS:
                continue
            cat = skill.get("category", "") or "general"
            by_category.setdefault(cat, []).append(skill)

        lines = []
        for category in sorted(by_category.keys()):
            lines.append(f"  {category}:")
            for skill in sorted(by_category[category], key=lambda s: s["name"]):
                if skill["description"]:
                    lines.append(f"    - {skill['name']}: {skill['description']}")
                else:
                    lines.append(f"    - {skill['name']}")
        return "\n".join(lines)

    def _build_core_skill_blocks(self) -> str:
        """Inline the full SKILL.md content of every core skill.

        These skills are mandatory and always-active — the model treats
        them as built-in capabilities, not load-on-demand recipes.
        """
        blocks: list[str] = []
        for name in self._CORE_SKILLS:
            try:
                # Core skills are inlined in full because they're the
                # baseline always-active capability; the public load_skill
                # now returns just a TOC for progressive disclosure.
                loaded = self.skills._load_full(name)
            except Exception:
                continue
            content = loaded.get("content") or ""
            if not content:
                continue
            blocks.append(f"## Core skill: `{name}` (always active)\n\n{content.strip()}")
        return "\n\n".join(blocks)

    def _build_tool_index(self) -> str:
        """Per-namespace method index for the system prompt.

        Eager full disclosure: each method gets its signature (with param
        types + defaults), one-line description, and the literal
        return-dict shape pulled from its TypedDict annotation. Cheap
        for our ~14-method surface and means the model never has to
        introspect to use a tool correctly.
        """
        import inspect as _inspect

        def _pretty_type(annotation: Any) -> str:
            """Strip noisy <class 'foo'> wrappers; preserve generics."""
            if annotation is _inspect.Parameter.empty:
                return "Any"
            if isinstance(annotation, type):
                return annotation.__name__
            text = str(annotation)
            if text.startswith("<class '") and text.endswith("'>"):
                text = text[len("<class '") : -len("'>")]
                if "." in text:
                    text = text.rsplit(".", 1)[-1]
            text = text.replace("typing.", "")
            return text

        def _render_signature(method: Any, name: str) -> str:
            sig = _inspect.signature(method)
            parts: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                p = f"{pname}: {_pretty_type(param.annotation)}"
                if param.default is not _inspect.Parameter.empty:
                    p += f" = {param.default!r}"
                parts.append(p)
            ret = _pretty_type(sig.return_annotation)
            return f"{name}({', '.join(parts)}) -> {ret}"

        def _render_return_shape(method: Any) -> str:
            """If the return is a TypedDict, render its key:type pairs.
            Falls back to empty string for plain types so we don't pollute
            the prompt for methods that don't have a structured return."""
            sig = _inspect.signature(method)
            ret = sig.return_annotation
            ann = getattr(ret, "__annotations__", None)
            if not ann:
                return ""
            keys = ", ".join(f'"{k}": {_pretty_type(v)}' for k, v in ann.items())
            return f"{{{keys}}}"

        sections: list[str] = []
        for namespace, tool_instance in self.programmatic_tools.items():
            if not hasattr(tool_instance, "__get_metadata__"):
                continue
            meta = tool_instance.__get_metadata__()
            ns_desc = (meta.get("description") or "").strip()
            sections.append(f"### `{namespace}` — {ns_desc}")
            for method_name in (meta.get("methods") or {}).keys():
                method = getattr(tool_instance, method_name, None)
                if method is None:
                    continue
                method_info = meta["methods"][method_name]
                method_desc = (method_info.get("description") or "").strip()
                sig_str = _render_signature(method, method_name)
                ret_shape = _render_return_shape(method)
                sections.append(f"- `{namespace}.{sig_str}`")
                sections.append(f"  {method_desc}")
                if ret_shape:
                    sections.append(f"  Returns: `{ret_shape}`")
            sections.append("")
        return "\n".join(sections).rstrip()

    def _build_storage_paths(self) -> dict[str, str]:
        """Resolve the storage directories the agent should know about.

        Honors $OKONOMI_HOME and any session-storage override in config.
        The model reads these paths, so they must reflect the user's
        actual filesystem layout — not the documented defaults.
        """
        import os

        okonomi_home = Path(os.getenv("OKONOMI_HOME", str(Path.home() / ".okonomi")))
        # SessionManager already resolved its own storage_dir using the
        # config override. Reuse that string so the prompt always matches
        # where sessions are *actually* being written.
        sessions_dir = str(self.session_manager.storage_dir)
        return {
            "sessions": sessions_dir,
            "research": str(okonomi_home / "research"),
            "skills": str(okonomi_home / "skills"),
            "favourites": str(okonomi_home / "favourites"),
        }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        tool_index = self._build_tool_index()
        skills_index = self._build_skills_index()
        core_skills = self._build_core_skill_blocks()
        paths = self._build_storage_paths()
        # f-string can't subscript with bare keys (`{paths[sessions]}` is
        # a NameError), so bind each path to its own local.
        sessions_dir = paths["sessions"]
        research_dir = paths["research"]
        skills_dir = paths["skills"]
        favourites_dir = paths["favourites"]

        return f"""You are an expert creative assistant for generating images and videos using code.

{core_skills}

## Programmatic Tools (used inside `execute_code`)

These namespaces are pre-imported as Python objects in the `execute_code`
sandbox. Methods are listed by name + one-line description so you know
*which* method to reach for. Parameter signatures and return-dict shapes
are NOT shown here — load them via Python introspection (rule 6) *before*
calling.

{tool_index}

## Skills (mandatory)

Before replying, scan the skill list below. If one clearly matches the task, you MUST
load it by calling `load_skill("skill-name")` directly. Then follow the loaded instructions step by step.

After difficult or iterative tasks, offer to save the approach as a skill using
`create_skill(name, content)`.

<available_skills>
{skills_index}
</available_skills>

If none match, proceed normally without loading a skill.

## Creative Workflow

Before generating any media, determine if the request is **direct** or **conceptual**:

### Direct Requests (proceed to generation)
User provides specific visual details: subject, composition, style, colors, lighting.
Examples:
- "A red fox sitting in snow, soft morning light, photorealistic"
- "Edit this image to change the background to a beach"

### Conceptual Requests (develop concept first)
User provides themes, moods, abstract ideas, or vague descriptions.
Examples:
- "Something cyberpunk"
- "Create art about loneliness"
- "Make a cool character"

**For conceptual requests, follow this workflow:**

1. **Research & Ideation** - Load the `research` skill for structured creative exploration. Use `web_search` for discovery and `web_fetch` to read specific pages.
2. **Determine Style** - Decide on the visual style (photorealistic, illustration, anime, painterly, etc.) based on the concept.
3. **Generate** - Write a layered prompt (subject, action, style, composition, lighting) and call `genmedia.run(...)` with the right tier-1 model from the `genmedia` skill.

## Image edit / reference prompts

For edit / reference endpoints (e.g. `openai/gpt-image-2/edit`, `fal-ai/flux-redux`), the prompt must explicitly state:
- **What to preserve**: style, colors, composition, subject identity
- **What to change**: specific elements to modify
- **Reference attribution**: "The style from image1", "The person from image2"

Examples:
- "Preserve the color palette and lighting from image1. Apply it to a new scene of a mountain landscape at dawn."
- "Keep the person's face and clothing from image1. Change the background to a futuristic cityscape."
- "Using the painterly style from image1, recreate the composition with autumn colors."

## Rules

1. Use `execute_code` to run Python code for image/video generation tools
2. Use `print()` to see results from code execution
3. Variables persist between executions — if code fails, prior variables are still available
4. All methods are synchronous - do NOT use `await`
5. **NEVER re-run `genmedia.run`, `browser.render`, or any other expensive call to recover from a parsing error.** These calls are expensive, slow, and cost money. The generation result is still sitting in your previous `result` variable — globals persist across `execute_code` calls. If the previous call failed with a `KeyError` / `AttributeError` / `TypeError` after the generation, write a *new* `execute_code` block that does `print(result)` and fixes the access pattern. If the generation call itself raised (network error, 500), inform the user and ask before retrying — do not silently re-run. Re-running a successful generation because of a parsing bug is the single most wasteful mistake you can make.
6. **The Programmatic Tools section above has every method's signature, parameter types, and return-dict shape.** Read it before writing code that calls a tool. Never guess `image_url` / `videos` / `url` — the literal return keys are right there. Use `inspect.getdoc(method)` only when you need the prose docstring (longer explanations, examples, edge-case notes); for signatures and return shapes the prompt is authoritative.
7. **Saving files**: `get_session_dir()` is in scope inside `execute_code` and returns the absolute path to the current session folder — use it for any file you create. Methods with `output_path` parameters (e.g. `browser.render_screenshot`) auto-save into that folder when `output_path=None`.
8. **Standalone tools** (`bash`, `search`, `web_search`, `web_fetch`, `search_sessions`, `search_artifacts`, `get_session`, `list_skills`, `load_skill`, `load_skill_section`, `load_skill_reference`, `create_skill`, `write_skill_file`, `patch_skill_file`, `delete_skill`, `launch_agent`, `list_agents`) — call directly, NOT through `execute_code`. For long-running shell commands (yt-dlp, ffmpeg, anything >30s), call `bash(command=..., background=True)` — the tool returns immediately with a `term-N` task id and you'll get a system note with the output when it finishes. **Cross-session memory ladder** (use in this order to keep token cost bounded): 1) `search_sessions(query)` / `search_artifacts(query, artifact_type=..., since_days=...)` — BM25-ranked FTS5 over a SQLite index, returns lean hit summaries; 2) `get_session(session_id)` — drill into one match for full metadata + artifacts list (NOT message history); 3) `bash(command="head -50 <dir>/messages.jsonl | jq ...")` — read specific portions of message history when the structured tools don't cover it. **Don't `cat` whole session.json files** — they can be 20-50KB; use `get_session` instead. **Skills are progressively disclosed**: `load_skill(name)` returns a TOC (intro + section slugs + reference filenames). Drill in with `load_skill_section(name, slug)` or `load_skill_reference(name, ref)` — only pull what the current task actually needs.
9. **Programmatic tools** (`browser`, `genmedia`) — use inside `execute_code` only. Image / video / audio / 3D generation flows through `genmedia.run("fal-ai/...", **kwargs)`. For local image / frame loading, just `from PIL import Image` or `import imageio.v3 as iio` directly — no wrapper needed. To *interpret* an image, mention its URL or path in your reply text — okonomi auto-attaches image references as multimodal content for you.
10. **Background agents**: Use `launch_agent(task, description)` to delegate independent work (e.g., generating multiple images in parallel). Check progress with `list_agents()`. Outputs are automatically merged back into the session.
11. **Media/expensive code always runs in the background after approval.** Any `execute_code` block touching `genmedia.`, `browser.render`, or `fal_client` is auto-dispatched as a background task (returns a `bg-exec-N` handle, no stdout, no `result` variable). After dispatch, **acknowledge in one short sentence and stop.** Do NOT write expected-output prose ("this will give you a vibrant cinematic look", "should produce a moody noir aesthetic", etc.) — those claims are wrong on failure and unnecessary on success. You'll be re-invoked with a completion notice when it finishes; describe the actual output then. You may also pass `background=True` explicitly for other long-running code.
12. **Local storage layout** — when asked to find or summarize prior work, search the right directory. These paths reflect the user's *actual* configuration (honoring `$OKONOMI_HOME` / session-storage override):
   - `{sessions_dir}` — full conversation logs and session metadata. Use this for "what did we discuss in past sessions".
   - `{research_dir}` — saved research reports (markdown files like `digital-decay-concept-brief.md`). Use this for "summarize my research" / "what concepts have I explored". The `research` skill writes here.
   - `{skills_dir}` — reusable skill definitions. Use this for "what skills exist".
   - `{favourites_dir}` — user-favourited assets.

   Examples (use the literal paths above, not `~/.okonomi/...`):
   ```
   search(pattern="<keyword>", path="{research_dir}", file_glob="*.md")
   bash(command="ls {research_dir}")
   ```

## Response Style

Keep replies short. Default to 1–3 sentences of plain prose. Skip
preambles ("Great question!", "I'll start by…"), restating the task,
and play-by-play of what you're about to do — the user can see your
tool calls in the UI. Don't bullet-list more than 4 items. Don't
explain the obvious result of a successful generation; just present
the asset URL and a one-line caption. Long replies belong only when
the user explicitly asks for an explanation, a walkthrough, or a list.

Be creative and help users generate stunning images and videos."""

    def _register_code_execution_tool(self, agent: Agent) -> None:
        """Register the code execution tool with the agent."""

        @agent.tool
        async def execute_code(
            ctx: RunContext[AgentDependencies],
            code: str,
            background: bool = False,
        ) -> str:
            """Execute Python code with access to all tool objects.

            The code runs in an environment with these tool objects available:
            - browser: Headless browser rendering (screenshot and video capture of HTML/JS)
            - genmedia: Run any fal.ai endpoint — image, video, audio, 3D — auto-tracks outputs

            For local image / frame work (PIL ops, imageio), just import
            the libraries directly from inside `execute_code`. For image
            interpretation (one or many), include the URLs / data URIs / local
            paths in your conversation reply or in your next user-facing
            message — okonomi extracts image references from text and
            attaches them to the model as multimodal content automatically.

            For past-session info, use the `search` tool (not a programmatic
            namespace) — e.g. `search(pattern="<keyword>", path="~/.okonomi/sessions/")`
            — instead of any sessions.* methods, which no longer exist.

            Args:
                code: Python code to execute. Use print() for output.
                background: When True, approval (if required) still happens in the foreground,
                    but execution is dispatched as a background task and the tool returns
                    immediately. A completion notification is injected as a new turn when the
                    task finishes. Use for long media generations when you don't need the
                    result to chain into the next code block. Defaults to False.

            Returns:
                The stdout output from the code execution, or error message.
                When background=True, returns a handle like "Dispatched as bg-exec-N ..."
            """
            # Mutate execution_globals in place — `ctx.deps.execution_globals`
            # is the same object as `self.execution_globals` on the agent, so
            # rebinding (`= {...}`) would silently break cross-turn variable
            # persistence (the agent would keep the empty original).
            #
            # Tool wrappers and helpers are reinstalled every call. They
            # close over the current ctx.deps (different on every turn);
            # gating reinstallation on "dict is empty" would mean a session
            # resume with restored user vars never installs them at all.
            # Cheap to re-create, identical to user vars in keyspace, no
            # collisions in practice.
            ctx.deps.execution_globals.setdefault("__builtins__", __builtins__)
            ctx.deps.execution_globals.setdefault("asyncio", asyncio)
            for namespace, tool_instance in ctx.deps.tools.items():
                ctx.deps.execution_globals[namespace] = ToolWrapper(tool_instance, ctx.deps, namespace)

            def get_session_dir() -> str:
                """Get the current session's directory path.

                Use this to save files (like stitched videos) to the session folder.

                Returns:
                    Absolute path to the current session directory.

                Example:
                    session_dir = get_session_dir()
                    html_path = f"{session_dir}/sketch.html"
                    with open(html_path, "w") as f:
                        f.write(html)
                """
                session_dir = ctx.deps.session_manager.session_dir(ctx.deps.session_id)
                session_dir.mkdir(parents=True, exist_ok=True)
                return str(session_dir)

            ctx.deps.execution_globals["get_session_dir"] = get_session_dir

            is_expensive = any(p in code for p in self._expensive_patterns)

            if self.approval_callback and not self.auto_approve and is_expensive:
                self.spinner_text = ""
                decision = await self.approval_callback(code)
                if decision is False:
                    if self._pending_tool_call is not None and self.emitter is not None:
                        try:
                            self.emitter.tool_finish(
                                self._pending_tool_call,
                                error="skipped by user",
                            )
                        except Exception:
                            pass
                        self._pending_tool_call = None
                    return "Code execution skipped by user."
                if decision == "__auto_approve__":
                    self.auto_approve = True
                elif isinstance(decision, str):
                    return f"User feedback before execution: {decision}\n\nRevise the code and try again."

            # Media/expensive code always backgrounds after approval passes.
            if is_expensive:
                background = True

            self.spinner_text = "Running..."

            def _exec_in_thread():
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        exec(code, ctx.deps.execution_globals)

                    stdout_output = stdout_capture.getvalue()
                    stderr_output = stderr_capture.getvalue()

                    result = ""
                    if stdout_output:
                        result += stdout_output
                    if stderr_output:
                        result += f"\nStderr:\n{stderr_output}"
                    return result.strip() if result.strip() else None, None
                except Exception:
                    return None, traceback.format_exc()

            if background:
                import threading
                import time

                self._bg_exec_counter += 1
                bg_id = f"bg-exec-{self._bg_exec_counter}"
                before_artifacts = len(ctx.deps.artifacts)

                # Long-lived buffers so /logs can snapshot during and after
                # the run. _run_bg redirects stdout/stderr into these, then
                # reads from them once exec returns.
                bg_stdout = io.StringIO()
                bg_stderr = io.StringIO()

                def _reader(out=bg_stdout, err=bg_stderr) -> str:
                    parts = []
                    s = out.getvalue()
                    if s:
                        parts.append(s)
                    e = err.getvalue()
                    if e:
                        parts.append(f"\n--- stderr ---\n{e}")
                    return "".join(parts) if parts else "(no output yet)"

                self._bg_logs[bg_id] = {
                    "kind": "bg-exec",
                    "label": _extract_bg_label(code),
                    "started_at": time.time(),
                    "ended_at": None,
                    "status": "running",
                    "_reader": _reader,
                }

                # Hand the ToolCall event from _display_tool_call to the bg
                # thread so it can resolve (ok/error + output) in-place when
                # the code finishes. This gives the user a single foreground
                # widget that represents the bg job throughout its lifetime.
                bound_tool_call = self._pending_tool_call
                self._pending_tool_call = None
                if bound_tool_call is not None and self.emitter is not None:
                    try:
                        bound_tool_call.args = f"{bound_tool_call.args} → {bg_id} running"
                        self.emitter.update(bound_tool_call)
                    except Exception:
                        pass

                def _run_bg():
                    error: str | None = None
                    output: str | None = None
                    new_artifacts = 0
                    try:
                        try:
                            with redirect_stdout(bg_stdout), redirect_stderr(bg_stderr):
                                exec(code, ctx.deps.execution_globals)
                            output = _reader().strip()
                            if output == "(no output yet)":
                                output = None
                        except Exception:
                            error = traceback.format_exc()
                            output = _reader().strip() or None
                        new_artifacts = len(ctx.deps.artifacts) - before_artifacts
                        log_entry = self._bg_logs.get(bg_id)
                        if log_entry is not None:
                            log_entry["status"] = "error" if error else "ok"
                            log_entry["ended_at"] = time.time()
                        if error:
                            print(f"\n{RED}✗ {bg_id} failed:\n{error}{RST}")
                        else:
                            tail = (
                                f" — {new_artifacts} artifact{'s' if new_artifacts != 1 else ''}"
                                if new_artifacts
                                else ""
                            )
                            print(f"\n{GREEN}✓ {bg_id} done{tail}{RST}")
                            if output and self.verbose:
                                print_ansi_box(output, label=f"{bg_id} output")
                        # Update the original ToolCall event so the user
                        # sees the bg job's result in-place in history.
                        if bound_tool_call is not None and self.emitter is not None:
                            try:
                                summary_str = (
                                    f"{new_artifacts} artifact{'s' if new_artifacts != 1 else ''}"
                                    if new_artifacts
                                    else "done"
                                )
                                merged_output = f"{bg_id} {summary_str}" + (f"\n\n{output}" if output else "")
                                self.emitter.tool_finish(
                                    bound_tool_call,
                                    output=merged_output,
                                    error=error,
                                )
                            except Exception:
                                pass
                    finally:
                        self._bg_execs.pop(bg_id, None)
                        produced_artifacts = list(ctx.deps.artifacts[before_artifacts:])
                        if self.on_bg_complete is not None:
                            try:
                                self.on_bg_complete(
                                    bg_id=bg_id,
                                    error=error,
                                    output=output,
                                    new_artifacts=new_artifacts,
                                    produced_artifacts=produced_artifacts,
                                )
                            except Exception as cb_err:
                                print(f"{RED}on_bg_complete callback failed: {cb_err}{RST}")

                thread = threading.Thread(target=_run_bg, daemon=True, name=f"okonomi-{bg_id}")
                self._bg_execs[bg_id] = thread
                thread.start()

                # Brief wait so fast failures (KeyError, fal arg validation,
                # 4xx, malformed loras) surface as the *actual* tool result
                # instead of letting the model narrate around an already-
                # broken job. 2s catches almost every fast failure without
                # noticeably stalling successful long-running jobs (which
                # take 10–90s on fal endpoints).
                thread.join(timeout=2.0)
                self.spinner_text = ""

                log_entry = self._bg_logs.get(bg_id, {})
                if not thread.is_alive() and log_entry.get("status") == "error":
                    reader = log_entry.get("_reader")
                    snapshot = ""
                    if callable(reader):
                        try:
                            out = reader()
                            if isinstance(out, str):
                                snapshot = out.strip()
                        except Exception:
                            snapshot = ""
                    return (
                        f"Background task {bg_id} FAILED during dispatch.\n\n"
                        f"{snapshot[-2000:] if snapshot else '(no output captured)'}\n\n"
                        "The job did not produce media. Do not narrate expected output. "
                        "Surface the error to the user and ask before retrying — "
                        "re-running an expensive call to recover from a parse-side error "
                        "(KeyError / TypeError on `result[...]`) is the worst kind of mistake; "
                        "inspect first."
                    )

                return (
                    f"Dispatched to background as {bg_id}. Execution is in flight.\n\n"
                    "Acknowledge dispatch in ONE short sentence and stop — do NOT narrate "
                    "what the result will look like, the aesthetic it will produce, or any "
                    "other forward-looking claim. You'll be re-invoked with a completion "
                    "notice when it finishes; describe the actual output then."
                )

            try:
                output, error = await asyncio.to_thread(_exec_in_thread)

                # Attach output/error back to the UI's ToolCall event so the
                # user sees failures without having to ask the model what
                # happened. The widget auto-expands on error status.
                if self._pending_tool_call is not None and self.emitter is not None:
                    pending = self._pending_tool_call
                    self._pending_tool_call = None
                    try:
                        self.emitter.tool_finish(pending, output=output or "", error=error)
                    except Exception:
                        pass

                if error:
                    if self.verbose:
                        print_ansi_box(f"Error executing code:\n{error}", label="error", color="31")
                    hint = ""
                    # If the code hit a parsing error *after* touching an
                    # expensive tool, remind the model that globals persist
                    # and it should NOT re-run the expensive call.
                    if any(p in code for p in self._expensive_patterns) and any(
                        kind in error for kind in ("KeyError", "AttributeError", "TypeError", "IndexError")
                    ):
                        hint = (
                            "\n\nHINT: The generation call likely succeeded — "
                            "the `result` variable is still in globals. DO NOT "
                            "re-run genmedia.run/browser.render/fal_client. Write a new "
                            "execute_code block with `print(result)` to inspect "
                            "its shape, then fix the access pattern."
                        )
                    return f"Error executing code:\n{error}{hint}"

                if self.verbose and output:
                    print_ansi_box(output, label="output")

                if output:
                    from .context import _strip_large_content

                    cleaned = _strip_large_content(output)
                    if len(cleaned) > 10_000:
                        cleaned = cleaned[:8_000] + "\n...[truncated]...\n" + cleaned[-1_500:]
                    return cleaned

                return "Code executed successfully (no output)"

            except asyncio.CancelledError:
                self.spinner_text = ""
                raise
            finally:
                self.spinner_text = ""

    def _register_tool(
        self,
        agent: Agent,
        tool_instance: Any,
        rename: dict[str, str] | None = None,
    ) -> None:
        """Register all public methods of a tool instance as direct agent tools.

        Uses each method's existing signature and docstring for schema generation.

        Args:
            agent: The Pydantic AI agent to register tools on.
            tool_instance: An object whose public methods become agent tools.
            rename: Optional mapping of method names to alternate tool names.
        """
        rename = rename or {}

        for name, method in inspect.getmembers(tool_instance, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            tool_name = rename.get(name, name)
            sig = inspect.signature(method)

            def make_wrapper(m, s, tn, agent_self=self):
                def wrapper(*args, **kwargs):
                    agent_self.spinner_text = f"Running {tn}..."
                    pending = agent_self._pending_tool_call
                    agent_self._pending_tool_call = None
                    try:
                        result = m(*args, **kwargs)
                        # Surface the return value back onto the UI event.
                        if pending is not None and agent_self.emitter is not None:
                            error_msg = None
                            output = ""
                            if isinstance(result, dict) and result.get("error"):
                                # Tools often return {"error": "..."} on failure
                                # instead of raising — treat as an error visibly.
                                error_msg = str(result["error"])
                            else:
                                try:
                                    output = _short_repr(result)
                                except Exception:  # noqa: BLE001
                                    output = repr(result)
                            try:
                                agent_self.emitter.tool_finish(pending, output=output, error=error_msg)
                            except Exception:  # noqa: BLE001
                                pass
                        return result
                    except Exception as exc:
                        # Tool raised — surface traceback to the UI so the
                        # user sees what went wrong before the model
                        # confabulates a recovery.
                        if pending is not None and agent_self.emitter is not None:
                            try:
                                agent_self.emitter.tool_finish(pending, error=f"{type(exc).__name__}: {exc}")
                            except Exception:  # noqa: BLE001
                                pass
                        raise
                    finally:
                        agent_self.spinner_text = ""

                wrapper.__name__ = tn
                wrapper.__qualname__ = tn
                wrapper.__doc__ = m.__doc__
                wrapper.__signature__ = s
                wrapper.__annotations__ = {
                    k: v.annotation for k, v in s.parameters.items() if v.annotation is not inspect.Parameter.empty
                }
                if s.return_annotation is not inspect.Signature.empty:
                    wrapper.__annotations__["return"] = s.return_annotation
                return wrapper

            agent.tool_plain(make_wrapper(method, sig, tool_name), name=tool_name)

    def _get_dependencies(self) -> AgentDependencies:
        """Create dependencies for an agent run."""
        return AgentDependencies(
            config=self.config,
            session_id=self.session_id,
            session_manager=self.session_manager,
            tools=self.programmatic_tools,
            artifacts=self.artifacts,
            tools_used=self.tools_used,
            models_used=self.models_used,
            verbose=self.verbose,
            execution_globals=self.execution_globals,
            on_artifact_added=self._eager_download_artifact,
        )

    def _register_bg_log(self, *, task_id: str, label: str, reader) -> None:
        """Called by sub-tools (TerminalTools, AgentTools) when a bg job
        starts. Records the job in the shared log registry so /jobs and
        /logs <id> see it alongside the bg-exec-N execute_code path.

        Kind is inferred from the task_id prefix so /jobs renders the
        right column without needing each call site to thread it through."""
        import time as _time

        prefix = task_id.split("-", 1)[0] if "-" in task_id else "task"
        self._bg_logs[task_id] = {
            "kind": prefix,
            "label": label,
            "started_at": _time.time(),
            "ended_at": None,
            "status": "running",
            "_reader": reader,
        }

    def _mark_bg_log_done(self, *, task_id: str, status: str) -> None:
        """Called when a registered bg job (term-N) finishes."""
        import time as _time

        entry = self._bg_logs.get(task_id)
        if entry is not None:
            entry["status"] = status
            entry["ended_at"] = _time.time()

    def _eager_download_artifact(self, artifact: Artifact) -> None:
        """Submit a background download for a remote artifact.

        Fires from AgentDependencies.add_artifact the moment a tool tracks an
        asset. By the time save_session runs, most assets are already on
        disk and `_download_session_assets` is a fast no-op. Idempotent —
        already-downloaded artifacts are skipped.
        """
        if not artifact.url or artifact.path:
            return

        def _run() -> None:
            try:
                session_dir = self.session_manager.session_dir(self.session_id)
                session_dir.mkdir(parents=True, exist_ok=True)
                local_path = self._download_asset(artifact.url, session_dir, artifact.artifact_type)
                artifact.path = local_path
            except Exception:
                pass

        try:
            future = self._asset_dl_pool.submit(_run)
            self._asset_dl_futures.append(future)
        except RuntimeError:
            _run()

    def _accumulate_usage(self, agent_run) -> None:
        """Fold one ``agent_run`` worth of usage + cost into ``self.token_usage``.

        Tokens: pydantic-ai's ``RunUsage.input_tokens`` is already the
        non-cached prompt count (cache hits/writes are reported
        separately), so we copy each bucket straight across.
        ``reasoning_tokens`` lives in ``usage.details`` for providers
        that report it; keys vary, so we try the common ones.

        Cost: OpenRouter (with ``openrouter_usage={"include": True}``)
        attaches the per-request USD cost as
        ``ModelResponse.provider_details["cost"]``. We sum it across
        every new ``ModelResponse`` produced this turn — captures
        multi-round tool-calling turns where the model is invoked many
        times per user message.
        """
        usage = agent_run.usage()
        details = getattr(usage, "details", {}) or {}
        reasoning = int(details.get("reasoning_tokens") or details.get("output_reasoning_tokens") or 0)
        # Back-compat: older pydantic-ai exposed request_tokens/response_tokens.
        in_toks = getattr(usage, "input_tokens", None)
        if in_toks is None:
            in_toks = getattr(usage, "request_tokens", 0)
        out_toks = getattr(usage, "output_tokens", None)
        if out_toks is None:
            out_toks = getattr(usage, "response_tokens", 0)

        cost = 0.0
        try:
            for msg in agent_run.new_messages():
                if isinstance(msg, ModelResponse) and msg.provider_details:
                    c = msg.provider_details.get("cost")
                    if c is not None:
                        cost += float(c)
        except Exception:
            pass

        self.token_usage.add(
            input_tokens=in_toks or 0,
            output_tokens=out_toks or 0,
            requests=getattr(usage, "requests", 0) or 1,
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
            reasoning_tokens=reasoning,
            cost_usd=cost,
        )

    def _sanitize_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Remove trailing messages with unprocessed tool calls.

        After cancellation the history may end with a ModelResponse containing
        ToolCallParts that were never answered with ToolReturnParts.  Passing
        such history back to the model triggers:
          "Cannot provide a new user prompt when the message history
           contains unprocessed tool calls."

        We walk backwards and strip any trailing ModelResponse that has tool
        calls, plus any ModelRequest that only contains tool returns (since
        those are orphaned without their preceding response).
        """
        trimmed = list(messages)
        while trimmed:
            last = trimmed[-1]
            if isinstance(last, ModelResponse):
                has_tool_calls = any(isinstance(p, ToolCallPart) for p in last.parts)
                if has_tool_calls:
                    trimmed.pop()
                    continue
            if isinstance(last, ModelRequest):
                only_tool_returns = all(isinstance(p, ToolReturnPart) for p in last.parts)
                if only_tool_returns:
                    trimmed.pop()
                    continue
            break
        return trimmed

    def _extract_code(self, part: ToolCallPart) -> str:
        """Extract code string from an execute_code tool call."""
        args = part.args
        if isinstance(args, dict):
            return args.get("code", "")
        if isinstance(args, str):
            import json

            try:
                parsed = json.loads(args)
                return parsed.get("code", "") if isinstance(parsed, dict) else ""
            except json.JSONDecodeError:
                return args
        if hasattr(args, "code"):
            return args.code
        return str(args)

    def _display_tool_call(self, part: ToolCallPart) -> None:
        """Display a tool call with appropriate formatting."""
        if part.tool_name == "execute_code":
            code = self._extract_code(part)
            if not code:
                return
            self._last_code = code
            needs_approval = self.approval_callback is not None and any(p in code for p in self._expensive_patterns)
            if self.verbose or needs_approval:
                if self.emitter is not None:
                    line_count = code.count("\n") + 1
                    # Code that triggers approval is shown expanded by default
                    # so the user can read what they're about to authorize
                    # without needing to press Ctrl+O.
                    self._pending_tool_call = self.emitter.tool_call(
                        "execute_code",
                        args=f"{line_count} lines",
                        details=code,
                        expanded=needs_approval,
                    )
                else:
                    print_code_box(code)
            else:
                line_count = code.count("\n") + 1
                if self.emitter is not None:
                    self._pending_tool_call = self.emitter.tool_call(
                        "execute_code",
                        args=f"{line_count} lines",
                        details=code,
                    )
                else:
                    print(f"{DIM}▶ execute_code ({line_count} lines) — Ctrl+O to view{RST}")
        elif part.tool_name == "load_skill":
            import json

            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            name = args.get("name", args) if isinstance(args, dict) else args
            if self.emitter is not None:
                self._pending_tool_call = self.emitter.tool_call("load_skill", args=str(name), details=str(args))
            else:
                print(f"{DIM}Loading skill: {name}{RST}")
        else:
            if self.emitter is not None:
                self._pending_tool_call = self.emitter.tool_call(
                    part.tool_name,
                    args=str(part.args) if part.args else "",
                    details=str(part.args) if part.args else "",
                )
            elif self.verbose:
                print(f"{DIM}{part.tool_name}({part.args}){RST}")
            else:
                print(f"{DIM}▶ {part.tool_name}{RST}")

    async def run_async(
        self,
        user_message: str,
        reset: bool = False,
        timeout: float | None = None,
    ) -> str:
        """Run the agent asynchronously with a user message.

        Supports cancellation via asyncio.CancelledError — partial message history
        and generated outputs are preserved when interrupted.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history
            timeout: Wall-clock seconds before this turn is cancelled. None
                uses ``config.timeouts.agent_turn``; pass 0 or a negative
                value to disable. On timeout, partial output and message
                history are preserved (same path as user-initiated cancel).

        Returns:
            The agent's response
        """
        if reset:
            self.message_history = []
            self.execution_globals = {}
            self.artifacts = []
            self.tools_used = set()
            self.models_used = set()
            self.token_usage = TokenUsage()

        self.task = user_message
        self._was_interrupted = False
        self._partial_output = ""
        deps = self._get_dependencies()
        final_output = ""

        if self.message_history:
            self.message_history = self._sanitize_history(self.message_history)

            # After interruption, strip the cancelled turn's partial model
            # response so the new instruction starts from a clean state.
            if self._was_interrupted:
                while self.message_history and isinstance(self.message_history[-1], ModelResponse):
                    self.message_history.pop()

            self.message_history, was_pruned = self.context.prune_if_needed(self.message_history)
            if was_pruned:
                print(f"{DIM}Context pruned (stripped large content){RST}")

            if self.context.should_compress_preflight(self.message_history):
                compressed, was_compressed = self.context.compress(self.message_history)
                if was_compressed:
                    print(f"{DIM}Context compressed: {len(self.message_history)} → {len(compressed)} messages{RST}")
                    self.message_history = compressed

        # When resuming after an interruption, prepend a priority marker
        # so the model treats the new message as a course correction.
        if self._was_interrupted and self.message_history:
            user_message = (
                "[The user interrupted the previous response to give new instructions. "
                "Stop what you were doing and focus on this message.]\n\n" + user_message
            )

        # Append skill creation nudge when threshold is reached
        if self._skill_nudge_interval > 0 and self._turns_since_skill >= self._skill_nudge_interval:
            user_message += (
                "\n\n[System note: You've completed several tool-heavy turns. "
                "If you developed a reusable approach, consider saving it as a "
                "skill with create_skill().]"
            )
            self._turns_since_skill = 0

        effective_timeout = timeout if timeout is not None else self.config.timeouts.agent_turn
        if effective_timeout is not None and effective_timeout <= 0:
            effective_timeout = None

        # Promote any image references in the user's text to real
        # multimodal parts so the model sees the actual pixels instead
        # of a bare URL/path string. Returns a plain string when no
        # images are found (no overhead for the common path).
        prompt_arg = _build_multimodal_prompt(user_message)

        try:
            async with self.agent.iter(
                prompt_arg,
                deps=deps,
                message_history=self.message_history if self.message_history else None,
            ) as agent_run:
                try:
                    timeout_cm = (
                        asyncio.timeout(effective_timeout)
                        if effective_timeout is not None
                        else contextlib.nullcontext()
                    )
                    async with timeout_cm:
                        async for node in agent_run:
                            if not Agent.is_model_request_node(node):
                                continue
                            self.spinner_text = "Thinking..."
                            async with node.stream(agent_run.ctx) as stream:
                                accumulated_text = ""
                                line_buf = ""

                                async for text in stream.stream_text(delta=True, debounce_by=None):
                                    if self.spinner_text:
                                        self.spinner_text = ""
                                    accumulated_text += text
                                    if self.emitter is not None:
                                        if self._current_stream is None:
                                            self._current_stream = self.emitter.assistant_start()
                                        self.emitter.assistant_append(self._current_stream, text)
                                    else:
                                        line_buf += text
                                        while "\n" in line_buf:
                                            line, line_buf = line_buf.split("\n", 1)
                                            print(f"{LLM}{line}{RST}")

                                self.spinner_text = ""

                                if self.emitter is not None:
                                    if self._current_stream is not None:
                                        self.emitter.assistant_done(self._current_stream)
                                        self._current_stream = None
                                else:
                                    if line_buf.strip():
                                        print(f"{LLM}{line_buf}{RST}")
                                    if accumulated_text.strip():
                                        print()

                                if accumulated_text.strip():
                                    final_output = accumulated_text

                                response = stream.response
                                if response and response.parts:
                                    for part in response.parts:
                                        if isinstance(part, ToolCallPart):
                                            self._display_tool_call(part)

                except (asyncio.CancelledError, TimeoutError) as exc:
                    timed_out = isinstance(exc, TimeoutError)
                    self._was_interrupted = True
                    self._partial_output = final_output
                    self.message_history = self._sanitize_history(agent_run.all_messages())
                    self.artifacts = deps.artifacts
                    self._accumulate_usage(agent_run)
                    if timed_out:
                        msg = f"Turn timed out after {effective_timeout:.0f}s. Partial output preserved."
                        if self.emitter is not None:
                            self.emitter.note(msg, level="warn")
                        else:
                            print(f"{YELLOW}{msg}{RST}")
                    return self._partial_output

                result = agent_run.result

                self._accumulate_usage(agent_run)

            # Update message history for conversation continuity
            self.message_history = result.all_messages()
            # Refresh the estimated history size so the status bar shows
            # the true persistent context — what will be sent on the
            # NEXT turn — rather than the cumulative input across the
            # round-trips of the last turn.
            from .context import estimate_tokens_rough

            try:
                self.token_usage.history_tokens = estimate_tokens_rough(self.message_history)
            except Exception:
                pass
            self.artifacts = deps.artifacts

            # Skill creation nudge: track tool-calling turns and remind
            # the model to consider saving approaches as skills.
            messages = result.all_messages()
            used_skill_manage = any(
                isinstance(part, ToolCallPart)
                and part.tool_name in ("create_skill", "write_skill_file", "patch_skill_file")
                for msg in messages
                if isinstance(msg, ModelResponse)
                for part in msg.parts
            )
            if used_skill_manage:
                self._turns_since_skill = 0
            elif deps.tools_used:
                self._turns_since_skill += 1

            # Return the final output (use result.output if we didn't capture it during streaming)
            return final_output if final_output else result.output

        except asyncio.CancelledError:
            return self._partial_output
        except Exception as e:
            print(f"{RED}Error: {e}{RST}")
            return f"Error: {str(e)}"

    def run(self, user_message: str, reset: bool = False) -> str:
        """Run the agent synchronously with a user message.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        return asyncio.run(self.run_async(user_message, reset=reset))

    def _generate_steps_summary(self) -> str:
        """Generate LLM summary of session actions."""
        return self.context.generate_steps_summary(self.message_history)

    def _download_asset(self, url: str, session_dir: Path, asset_type: str) -> str:
        """Download an asset from URL to the session's asset subdirectory."""
        try:
            from .session import asset_subdir_for_file

            parsed = urlparse(url)
            url_filename = Path(parsed.path).name

            if Path(url_filename).suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".gif",
                ".mp4",
                ".webm",
                ".mov",
            }:
                filename = url_filename
            else:
                ext = ".jpg" if asset_type == "image" else ".mp4"
                filename = f"{url_filename}{ext}"

            subdir = asset_subdir_for_file(filename)
            asset_dir = session_dir / "assets" / subdir
            asset_dir.mkdir(parents=True, exist_ok=True)
            local_path = asset_dir / filename

            with httpx.Client(timeout=60.0) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)

            return str(local_path)
        except Exception:
            return ""

    def _download_session_assets(self, session_dir: Path) -> None:
        """Download all remote artifacts to the session directory.

        Crash-resistant: a failed download (network blip, fal expiry,
        etc.) does not abort the save. The artifact keeps its remote URL;
        the next save will retry the download.
        """
        for artifact in self.artifacts:
            if not artifact.url or artifact.path:
                continue
            try:
                local_path = self._download_asset(artifact.url, session_dir, artifact.artifact_type)
                artifact.path = local_path
            except Exception:
                continue

    def save_session(self, status: SessionStatus = "completed", download_assets: bool = True) -> None:
        """Save the current session state.

        Args:
            status: Status of the session (completed, error, abandoned)
            download_assets: When True (default), synchronously download
                any tracked outputs that don't yet have a ``local_path``.
                Set False on shutdown / atexit paths so quitting doesn't
                wait on slow network downloads — the eager bg pool has
                been pulling these all along, and any still-missing files
                will be retried on the next session resume.
        """
        session_dir = self.session_manager.session_dir(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        if download_assets:
            self._download_session_assets(session_dir)

        # Carry forward identity-shaping fields so mid-session title
        # regen doesn't relocate the directory or reset created_at.
        existing = self.session_manager.load_session(self.session_id, with_history=False)
        existing_dir_name = existing.dir_name if existing else ""
        existing_title = existing.title if existing else ""
        existing_created = existing.created_at if existing else ""

        now = datetime.now().isoformat()
        session = Session(
            session_id=self.session_id,
            task_input=self.task,
            status=status,
            created_at=existing_created or now,
            updated_at=now,
            num_artifacts=len(self.artifacts),
            model=self.config.agent.model_id,
            input_tokens=self.token_usage.input_tokens,
            output_tokens=self.token_usage.output_tokens,
            cache_read_tokens=self.token_usage.cache_read_tokens,
            cache_write_tokens=self.token_usage.cache_write_tokens,
            reasoning_tokens=self.token_usage.reasoning_tokens,
            requests=self.token_usage.requests,
            cost_usd=self.token_usage.cost_usd or None,
            # Body
            steps_summary=self._generate_steps_summary(),
            title=existing_title,
            dir_name=existing_dir_name,
            artifacts=list(self.artifacts),
            message_history=self.message_history if self.message_history else None,
        )

        # One-shot titling: title is set on first save and never regenerated.
        if not session.title:
            from .utils import generate_title_from_messages

            title = generate_title_from_messages(self.message_history, fallback=self.task)
            if title:
                session.title = title

        self.session_manager.save_session(session)

        # Pickle user-bound execution_globals (vars the model created in
        # execute_code blocks) so they survive a quit + resume. Tool
        # wrappers and helpers (genmedia/web/get_session_dir/...) are
        # re-injected on the next execute_code call — we exclude them here.
        self._serialize_user_globals()

    def _serialize_user_globals(self) -> None:
        """Save picklable user-bound variables from ``execution_globals`` to
        ``<session_dir>/globals.pkl``. Tool wrappers, modules, functions,
        and anything that fails to pickle are silently skipped — the goal
        is to recover the most-used patterns (URLs, paths, lists of
        descriptions, intermediate dicts), not arbitrary live objects.
        """
        import pickle
        import types

        if not self.execution_globals:
            return

        SKIP_KEYS = {"__builtins__", "asyncio", "get_session_dir"}
        SKIP_KEYS |= set(self.programmatic_tools.keys())  # tool wrappers

        snapshot: dict[str, Any] = {}
        for key, value in self.execution_globals.items():
            if key in SKIP_KEYS or key.startswith("__"):
                continue
            if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
                continue
            # ToolWrappers and Tool instances both carry _tool_instance
            # or _set_deps — drop anything that smells tool-ish.
            if hasattr(value, "_tool_instance") or hasattr(value, "_set_deps"):
                continue
            try:
                pickle.dumps(value)
            except Exception:
                continue
            snapshot[key] = value

        try:
            session_dir = self.session_manager.session_dir(self.session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            with open(session_dir / "globals.pkl", "wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _restore_user_globals(self) -> None:
        """Counterpart to ``_serialize_user_globals``. Reads
        ``<session_dir>/globals.pkl`` if present and merges into
        ``self.execution_globals``. Tool wrappers + helpers will be
        reinstalled on the next ``execute_code`` call."""
        import pickle

        try:
            session_dir = self.session_manager.session_dir(self.session_id)
            globals_path = session_dir / "globals.pkl"
            if not globals_path.exists():
                return
            with open(globals_path, "rb") as f:
                snapshot = pickle.load(f)
            if isinstance(snapshot, dict):
                self.execution_globals.update(snapshot)
        except Exception:
            pass

    def load_session(self, session_id: str) -> bool:
        """Load agent state from a saved session.

        Restores the full message history when available so the model
        sees its prior tool calls, intermediate reasoning, and produced
        artifacts directly. The next-turn ``run_async`` already runs
        prune + compress on the restored history, so oversized sessions
        get trimmed there instead of being stripped here.

        For legacy sessions without ``messages.jsonl`` (older saves), we
        fall back to a synthesized summary message — better than nothing
        but the agent will know much less about what happened.

        Args:
            session_id: ID of the session to load

        Returns:
            True if session was loaded successfully
        """
        session = self.session_manager.load_session(session_id)

        if not session:
            return False

        self.session_id = session_id
        self.task = session.task_input
        self.artifacts = session.artifacts

        if session.message_history:
            # Real conversation. Strip any orphaned tool calls at the tail
            # (interrupted save) — same hygiene we apply mid-turn.
            self.message_history = self._sanitize_history(list(session.message_history))
            if not self.message_history:
                # Sanitization wiped everything (rare). Fall back to summary.
                self.message_history = self.context.build_resume_context(
                    session_id=session_id,
                    task=session.task_input,
                    steps_summary=session.steps_summary,
                    artifacts=session.artifacts,
                )
        else:
            # Legacy session — no jsonl. Synthesize from metadata.
            self.message_history = self.context.build_resume_context(
                session_id=session_id,
                task=session.task_input,
                steps_summary=session.steps_summary,
                artifacts=session.artifacts,
            )

        self.token_usage = TokenUsage(
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            total_tokens=session.input_tokens + session.output_tokens,
            cache_read_tokens=session.cache_read_tokens,
            cache_write_tokens=session.cache_write_tokens,
            reasoning_tokens=session.reasoning_tokens,
            requests=session.requests,
            cost_usd=float(session.cost_usd or 0.0),
        )

        # Restore Python locals the model bound during the prior session.
        # Tool wrappers / helpers are re-injected on the next execute_code
        # call, so we only need user-bound vars here.
        self._restore_user_globals()

        return True
