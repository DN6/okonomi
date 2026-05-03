"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM interactions
with a local Python code executor that has access to all tool instances.
"""

import asyncio
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
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
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
from .display import DIM, LLM, RST, RED, GREEN, print_ansi_box, print_code_box
from .session import Artifact, GeneratedOutput, Session, SessionManager, SessionMetadata
from .tools.agents import AgentTools
from .tools._browser_renderer import BrowserRendererTools
from .tools._image_processing import EditingTools
from .tools._image_generation import ImageGenerationTools
from .tools._prompt import PromptTools
from .tools._research import ResearchTools
from .tools.skills import SkillTools
from .tools._video import VideoTools
from .tools.terminal import TerminalTools


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
    """Tracks token usage across agent runs."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    last_turn_input: int = 0
    history_tokens: int = 0  # estimated size of the persistent message_history

    def add(self, input_tokens: int, output_tokens: int, requests: int = 1) -> None:
        """Add usage from a single run.

        ``input_tokens`` is the *cumulative* input across every model
        round-trip in the agent turn (pydantic-ai's
        ``usage.request_tokens``). It is NOT the size of the persisted
        message history — see ``history_tokens`` for that, set by
        ``OkonomiAgent.run_async`` after pruning/compression.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.requests += requests
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

    config: OkonomiConfig
    session_id: str
    session_manager: SessionManager
    tools: dict[str, Any] = field(default_factory=dict)
    generated_outputs: list[GeneratedOutput] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    input_assets: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)
    models_used: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    verbose: bool = False
    # Persistent state for code execution across multiple calls
    execution_globals: dict[str, Any] = field(default_factory=dict)
    # Optional hook called from add_output so the agent can kick off an
    # eager background download of the asset. Set by OkonomiAgent in
    # _get_dependencies. Signature: (output) -> None, must not raise.
    on_output_added: Optional[Any] = None

    def add_output(self, output: GeneratedOutput) -> None:
        """Track a generated output. Triggers eager bg download via hook."""
        self.generated_outputs.append(output)
        if output.tool_name:
            self.tools_used.add(output.tool_name)
        if output.model_id:
            self.models_used.add(output.model_id)
        if self.on_output_added is not None:
            try:
                self.on_output_added(output)
            except Exception:
                pass

    def add_artifact(self, path: str, artifact_type: str, description: str = "", url: str = "") -> None:
        """Track a created artifact (file, research report, skill, etc.)."""
        self.artifacts.append(
            Artifact(
                path=path,
                artifact_type=artifact_type,
                description=description,
                timestamp=datetime.now().isoformat(),
                url=url,
            )
        )

    def add_input_asset(self, url: str) -> None:
        """Track an input asset (image/video URL)."""
        if url and url not in self.input_assets:
            self.input_assets.append(url)

    def add_error(self, error: str) -> None:
        """Track an error that occurred during execution."""
        self.errors.append(error)

    def get_tool(self, namespace: str) -> Any:
        """Get a tool instance by namespace."""
        return self.tools.get(namespace)


class ToolResult(dict):
    """Dict subclass that shows available keys on KeyError."""

    def __missing__(self, key):
        available = ", ".join(repr(k) for k in self.keys())
        raise KeyError(f"Key {key!r} not found. Available keys: {available}")


def _extract_bg_label(code: str) -> str:
    """First identifiable expensive call in the code, used as a human label.

    Order matters: prefer a `genmedia run <model-id>` match over a generic
    namespace match, since the model id is a far more useful label than
    "run". Falls back to "code block" when nothing recognisable is found.
    """
    import re

    # `genmedia run fal-ai/flux/dev …` → label = `fal-ai/flux/dev`
    gm = re.search(r"\bgenmedia\s+run\s+(\S+)", code)
    if gm:
        return gm.group(1)

    match = re.search(
        r"\b(image_generation|video|browser|image_processing|prompt|research|sessions)\.(\w+)",
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
        session_dir = self._deps.session_manager.get_session_dir(self._deps.session_id)
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

                for key in ["image", "image_url", "video", "video_url", "first_frame", "last_frame"]:
                    if key in kwargs and kwargs[key]:
                        self._deps.add_input_asset(kwargs[key])

                if isinstance(result, dict):
                    self._track_outputs(result, name, kwargs)
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

    def _track_outputs(self, result: dict, method_name: str, kwargs: dict) -> None:
        """Extract and track generated outputs from tool results."""
        now = datetime.now().isoformat()
        tool_name = f"{self._namespace}.{method_name}"
        prompt = kwargs.get("prompt", "")

        output_path = result.get("output_path", "")

        if "images" in result:
            for img_url in result["images"]:
                local = output_path if output_path and Path(output_path).exists() else ""
                output = GeneratedOutput(
                    asset_url=img_url,
                    asset_type="image",
                    prompt=prompt,
                    tool_name=tool_name,
                    model_id=result.get("model_id", ""),
                    timestamp=now,
                    local_path=local,
                )
                self._deps.add_output(output)
                self._deps.add_artifact(
                    path=local or img_url,
                    artifact_type="image",
                    description=prompt[:200],
                    url=img_url,
                )

        if "video" in result:
            local = output_path if output_path and Path(output_path).exists() else ""
            output = GeneratedOutput(
                asset_url=result["video"],
                asset_type="video",
                prompt=prompt,
                tool_name=tool_name,
                model_id=result.get("model_id", ""),
                timestamp=now,
                local_path=local,
            )
            self._deps.add_output(output)
            self._deps.add_artifact(
                path=local or result["video"],
                artifact_type="video",
                description=prompt[:200],
                url=result["video"],
            )

        if "screenshot" in result:
            local = output_path if output_path and Path(output_path).exists() else ""
            output = GeneratedOutput(
                asset_url=result["screenshot"],
                asset_type="image",
                prompt=prompt,
                tool_name=tool_name,
                timestamp=now,
                local_path=local,
            )
            self._deps.add_output(output)
            self._deps.add_artifact(
                path=local or result["screenshot"],
                artifact_type="image",
                description=prompt[:200],
            )


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
        self.session_id = session_id or self.session_manager.create_session_id()
        self._setup_tools()
        self.message_history: list[ModelMessage] = []
        self.task: str = ""
        self.verbose = verbose

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
        self.generated_outputs: list[GeneratedOutput] = []
        self.artifacts: list[Artifact] = []
        self.input_assets: list[str] = []
        self.tools_used: set[str] = set()
        self.models_used: set[str] = set()
        self.errors: list[str] = []

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
            "image_generation.",
            "video.",
            "browser.render",
            "fal_client",
            # genmedia is a CLI front-end for fal — invoking
            # `genmedia run <id>` triggers the same approval +
            # bg-exec auto-dispatch as a direct fal call.
            "genmedia run",
        ]

        self._bg_exec_counter = 0
        # Background execute_code jobs keyed by bg_id. Plain threads (not asyncio
        # tasks) so they survive the per-turn event loop teardown in the CLI.
        import threading as _threading_mod

        self._bg_execs: dict[str, _threading_mod.Thread] = {}
        # bg_id → short human description (first tool-method called in the code
        # block, e.g. "image_generation.text_to_image") so the UI can show
        # something more useful than "bg-exec-1".
        self._bg_exec_meta: dict[str, str] = {}

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
        # Programmatic tools — called from inside execute_code
        self.programmatic_tools: dict[str, Any] = {
            "browser": BrowserRendererTools(self.config),
            "image_processing": EditingTools(self.config),
            "image_generation": ImageGenerationTools(self.config),
            "prompt": PromptTools(self.config),
            "research": ResearchTools(self.config),
            "video": VideoTools(self.config),
        }

        # Standalone tools — direct agent tools (one call per round-trip)
        self.terminal = TerminalTools(self.config)
        self.agents = AgentTools(self.config)
        self.skills = SkillTools(
            self.config,
            available_tools=set(self.programmatic_tools.keys()) | {"terminal", "skills"},
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
        self.agents._bind(self)
        self._register_tool(agent, self.agents, rename={"launch": "launch_agent", "list_agents": "list_agents"})

        # Programmatic tools (available inside execute_code)
        self._register_code_execution_tool(agent)

        return agent

    def _create_model(self) -> CachedOpenRouterModel:
        return CachedOpenRouterModel(self.config.agent.model_id)

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
                loaded = self.skills.load_skill(name)
            except Exception:
                continue
            content = loaded.get("instructions") or loaded.get("content") or ""
            if not content:
                continue
            blocks.append(
                f"## Core skill: `{name}` (always active)\n\n{content.strip()}"
            )
        return "\n\n".join(blocks)

    def _build_namespace_list(self) -> str:
        """Comma-separated namespace names. Used by callers that want a
        compact one-liner. Auto-syncs when programmatic_tools changes."""
        return ", ".join(f"`{ns}`" for ns in self.programmatic_tools.keys())

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
                text = text[len("<class '"):-len("'>")]
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

1. **Research & Ideation** - Load the `deep-concept-research` skill for structured creative exploration. Use `research.web_search()` for inspiration and references.
2. **Determine Style** - Decide on the visual style (photorealistic, illustration, anime, painterly, etc.) based on the concept.
3. **Create Prompt** - Use `prompt.create_prompt()` with context from your research.
4. **Generate** - Proceed to image generation with the developed concept.

## Tool-Specific Requirements

### edit_image & reference_image
These tools require **structured prompts** that explicitly state:
- **What to preserve**: style, colors, composition, subject identity
- **What to change**: specific elements to modify
- **Reference attribution**: "The style from image1", "The person from image2"

Example structured prompts:
- "Preserve the color palette and lighting from image1. Apply it to a new scene of a mountain landscape at dawn."
- "Keep the person's face and clothing from image1. Change the background to a futuristic cityscape."
- "Using the painterly style from image1, recreate the composition with autumn colors."

## Rules

1. Use `execute_code` to run Python code for image/video generation tools
2. Use `print()` to see results from code execution
3. Variables persist between executions — if code fails, prior variables are still available
4. All methods are synchronous - do NOT use `await`
5. **NEVER re-run `image_generation.*`, `video.*`, `browser.render`, or `fal_client` calls to recover from an error.** These calls are expensive, slow, and cost money. The generation result is still sitting in your previous `result` variable — globals persist across `execute_code` calls. If the previous call failed with a `KeyError` / `AttributeError` / `TypeError` after the generation, write a *new* `execute_code` block that does `print(result)` and fixes the access pattern. If the generation call itself raised (network error, 500), inform the user and ask before retrying — do not silently re-run. Re-running a successful generation because of a parsing bug is the single most wasteful mistake you can make.
6. **The Programmatic Tools section above has every method's signature, parameter types, and return-dict shape.** Read it before writing code that calls a tool. Never guess `image_url` / `videos` / `url` — the literal return keys are right there. Use `inspect.getdoc(method)` only when you need the prose docstring (longer explanations, examples, edge-case notes); for signatures and return shapes the prompt is authoritative.
7. **Saving files**: All tools have a `session_dir` property (e.g., `video.session_dir`) for the current session folder.
   Methods with `output_path` parameters auto-save to session directory when passed `output_path=None`.
8. **Standalone tools** (`run`, `run_background`, `search`, `load_skill`, `create_skill`, `edit_skill`, `list_skills`, `launch_agent`, `list_agents`) — call directly, NOT through `execute_code`.
9. **Programmatic tools** (`browser`, `image_processing`, `image_generation`, `prompt`, `research`, `video`) — use inside `execute_code` only.
11. **Background agents**: Use `launch_agent(task, description)` to delegate independent work (e.g., generating multiple images in parallel). Check progress with `list_agents()`. Outputs are automatically merged back into the session.
12. **Media/expensive code always runs in the background after approval.** Any `execute_code` block touching `image_generation.`, `video.`, `browser.render`, or `fal_client` is auto-dispatched as a background task (returns a `bg-exec-N` handle, no stdout, no `result` variable). Do NOT claim the media is ready in that turn. You'll be re-invoked with a completion notice when it finishes; the artifact will already be in the session. You may also pass `background=True` explicitly for other long-running code.
10. **Local storage layout** — when asked to find or summarize prior work, search the right directory. These paths reflect the user's *actual* configuration (honoring `$OKONOMI_HOME` / session-storage override):
   - `{sessions_dir}` — full conversation logs and session metadata. Use this for "what did we discuss in past sessions".
   - `{research_dir}` — saved research reports (markdown files like `digital-decay-concept-brief.md`). Use this for "summarize my research" / "what concepts have I explored". The `concept_research` skill writes here.
   - `{skills_dir}` — reusable skill definitions. Use this for "what skills exist".
   - `{favourites_dir}` — user-favourited assets.

   Examples (use the literal paths above, not `~/.okonomi/...`):
   ```
   search(pattern="<keyword>", path="{research_dir}", file_glob="*.md")
   run(command="ls {research_dir}")
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
            - image_processing: Image loading, segmentation, layer decomposition
            - image_generation: Text-to-image, image-to-image, editing
            - prompt: Prompt creation and optimization
            - research: Web search
            - video: Video generation

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
            # Initialize execution globals with tool wrappers if not already done
            if not ctx.deps.execution_globals:
                ctx.deps.execution_globals = {
                    "__builtins__": __builtins__,
                    "asyncio": asyncio,
                }
                # Add wrapped tools
                for namespace, tool_instance in ctx.deps.tools.items():
                    ctx.deps.execution_globals[namespace] = ToolWrapper(tool_instance, ctx.deps, namespace)

                def get_session_dir() -> str:
                    """Get the current session's directory path.

                    Use this to save files (like stitched videos) to the session folder.

                    Returns:
                        Absolute path to the current session directory.

                    Example:
                        session_dir = get_session_dir()
                        output_path = f"{session_dir}/final_video.mp4"
                        result = video.stitch_videos(video_urls, output_path)
                    """
                    session_dir = ctx.deps.session_manager.get_session_dir(ctx.deps.session_id)
                    session_dir.mkdir(parents=True, exist_ok=True)
                    return str(session_dir)

                def track_output(
                    url: str,
                    asset_type: str,
                    prompt: str = "",
                    model_id: str = "",
                    tool_name: str = "genmedia",
                ) -> None:
                    """Register a generated asset URL with the session.

                    Use after every successful media generation (e.g.
                    `genmedia run` or any other path that produces a
                    URL). Triggers eager background download into the
                    session directory, surfaces in `/assets`, and
                    counts toward bg-exec completion summaries.

                    Args:
                        url: The remote URL of the generated asset
                            (https://fal.media/..., …).
                        asset_type: One of "image", "video", "audio",
                            "3d", "other".
                        prompt: The prompt or input that produced this
                            asset (helps identify it in /assets).
                        model_id: The model that generated it (e.g.
                            "fal-ai/flux/dev"). Optional but useful.
                        tool_name: Defaults to "genmedia". Override
                            when calling from a different code path.

                    Call once per produced URL. Call multiple times if
                    a single generation returned several assets.
                    """
                    if not url:
                        return
                    output = GeneratedOutput(
                        asset_url=url,
                        asset_type=asset_type or "other",
                        prompt=prompt or "",
                        tool_name=tool_name or "",
                        model_id=model_id or "",
                        timestamp=datetime.now().isoformat(),
                    )
                    ctx.deps.add_output(output)

                def track_artifact(
                    path: str,
                    artifact_type: str = "other",
                    description: str = "",
                    url: str = "",
                ) -> None:
                    """Register a local file artifact with the session.

                    Use for local files the agent created (HTML
                    sketches, stitched videos, downloaded sources)
                    that should appear in session metadata.
                    """
                    if not path:
                        return
                    ctx.deps.add_artifact(
                        path=path,
                        artifact_type=artifact_type or "other",
                        description=description or "",
                        url=url or "",
                    )

                ctx.deps.execution_globals["get_session_dir"] = get_session_dir
                ctx.deps.execution_globals["track_output"] = track_output
                ctx.deps.execution_globals["track_artifact"] = track_artifact

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

                self._bg_exec_counter += 1
                bg_id = f"bg-exec-{self._bg_exec_counter}"
                before_outputs = len(ctx.deps.generated_outputs)
                before_artifacts = len(ctx.deps.artifacts)

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
                    new_outputs = 0
                    new_artifacts = 0
                    try:
                        output, error = _exec_in_thread()
                        new_outputs = len(ctx.deps.generated_outputs) - before_outputs
                        new_artifacts = len(ctx.deps.artifacts) - before_artifacts
                        if error:
                            print(f"\n{RED}✗ {bg_id} failed:\n{error}{RST}")
                        else:
                            summary = []
                            if new_outputs:
                                summary.append(f"{new_outputs} output{'s' if new_outputs != 1 else ''}")
                            if new_artifacts:
                                summary.append(f"{new_artifacts} artifact{'s' if new_artifacts != 1 else ''}")
                            tail = f" — {', '.join(summary)}" if summary else ""
                            print(f"\n{GREEN}✓ {bg_id} done{tail}{RST}")
                            if output and self.verbose:
                                print_ansi_box(output, label=f"{bg_id} output")
                        # Update the original ToolCall event so the user
                        # sees the bg job's result in-place in history.
                        if bound_tool_call is not None and self.emitter is not None:
                            try:
                                parts = []
                                if new_outputs:
                                    parts.append(f"{new_outputs} output{'s' if new_outputs != 1 else ''}")
                                if new_artifacts:
                                    parts.append(f"{new_artifacts} artifact{'s' if new_artifacts != 1 else ''}")
                                summary_str = ", ".join(parts) if parts else "done"
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
                        self._bg_exec_meta.pop(bg_id, None)
                        # Snapshot the actual entries appended by this bg run
                        # so the callback can name them precisely instead of
                        # leaving the agent to guess from session-wide state.
                        produced_outputs = list(ctx.deps.generated_outputs[before_outputs:])
                        produced_artifacts = list(ctx.deps.artifacts[before_artifacts:])
                        if self.on_bg_complete is not None:
                            try:
                                self.on_bg_complete(
                                    bg_id=bg_id,
                                    error=error,
                                    output=output,
                                    new_outputs=new_outputs,
                                    new_artifacts=new_artifacts,
                                    produced_outputs=produced_outputs,
                                    produced_artifacts=produced_artifacts,
                                )
                            except Exception as cb_err:
                                print(f"{RED}on_bg_complete callback failed: {cb_err}{RST}")

                thread = threading.Thread(target=_run_bg, daemon=True, name=f"okonomi-{bg_id}")
                self._bg_execs[bg_id] = thread
                # Scrape a short label for the UI ("image_generation.text_to_image",
                # "video.generate", etc.) so running bg work is identifiable.
                self._bg_exec_meta[bg_id] = _extract_bg_label(code)
                thread.start()
                self.spinner_text = ""
                return (
                    f"Dispatched to background as {bg_id}. Execution is in flight — "
                    "do NOT claim the media is ready yet. You will be notified on "
                    "completion via a new turn; any generated outputs are merged "
                    "into the session automatically."
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
                            "re-run image_generation/video/fal_client. Write a new "
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
            generated_outputs=self.generated_outputs,
            artifacts=self.artifacts,
            input_assets=self.input_assets,
            tools_used=self.tools_used,
            models_used=self.models_used,
            errors=self.errors,
            verbose=self.verbose,
            execution_globals=self.execution_globals,
            on_output_added=self._eager_download_output,
        )

    def _eager_download_output(self, output: GeneratedOutput) -> None:
        """Submit a background download for a generated output.

        Fires from AgentDependencies.add_output the moment a tool tracks an
        asset. By the time save_session runs, most assets are already on
        disk and `_download_session_assets` is a fast no-op. Idempotent —
        already-downloaded outputs are skipped.
        """
        if not output.asset_url or output.local_path:
            return

        def _run() -> None:
            try:
                session_dir = self.session_manager.get_session_dir(self.session_id)
                session_dir.mkdir(parents=True, exist_ok=True)
                local_path = self._download_asset(output.asset_url, session_dir, output.asset_type)
                # Mutating from a worker thread is safe: GeneratedOutput is
                # a plain dataclass with no other writers for local_path.
                output.local_path = local_path
            except Exception:
                # Leave local_path empty so save_session retries later.
                pass

        try:
            future = self._asset_dl_pool.submit(_run)
            self._asset_dl_futures.append(future)
        except RuntimeError:
            # Pool was shut down (process exiting) — fall back to sync.
            _run()

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
            needs_approval = self.approval_callback is not None and any(
                p in code for p in self._expensive_patterns
            )
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

    async def run_async(self, user_message: str, reset: bool = False) -> str:
        """Run the agent asynchronously with a user message.

        Supports cancellation via asyncio.CancelledError — partial message history
        and generated outputs are preserved when interrupted.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        if reset:
            self.message_history = []
            self.execution_globals = {}
            self.generated_outputs = []
            self.artifacts = []
            self.input_assets = []
            self.tools_used = set()
            self.models_used = set()
            self.errors = []
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

        try:
            async with self.agent.iter(
                user_message,
                deps=deps,
                message_history=self.message_history if self.message_history else None,
            ) as agent_run:
                try:
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

                except asyncio.CancelledError:
                    self._was_interrupted = True
                    self._partial_output = final_output
                    self.message_history = self._sanitize_history(agent_run.all_messages())
                    self.generated_outputs = deps.generated_outputs
                    self.artifacts = deps.artifacts
                    usage = agent_run.usage()
                    self.token_usage.add(
                        input_tokens=usage.request_tokens or 0,
                        output_tokens=usage.response_tokens or 0,
                        requests=usage.requests or 1,
                    )
                    return self._partial_output

                result = agent_run.result

                # Track token usage
                usage = agent_run.usage()
                self.token_usage.add(
                    input_tokens=usage.request_tokens or 0,
                    output_tokens=usage.response_tokens or 0,
                    requests=usage.requests or 1,
                )

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
            self.generated_outputs = deps.generated_outputs
            self.artifacts = deps.artifacts

            # Skill creation nudge: track tool-calling turns and remind
            # the model to consider saving approaches as skills.
            messages = result.all_messages()
            used_skill_manage = any(
                isinstance(part, ToolCallPart) and part.tool_name in ("create_skill", "edit_skill")
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
        """Download all generated outputs to the session directory.

        Crash-resistant: a failed download (network blip, fal expiry,
        etc.) does not abort the save. The output keeps its remote URL;
        the next save will retry the download.
        """
        for output in self.generated_outputs:
            if not output.asset_url or output.local_path:
                continue
            try:
                local_path = self._download_asset(output.asset_url, session_dir, output.asset_type)
                output.local_path = local_path
            except Exception:
                # Swallow — preserve URL for retry, keep iterating.
                continue

    def save_session(self, status: str = "completed", download_assets: bool = True) -> None:
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
        # Get session directory and ensure it exists
        session_dir = self.session_manager.get_session_dir(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        if download_assets:
            self._download_session_assets(session_dir)

        now = datetime.now().isoformat()
        metadata = SessionMetadata(
            session_id=self.session_id,
            task_input=self.task,
            status=status,
            created_at=now,
            updated_at=now,
            num_outputs=len(self.generated_outputs),
            num_artifacts=len(self.artifacts),
            tools_used=list(self.tools_used),
            models_used=list(self.models_used),
            has_input_assets=len(self.input_assets) > 0,
            token_usage={
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "requests": self.token_usage.requests,
            },
        )

        session = Session(
            metadata=metadata,
            generated_outputs=self.generated_outputs,
            artifacts=self.artifacts,
            input_assets=self.input_assets,
            steps_summary=self._generate_steps_summary(),
            errors=self.errors,
            message_history=self.message_history if self.message_history else None,
        )

        if not metadata.title and self.task:
            from .utils import generate_title

            title = generate_title(self.task)
            if title:
                metadata.title = title

        self.session_manager.save_session(session)

    def load_session(self, session_id: str) -> bool:
        """Load agent state from a saved session.

        Restores outputs, metadata, and injects a summary as context
        rather than restoring the full message history (saves tokens).
        The system prompt includes generated outputs via _build_session_context().

        Args:
            session_id: ID of the session to load

        Returns:
            True if session was loaded successfully
        """
        session = self.session_manager.load_session(session_id)

        if not session:
            return False

        self.session_id = session_id
        self.task = session.metadata.task_input
        self.generated_outputs = session.generated_outputs
        self.artifacts = session.artifacts
        self.input_assets = session.input_assets
        self.tools_used = set(session.metadata.tools_used)
        self.models_used = set(session.metadata.models_used)
        self.errors = session.errors

        self.message_history = self.context.build_resume_context(
            session_id=session_id,
            task=session.metadata.task_input,
            steps_summary=session.steps_summary,
            generated_outputs=session.generated_outputs,
            artifacts=session.artifacts,
        )

        if session.metadata.token_usage:
            tu = session.metadata.token_usage
            self.token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
                total_tokens=tu.get("total_tokens", 0),
                requests=tu.get("requests", 0),
            )

        return True
