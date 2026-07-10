"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM
interactions, along with its dependency container (``AgentDependencies``)
and asset-download helpers. The class is assembled from focused modules:
the turn loop and event wiring live here; code execution (``execution``)
is a mixin; session persistence (``persistence``) is a composed
``SessionStore``; system-prompt construction (``prompt``), provider glue
(``provider``), tool wrappers (``wrappers``), and image-reference parsing
(``multimodal``) are plain modules.
"""

import asyncio
import contextlib
import inspect
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from ..context import _CHARS_PER_TOKEN, ContextManager, count_tokens
from ..events import AssistantStream, EventEmitter, SystemNote, ToolCall, TurnEnd, TurnStart
from ..session import (
    Artifact,
    OkonomiState,
    SessionInfo,
    SessionManager,
    SessionStatus,
    asset_subdir_for_file,
    new_session_id,
)
from ..tools import (
    AgentTools,
    BrowserRendererTools,
    FileTools,
    GenmediaTools,
    SessionsTools,
    SkillTools,
    TerminalTools,
    WebTools,
)
from .execution import CodeExecutionMixin
from .multimodal import _parse_user_message
from .persistence import SessionStore, strip_unanswered_tool_calls
from .prompt import build_system_prompt
from .provider import CachedOpenRouterModel
from .wrappers import _short_repr


@dataclass
class AgentDependencies:
    """Dependencies injected into pydantic-ai tool calls and execute_code.

    Holds a reference to the live ``OkonomiState`` so tool wrappers can
    record artifacts and tool usage in one place, plus the session
    manager (for path resolution) and the eager-download hook.
    """

    state: OkonomiState
    session_manager: SessionManager
    tools: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    # Fired from add_artifact so the agent can kick off an eager
    # background download for artifacts with a remote ``url``.
    # Signature: (artifact: Artifact) -> None, must not raise.
    on_artifact_added: Any | None = None

    def add_artifact(self, artifact: Artifact) -> None:
        """Track an artifact (AI-generated media OR locally-written file).

        Stamps a timestamp if missing, records the producing tool name,
        and fires the eager-download hook for artifacts with a remote
        ``url``.
        """
        if not artifact.timestamp:
            artifact.timestamp = datetime.now().isoformat()
        self.state.artifacts.append(artifact)

        if artifact.tool_name:
            self.state.tools_used.add(artifact.tool_name)

        if self.on_artifact_added is not None:
            try:
                self.on_artifact_added(artifact)
            except Exception:
                pass


# Maps a file extension to the artifact_type used by /assets and search_artifacts.
# execute_code writes media straight to the session dir (PIL/matplotlib/ffmpeg)
# without going through add_artifact; a post-run sweep keyed on this table is what
# turns those files into tracked assets.
_MEDIA_EXT_TO_TYPE = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".mp4": "video",
    ".webm": "video",
    ".mov": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".glb": "3d",
    ".gltf": "3d",
    ".obj": "3d",
    ".ply": "3d",
    ".html": "html",
    ".htm": "html",
}

_KNOWN_ASSET_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov"}


def download_asset(url: str, session_dir: Path, asset_type: str) -> str:
    """Download ``url`` into ``<session_dir>/assets/<subdir>/`` and return
    the local path. Returns ``""`` on any failure (network blip, fal
    expiry) — callers retry on the next save."""
    try:
        url_filename = Path(urlparse(url).path).name
        if Path(url_filename).suffix.lower() in _KNOWN_ASSET_EXTS:
            filename = url_filename
        else:
            ext = ".jpg" if asset_type == "image" else ".mp4"
            filename = f"{url_filename}{ext}"

        asset_dir = session_dir / "assets" / asset_subdir_for_file(filename)
        asset_dir.mkdir(parents=True, exist_ok=True)
        local_path = asset_dir / filename

        with httpx.Client(timeout=60.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            local_path.write_bytes(response.content)

        return str(local_path)
    except Exception:
        return ""


def download_session_assets(artifacts: list[Artifact], session_dir: Path) -> None:
    """Download every remote artifact missing a local path into ``session_dir``.

    Mutates each artifact in place — sets ``path`` on success, leaves it
    empty on failure so the next save can retry.
    """
    for artifact in artifacts:
        if not artifact.url or artifact.path:
            continue
        local_path = download_asset(artifact.url, session_dir, artifact.artifact_type)
        if local_path:
            artifact.path = local_path


class OkonomiAgent(CodeExecutionMixin):
    """Pydantic AI-based agent with local code execution.

    This agent provides:
    - Code execution with tool instances injected into the environment
    - Session management for continuity
    - Message history for multi-turn conversations
    """

    def __init__(
        self,
        *,
        model_id: str = "anthropic/claude-sonnet-4-5",
        timeout: float = 600.0,
        compress_at_tokens: int = 170_000,
        storage_dir: Path | None = None,
        session_id: str | None = None,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.timeout = timeout
        self.compress_at_tokens = compress_at_tokens
        self.session_manager = SessionManager(storage_dir=storage_dir)
        self.state = OkonomiState(info=SessionInfo(session_id=session_id or new_session_id(), model=model_id))
        self.verbose = verbose
        self._setup_tools()

        # Background pool for eager asset downloads. Daemon threads —
        # don't block process exit. Cap at 4 so a burst of generations
        # doesn't saturate bandwidth.
        self._asset_dl_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="okonomi-dl")
        # Tracked so quit can bounded-wait for in-flight downloads.
        self._asset_dl_futures: list[Future] = []

        # Set on cancel/timeout so the NEXT turn knows to prepend a
        # "you were interrupted, focus on the new instruction" marker.
        self._was_interrupted: bool = False

        self.context_manager = ContextManager(
            model_id=model_id,
            compress_at_tokens=compress_at_tokens,
        )
        # Session persistence, composed rather than inherited — owns no
        # agent state; the live state is passed per save/load call.
        self.store = SessionStore(
            self.session_manager,
            self.context_manager,
            tool_namespaces=set(self.programmatic_tools),
        )
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
        self._bg_execs: dict[str, threading.Thread] = {}
        # Per-job log registry covering bg-exec-N (Python execute_code jobs)
        # and term-N (terminal.bash background jobs). Survives completion so
        # /logs <id> can read after a job finishes. Each entry:
        #   {"kind", "label", "started_at", "ended_at", "status", "_reader"}
        # `_reader` is a zero-arg callable returning the current snapshot.
        self._bg_logs: dict[str, dict[str, Any]] = {}

        # Event bus the agent pushes typed UI events onto. Always present;
        # callers subscribe what they want (the REPL transcript, or
        # nothing). Default has zero subscribers — events accumulate in
        # ``emitter.history()`` and nothing is rendered.
        self.emitter = EventEmitter()
        self._turn_count = 0
        # Let context compression surface its user-facing warning through the
        # same event pipe instead of a raw print (built before the emitter).
        self.context_manager.emitter = self.emitter
        self._current_stream: Any = None
        # Tracks the ToolCall event for the in-flight execute_code so we can
        # attach its output/error back to the UI when the tool returns.
        self._pending_tool_calls: list[Any] = []
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

    def reload(self) -> dict[str, int]:
        """Re-read skills from disk and rebuild the pydantic-ai Agent so
        the next turn picks up changes without restarting the process.

        Catches: new/edited/deleted skill markdown, sharpened tool
        docstrings, edited system-prompt rules. Does NOT hot-reload tool
        method bodies — for that, restart.

        Returns a dict with counts: ``{"skills": N, "tools": M}``.
        """
        self.skills._invalidate_cache()
        self.agent = self._create_agent()
        skill_count = len(self.skills.list_skills().get("skills", []))
        tool_count = len(self.agent._function_toolset.tools)
        return {"skills": skill_count, "tools": tool_count}

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
            "browser": BrowserRendererTools(),
            "genmedia": GenmediaTools(),
        }

        # Standalone tools — direct agent tools (one call per round-trip)
        self.terminal = TerminalTools()
        self.terminal._on_bg_register = self._register_bg_log
        self.terminal._on_bg_finish = self._mark_bg_log_done
        self.web = WebTools()
        self.sessions = SessionsTools()
        self.agents = AgentTools()
        self.files = FileTools()
        self.skills = SkillTools(
            available_tools=set(self.programmatic_tools.keys()) | {"terminal", "web", "sessions", "skills", "files"},
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
        self._register_tool(agent, self.files)
        self.agents._bind(self)
        self._register_tool(agent, self.agents, rename={"launch": "launch_agent", "list_agents": "list_agents"})

        # Programmatic tools (available inside execute_code)
        self._register_code_execution_tool(agent)

        return agent

    def _get_system_prompt(self) -> str:
        return build_system_prompt(
            skills=self.skills,
            programmatic_tools=self.programmatic_tools,
            session_manager=self.session_manager,
        )

    # Per-HTTP-request timeout for the OpenRouter client. Safety belt
    # against a single hung API call — distinct from ``self.timeout``
    # (which budgets the whole multi-round turn). 60s is generous for a
    # single completion; tools have their own bg/term mechanism for
    # genuinely long work.
    _REQUEST_TIMEOUT_S = 60.0

    def _create_model(self) -> CachedOpenRouterModel:
        # `openrouter_usage={"include": True}` makes OpenRouter return
        # `usage.cost` (actual USD spent on the call) in each response.
        # We pull that out per-turn in `_accumulate_usage` and persist
        # it as `actual_cost_usd` — authoritative billing without
        # maintaining a local pricing table.
        return CachedOpenRouterModel(
            self.model_id,
            settings=OpenRouterModelSettings(
                timeout=self._REQUEST_TIMEOUT_S,
                openrouter_usage={"include": True},
            ),
        )

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
                    pending = agent_self._pending_tool_calls.pop(0) if agent_self._pending_tool_calls else None
                    try:
                        result = m(*args, **kwargs)
                        # Surface the return value back onto the UI event.
                        if pending is not None:
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
                                agent_self.emitter.emit(pending.finish(output=output, error=error_msg))
                            except Exception:  # noqa: BLE001
                                pass
                        return result
                    except Exception as exc:
                        # Tool raised — surface traceback to the UI so the
                        # user sees what went wrong before the model
                        # confabulates a recovery.
                        if pending is not None:
                            try:
                                agent_self.emitter.emit(pending.finish(error=f"{type(exc).__name__}: {exc}"))
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
        return AgentDependencies(
            state=self.state,
            session_manager=self.session_manager,
            tools=self.programmatic_tools,
            verbose=self.verbose,
            on_artifact_added=self._eager_download_artifact,
        )

    def _register_bg_log(self, *, task_id: str, label: str, reader) -> None:
        """Called by sub-tools (TerminalTools, AgentTools) when a bg job
        starts. Records the job in the shared log registry so /jobs and
        /logs <id> see it alongside the bg-exec-N execute_code path.

        Kind is inferred from the task_id prefix so /jobs renders the
        right column without needing each call site to thread it through."""
        prefix = task_id.split("-", 1)[0] if "-" in task_id else "task"
        self._bg_logs[task_id] = {
            "kind": prefix,
            "label": label,
            "started_at": time.time(),
            "ended_at": None,
            "status": "running",
            "_reader": reader,
        }

    def _mark_bg_log_done(self, *, task_id: str, status: str) -> None:
        """Called when a registered bg job (term-N) finishes."""
        entry = self._bg_logs.get(task_id)
        if entry is not None:
            entry["status"] = status
            entry["ended_at"] = time.time()

    def _sync_session_assets(self, deps: "AgentDependencies", *, tool_name: str = "execute_code") -> int:
        """Register media files in the session dir that aren't tracked yet.

        execute_code writes images/video/audio/etc. straight to the session
        directory (PIL.save, matplotlib.savefig, ffmpeg, imageio) without ever
        calling add_artifact, so those outputs never reach /assets, the SQLite
        index, or search_artifacts. This sweep walks the session dir and adds an
        artifact for every media file not already represented by a tracked
        path. Idempotent: dedup is keyed on the resolved path, so re-running it
        each turn only ever picks up genuinely new files (already-tracked
        downloads and prior sweeps are skipped). Returns the number added.
        """
        try:
            session_dir = deps.session_manager.session_dir(deps.state.info.session_id)
        except Exception:
            return 0
        if not session_dir.exists():
            return 0

        tracked: set[str] = set()
        for a in deps.state.artifacts:
            if not a.path:
                continue
            try:
                tracked.add(str(Path(a.path).resolve()))
            except Exception:
                tracked.add(a.path)

        added = 0
        for p in sorted(session_dir.rglob("*")):
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            artifact_type = _MEDIA_EXT_TO_TYPE.get(p.suffix.lower())
            if artifact_type is None:
                continue
            try:
                resolved = str(p.resolve())
            except Exception:
                resolved = str(p)
            if resolved in tracked:
                continue
            tracked.add(resolved)
            deps.add_artifact(
                Artifact(
                    artifact_type=artifact_type,
                    path=str(p),
                    tool_name=tool_name,
                    timestamp=datetime.now().isoformat(),
                )
            )
            added += 1
        return added

    def _eager_download_artifact(self, artifact: Artifact) -> None:
        """Submit a background download for a remote artifact.

        Fires from AgentDependencies.add_artifact the moment a tool tracks an
        asset. By the time save_session runs, most assets are already on
        disk and ``download_session_assets`` is a fast no-op. Idempotent —
        already-downloaded artifacts are skipped.
        """
        if not artifact.url or artifact.path:
            return

        def _run() -> None:
            try:
                session_dir = self.session_manager.session_dir(self.state.info.session_id)
                session_dir.mkdir(parents=True, exist_ok=True)
                local_path = download_asset(artifact.url, session_dir, artifact.artifact_type)
                artifact.path = local_path
            except Exception:
                pass

        try:
            future = self._asset_dl_pool.submit(_run)
            self._asset_dl_futures.append(future)
            future.add_done_callback(self._asset_dl_futures.remove)
        except RuntimeError:
            _run()

    def _accumulate_usage(self, agent_run) -> None:
        """Fold one ``agent_run`` worth of usage + cost into ``self.state.info.tokens``.

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

        self.state.info.record_usage(
            input_tokens=in_toks or 0,
            output_tokens=out_toks or 0,
            requests=getattr(usage, "requests", 0) or 1,
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
            reasoning_tokens=reasoning,
            cost_usd=cost,
        )

    def _extract_code(self, part: ToolCallPart) -> str:
        """Extract code string from an execute_code tool call."""
        args = part.args
        if isinstance(args, dict):
            return args.get("code", "")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed.get("code", "") if isinstance(parsed, dict) else ""
            except json.JSONDecodeError:
                return args
        if hasattr(args, "code"):
            return args.code
        return str(args)

    def _finish_pending_tool_call(self, *, output: str = "", error: str | None = None) -> None:
        """Mark the next pending tool-call widget as complete. Used by paths
        that resolve a tool call without invoking it (approval skip /
        feedback) or before the wrapper has a chance to."""
        if not self._pending_tool_calls:
            return
        pending = self._pending_tool_calls.pop(0)
        try:
            self.emitter.emit(pending.finish(output=output, error=error))
        except Exception:
            pass

    def _display_tool_call(self, part: ToolCallPart) -> None:
        """Emit a ToolCall event for ``part``. The subscriber (TUI widget
        or stdout printer) handles rendering."""
        call_id = getattr(part, "tool_call_id", "") or ""
        if part.tool_name == "execute_code":
            code = self._extract_code(part)
            if not code:
                return
            line_count = code.count("\n") + 1
            ev = self.emitter.emit(
                ToolCall(
                    "execute_code",
                    args=f"{line_count} lines",
                    details=code,
                    tool_call_id=call_id,
                )
            )
        elif part.tool_name == "load_skill":
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            name = args.get("name", args) if isinstance(args, dict) else args
            ev = self.emitter.emit(ToolCall("load_skill", args=str(name), details=str(args), tool_call_id=call_id))
        else:
            args_str = str(part.args) if part.args else ""
            ev = self.emitter.emit(ToolCall(part.tool_name, args=args_str, details=args_str, tool_call_id=call_id))
        self._pending_tool_calls.append(ev)

    async def run_async(self, user_message: str, reset: bool = False) -> str:
        """Run the agent asynchronously with a user message.

        Turns are bounded by ``self.timeout`` seconds (0 disables). On
        timeout, partial output and message history are preserved —
        same path as user-initiated cancel.
        """
        if reset:
            self.state = OkonomiState(info=SessionInfo(session_id=self.state.info.session_id, model=self.model_id))

        self._turn_count += 1
        turn = self._turn_count
        self.emitter.emit(TurnStart(turn=turn))

        # Snapshot the interrupt flag before clearing it; the checks below
        # need the prior turn's value.
        prior_interrupted = self._was_interrupted
        self._was_interrupted = False
        # "Approve all" applies to this turn only — a new user message
        # implicitly resets it.
        self.auto_approve = False
        self.state.info.task_input = user_message
        deps = self._get_dependencies()

        if self.state.context:
            self.state.context = strip_unanswered_tool_calls(self.state.context)
            if prior_interrupted:
                # Drop the cancelled turn's trailing assistant response so
                # the new instruction starts from a clean state.
                while self.state.context and isinstance(self.state.context[-1], ModelResponse):
                    self.state.context.pop()
            compressed = self.context_manager.compress(self.state.context)
            if compressed is not None:
                self.emitter.emit(
                    SystemNote(f"Context compressed: {len(self.state.context)} → {len(compressed)} messages")
                )
                self.state.context = compressed

        if prior_interrupted and self.state.context:
            user_message = (
                "[The user interrupted the previous response to give new instructions. "
                "Stop what you were doing and focus on this message.]\n\n" + user_message
            )

        if self._skill_nudge_interval > 0 and self._turns_since_skill >= self._skill_nudge_interval:
            user_message += (
                "\n\n[System note: You've completed several tool-heavy turns. "
                "If you developed a reusable approach, consider saving it as a "
                "skill with create_skill().]"
            )
            self._turns_since_skill = 0

        prompt = _parse_user_message(user_message, emitter=self.emitter)
        timeout_manager = asyncio.timeout(self.timeout) if self.timeout > 0 else contextlib.nullcontext()

        try:
            async with self.agent.iter(
                prompt,
                deps=deps,
                message_history=self.state.context or None,
            ) as agent_run:
                try:
                    async with timeout_manager:
                        async for node in agent_run:
                            if not Agent.is_model_request_node(node):
                                continue
                            self.spinner_text = "Thinking..."
                            await self._stream_model_response(node, agent_run.ctx)
                except (asyncio.CancelledError, TimeoutError) as exc:
                    self._handle_interruption(agent_run, deps, isinstance(exc, TimeoutError))
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    return ""

                result = agent_run.result

            messages = result.all_messages()
            self._commit_turn(agent_run, deps, messages)
            self._update_skill_nudge(messages, deps)
            return result.output

        except Exception as e:
            self.emitter.emit(SystemNote(f"Error: {e}", level="error"))
            return f"Error: {str(e)}"
        finally:
            # The turn is over on every exit path (normal, cancel, timeout,
            # error). Clear the spinner so the status bar can't get stuck on
            # a stale "Thinking..." while the agent is actually idle.
            self.spinner_text = ""
            self.emitter.emit(TurnEnd(turn=turn))

    def _handle_interruption(self, agent_run, deps: AgentDependencies, timed_out: bool) -> None:
        """Commit partial state and surface a timeout note. Caller decides
        whether to re-raise CancelledError or return an empty string."""
        self._was_interrupted = True
        self._commit_turn(agent_run, deps, strip_unanswered_tool_calls(agent_run.all_messages()))
        if timed_out:
            self.emitter.emit(
                SystemNote(
                    f"Turn timed out after {self.timeout:.0f}s. Partial state preserved.",
                    level="warn",
                )
            )

    async def _stream_model_response(self, node, ctx) -> None:
        """Stream one model-request node's text output into the emitter
        as a single growing AssistantStream event, then display any
        accompanying tool calls."""
        async with node.stream(ctx) as stream:
            async for text in stream.stream_text(delta=True, debounce_by=None):
                if self.spinner_text:
                    self.spinner_text = ""
                if self._current_stream is None:
                    self._current_stream = AssistantStream()
                self._current_stream.text += text
                self.emitter.emit(self._current_stream)

            self.spinner_text = ""
            if self._current_stream is not None:
                self._current_stream.done = True
                self.emitter.emit(self._current_stream)
                self._current_stream = None

            response = stream.response
            if response and response.parts:
                for part in response.parts:
                    if isinstance(part, ToolCallPart):
                        self._display_tool_call(part)

    def _commit_turn(self, agent_run, deps: AgentDependencies, post_messages: list) -> None:
        """Fold the run's results into ``self.state``. Shared by the
        normal-completion and cancellation paths."""
        self._accumulate_usage(agent_run)
        try:
            self.state.message_history.extend(list(agent_run.new_messages()))
        except Exception:
            pass
        self.state.context = post_messages
        try:
            ctx_toks = count_tokens(self.state.context)
            system_toks = len(self._get_system_prompt()) // _CHARS_PER_TOKEN
            self.state.info.tokens.context_tokens = ctx_toks + system_toks
        except Exception:
            pass

    def _update_skill_nudge(self, messages: list, deps: AgentDependencies) -> None:
        """Track tool-heavy turns so the periodic 'save as a skill' nudge
        fires at the right cadence."""
        used_skill_manage = any(
            isinstance(part, ToolCallPart)
            and part.tool_name in ("create_skill", "write_skill_file", "patch_skill_file")
            for msg in messages
            if isinstance(msg, ModelResponse)
            for part in msg.parts
        )
        if used_skill_manage:
            self._turns_since_skill = 0
        elif deps.state.tools_used:
            self._turns_since_skill += 1

    def run(self, user_message: str, reset: bool = False) -> str:
        """Run the agent synchronously with a user message.

        Args:
            user_message: The user's message/task
            reset: Whether to reset conversation history

        Returns:
            The agent's response
        """
        return asyncio.run(self.run_async(user_message, reset=reset))

    def save_session(
        self,
        status: SessionStatus = "completed",
        download_assets: bool = True,
        skip_summary: bool = False,
    ) -> None:
        """Persist the current state via the composed ``SessionStore``.
        See :meth:`SessionStore.save` for parameter semantics."""
        self.store.save(
            self.state,
            model_id=self.model_id,
            status=status,
            download_assets=download_assets,
            skip_summary=skip_summary,
        )

    def load_session(self, session_id: str) -> bool:
        """Load a saved session and install it as the live state.

        Returns True if the session was loaded successfully. See
        :meth:`SessionStore.load` for what restoration involves."""
        state = self.store.load(session_id)
        if state is None:
            return False
        self.state = state
        return True
