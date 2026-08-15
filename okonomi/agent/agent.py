"""Pydantic AI-based agent for Okonomi with code execution.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM
interactions, along with its runtime state (``OkonomiState``), dependency
container (``AgentDependencies``) and asset-download helpers. The class
is assembled from focused modules: the turn loop, event wiring, and
session save/load live here; code execution is a composed
``CodeExecutor`` (``execute_code``); system-prompt construction
(``prompt``), provider glue (``provider``), tool wrappers
(``wrappers``), and image-reference parsing (``multimodal``) are plain
modules.
"""

import asyncio
import contextlib
import inspect
import json
import pickle
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import Model

from ..context import _CHARS_PER_TOKEN, ContextManager, count_tokens, strip_unanswered_tool_calls
from ..events import (
    ApprovalRequest,
    AssistantStream,
    Event,
    EventEmitter,
    Interrupt,
    JobFinished,
    JobStarted,
    StatusChanged,
    SystemNote,
    ToolCallDisplay,
    TurnEnd,
    TurnStart,
)
from ..session import Artifact, Session, SessionStatus, asset_subdir_for_file
from ..tools import (
    AgentTools,
    FileTools,
    SessionsTools,
    SkillTools,
    TerminalTools,
    WebTools,
)
from .execute_code import CodeExecutor
from .multimodal import _parse_user_message
from .prompt import build_system_prompt
from .provider import build_model
from ..config import resolve_model_id, select_provider


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


@dataclass
class OkonomiState:
    """The live, mutable conversation state — the session's body,
    attached as ``Session.state`` by the agent.

    The persistent fields are written out by ``Session.save``; the
    non-persistent ones are rebuilt on resume.
    """

    # ── persistent — written out on every save ───────────────────────────
    # Append-only audit log: every message ever produced this session.
    # Persisted as event rows (the new tail appended each save).
    # Compression never touches it.
    message_history: list[Any] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    # REPL state across execute_code turns (model-bound locals).
    # Persisted best-effort to globals.pkl (picklable values only).
    execution_globals: dict[str, Any] = field(default_factory=dict)

    # ── non-persistent — recomputed on resume ────────────────────────────
    # Working set sent to the LLM each turn. Sanitized and compressed
    # as needed; recomputed from message_history on resume.
    context: list[Any] = field(default_factory=list)


@dataclass
class AgentDependencies:
    """Dependencies injected into pydantic-ai tool calls and execute_code.

    Holds a reference to the live ``Session`` — its attached state is
    where tool wrappers record artifacts, and ``session.session_dir``
    resolves paths — plus the eager-download hook.
    """

    session: Session
    verbose: bool = False
    # Fired from add_artifact so the agent can kick off an eager
    # background download for artifacts with a remote ``url``.
    # Signature: (artifact: Artifact) -> None, must not raise.
    on_artifact_added: Any | None = None

    @property
    def state(self) -> OkonomiState:
        return self.session.state

    def add_artifact(self, artifact: Artifact) -> None:
        """Track an artifact (AI-generated media OR locally-written file).

        Stamps a timestamp if missing and fires the eager-download hook
        for artifacts with a remote ``url``.
        """
        if not artifact.timestamp:
            artifact.timestamp = datetime.now().isoformat()
        self.state.artifacts.append(artifact)

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


class OkonomiAgent:
    """Pydantic AI-based agent with local code execution.

    This agent provides:
    - Code execution with tool instances injected into the environment
    - Session management for continuity
    - Message history for multi-turn conversations
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        timeout: float = 600.0,
        compress_at_tokens: int = 170_000,
        session_id: str | None = None,
        verbose: bool = False,
    ):
        self.provider = select_provider()
        self.model_id = model_id or resolve_model_id(self.provider)
        self.timeout = timeout
        self.compress_at_tokens = compress_at_tokens
        self.session = Session(session_id or "", model=model_id, state=OkonomiState())
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
        self._status: str = ""

        # Approval gating flags + ApprovalRequest flow. Currently unused:
        # execute_code no longer gates (the expensive-tool path it guarded
        # was removed). The flow is still wired to the TUI; kept pending a
        # decision to remove the approval system entirely.
        self.approvals_enabled: bool = False
        self.auto_approve: bool = False

        # Owns execute_code: in-process exec, approval gating, bg-exec-N
        # threads. Composed rather than inherited so its state (live
        # threads, dispatch counter) has a single owner.
        self.executor = CodeExecutor(self)

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
        # Stream-event persistence cursors: system notes are flushed from
        # the emitter's history at save (everything past this index);
        # interrupts queue here until the next save.
        self._persisted_note_count = 0
        self._pending_stream_events: list[Event] = []
        # Let context compression surface its user-facing warning through the
        # same event pipe instead of a raw print (built before the emitter).
        self.context_manager.emitter = self.emitter
        self._current_stream: Any = None
        # Tracks the ToolCall event for the in-flight execute_code so we can
        # attach its output/error back to the UI when the tool returns.
        self._pending_tool_calls: list[Any] = []
        # Background-job completions surface as JobFinished events on the
        # emitter (see _job_finished) — subscribers react (e.g. the TUI
        # injects a follow-up turn); nothing polls or registers callbacks.
        self.terminal._on_bg_complete = self._job_finished

        self.agent = self._create_agent()

    @property
    def state(self) -> OkonomiState:
        """The live conversation state — owned by ``self.session``."""
        return self.session.state

    @property
    def status(self) -> str:
        """Current working-state label. Assignments emit ``StatusChanged``
        so subscribers get pushed the change; the attribute remains
        readable for pull-style status bars."""
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        if value == self._status:
            return
        self._status = value
        emitter = getattr(self, "emitter", None)
        if emitter is not None:
            emitter.emit(StatusChanged(value))

    async def _request_approval(self, code: str) -> bool | str:
        """Emit an ``ApprovalRequest`` and suspend until a subscriber
        answers via its ``respond`` handle (thread-safe, one-shot)."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool | str] = loop.create_future()

        def respond(decision: bool | str) -> None:
            def _resolve() -> None:
                if not fut.done():
                    fut.set_result(decision)

            loop.call_soon_threadsafe(_resolve)

        self.emitter.emit(ApprovalRequest(code=code, respond=respond))
        return await fut

    def _job_finished(
        self,
        *,
        bg_id: str,
        error: str | None = None,
        output: str | None = None,
        new_artifacts: int = 0,
        produced_artifacts: list | None = None,
    ) -> None:
        """Emit ``JobFinished`` for a completed background job (bg-exec
        blocks and term commands both route here)."""
        refs = [{"artifact_type": a.artifact_type, "path": a.path, "url": a.url} for a in produced_artifacts or []]
        self.emitter.emit(
            JobFinished(
                job_id=bg_id,
                status="error" if error else "ok",
                error=error,
                output=output,
                new_artifacts=new_artifacts,
                artifact_refs=refs,
            )
        )

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
        Standalone tools are registered directly on the agent (one call per
        model round-trip). ``execute_code`` is a plain Python REPL — the
        model imports libraries directly; no tool objects are injected.
        """
        # Standalone tools — direct agent tools (one call per round-trip)
        self.terminal = TerminalTools()
        self.terminal._on_bg_register = self._register_bg_log
        self.terminal._on_bg_finish = self._mark_bg_log_done
        self.web = WebTools()
        self.sessions = SessionsTools()
        self.agents = AgentTools()
        self.files = FileTools()
        self.skills = SkillTools(
            available_tools={"terminal", "web", "sessions", "skills", "files"},
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

        # The execute_code REPL tool
        self.executor.register_tool(agent)

        return agent

    def _get_system_prompt(self) -> str:
        return build_system_prompt(skills=self.skills)

    # Per-HTTP-request timeout for the OpenRouter client. Safety belt
    # against a single hung API call — distinct from ``self.timeout``
    # (which budgets the whole multi-round turn). 60s is generous for a
    # single completion; tools have their own bg/term mechanism for
    # genuinely long work.
    _REQUEST_TIMEOUT_S = 60.0

    def _create_model(self) -> Model:
        return build_model(self.provider, self.model_id, self._REQUEST_TIMEOUT_S)

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
                    agent_self.status = f"Running {tn}..."
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
                        agent_self.status = ""

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
            session=self.session,
            verbose=self.verbose,
            on_artifact_added=self._eager_download_artifact,
        )

    def _register_bg_log(self, *, task_id: str, label: str, reader, kind: str | None = None) -> None:
        """Called when any bg job starts (bg-exec blocks, term commands,
        launched agents, /bg tasks). Records the job in the shared log
        registry so /jobs and /logs <id> see it.

        Kind defaults to the task_id prefix so /jobs renders the right
        column without each call site threading it through."""
        prefix = kind or (task_id.split("-", 1)[0] if "-" in task_id else "task")
        self._bg_logs[task_id] = {
            "kind": prefix,
            "label": label,
            "started_at": time.time(),
            "ended_at": None,
            "status": "running",
            "_reader": reader,
        }
        self.emitter.emit(JobStarted(job_id=task_id, kind=prefix, label=label))

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
            session_dir = deps.session.session_dir
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
            # Locally-written files carry no generation prompt, so stamp a
            # cheap description at sweep time — filename + current task —
            # or they'd be findable only by path substring.
            task = deps.session.task_input
            description = f"{p.name} — written during: {task[:200]}" if task else p.name
            deps.add_artifact(
                Artifact(
                    artifact_type=artifact_type,
                    path=str(p),
                    tool_name=tool_name,
                    description=description,
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
                session_dir = self.session.session_dir
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
        """Fold one ``agent_run`` worth of usage + cost into ``self.session.tokens``.

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
        usage = agent_run.usage
        details = getattr(usage, "details", {}) or {}
        reasoning = int(details.get("reasoning_tokens") or details.get("output_reasoning_tokens") or 0)
        in_toks = usage.input_tokens or 0
        out_toks = usage.output_tokens or 0

        cost = 0.0
        try:
            for msg in agent_run.new_messages():
                if isinstance(msg, ModelResponse) and msg.provider_details:
                    c = msg.provider_details.get("cost")
                    if c is not None:
                        cost += float(c)
        except Exception:
            pass

        self.session.record_usage(
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
                ToolCallDisplay(
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
            ev = self.emitter.emit(
                ToolCallDisplay("load_skill", args=str(name), details=str(args), tool_call_id=call_id)
            )
        else:
            args_str = str(part.args) if part.args else ""
            ev = self.emitter.emit(
                ToolCallDisplay(part.tool_name, args=args_str, details=args_str, tool_call_id=call_id)
            )
        self._pending_tool_calls.append(ev)

    async def run_async(self, user_message: str, reset: bool = False) -> str:
        """Run the agent asynchronously with a user message.

        Turns are bounded by ``self.timeout`` seconds (0 disables). On
        timeout, partial output and message history are preserved —
        same path as user-initiated cancel.
        """
        if reset:
            self.session = Session(self.session.session_id, model=self.model_id, state=OkonomiState())

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
        self.session.task_input = user_message
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

        prompt = _parse_user_message(user_message, emitter=self.emitter)
        timeout = asyncio.timeout(self.timeout) if self.timeout > 0 else contextlib.nullcontext()

        try:
            async with self.agent.iter(
                prompt,
                deps=deps,
                message_history=self.state.context or None,
            ) as agent_run:
                try:
                    async with timeout:
                        async for node in agent_run:
                            if not Agent.is_model_request_node(node):
                                continue
                            self.status = "Thinking..."
                            await self._stream_model_response(node, agent_run.ctx)

                except (asyncio.CancelledError, TimeoutError) as exc:
                    self._handle_interruption(agent_run, deps, isinstance(exc, TimeoutError))
                    if isinstance(exc, asyncio.CancelledError):
                        raise

                    return ""

                result = agent_run.result

            messages = result.all_messages()
            self._commit_turn(agent_run, deps, messages)
            return result.output

        except Exception as e:
            self.emitter.emit(SystemNote(f"Error: {e}", level="error"))
            return f"Error: {str(e)}"
        finally:
            # The turn is over on every exit path (normal, cancel, timeout,
            # error). Clear the spinner so the status bar can't get stuck on
            # a stale "Thinking..." while the agent is actually idle.
            self.status = ""
            self.emitter.emit(TurnEnd(turn=turn))

    def _handle_interruption(self, agent_run, deps: AgentDependencies, timed_out: bool) -> None:
        """Commit partial state and surface a timeout note. Caller decides
        whether to re-raise CancelledError or return an empty string."""
        self._was_interrupted = True
        self._pending_stream_events.append(Interrupt("timeout" if timed_out else "cancelled"))
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
                if self.status:
                    self.status = ""
                if self._current_stream is None:
                    self._current_stream = AssistantStream()
                self._current_stream.text += text
                self.emitter.emit(self._current_stream)

            self.status = ""
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
            self.session.tokens.context_tokens = ctx_toks + system_toks
        except Exception:
            pass

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
    ) -> None:
        """Persist the live state.

        Args:
            status: Status of the session (completed, error, abandoned)
            download_assets: When True (default), synchronously download
                any tracked outputs that don't yet have a ``local_path``.
                Set False on shutdown / atexit paths so quitting doesn't
                wait on slow network downloads — the eager bg pool has
                been pulling these all along, and any still-missing files
                will be retried on the next session resume.
        """
        session = self.session

        if download_assets:
            download_session_assets(session.state.artifacts, session.session_dir)

        # Stamp the run-owned fields onto the live session; identity
        # fields (title / created_at) are write-once at the
        # DB layer, so this is a plain write.
        session.status = status
        session.model = self.model_id

        session.save()
        self._flush_stream_events()
        self._serialize_user_globals()

        # Titles are generated out-of-band by a small model, fire-and-forget,
        # so the agent never blocks on (or owns) titling. Refresh on every
        # completed turn so the title tracks the whole conversation.
        if status == "completed" and session.state.message_history:
            from ..utils import _maybe_update_session_title_async

            _maybe_update_session_title_async(session, session.state.message_history)

    def load_session(self, session_id: str) -> bool:
        """Load a saved session and install it as the live state.

        Restores the full message history so the model sees its prior
        tool calls, intermediate reasoning, and produced artifacts
        directly. The next-turn ``run_async`` already runs prune +
        compress on the restored history, so oversized sessions get
        trimmed there instead of being stripped here.

        Returns True if the session was loaded successfully.
        """
        session = Session.load(session_id)
        if session is None:
            return False
        state = OkonomiState(
            message_history=session.load_messages(),
            artifacts=session.load_artifacts(),
        )
        session.state = state

        # Raw log restored as-is (append-only audit record). Working set
        # is a sanitized view — strip orphan tool calls at the tail from
        # an interrupted save. Compression on the working set fires on
        # the next turn if oversized.
        state.context = strip_unanswered_tool_calls(list(state.message_history))
        self.session = session
        # Notes emitted before the switch belong to the previous session.
        self._persisted_note_count = len(self.emitter.history())
        self._pending_stream_events = []

        # Restore Python locals the model bound during the prior session.
        # Tool wrappers / helpers are re-injected on the next execute_code
        # call, so we only need user-bound vars here.
        self._restore_user_globals()
        return True

    def _flush_stream_events(self) -> None:
        """Persist stream events accumulated since the last save: system
        notes gathered from the emitter, plus pending interrupts."""
        history = self.emitter.history()
        notes = [ev for ev in history[self._persisted_note_count :] if isinstance(ev, SystemNote)]
        self._persisted_note_count = len(history)
        pending, self._pending_stream_events = self._pending_stream_events, []
        if notes or pending:
            self.session.append_stream_events([*notes, *pending])

    def _serialize_user_globals(self) -> None:
        """Save picklable user-bound variables from ``execution_globals`` to
        ``<session_dir>/globals.pkl``. Tool wrappers, modules, functions,
        and anything that fails to pickle are silently skipped — the goal
        is to recover the most-used patterns (URLs, paths, lists of
        descriptions, intermediate dicts), not arbitrary live objects.
        """
        state = self.state
        if not state.execution_globals:
            return

        SKIP_KEYS = {"__builtins__", "asyncio", "get_session_dir"}

        candidates: dict[str, Any] = {}
        for key, value in state.execution_globals.items():
            if key in SKIP_KEYS or key.startswith("__"):
                continue
            if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
                continue
            # Tool instances expose __get_metadata__ (ToolMetadataMixin) —
            # never pickle them; they're re-bound fresh each execute_code call.
            if hasattr(value, "__get_metadata__"):
                continue
            candidates[key] = value

        if not candidates:
            return

        try:
            session_dir = self.session.session_dir
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / "globals.pkl"
            try:
                with open(path, "wb") as f:
                    pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
                return
            except Exception:
                pass
            # One of the values is unpicklable — fall back to per-key.
            clean: dict[str, Any] = {}
            for key, value in candidates.items():
                try:
                    pickle.dumps(value)
                except Exception:
                    continue
                clean[key] = value
            with open(path, "wb") as f:
                pickle.dump(clean, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _restore_user_globals(self) -> None:
        """Counterpart to ``_serialize_user_globals``. Reads
        ``<session_dir>/globals.pkl`` if present and merges into
        ``state.execution_globals``. Tool wrappers + helpers will be
        reinstalled on the next ``execute_code`` call."""
        try:
            globals_path = self.session.session_dir / "globals.pkl"
            if not globals_path.exists():
                return
            with open(globals_path, "rb") as f:
                snapshot = pickle.load(f)
            if isinstance(snapshot, dict):
                self.state.execution_globals.update(snapshot)
        except Exception:
            pass
