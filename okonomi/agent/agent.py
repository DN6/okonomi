"""Pydantic AI-based agent for Okonomi.

This module provides the OkonomiAgent class that uses Pydantic AI for LLM
interactions, along with its runtime state (``OkonomiState``), dependency
container (``AgentDependencies``), artifact-download helpers and
image-reference parsing. The class is assembled from focused modules:
the turn loop, event wiring, and session save/load live here; code runs
through the sandboxed ``bash`` tool (``tools.terminal``); system-prompt
construction (``prompt``) and provider glue (``provider``) are plain
modules.
"""

import asyncio
import base64
import contextlib
import hashlib
import inspect
import itertools
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    BinaryContent,
    EnqueuedMessagesEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model

from ..config import resolve_model_id, select_provider
from ..context import _CHARS_PER_TOKEN, ContextManager, count_tokens, strip_unanswered_tool_calls
from ..events import (
    ApprovalRequest,
    AssistantStream,
    Event,
    EventEmitter,
    Interrupt,
    StatusChanged,
    SystemNote,
    ToolCallDisplay,
    TurnEnd,
    TurnStart,
)
from ..mcp import build_toolsets as build_mcp_toolsets
from ..session import Artifact, Session, SessionStatus, artifact_subdir_for_file
from ..tools import (
    AgentTools,
    FileTools,
    SessionsTools,
    SkillTools,
    TerminalTools,
    ViewerTools,
    WebTools,
)
from ..tools._command_guard import approve_always, check_command, dangerous_matches, load_approval_rules
from ..traces import TracingModel
from ..utils import redact_secrets
from .prompt import build_system_prompt
from .provider import build_model
from .tasks import TaskRegistry


# A tool call still running after this many seconds moves to the background
# as a ``tool-N`` task so the main thread stays free: the model gets a
# ``{"pending": "tool-N"}`` placeholder and ends its reply, and the result
# arrives later as a follow-up message (``TaskFinished`` → ``tool_notice``),
# the same path as every other background task. Short calls stay inline so
# the model can chain them.
TOOL_DETACH_AFTER = 30.0
_detach_counter = itertools.count(1)


def _tool_result_display(part: ToolReturnPart | RetryPromptPart) -> dict[str, str | None]:
    """``ToolCallDisplay.finish`` kwargs for a tool result. Tools often
    return ``{"error": ...}`` instead of raising, so that shows as an error
    too; a detached call shows its task id instead of the placeholder dict."""
    if isinstance(part, RetryPromptPart):
        return {"error": part.content if isinstance(part.content, str) else str(part.content)}
    content = part.content
    if isinstance(content, dict) and content.get("error"):
        return {"error": str(content["error"])}
    if isinstance(content, dict) and content.get("pending"):
        return {"output": f"still running — moved to background as {content['pending']}; its result arrives as a follow-up message"}
    return {"output": _short_repr(content)}


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

    # ── non-persistent — recomputed on resume ────────────────────────────
    # Working set sent to the LLM each turn. Sanitized and compressed
    # as needed; recomputed from message_history on resume.
    context: list[Any] = field(default_factory=list)


@dataclass
class AgentDependencies:
    """Dependencies injected into pydantic-ai tool calls.

    Holds a reference to the live ``Session`` — its attached state is
    where tool wrappers record artifacts, and ``session.session_dir``
    resolves paths — plus the eager-download hook.
    """

    session: Session
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


# Maps a file extension to the artifact_type used by the viewer and search_artifacts.
# bash commands write media straight to the session dir (PIL/matplotlib/ffmpeg)
# without going through add_artifact; a post-run sweep keyed on this table is what
# turns those files into tracked artifacts.
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
    # Briefs, prompt packages and research strands: tracked so a later
    # generation that reads one links back to it in the artifact lineage.
    ".md": "research",
}

_KNOWN_ARTIFACT_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm", ".mov"}


def download_artifact(url: str, artifacts_dir: Path, artifact_type: str) -> str:
    """Download ``url`` into ``<artifacts_dir>/<subdir>/`` and return the
    local path. Returns ``""`` on any failure (network blip, fal expiry) —
    callers retry on the next save."""
    try:
        url_filename = Path(urlparse(url).path).name
        if Path(url_filename).suffix.lower() in _KNOWN_ARTIFACT_EXTS:
            filename = url_filename
        else:
            ext = ".jpg" if artifact_type == "image" else ".mp4"
            filename = f"{url_filename}{ext}"

        artifact_dir = artifacts_dir / artifact_subdir_for_file(filename)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        local_path = artifact_dir / filename

        with httpx.Client(timeout=60.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            local_path.write_bytes(response.content)

        return str(local_path)
    except Exception:
        return ""


def download_session_artifacts(artifacts: list[Artifact], artifacts_dir: Path) -> None:
    """Download every remote artifact missing a local path into ``artifacts_dir``.

    Mutates each artifact in place — sets ``path`` on success, leaves it
    empty on failure so the next save can retry.
    """
    for artifact in artifacts:
        if not artifact.url or artifact.path:
            continue
        local_path = download_artifact(artifact.url, artifacts_dir, artifact.artifact_type)
        if local_path:
            artifact.path = local_path


# The user can mention image URLs, data: URIs, or local paths in plain text;
# the helpers below resolve them into pydantic-ai multimodal content parts.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")
_IMAGE_REF_RE = re.compile(
    r"""(?:https?://|file://|data:image/[a-zA-Z]+;base64,)[^\s<>"'`]+""",
    re.IGNORECASE,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpg|jpeg|webp|gif|bmp|tiff)\b", re.IGNORECASE)
_PATH_ANCHOR_RE = re.compile(r"(?:[/~]|\./)")
_SUPPORTED_IMAGE_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _resolve_image(ref: str):
    """Return a pydantic-ai content part for ``ref``, or None for dead
    paths / unsupported formats."""
    if ref.startswith(("http://", "https://", "data:")):
        return ImageUrl(url=ref)
    # Finder drag-and-drop yields file:///foo/bar%20baz.png — decode it.
    path_str = unquote(ref[7:]) if ref.startswith("file://") else ref
    media_type, _ = mimetypes.guess_type(path_str)
    if media_type not in _SUPPORTED_IMAGE_MIME:
        return None
    try:
        data = Path(path_str).expanduser().read_bytes()
    except (OSError, ValueError):
        return None
    return BinaryContent(data=data, media_type=media_type)


def _iter_image_refs(text: str):
    """Yield image refs (URLs, data: URIs, validated local paths) found
    in ``text``, deduplicated. Local paths are robust to spaces,
    backslash-escapes, and surrounding quotes — we walk back from each
    image-extension occurrence to a path-start anchor and validate
    against the filesystem before yielding."""
    seen: set[str] = set()

    for m in _IMAGE_REF_RE.finditer(text):
        ref = m.group(0).rstrip(".,;:)\"'")
        is_image = ref.startswith("data:image/") or ref.split("?", 1)[0].lower().endswith(_IMAGE_EXTS)
        if is_image and ref not in seen:
            seen.add(ref)
            yield ref

    for ext_m in _IMAGE_EXT_RE.finditer(text):
        ext_end = ext_m.end()
        line_start = text.rfind("\n", 0, ext_end) + 1
        for anchor in _PATH_ANCHOR_RE.finditer(text, line_start, ext_end):
            raw = text[anchor.start() : ext_end].strip("\"'").replace("\\ ", " ")
            try:
                p = Path(raw).expanduser()
            except (OSError, ValueError):
                continue
            if p.is_file():
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    yield key
                break


def _parse_user_message(text: str, emitter=None):
    """If ``text`` mentions images, return ``[text, *image_parts]`` so the
    model sees both the pixels and the string references it can pass
    downstream. Returns ``text`` unchanged when no images resolve.

    Dead refs (path missing, unsupported format) surface as SystemNote
    events so the user sees *why* their image wasn't attached.
    """
    parts: list = []
    dead: list[str] = []
    for ref in _iter_image_refs(text):
        part = _resolve_image(ref)
        if part is None:
            dead.append(ref)
        else:
            parts.append(part)

    if emitter is not None:
        for ref in dead:
            emitter.emit(
                SystemNote(
                    f"Couldn't attach image {ref!r} — file missing, unreadable, "
                    "or unsupported format (need png/jpg/gif/webp).",
                    level="warn",
                )
            )

    return [text, *parts] if parts else text


def store_references(session: Session, refs: list[str]) -> list[Artifact]:
    """Keep the images the user handed in. A reference is a pick made
    before anything was generated, so it is copied into
    ``session.references_dir`` and tracked with ``tool_name="user"``: it
    then survives URL expiry, shows in the viewer, and roots the lineage
    of whatever is generated from it. Paths already inside this
    session's artifacts dir are the user pointing at an output, not
    supplying one, and are left alone. Returns the artifacts added."""
    artifacts_root = session.artifacts_dir.resolve()
    references_dir = session.references_dir.resolve()
    known = {a.url for a in session.state.artifacts if a.url} | {a.path for a in session.state.artifacts if a.path}
    # The same image handed over twice, by any path or URL, is kept once.
    kept = {hashlib.sha1(p.read_bytes()).hexdigest() for p in references_dir.glob("*") if p.is_file()}
    added: list[Artifact] = []
    for ref in refs:
        if ref in known:
            continue
        if ref.startswith("data:image/"):
            media_type, payload = ref[5:].split(";base64,", 1)
            data = base64.b64decode(payload)
            name = f"pasted-{hashlib.sha1(data).hexdigest()[:8]}.{media_type.split('/', 1)[1]}"
            url = ""
        elif ref.startswith(("http://", "https://")):
            with httpx.Client(timeout=30.0) as client:
                response = client.get(ref, follow_redirects=True)
                response.raise_for_status()
            data = response.content
            name = Path(urlparse(ref).path).name or f"{hashlib.sha1(ref.encode()).hexdigest()[:8]}.jpg"
            url = ref
        else:
            src = Path(unquote(ref[7:]) if ref.startswith("file://") else ref).expanduser().resolve()
            if str(src) in known or src.is_relative_to(artifacts_root):
                continue
            data = src.read_bytes()
            name = src.name
            url = ""
        digest = hashlib.sha1(data).hexdigest()
        if digest in kept:
            continue
        dest = references_dir / name
        if dest.exists():
            dest = dest.with_name(f"{digest[:8]}-{name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        kept.add(digest)
        artifact = Artifact(
            artifact_type="image",
            path=str(dest),
            url=url,
            tool_name="user",
            description=f"reference supplied by the user: {ref[:200]}",
            timestamp=datetime.now().isoformat(),
        )
        session.state.artifacts.append(artifact)
        known.update({ref, str(dest), url})
        added.append(artifact)
    return added


class OkonomiAgent:
    """Pydantic AI-based agent.

    This agent provides:
    - Sandboxed shell execution via the bash tool
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
    ):
        self.provider = select_provider()
        self.model_id = model_id or resolve_model_id(self.provider)
        self.timeout = timeout
        self.compress_at_tokens = compress_at_tokens
        self.session = Session(session_id or "", model=model_id, state=OkonomiState())
        self._setup_tools()

        # Background pool for eager artifact downloads. Daemon threads —
        # don't block process exit. Cap at 4 so a burst of generations
        # doesn't saturate bandwidth.
        self._artifact_dl_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="okonomi-dl")

        # Set on cancel/timeout so the NEXT turn knows to prepend a
        # "you were interrupted, focus on the new instruction" marker.
        self._was_interrupted: bool = False
        # Set when the running turn is cancelled or times out, so tool calls
        # still attached to it stop instead of detaching into tasks whose
        # completion would start an unasked-for turn. Replaced every turn.
        self._turn_cancelled = threading.Event()
        # Set when the user sends a message mid-tool: running calls move to
        # the background now so the model reads the message without waiting
        # on them. Replaced every turn.
        self._detach_now = threading.Event()
        # Reply text the model had streamed when its turn was cut off; the
        # next user message tells the model where it was interrupted.
        self._interrupted_partial: str = ""
        # Why the last turn ended without a real reply (timeout or a raised
        # error), else None. ``run_async`` reports both as an ordinary return
        # value, so a caller that must tell failure from success reads this.
        self.last_turn_error: str | None = None

        self.context_manager = ContextManager(
            model_id=model_id,
            compress_at_tokens=compress_at_tokens,
        )
        self._status: str = ""

        # Event bus the agent pushes typed UI events onto. Always present;
        # callers subscribe what they want (the REPL transcript, or
        # nothing). Default has zero subscribers — events accumulate in
        # ``emitter.history()`` and nothing is rendered.
        self.emitter = EventEmitter()
        # Shared background-task registry (term-N, launched
        # agents, /bg tasks). /tasks and the status bar read it.
        self.tasks = TaskRegistry(self.emitter)
        self._turn_count = 0
        # Stream-event persistence cursors: system notes are flushed from
        # the emitter's history at save (everything past this index);
        # interrupts queue here until the next save.
        self._persisted_note_count = 0
        self._pending_stream_events: list[Event] = []
        # Background task id (term-N / tool-N) → the tool call that started
        # it, so files it writes are stamped with their producing call.
        self._task_tool_call_ids: dict[str, str] = {}
        # Let context compression surface its user-facing warning through the
        # same event pipe instead of a raw print (built before the emitter).
        self.context_manager.emitter = self.emitter
        self._current_stream: Any = None
        # Open ToolCall widgets by tool_call_id, closed when pydantic-ai
        # reports the call's result (or the turn ends without one).
        self._open_tool_calls: dict[str, ToolCallDisplay] = {}
        # The turn in flight, so messages sent mid-turn can be enqueued into
        # it; and those messages by enqueue id until the turn picks them up.
        self._agent_run: Any = None
        self._enqueued: dict[str, str] = {}
        # Tools run on this pool so a call that outlives TOOL_DETACH_AFTER can
        # keep running as a background task after the turn moves on.
        self._tool_pool = ThreadPoolExecutor(thread_name_prefix="okonomi-tool")
        # Set by a UI that answers ApprovalRequest events. Without one, bash
        # refuses dangerous commands instead of waiting for an answer that
        # never comes.
        self.can_prompt = False
        self._session_approvals: set[str] = set()
        # MCP connections held open across turns (see _ensure_mcp_connected),
        # with the event loop and toolsets they were opened for.
        self._mcp_stack: contextlib.AsyncExitStack | None = None
        self._mcp_stack_key: tuple[Any, Any] | None = None
        # Background-task completions surface as TaskFinished events on the
        # emitter (see _task_finished) — subscribers react (e.g. the TUI
        # injects a follow-up turn); nothing polls or registers callbacks.
        self.terminal._on_bg_complete = self._task_finished

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
        self.emitter.emit(StatusChanged(value))

    def _task_finished(
        self,
        *,
        bg_id: str,
        error: str | None = None,
        output: str | None = None,
        produced_artifacts: list | None = None,
    ) -> None:
        """Delegate to the shared TaskRegistry (see ``okonomi.agent.tasks``),
        sweeping the session dir first so the finished task's outputs are
        registered as artifacts before the completion event fires."""
        try:
            self._sync_session_artifacts(self._get_dependencies(), tool_call_id=self._task_tool_call_ids.pop(bg_id, ""))
        except Exception:  # noqa: BLE001
            pass
        self.tasks.finished(
            bg_id=bg_id,
            error=error,
            output=output,
            produced_artifacts=produced_artifacts,
        )

    def reload(self) -> dict[str, int]:
        """Re-read skills from disk and rebuild the pydantic-ai Agent so
        the next turn picks up changes without restarting the process.

        Catches: new/edited/deleted skill markdown, sharpened tool
        docstrings, edited system-prompt rules. Does NOT hot-reload tool
        method bodies — for that, restart.

        Returns a dict with counts: ``{"skills": N, "tools": M, "mcp": K}``.
        """
        self.skills._invalidate_cache()
        self.agent = self._create_agent()
        skill_count = len(self.skills.list_skills().get("skills", []))
        tool_count = len(self.agent._function_toolset.tools)
        return {"skills": skill_count, "tools": tool_count, "mcp": len(self._mcp_toolsets)}

    def _setup_tools(self) -> None:
        """Create all tool instances.

        All tools are standalone: registered directly on the agent, one
        call per model round-trip. ``bash`` is the execution surface —
        subprocesses get a scrubbed env, $OKONOMI_SESSION_DIR (the
        session's artifacts dir) and $OKONOMI_SCRATCH_DIR.
        """

        def _artifacts_dir() -> str:
            d = self.session.artifacts_dir
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

        def _scratch_dir() -> str:
            d = self.session.scratch_dir
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

        # Standalone tools — direct agent tools (one call per round-trip)
        self.terminal = TerminalTools()
        self.terminal._on_bg_register = self._register_bg_log
        self.terminal._on_bg_finish = self._mark_bg_log_done
        self.terminal._session_dir_fn = _artifacts_dir
        self.terminal._scratch_dir_fn = _scratch_dir
        self.web = WebTools()
        self.sessions = SessionsTools(lambda: self.session)
        self.agents = AgentTools()
        self.files = FileTools()
        self.viewer = ViewerTools(lambda: self.session)
        self.skills = SkillTools(
            available_tools={"terminal", "web", "sessions", "skills", "files"},
        )

    def _create_agent(self) -> Agent[AgentDependencies, str]:
        """Create and configure the Pydantic AI agent."""
        model = self._create_model()
        self._mcp_toolsets = build_mcp_toolsets(process_tool_call=self._process_mcp_tool_call)
        agent = Agent(
            model,
            deps_type=AgentDependencies,
            system_prompt=self._get_system_prompt(),
            toolsets=self._mcp_toolsets,
        )

        # Standalone tools
        self._register_tool(agent, self.skills)
        self._register_tool(agent, self.terminal)
        self._register_tool(agent, self.web)
        self._register_tool(agent, self.sessions)
        self._register_tool(agent, self.files)
        self._register_tool(agent, self.viewer)
        self.agents._bind(self)
        self._register_tool(agent, self.agents, rename={"launch": "launch_agent"})

        return agent

    def _get_system_prompt(self) -> str:
        return build_system_prompt(skills=self.skills)

    # Per-HTTP-request timeout for the provider client. Safety belt
    # against a single hung API call — distinct from ``self.timeout``
    # (which budgets the whole multi-round turn). 60s is generous for a
    # single completion; tools have their own bg/term mechanism for
    # genuinely long work.
    _REQUEST_TIMEOUT_S = 60.0

    def _create_model(self) -> Model:
        return TracingModel(build_model(self.provider, self.model_id, self._REQUEST_TIMEOUT_S), lambda: self.session)

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
            # ``_``-prefixed params are wrapper-to-tool plumbing (bash's
            # ``_chunks_sink`` / ``_kill``), not arguments the model may set.
            sig = sig.replace(parameters=[p for p in sig.parameters.values() if not p.name.startswith("_")])

            def make_wrapper(m, s, tn, agent_self=self):
                def wrapper(ctx: RunContext[AgentDependencies], *args, **kwargs):
                    chunks: list[str] = []
                    cancelled = agent_self._turn_cancelled
                    detach_now = agent_self._detach_now
                    kill = threading.Event()
                    if tn == "bash":
                        # Asked here rather than inside bash: time spent
                        # waiting on the user must not count toward the
                        # detach deadline below.
                        command = kwargs.get("command", args[0] if args else "")
                        pending = agent_self._unapproved_patterns(command)
                        if pending and agent_self.can_prompt:
                            denial = agent_self._ask_approval(command, pending)
                            if denial is not None:
                                return denial
                        kwargs["_approved"] = not pending or agent_self.can_prompt
                        # Live output for /tasks on tool-N. ``kill`` replaces the
                        # shared Ctrl+C flag: it is only set while the call is
                        # still attached, so a detached call survives later stops.
                        kwargs["_chunks_sink"] = chunks
                        kwargs["_kill"] = kill
                    fut = agent_self._tool_pool.submit(m, *args, **kwargs)
                    deadline = time.monotonic() + TOOL_DETACH_AFTER
                    while True:
                        try:
                            result = fut.result(timeout=0.1)
                            break
                        except FutureTimeout:
                            if cancelled.is_set():
                                kill.set()
                                return {"error": "interrupted by the user"}
                            if detach_now.is_set() or time.monotonic() > deadline:
                                return agent_self._detach_tool_call(fut, tn, chunks, ctx.tool_call_id or "")
                    if tn == "bash":
                        if isinstance(result, dict) and result.get("task_id"):
                            agent_self._task_tool_call_ids[result["task_id"]] = ctx.tool_call_id or ""
                        # bash writes media straight into the session
                        # dir; sweep so outputs register as artifacts.
                        try:
                            agent_self._sync_session_artifacts(
                                agent_self._get_dependencies(), tool_call_id=ctx.tool_call_id or ""
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    return result

                ctx_param = inspect.Parameter(
                    "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext[AgentDependencies]
                )
                wrapper.__name__ = tn
                wrapper.__qualname__ = tn
                wrapper.__doc__ = m.__doc__
                wrapper.__signature__ = s.replace(parameters=[ctx_param, *s.parameters.values()])
                wrapper.__annotations__ = {
                    k: v.annotation for k, v in s.parameters.items() if v.annotation is not inspect.Parameter.empty
                }
                wrapper.__annotations__["ctx"] = RunContext[AgentDependencies]
                if s.return_annotation is not inspect.Signature.empty:
                    wrapper.__annotations__["return"] = s.return_annotation
                return wrapper

            agent.tool(make_wrapper(method, sig, tool_name), name=tool_name)

    def _unapproved_patterns(self, command: str) -> list[str]:
        """Dangerous patterns in ``command`` the user has not approved for
        this session or always. Empty when approvals are off or the command
        is hard-blocked anyway (bash reports that block itself)."""
        if check_command(command, approved=True)[0] is False:
            return []
        if check_command(command)[0]:
            return []
        approved = self._session_approvals | load_approval_rules()["allow"]
        return [key for key in dangerous_matches(command) if key not in approved]

    def _ask_approval(self, command: str, patterns: list[str]) -> dict | None:
        """Block until the user answers an ApprovalRequest for ``command``.
        Returns None to run it, or the tool result to return instead."""
        answered = threading.Event()
        box: dict[str, Any] = {}

        def _respond(decision: bool | str) -> None:
            if not answered.is_set():
                box["decision"] = decision
                answered.set()

        self.emitter.emit(ApprovalRequest(code=command, reason=", ".join(patterns), respond=_respond))
        while not answered.wait(0.1):
            if self._turn_cancelled.is_set():
                return {"error": "interrupted by the user"}

        decision = box["decision"]
        if decision is True:
            return None
        if decision == "__session__":
            self._session_approvals.update(patterns)
            return None
        if decision == "__always__":
            approve_always(patterns)
            return None
        if decision is False:
            return {"error": f"The user denied this command ({', '.join(patterns)}). Do not retry or rephrase it; ask the user how to proceed."}
        return {"error": f"The user did not run this command and replied: {decision}"}

    async def _process_mcp_tool_call(self, ctx: RunContext[AgentDependencies], call_tool, name: str, args: dict):
        """Apply TOOL_DETACH_AFTER to MCP calls, which bypass
        ``_register_tool``'s wrapper. The call keeps running on this event
        loop after the turn ends; that works because the connection is held
        open across turns by ``_ensure_mcp_connected``."""
        task = asyncio.ensure_future(call_tool(name, args))
        try:
            done, _ = await asyncio.wait({task}, timeout=TOOL_DETACH_AFTER)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if done:
            return task.result()
        return self._detach_tool_call(task, ctx.tool_name or name, [])

    async def _ensure_mcp_connected(self) -> None:
        """Hold MCP connections open across turns, so a call moved to the
        background outlives the turn that made it (each pydantic-ai run only
        enters toolsets for its own duration; this extra enter keeps the
        ref-counted connection alive). A connection belongs to the event
        loop that opened it: a run on another loop (one-shot ``asyncio.run``,
        a sub-agent thread) or after ``reload()`` rebuilt the toolsets opens
        a fresh one."""
        key = (asyncio.get_running_loop(), tuple(self._mcp_toolsets))
        if self._mcp_stack_key == key:
            return
        await self.close_mcp()
        if not self._mcp_toolsets:
            return
        async with contextlib.AsyncExitStack() as stack:
            for toolset in self._mcp_toolsets:
                await stack.enter_async_context(toolset)
            self._mcp_stack = stack.pop_all()
        self._mcp_stack_key = key

    async def close_mcp(self) -> None:
        """Close held MCP connections. Must run on the loop that opened
        them; a stack from an already-closed loop is just dropped."""
        stack, key = self._mcp_stack, self._mcp_stack_key
        self._mcp_stack = self._mcp_stack_key = None
        if stack is not None and key is not None and key[0] is asyncio.get_running_loop():
            await stack.aclose()

    def _detach_tool_call(
        self, fut: Future | asyncio.Future, tool_name: str, chunks: list[str], tool_call_id: str = ""
    ) -> dict:
        """Register a slow tool call as a ``tool-N`` task and return the
        placeholder the model sees instead of its result. The result is
        reported through ``_task_finished`` when the call completes."""
        task_id = f"tool-{next(_detach_counter)}"
        self._task_tool_call_ids[task_id] = tool_call_id
        box: dict[str, Any] = {}

        def _reader() -> str:
            if tool_name == "bash":
                joined = "".join(chunks)
                if "__CWD=" in joined:
                    joined = joined.rsplit("__CWD=", 1)[0]
                return redact_secrets(joined) or "(no output yet)"
            return _short_repr(box["result"]) if "result" in box else "(running — this tool has no live output)"

        self.tasks.register(task_id=task_id, label=tool_name, reader=_reader, kind="tool")

        def _done(f: Future | asyncio.Future) -> None:
            try:
                result = f.result()
                error = str(result["error"]) if isinstance(result, dict) and result.get("error") else None
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                result = None
                error = f"{type(exc).__name__}: {exc}"
            box["result"] = result if error is None else {"error": error}
            self.tasks.mark_done(task_id=task_id, status="error" if error else "ok")
            self._task_finished(bg_id=task_id, error=error, output=None if error else _short_repr(result))

        fut.add_done_callback(_done)
        return {
            "pending": task_id,
            "message": (
                f"{tool_name} is still running after {TOOL_DETACH_AFTER:.0f}s and has moved to the background as "
                f"{task_id}. Do not poll, wait, or re-run it. Tell the user it is running and end your reply; "
                "its result arrives as a [system] message when it finishes."
            ),
        }

    def _get_dependencies(self) -> AgentDependencies:
        return AgentDependencies(
            session=self.session,
            on_artifact_added=self._eager_download_artifact,
        )

    def _register_bg_log(self, *, task_id: str, label: str, reader) -> None:
        """Delegate to the shared TaskRegistry (see ``okonomi.agent.tasks``)."""
        self.tasks.register(task_id=task_id, label=label, reader=reader)

    def _mark_bg_log_done(self, *, task_id: str, status: str) -> None:
        """Delegate to the shared TaskRegistry (see ``okonomi.agent.tasks``)."""
        self.tasks.mark_done(task_id=task_id, status=status)

    def _sync_session_artifacts(self, deps: "AgentDependencies", tool_call_id: str = "") -> int:
        """Register media files in the session dir that aren't tracked yet.

        bash commands write images/video/audio/etc. straight to the session
        directory (PIL.save, matplotlib.savefig, ffmpeg, imageio) without ever
        calling add_artifact, so those outputs never reach the viewer, the SQLite
        index, or search_artifacts. This sweep walks the session dir and adds an
        artifact for every media file not already represented by a tracked
        path. Idempotent: dedup is keyed on the resolved path, so re-running it
        each turn only ever picks up genuinely new files (already-tracked
        downloads and prior sweeps are skipped). Returns the number added.
        """
        try:
            artifacts_dir = deps.session.artifacts_dir
        except Exception:
            return 0
        if not artifacts_dir.exists():
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
        for p in sorted(artifacts_dir.rglob("*")):
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
                    tool_name="bash",
                    description=description,
                    timestamp=datetime.now().isoformat(),
                    tool_call_id=tool_call_id,
                )
            )
            added += 1
        return added

    def _eager_download_artifact(self, artifact: Artifact) -> None:
        """Submit a background download for a remote artifact.

        Fires from AgentDependencies.add_artifact the moment a tool tracks an
        artifact. By the time save_session runs, most artifacts are already on
        disk and ``download_session_artifacts`` is a fast no-op. Idempotent —
        already-downloaded artifacts are skipped.
        """
        if not artifact.url or artifact.path:
            return

        def _run() -> None:
            try:
                artifacts_dir = self.session.artifacts_dir
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                local_path = download_artifact(artifact.url, artifacts_dir, artifact.artifact_type)
                artifact.path = local_path
            except Exception:
                pass

        try:
            self._artifact_dl_pool.submit(_run)
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

        self.session.tokens.add(
            input_tokens=in_toks,
            output_tokens=out_toks,
            requests=getattr(usage, "requests", 0) or 1,
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0) or 0,
            reasoning_tokens=reasoning,
            cost_usd=cost,
        )

    def _on_event(self, ev: Any) -> None:
        """Translate one pydantic-ai stream event into UI events. Covers
        every toolset (function tools, MCP) because pydantic-ai emits the
        same call/result events for all of them, keyed by tool_call_id."""
        match ev:
            case PartStartEvent(part=TextPart(content=text)) | PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                self._stream_text(text)
            case FunctionToolCallEvent(part=call):
                self._open_tool_call(call)
            case EnqueuedMessagesEvent(enqueue_id=enqueue_id):
                if self._enqueued.pop(enqueue_id, None) is not None:
                    self.emitter.emit(SystemNote("↳ Message picked up by the running turn."))
            case FunctionToolResultEvent(part=result):
                widget = self._open_tool_calls.pop(result.tool_call_id, None)
                if widget is not None:
                    self.emitter.emit(widget.finish(**_tool_result_display(result)))
                if not self._open_tool_calls:
                    self.status = ""

    def enqueue(self, text: str) -> bool:
        """Deliver ``text`` into the running turn at its next model request,
        so the user can redirect or add to a turn without cancelling it. If
        the model was about to finish, it answers the message instead.
        Must be called on the event loop driving ``run_async``. Returns
        False when no turn is running."""
        if self._agent_run is None:
            return False
        prompt = _parse_user_message(text, emitter=self.emitter)
        self._record_references(text)
        enqueue_id = self._agent_run.enqueue(*prompt) if isinstance(prompt, list) else self._agent_run.enqueue(prompt)
        if enqueue_id is not None:
            self._enqueued[enqueue_id] = text
        return True

    def _record_references(self, text: str) -> None:
        """Copy any images the user mentioned into the session's
        ``artifacts/references/``, off the event loop since a URL means
        a download. Failures only warn: the model already has the pixels."""
        refs = list(_iter_image_refs(text))
        if not refs:
            return

        def _store() -> None:
            try:
                store_references(self.session, refs)
            except Exception as e:  # noqa: BLE001
                self.emitter.emit(SystemNote(f"Couldn't keep a reference image: {e}", level="warn"))

        self._tool_pool.submit(_store)

    def detach_running_tools(self) -> None:
        """Move the running turn's in-flight tool calls to the background
        now (as ``tool-N`` tasks), so a message just enqueued reaches the
        model without waiting on them."""
        self._detach_now.set()

    def take_undelivered(self) -> list[str]:
        """Messages enqueued into a turn that ended (or was cancelled)
        before reaching its next model request; the caller resubmits them
        as new turns."""
        texts = list(self._enqueued.values())
        self._enqueued.clear()
        return texts

    @property
    def streaming_text(self) -> bool:
        """True while the model is mid-reply with no tool running — the
        moment a new user message should cut in rather than wait."""
        return self._current_stream is not None and not self._open_tool_calls

    def _stream_text(self, text: str) -> None:
        if not text:
            return
        self.status = ""
        if self._current_stream is None:
            self._current_stream = AssistantStream()
        self._current_stream.text += text
        self.emitter.emit(self._current_stream)

    def _end_text_stream(self) -> None:
        if self._current_stream is not None:
            self._current_stream.done = True
            self.emitter.emit(self._current_stream)
            self._current_stream = None

    def _open_tool_call(self, part: ToolCallPart) -> None:
        call_id = part.tool_call_id
        if part.tool_name == "load_skill":
            try:
                args: Any = part.args_as_dict()
            except ValueError:
                args = part.args
            name = args.get("name", args) if isinstance(args, dict) else args
            ev = ToolCallDisplay("load_skill", args=str(name), details=str(args), tool_call_id=call_id)
        else:
            args_str = str(part.args) if part.args else ""
            ev = ToolCallDisplay(part.tool_name, args=args_str, details=args_str, tool_call_id=call_id)
        self._open_tool_calls[call_id] = self.emitter.emit(ev)
        self.status = f"Running {part.tool_name}..."

    def _close_open_tool_calls(self, error: str) -> None:
        """A run that raises or is cancelled mid-tool produces no result
        event; close its widgets so none are left spinning."""
        for widget in self._open_tool_calls.values():
            self.emitter.emit(widget.finish(error=error))
        self._open_tool_calls.clear()

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
        self.last_turn_error = None
        self.emitter.emit(TurnStart(turn=turn))

        # Snapshot the interrupt flag before clearing it; the checks below
        # need the prior turn's value.
        prior_interrupted = self._was_interrupted
        self._was_interrupted = False
        self._turn_cancelled = threading.Event()
        self._detach_now = threading.Event()
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

        partial, self._interrupted_partial = self._interrupted_partial, ""
        # Task notices ("[system] ...") are not the user speaking.
        if prior_interrupted and self.state.context and not user_message.startswith("[system]"):
            said = f" You had started replying: \"{partial.strip()}\"" if partial.strip() else ""
            user_message = (
                f"[The user interrupted the previous response to give new instructions.{said} "
                "Stop what you were doing and focus on this message.]\n\n" + user_message
            )

        prompt = _parse_user_message(user_message, emitter=self.emitter)
        self._record_references(user_message)
        timeout = asyncio.timeout(self.timeout) if self.timeout > 0 else contextlib.nullcontext()

        try:
            await self._ensure_mcp_connected()
            async with self.agent.iter(
                prompt,
                deps=deps,
                message_history=self.state.context or None,
            ) as agent_run:
                self._agent_run = agent_run
                try:
                    async with timeout:
                        async for node in agent_run:
                            if Agent.is_model_request_node(node):
                                self.status = "Thinking..."
                            elif not Agent.is_call_tools_node(node):
                                continue
                            async with node.stream(agent_run.ctx) as events:
                                async for ev in events:
                                    self._on_event(ev)
                            self._end_text_stream()

                except (asyncio.CancelledError, TimeoutError) as exc:
                    self._handle_interruption(agent_run, isinstance(exc, TimeoutError))
                    if isinstance(exc, asyncio.CancelledError):
                        raise

                    self.last_turn_error = f"turn timed out after {self.timeout:.0f}s"
                    return ""

                result = agent_run.result

            messages = result.all_messages()
            self._commit_turn(agent_run, messages)
            return result.output

        except Exception as e:
            self._close_open_tool_calls(f"{type(e).__name__}: {e}")
            self.emitter.emit(SystemNote(f"Error: {e}", level="error"))
            self.last_turn_error = f"{type(e).__name__}: {e}"
            return f"Error: {str(e)}"
        finally:
            # The turn is over on every exit path (normal, cancel, timeout,
            # error). Clear the spinner so the status bar can't get stuck on
            # a stale "Thinking..." while the agent is actually idle.
            self._agent_run = None
            self._end_text_stream()
            self._close_open_tool_calls("interrupted")
            self.status = ""
            self.emitter.emit(TurnEnd(turn=turn))

    def _handle_interruption(self, agent_run, timed_out: bool) -> None:
        """Commit partial state and surface a timeout note. Caller decides
        whether to re-raise CancelledError or return an empty string."""
        self._was_interrupted = True
        self._turn_cancelled.set()
        self._interrupted_partial = self._current_stream.text if self._current_stream is not None else ""
        self._pending_stream_events.append(
            Interrupt("timeout" if timed_out else "cancelled", partial=self._interrupted_partial)
        )
        self._commit_turn(agent_run, strip_unanswered_tool_calls(agent_run.all_messages()))
        if timed_out:
            self.emitter.emit(
                SystemNote(
                    f"Turn timed out after {self.timeout:.0f}s. Partial state preserved.",
                    level="warn",
                )
            )

    def _commit_turn(self, agent_run, post_messages: list) -> None:
        """Fold the run's results into ``self.state``. Shared by the
        normal-completion and cancellation paths."""
        self._accumulate_usage(agent_run)
        try:
            # A cancelled turn ends on a tool call with no return; persisting
            # it would replay as a tool that never finishes.
            self.state.message_history.extend(strip_unanswered_tool_calls(list(agent_run.new_messages())))
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
        download_artifacts: bool = True,
    ) -> None:
        """Persist the live state.

        Args:
            status: Status of the session (completed, error, abandoned)
            download_artifacts: When True (default), synchronously download
                any tracked outputs that don't yet have a ``local_path``.
                Set False on shutdown / atexit paths so quitting doesn't
                wait on slow network downloads — the eager bg pool has
                been pulling these all along, and any still-missing files
                will be retried on the next session resume.
        """
        session = self.session

        if download_artifacts:
            download_session_artifacts(session.state.artifacts, session.artifacts_dir)

        # Stamp the run-owned fields onto the live session; identity
        # fields (title / created_at) are write-once at the
        # DB layer, so this is a plain write.
        session.status = status
        session.model = self.model_id

        session.save()
        self._flush_stream_events()

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
