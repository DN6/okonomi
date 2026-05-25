"""TuiSession — drives the agent from the Textual app.

Replaces InteractiveSession (prompt_toolkit). Responsibilities:
  - Own the EventEmitter the agent writes to.
  - Run the agent in a worker thread with a per-turn asyncio loop.
  - Bridge user input from the Textual app → agent queue.
  - Bridge approval requests from the agent → Textual modal → decision back to agent.
  - Save session on exit, handle Ctrl+C interrupt, Ctrl+D quit.
"""

from __future__ import annotations

import asyncio
import threading
from queue import Queue
from typing import Any

from ..agent import OkonomiAgent
from ..session import SessionStatus
from .app import OkonomiApp
from .events import Banner, EventEmitter


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run remaining tasks briefly, then close the loop."""
    try:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if pending:
            for t in pending:
                t.cancel()

            async def _wait() -> None:
                await asyncio.gather(*pending, return_exceptions=True)

            loop.run_until_complete(_wait())
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        try:
            loop.close()
        except Exception:
            pass


class TuiSession:
    """Textual-driven interactive session for OkonomiAgent."""

    def __init__(self, agent: OkonomiAgent) -> None:
        self.agent = agent
        self.emitter = EventEmitter()
        self.agent.emitter = self.emitter

        self._input_queue: Queue[str | None] = Queue()
        self._agent_running = threading.Event()
        self._agent_lock = threading.Lock()
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._agent_task: asyncio.Task | None = None

        self._approval_queue: Queue[Any] = Queue()
        self.agent.approval_callback = self._approval_callback
        self.agent.on_bg_complete = self._on_bg_exec_complete
        self.agent.terminal._on_bg_complete = self._on_bg_exec_complete

        self._app: OkonomiApp | None = None
        self._worker: threading.Thread | None = None
        self._next_reset: bool = True

        # Wall-clock start of the in-flight turn. Used by the status bar
        # to render an elapsed timer ("Thinking… 1m 23s"). None when idle.
        self._turn_started_at: float | None = None

        # Background full-agent tasks (not to be confused with the agent's
        # bg-exec threads for expensive code). Keyed by bg-N id.
        self._bg_tasks: dict[str, threading.Thread] = {}
        self._bg_task_meta: dict[str, str] = {}  # task_id → truncated prompt
        self._bg_counter = 0

    # --- Agent worker ---

    def _worker_loop(self) -> None:
        """Pull messages from the queue, run the agent per message."""
        while True:
            msg = self._input_queue.get()
            if msg is None:
                break
            reset = self._next_reset
            self._next_reset = False
            self._run_agent_turn(msg, reset=reset)

    def _run_agent_turn(self, user_message: str, reset: bool) -> None:
        import time as _time

        self._turn_started_at = _time.monotonic()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._agent_lock:
            self._agent_loop = loop
        self._agent_running.set()
        try:
            task = loop.create_task(self.agent.run_async(user_message, reset=reset))
            with self._agent_lock:
                self._agent_task = task
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            self.emitter.note("Interrupted. History preserved.", level="warn")
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            # Keep the first line (summary) visible; stash full traceback
            # as an expandable tool-call-style event. Users can report a
            # concrete error instead of "the agent broke."
            self.emitter.note(f"Agent error: {type(e).__name__}: {e}", level="error")
            trace_event = self.emitter.tool_call(
                "traceback",
                args="agent failure",
                details=tb,
            )
            self.emitter.tool_finish(trace_event, error="see traceback above")
        finally:
            with self._agent_lock:
                self._agent_task = None
                self._agent_loop = None
            try:
                _shutdown_loop(loop)
            except Exception:
                pass
            self._agent_running.clear()
            self._turn_started_at = None
            # Persist session state after every turn — completion, error,
            # or interrupt. Sessions disappearing because the user Ctrl+D'd
            # before the once-on-exit save is unacceptable.
            self._safe_save_session(status="completed")

    def _safe_save_session(self, *, status: SessionStatus, download_assets: bool = False) -> None:
        """Save the session swallowing any exception.

        Called from multiple places (per-turn finally, app exit, atexit)
        so it must never raise — a save error must not crash the session.
        ``download_assets`` defaults to False because the agent's eager
        background download pool keeps assets up to date independently;
        making save_session also block on downloads is what was making
        quit slow.
        """
        try:
            self.agent.save_session(status=status, download_assets=download_assets)
        except Exception:  # noqa: BLE001
            pass

    def cancel_current_turn(self) -> None:
        """Cancel the in-flight agent turn (Ctrl+C)."""
        with self._agent_lock:
            loop = self._agent_loop
            task = self._agent_task
        if loop and task and not task.done():
            loop.call_soon_threadsafe(task.cancel)

    # --- Submit / approval / bg-exec hooks ---

    def _on_submit(self, text: str) -> None:
        """Handle user text from the Textual app.

        Runs in the Textual thread; push work to the worker via queue.
        Intercepts slash commands before they reach the agent.
        """
        from . import commands

        self.emitter.divider()
        self.emitter.user(text)

        if commands.is_slash_command(text):
            result = commands.handle(text, self.agent, self.emitter)
            if result is None:
                return  # handled locally, no agent turn
            if result == "__exit__":
                if self._app is not None:
                    self._app.exit()
                return
            if result == "__session_picker__":
                if self._app is not None:
                    self._app.show_session_picker(self._resume_session)
                return
            if result.startswith("__bg__:"):
                self._spawn_bg_task(result[len("__bg__:") :])
                return
            if result.startswith("__btw__:"):
                self._inject_side_question(result[len("__btw__:") :])
                return
            # Command returned follow-on text — forward to the agent.
            text = result

        self._input_queue.put(text)

    def _inject_side_question(self, question: str) -> None:
        """Cancel the in-flight turn and queue the question as a course
        correction. The agent's run_async path already handles
        ``_was_interrupted`` by prepending the new message with a
        priority marker, so this works with no agent-side changes.
        """
        if self._agent_running.is_set():
            self.cancel_current_turn()
            # Tag the message so the agent knows this is a side question
            # rather than a routine continuation. The "interrupted"
            # framing in run_async will pick it up.
            tagged = f"[Side question while you were working]\n\n{question}"
        else:
            tagged = question
        self._input_queue.put(tagged)

    def _spawn_bg_task(self, task_input: str) -> None:
        """Run a task in the background with a separate OkonomiAgent.

        Mirrors the prompt_toolkit CLI's _run_bg_task. Outputs merge back
        into the parent agent's session on completion.
        """
        from ..agent import OkonomiAgent

        self._bg_counter += 1
        task_id = f"bg-{self._bg_counter}"
        parent = self.agent
        approval_cb = parent.approval_callback
        on_bg_complete_cb = parent.on_bg_complete

        self.emitter.note(f"Background task {task_id} started: {task_input[:80]}")

        def _run() -> None:
            bg_agent = OkonomiAgent(
                parent.config,
                session_id=parent.session_id,
            )
            bg_agent.approval_callback = approval_cb
            bg_agent.on_bg_complete = on_bg_complete_cb
            # Use an isolated emitter so the bg agent's tool calls and
            # streaming text do NOT show up in the foreground history.
            # Only the completion note (emitted by THIS thread via
            # self.emitter, below) surfaces to the user.
            bg_agent.emitter = EventEmitter()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(bg_agent.run_async(task_input, reset=True))
                parent.artifacts.extend(bg_agent.artifacts)
                parent.tools_used.update(bg_agent.tools_used)
                n = len(bg_agent.artifacts)
                self.emitter.note(f"Background task {task_id} done — {n} artifact{'s' if n != 1 else ''}")
            except Exception as e:  # noqa: BLE001
                self.emitter.note(f"Background task {task_id} failed: {e}", level="error")
            finally:
                _shutdown_loop(loop)
                self._bg_tasks.pop(task_id, None)
                self._bg_task_meta.pop(task_id, None)

        thread = threading.Thread(target=_run, daemon=True, name=f"okonomi-{task_id}")
        self._bg_tasks[task_id] = thread
        self._bg_task_meta[task_id] = task_input[:50] + "…" if len(task_input) > 50 else task_input
        thread.start()

    def _resume_session(self, session_id: str) -> None:
        """Callback from the session picker. Loads a past session and
        replays its conversation into the chat view so the user can
        actually see where they left off (otherwise the scrollback is
        empty even though the model has full context)."""
        if not self.agent.load_session(session_id):
            self.emitter.note(f"Failed to load session: {session_id}", level="error")
            return

        self._replay_message_history()
        meta = self.agent.session_manager.load_session(session_id, with_history=False)
        title = (meta.title if meta else "") or ""
        label = f"{session_id[:10]} · {title}" if title else session_id
        self.emitter.note(f"↑ Resumed: {label}")

    def _replay_message_history(self) -> None:
        """Re-emit the saved conversation as visible chat events.

        Walks ``agent.message_history`` and emits one ``UserMessage`` per
        user prompt, one ``AssistantStream`` per model response (plus
        any tool calls), and attaches tool returns to the corresponding
        widget. Synthetic injections (resume markers, `[system]` notes,
        interruption markers) are skipped — they're scaffolding, not
        user-visible turns.
        """
        history = getattr(self.agent, "message_history", None) or []
        if not history:
            return

        try:
            from pydantic_ai.messages import (
                ModelRequest,
                ModelResponse,
                TextPart,
                ToolCallPart,
                ToolReturnPart,
                UserPromptPart,
            )
        except Exception:
            return

        SYNTHETIC_PREFIXES = (
            "[Resumed",
            "[system]",
            "[The user interrupted",
        )
        pending_tool_calls: dict[str, Any] = {}
        any_emitted = False

        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        text = part.content if isinstance(part.content, str) else str(part.content)
                        if text.strip().startswith(SYNTHETIC_PREFIXES):
                            continue
                        if any_emitted:
                            self.emitter.divider()
                        self.emitter.user(text)
                        any_emitted = True
                    elif isinstance(part, ToolReturnPart):
                        tc = pending_tool_calls.pop(getattr(part, "tool_call_id", ""), None)
                        if tc is None:
                            continue
                        content = part.content if isinstance(part.content, str) else str(part.content)
                        try:
                            self.emitter.tool_finish(tc, output=content[:2000])
                        except Exception:
                            pass
            elif isinstance(msg, ModelResponse):
                stream = None
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        chunk = part.content
                        if not chunk or not chunk.strip():
                            continue
                        if stream is None:
                            stream = self.emitter.assistant_start()
                        self.emitter.assistant_append(stream, chunk)
                        any_emitted = True
                    elif isinstance(part, ToolCallPart):
                        args_repr = repr(part.args) if not isinstance(part.args, str) else part.args
                        tc = self.emitter.tool_call(
                            tool_name=part.tool_name,
                            args=args_repr[:80],
                            details=args_repr[:5000],
                        )
                        pending_tool_calls[getattr(part, "tool_call_id", "")] = tc
                        any_emitted = True
                if stream is not None:
                    self.emitter.assistant_done(stream)

    async def _approval_callback(self, code: str) -> bool | str:
        """Agent asks: is this expensive code ok to run?

        Called from the agent's worker thread (inside the per-turn event loop).
        Blocks until the UI replies via _resolve_approval(). The callback is
        async, but we block with asyncio.to_thread since the queue is sync.
        """
        if self._app is None:
            return True
        self._app.call_from_thread(self._app.show_approval, code, self._resolve_approval)
        decision = await asyncio.to_thread(self._approval_queue.get)
        return decision

    def _resolve_approval(self, decision: bool | str) -> None:
        """Called by the app when the user responds to an approval modal."""
        self._approval_queue.put(decision)

    def _on_bg_exec_complete(
        self,
        *,
        bg_id: str,
        error: str | None,
        output: str | None,
        new_artifacts: int,
        produced_artifacts: list | None = None,
    ) -> None:
        """Inject a synthetic turn so the agent can surface completed media.

        Names the *specific* assets this bg task produced so the agent
        doesn't grab unrelated older items from session state.
        """
        if error:
            msg = (
                f"[system] Background task {bg_id} failed:\n{error}\n\n"
                "DO NOT re-run genmedia.run, browser.render, or any "
                "other expensive generation call to recover. If this "
                "is a parsing error (KeyError / AttributeError / TypeError "
                "after the generation line), the generation already "
                "succeeded — the `result` variable is still in globals from "
                "the bg run. Write a new execute_code block with "
                "`print(result)` to inspect its shape and fix the access. "
                "If the generation call itself raised (network/500/timeout), "
                "surface the error to the user and ask before retrying."
            )
            self._input_queue.put(msg)
            return

        asset_lines: list[str] = []
        for a in produced_artifacts or []:
            path = getattr(a, "path", "") or ""
            url = getattr(a, "url", "") or ""
            kind = getattr(a, "artifact_type", "") or "artifact"
            ref = path or url
            if ref:
                asset_lines.append(f"  - {kind}: {ref}")
        produced_block = "\n".join(asset_lines) if asset_lines else "  (none)"

        msg = (
            f"[system] Background task {bg_id} completed.\n\n"
            f"Assets produced by THIS task:\n{produced_block}\n\n"
            "Use ONLY these specific URLs/paths when presenting to the user. "
            "Do NOT reference older session assets and do NOT regenerate. "
            "The list above is authoritative for what "
            f"{bg_id} just produced."
        )
        self._input_queue.put(msg)

    # --- Entry points ---

    async def run(self, initial_message: str = "") -> None:
        # Last-ditch save: even if the process is killed via SIGTERM /
        # uncaught exception / Textual hang, atexit fires before the
        # interpreter exits.
        import atexit

        atexit.register(self._safe_save_session, status="abandoned")

        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="okonomi-agent")
        self._worker.start()

        # Silently prune empty sessions left over from previous runs.
        try:
            self.agent.session_manager.prune_empty_sessions()
        except Exception:  # noqa: BLE001
            pass

        # Sakura wave banner + one-line welcome.
        self.emitter.emit(Banner())
        self.emitter.note(
            "type a message or /help. Ctrl+D to exit, Ctrl+O to expand the latest tool call.",
        )

        # Surface a one-time install hint when the genmedia CLI isn't on
        # PATH — it's the recommended path for fal-backed media gen.
        # Don't crash; the user might be doing a no-fal session.
        import shutil as _shutil

        if _shutil.which("genmedia") is None:
            self.emitter.note(
                "genmedia CLI not found on $PATH. Install with:\n"
                "  curl -fsSL https://genmedia.sh/install | sh\n"
                "Until then, image/video generation via the genmedia "
                "skill won't work — local tools (browser, prompt, "
                "research) still do.",
                level="warn",
            )

        def _spinner_label() -> str:
            # Always show liveness while a turn is in flight, even during
            # the gaps when agent.spinner_text is briefly empty (between
            # tool calls, mid-stream). Append an elapsed timer ("1m 23s")
            # so long-running turns feel honest about how long they've
            # been working.
            txt = getattr(self.agent, "spinner_text", "") or ""
            if not txt and self._agent_running.is_set():
                txt = "Working..."
            if not txt:
                return ""
            started = self._turn_started_at
            if started is None:
                return txt
            import time as _time

            elapsed = int(_time.monotonic() - started)
            m, s = divmod(elapsed, 60)
            stamp = f"{m}m {s}s" if m else f"{s}s"
            return f"{txt}  ({stamp})"

        def _usage_label() -> str:
            usage = getattr(self.agent, "token_usage", None)
            if usage is None or not getattr(usage, "total_tokens", 0):
                return ""
            return str(usage)

        def _bg_tasks() -> list[tuple[str, str]]:
            """List of (id, label) pairs for in-flight bg-exec + /bg tasks.

            Filters out threads that have finished but not yet been cleared
            from the dicts (race around thread.is_alive during shutdown).
            """
            out: list[tuple[str, str]] = []
            # Agent-level bg-exec threads (expensive execute_code blocks).
            # Label comes from the shared _bg_logs registry — same source
            # of truth /jobs uses, so the bottom bar matches the slash command.
            execs = getattr(self.agent, "_bg_execs", {}) or {}
            logs = getattr(self.agent, "_bg_logs", {}) or {}
            for bg_id, thread in list(execs.items()):
                if thread and thread.is_alive():
                    label = (logs.get(bg_id) or {}).get("label", "")
                    out.append((bg_id, label))
            # Session-level /bg full-agent tasks
            for task_id, thread in list(self._bg_tasks.items()):
                if thread and thread.is_alive():
                    out.append((task_id, self._bg_task_meta.get(task_id, "")))
            return out

        self._app = OkonomiApp(
            self.emitter,
            on_submit=self._on_submit,
            spinner_label_fn=_spinner_label,
            usage_fn=_usage_label,
            bg_tasks_fn=_bg_tasks,
        )
        self._app.session = self  # type: ignore[attr-defined]

        if initial_message:
            self._on_submit(initial_message)

        try:
            # inline=True renders the app into the regular terminal output
            # instead of the alternate-screen buffer, so shell history stays
            # visible above and the app joins normal scrollback on exit.
            # mouse=False disables Textual's mouse capture so the terminal's
            # own text selection / copy-paste keeps working.
            await self._app.run_async(
                inline=True,
                inline_no_clear=True,
                mouse=False,
            )
        finally:
            self._input_queue.put(None)
            if self._agent_running.is_set():
                self.cancel_current_turn()

        # Drop in-flight downloads on the floor — URLs are persisted in
        # the DB, resume retries anything missing. Then save metadata
        # only (no synchronous download sweep) so quit returns instantly.
        try:
            self.agent._asset_dl_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        self._safe_save_session(status="completed", download_assets=False)
