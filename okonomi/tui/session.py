"""TuiSession — drives the agent from the streaming-scrollback REPL.

Owns a persistent worker loop on a daemon thread; user messages are
serialized through an ``asyncio.Queue`` and consumed one turn at a
time. Cross-thread coordination (cancel, submit, approval reply) marshals
through ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout

from ..agent import OkonomiAgent
from ..events import Banner, Divider, EventEmitter, SystemNote, ToolCall, UserMessage
from ..session import SessionStatus
from .app import ReplApp
from .asset_actions import handle_asset_action
from .transcript import replay_history


class TuiSession:
    """Streaming-scrollback REPL session for OkonomiAgent."""

    def __init__(self, agent: OkonomiAgent) -> None:
        self.agent = agent
        self.emitter = agent.emitter

        # Persistent worker loop on its own thread. Turns are submitted
        # via run_coroutine_threadsafe, serialized through an asyncio
        # queue consumed by _worker_loop. Cancellation flips a flag
        # on self._current_task via the same loop.
        self._loop = asyncio.new_event_loop()
        self._agent_thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="okonomi-agent")
        self._agent_thread.start()

        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._current_task: asyncio.Task | None = None
        self._worker_loop_future = asyncio.run_coroutine_threadsafe(self._worker_loop(), self._loop)

        # The inline approval UI is single-slot (one input box), so
        # concurrent approvals — e.g. a background task and the foreground
        # turn — must serialize. A thread lock, not asyncio.Lock: the
        # callers may run on different loops/threads (bg-task agents).
        self._approval_ui_lock = threading.Lock()
        self.agent.approval_callback = self._approval_callback
        self.agent.on_bg_complete = self._on_bg_exec_complete
        self.agent.terminal._on_bg_complete = self._on_bg_exec_complete

        # First-turn reset wipes message_history/artifacts/etc. for a
        # fresh start. Skip it if the agent was already loaded (CLI
        # ``--resume`` / ``--session`` path) — otherwise the first
        # prompt blows away the resumed state.
        self._next_reset: bool = not bool(self.agent.state.message_history)

        # Wall-clock start of the in-flight turn. Used by the status bar
        # to render an elapsed timer ("Thinking… 1m 23s"). None when idle.
        self._turn_started_at: float | None = None

        # Background full-agent tasks (not the agent's per-execute_code
        # bg-exec threads). Keyed by bg-N id.
        self._bg_tasks: dict[str, threading.Thread] = {}
        self._bg_counter = 0

        # The prompt_toolkit streaming-scrollback REPL front-end. Prints the
        # EventEmitter stream into native terminal scrollback and owns the
        # bottom input chrome.
        self._app = ReplApp(
            self.emitter,
            on_submit=self._on_submit,
            spinner_label_fn=self._spinner_label,
            usage_fn=self._usage_label,
            bg_tasks_fn=self._bg_tasks_list,
        )
        self._app.session = self  # type: ignore[attr-defined]

    # --- Agent worker ---

    @property
    def _agent_running(self) -> bool:
        task = self._current_task
        return task is not None and not task.done()

    async def _worker_loop(self) -> None:
        """Pull messages from the asyncio queue, run one turn per message."""
        while True:
            msg = await self._input_queue.get()
            if msg is None:
                break
            reset = self._next_reset
            self._next_reset = False
            await self._run_agent_turn(msg, reset=reset)

    async def _run_agent_turn(self, user_message: str, reset: bool) -> None:
        self._turn_started_at = time.monotonic()
        self._current_task = asyncio.current_task()
        status: SessionStatus = "completed"
        try:
            await self.agent.run_async(user_message, reset=reset)
        except asyncio.CancelledError:
            status = "interrupted"
            self.emitter.emit(SystemNote("Interrupted. History preserved.", level="warn"))
        except Exception as e:  # noqa: BLE001
            import traceback

            status = "error"
            tb = traceback.format_exc()
            self.emitter.emit(SystemNote(f"Agent error: {type(e).__name__}: {e}", level="error"))
            trace_event = self.emitter.emit(ToolCall("traceback", args="agent failure", details=tb))
            self.emitter.emit(trace_event.finish(error="see traceback above"))
        finally:
            self._current_task = None
            self._turn_started_at = None
            # Off-load to a thread — save can do an LLM call (steps
            # summary) and a pickle dump that easily takes seconds.
            # Blocking the consumer here would stall queued submits and
            # cancellations marshalled through the same loop.
            await asyncio.to_thread(self._safe_save_session, status=status)

    def _safe_save_session(
        self,
        *,
        status: SessionStatus,
        download_assets: bool = False,
        skip_summary: bool = False,
    ) -> None:
        """Save the session swallowing any exception.

        Called from multiple places (per-turn finally, app exit, atexit)
        so it must never raise — a save error must not crash the session.
        ``download_assets`` defaults to False because the agent's eager
        background download pool keeps assets up to date independently.
        ``skip_summary`` skips the LLM steps-summary call (set True on
        shutdown so exit doesn't block on a 10–60s OpenRouter request).
        """
        try:
            self.agent.save_session(
                status=status,
                download_assets=download_assets,
                skip_summary=skip_summary,
            )
        except Exception:  # noqa: BLE001
            pass

    def cancel_current_turn(self) -> None:
        """Cancel the in-flight agent turn (Ctrl+C).

        Marshalled through the worker loop so the task-read and the
        cancel-call happen on the same thread that owns the task.
        """

        def _do_cancel() -> None:
            task = self._current_task
            if task is not None and not task.done():
                task.cancel()

        self._loop.call_soon_threadsafe(_do_cancel)

    # --- Submit / approval / bg-exec hooks ---

    def _on_submit(self, text: str) -> None:
        """Handle user text from the REPL.

        Runs on the prompt_toolkit loop; push work to the worker via queue.
        Intercepts slash commands before they reach the agent.
        """
        from . import commands

        self.emitter.emit(Divider())
        self.emitter.emit(UserMessage(text))

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
            if result == "__asset_picker__":
                if self._app is not None:
                    self._app.show_asset_picker(
                        list(self.agent.state.artifacts),
                        lambda action, paths: handle_asset_action(action, paths, self.agent, self.emitter),
                    )
                return
            if result == "__verbose__":
                toggle = getattr(self._app, "toggle_verbose", None)
                if toggle is not None:
                    toggle()
                else:
                    self.emitter.emit(SystemNote("`/verbose` is only available in the REPL.", level="warn"))
                return
            if result.startswith("__bg__:"):
                self._spawn_bg_task(result[len("__bg__:") :])
                return
            if result.startswith("__btw__:"):
                self._inject_side_question(result[len("__btw__:") :])
                return

            # Command returned follow-on text — forward to the agent.
            text = result

        self._submit(text)

    def _submit(self, msg: str | None) -> None:
        """Schedule a message onto the worker loop's input queue.

        ``None`` is the shutdown sentinel — the consumer breaks on it.
        """
        self._loop.call_soon_threadsafe(self._input_queue.put_nowait, msg)

    def _inject_side_question(self, question: str) -> None:
        """Cancel the in-flight turn and queue the question as a course
        correction. The agent's run_async path already handles
        ``_was_interrupted`` by prepending the new message with a
        priority marker, so this works with no agent-side changes.
        """
        if self._agent_running:
            self.cancel_current_turn()
            # Tag the message so the agent knows this is a side question
            # rather than a routine continuation. The "interrupted"
            # framing in run_async will pick it up.
            tagged = f"[Side question while you were working]\n\n{question}"
        else:
            tagged = question
        self._submit(tagged)

    def _spawn_bg_task(self, task_input: str) -> None:
        """Run a task in the background with a separate OkonomiAgent.

        Mirrors the prompt_toolkit CLI's _run_bg_task. Outputs merge back
        into the parent agent's session on completion.
        """
        self._bg_counter += 1
        task_id = f"bg-{self._bg_counter}"
        parent = self.agent
        approval_cb = parent.approval_callback
        on_bg_complete_cb = parent.on_bg_complete

        self.emitter.emit(SystemNote(f"Background task {task_id} started: {task_input[:80]}"))

        def _run() -> None:
            bg_agent = OkonomiAgent(
                model_id=parent.model_id,
                timeout=parent.timeout,
                compress_at_tokens=parent.compress_at_tokens,
                session_id=parent.state.info.session_id,
            )
            bg_agent.approval_callback = approval_cb
            bg_agent.on_bg_complete = on_bg_complete_cb
            # Isolated emitter — the bg agent's tool calls and streaming
            # text don't show up in the foreground history; only the
            # completion note (below) surfaces to the user.
            bg_agent.emitter = EventEmitter()
            try:
                asyncio.run(bg_agent.run_async(task_input, reset=True))
                parent.state.artifacts.extend(bg_agent.state.artifacts)
                parent.state.tools_used.update(bg_agent.state.tools_used)
                n = len(bg_agent.state.artifacts)
                self.emitter.emit(SystemNote(f"Background task {task_id} done — {n} artifact{'s' if n != 1 else ''}"))
            except Exception as e:  # noqa: BLE001
                self.emitter.emit(SystemNote(f"Background task {task_id} failed: {e}", level="error"))
            finally:
                self._bg_tasks.pop(task_id, None)

        label = task_input[:50] + "…" if len(task_input) > 50 else task_input
        thread = threading.Thread(target=_run, daemon=True, name=label)
        self._bg_tasks[task_id] = thread
        thread.start()

    def _resume_session(self, session_id: str) -> None:
        """Callback from the session picker. Loads a past session and
        replays its conversation into the chat view so the user can
        actually see where they left off (otherwise the scrollback is
        empty even though the model has full context)."""
        if not self.agent.load_session(session_id):
            self.emitter.emit(SystemNote(f"Failed to load session: {session_id}", level="error"))
            return

        # Skip the first-turn reset — it would wipe the freshly-loaded
        # artifacts and message history.
        self._next_reset = False

        replay_history(self.agent.state.message_history, self.emitter)
        meta = self.agent.session_manager.get_info(session_id)
        title = (meta.title if meta else "") or ""
        label = f"{session_id[:10]} · {title}" if title else session_id
        self.emitter.emit(SystemNote(f"↑ Resumed: {label}"))

    async def _approval_callback(self, code: str) -> bool | str:
        """Agent asks: is this expensive code ok to run?

        Each request gets a private Future captured in the resolve closure
        so a decision can never be delivered to the wrong waiter. The UI
        lock serializes requests because only one approval prompt can be
        shown at a time; a concurrent caller (a background task agent)
        waits for its turn.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool | str] = loop.create_future()

        def _resolve(decision: bool | str) -> None:
            loop.call_soon_threadsafe(fut.set_result, decision)

        await asyncio.to_thread(self._approval_ui_lock.acquire)
        try:
            self._app.call_from_thread(self._app.show_approval, code, _resolve)
            return await fut
        finally:
            self._approval_ui_lock.release()

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
            self._submit(msg)
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
        self._submit(msg)

    # --- Entry points ---

    # --- Status-bar callables passed to ReplApp ---

    def _spinner_label(self) -> str:
        """Working-state label with elapsed timer. Always shows liveness
        while a turn is in flight (even when ``agent.spinner_text`` is
        briefly empty between tool calls)."""
        txt = self.agent.spinner_text or ""
        if not txt and self._agent_running:
            txt = "Working..."
        if not txt:
            return ""
        if self._turn_started_at is None:
            return txt
        elapsed = int(time.monotonic() - self._turn_started_at)
        m, s = divmod(elapsed, 60)
        stamp = f"{m}m {s}s" if m else f"{s}s"
        return f"{txt}  ({stamp})"

    def _usage_label(self) -> str:
        usage = self.agent.state.info.tokens
        if not usage.total_tokens:
            return ""
        return str(usage)

    def _bg_tasks_list(self) -> list[tuple[str, str]]:
        """(id, label) pairs for in-flight bg-exec + /bg tasks. Threads
        that have finished but not yet been popped from their dicts are
        filtered (race around thread.is_alive during shutdown)."""
        out: list[tuple[str, str]] = []
        for bg_id, thread in list(self.agent._bg_execs.items()):
            if thread and thread.is_alive():
                label = (self.agent._bg_logs.get(bg_id) or {}).get("label", "")
                out.append((bg_id, label))
        for task_id, thread in list(self._bg_tasks.items()):
            if thread and thread.is_alive():
                out.append((task_id, thread.name))
        return out

    # --- Entry points ---

    async def run(self, initial_message: str = "") -> None:
        import atexit
        import shutil

        # Last-ditch save: atexit fires before interpreter exit even on
        # SIGTERM, an uncaught exception, or a hung front-end.
        atexit.register(self._safe_save_session, status="abandoned", skip_summary=True)

        try:
            self.agent.session_manager.prune_empty_sessions()
        except Exception:  # noqa: BLE001
            pass

        self.emitter.emit(Banner())
        self.emitter.emit(SystemNote(
            "type a message or /help. Ctrl+D to exit. Scroll and select text "
            "natively; /verbose toggles full tool output.",
        ))

        if shutil.which("genmedia") is None:
            self.emitter.emit(SystemNote(
                "genmedia CLI not found on $PATH. Install with:\n"
                "  curl -fsSL https://genmedia.sh/install | sh\n"
                "Until then, image/video generation via the genmedia "
                "skill won't work — local tools (browser, prompt, "
                "research) still do.",
                level="warn",
            ))

        if initial_message:
            self._on_submit(initial_message)

        try:
            await self._app.run_async()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Tear down the worker loop and persist a final save.

        Cancels any in-flight turn, posts the consumer-stop sentinel,
        stops the loop, joins the thread, drops in-flight asset
        downloads, and saves session metadata (URLs survive in the DB;
        resume retries anything missing)."""
        if self._agent_running:
            self.cancel_current_turn()
        self._submit(None)
        try:
            self._worker_loop_future.result(timeout=2.0)
        except FutureTimeout:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._agent_thread.join(timeout=2.0)
        try:
            self.agent._asset_dl_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        self._safe_save_session(status="completed", download_assets=False, skip_summary=True)
